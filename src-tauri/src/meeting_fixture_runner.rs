//! Driver-free acceptance checks against the actual Tauri backend and scheduler.
//! The WebDriver suite separately checks IPC, rendering, and recovery controls.
use crate::{meeting, meeting_fixture as fixture, meeting_work, store, AppState};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};

type Result<T = ()> = std::result::Result<T, String>;

pub fn launch_if_requested(app: &AppHandle) {
    let args: Vec<_> = std::env::args().collect();
    let Some(pos) = args.iter().position(|a| a == "--meeting-fixture-scenario") else {
        return;
    };
    let app = app.clone();
    let case = args.get(pos + 1).cloned().unwrap_or_default();
    let id = args.get(pos + 2).cloned().unwrap_or_default();
    std::thread::spawn(move || {
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&app, &case, &id)))
                .unwrap_or_else(|_| Err("native scenario panicked".into()));
        match result {
            Ok(()) => {
                println!("MEETING_FIXTURE_PASS {case}");
                app.exit(0);
            }
            Err(error) => {
                eprintln!("MEETING_FIXTURE_FAIL {case}: {error}");
                app.exit(1);
            }
        }
    });
}

fn ensure(ok: bool, message: &str) -> Result {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn wait(mut check: impl FnMut() -> Result<bool>) -> Result {
    let deadline = Instant::now() + Duration::from_secs(12);
    while !check()? {
        if Instant::now() >= deadline {
            return Err("native condition timed out".into());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    Ok(())
}
fn control(app: &AppHandle, id: &str, action: fixture::Action) -> Result {
    fixture::meeting_fixture(app.clone(), id.into(), action).map(|_| ())
}
fn status(app: &AppHandle, id: &str) -> Result<meeting_work::Status> {
    meeting::statuses(app)?
        .into_iter()
        .find(|r| r.meeting_id == id)
        .ok_or("missing work status".into())
}
fn get(app: &AppHandle, id: &str) -> Result<store::Meeting> {
    store::get_meeting(&app.state::<store::Db>(), id)?.ok_or("missing meeting".into())
}
fn create(app: &AppHandle, id: &str) -> Result {
    store::create_meeting(
        &app.state::<store::Db>(),
        &serde_json::from_value(serde_json::json!({
            "id":id, "title":id, "startedAt":crate::util::now_iso()
        }))
        .map_err(|e| e.to_string())?,
    )
}
fn start(app: &AppHandle, id: &str) -> Result {
    meeting::start_with_options(
        app,
        meeting::StartOptions {
            meeting_id: id.into(),
            stt_model: format!("{}{id}", fixture::MODEL_PREFIX),
            language: Some("en".into()),
            llm_model: None,
            embedding_model: Some(String::new()),
        },
    )
}
fn feed(app: &AppHandle, id: &str, seconds: i64) -> Result {
    control(app, id, fixture::Action::Feed)?;
    wait(|| Ok(status(app, id)?.mic_samples == seconds * 16_000))
}
fn retry(app: &AppHandle, id: &str) -> Result {
    wait(|| match meeting::retry(app, id) {
        Ok(()) => Ok(true),
        Err(e) if e.contains("previous inference call") => Ok(false),
        Err(e) => Err(e),
    })
}
fn complete(app: &AppHandle, id: &str, count: usize) -> Result {
    wait(|| Ok(status(app, id)?.state == "complete"))?;
    let m = get(app, id)?;
    ensure(
        m.segments.len() == count,
        "missing or duplicate transcript rows",
    )?;
    for (i, s) in m.segments.iter().enumerate() {
        ensure(
            s.source == "mic"
                && s.start_ms == i as i64 * 1000
                && s.end_ms == (i as i64 + 1) * 1000
                && s.text == "Fixture speech survived.",
            "incorrect recovered transcript range or PCM",
        )?;
    }
    let n = crate::audio_store::sample_count(&crate::audio_store::mic_path(app, id)?)?;
    let pcm = crate::audio_store::read_samples(&crate::audio_store::mic_path(app, id)?, 0, n)?;
    ensure(
        pcm.iter().all(|s| (*s - 0.25).abs() < 0.0001),
        "saved PCM was corrupted",
    )?;
    ensure(
        n == count as i64 * 16_000 && status(app, id)?.pending == 0,
        "incorrect saved audio or pending count",
    )
}
fn remove(app: &AppHandle, id: &str) -> Result {
    control(app, id, fixture::Action::Clear)?;
    store::delete_meeting(&app.state::<store::Db>(), id)?;
    crate::audio_store::delete(app, id);
    Ok(())
}
fn in_flight(app: &AppHandle, id: &str) -> bool {
    fixture::get(app, id).in_flight.load(Ordering::SeqCst) == 1
}

fn run(app: &AppHandle, case: &str, base: &str) -> Result {
    // The launcher supplies a unique filename-safe ID. Never use a real profile.
    ensure(
        app.config().identifier == "com.unsound.e2e"
            && base.starts_with("e2e-")
            && base.len() < 64
            && base.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-'),
        "invalid isolated scenario",
    )?;
    match case {
        "soak" | "soak-recover" => return crate::meeting_bench::run(app,case,base),
        "recording-indicator" => {
            let window = app.get_webview_window("meeting-indicator").ok_or("missing indicator window")?;
            let visible = || window.is_visible().map_err(|e| e.to_string());
            ensure(!visible()? && meeting::recording_status(app).is_none(), "idle indicator is visible")?;
            create(app, base)?;
            start(app, base)?;
            wait(|| visible())?;
            wait(|| Ok(meeting::recording_status(app).is_some_and(|s| s.meeting_id == base && s.elapsed_ms >= 1000)))?;
            if let Some(main) = app.get_webview_window("main") {
                main.hide().map_err(|e| e.to_string())?;
            }
            ensure(visible()?, "hiding main hid the recording indicator")?;
            // Exercise the same command used by the indicator, including stale clicks.
            let stopped = tauri::async_runtime::block_on(crate::meeting_stop(app.clone(), Some(base.into())))?;
            ensure(stopped.is_some_and(|m| m.id == base), "indicator stop did not return the meeting")?;
            wait(|| Ok(!visible()? && meeting::recording_status(app).is_none()))?;

            let next = format!("{base}-next");
            create(app, &next)?;
            start(app, &next)?;
            wait(|| visible())?;
            ensure(tauri::async_runtime::block_on(crate::meeting_stop(app.clone(), Some(base.into())))?.is_none(), "stale indicator stopped a newer meeting")?;
            ensure(visible()? && meeting::is_recording(app), "stale stop hid the active indicator")?;
            control(app, &next, fixture::Action::Disconnect)?;
            wait(|| Ok(!visible()? && meeting::recording_status(app).is_none()))?;
            wait(|| Ok(!in_flight(app, base) && !in_flight(app, &next)))?;
            remove(app, base)?;
            remove(app, &next)?;
        }
        "failures" => {
            let id = format!("{base}-stall");
            create(app, &id)?;
            control(app, &id, fixture::Action::Stall)?;
            start(app, &id)?;
            feed(app, &id, 1)?;
            wait(|| Ok(in_flight(app, &id)))?;
            control(app, &id, fixture::Action::Watchdog)?;
            wait(|| Ok(status(app, &id)?.state == "error"))?;
            ensure(
                status(app, &id)?.pending == 1 && get(app, &id)?.segments.is_empty(),
                "stalled work was lost",
            )?;
            meeting::stop(app)?;
            control(app, &id, fixture::Action::Clear)?;
            retry(app, &id)?;
            complete(app, &id, 1)?;
            remove(app, &id)?;

            let first = format!("{base}-first");
            let second = format!("{base}-second");
            create(app, &first)?;
            control(app, &first, fixture::Action::StallUncancellable)?;
            start(app, &first)?;
            feed(app, &first, 1)?;
            wait(|| Ok(in_flight(app, &first)))?;
            let begin = Instant::now();
            meeting::stop(app)?;
            println!(
                "MEETING_FIXTURE_METRIC stop_ms={}",
                begin.elapsed().as_millis()
            );
            ensure(
                begin.elapsed() < Duration::from_millis(2500) && in_flight(app, &first),
                "stop waited for inference",
            )?;
            create(app, &second)?;
            start(app, &second)?;
            feed(app, &second, 1)?;
            ensure(
                fixture::get(app, &second).calls.load(Ordering::SeqCst) == 0,
                "stuck worker spawned a replacement",
            )?;
            meeting::pause(app, &first)?;
            control(app, &first, fixture::Action::Release)?;
            meeting::stop(app)?;
            complete(app, &second, 1)?;
            ensure(
                get(app, &first)?.segments.is_empty(),
                "late canceled result committed",
            )?;
            retry(app, &first)?;
            complete(app, &first, 1)?;
            remove(app, &first)?;
            remove(app, &second)?;

            for (suffix, fault) in [
                ("transcript", fixture::Action::FailTranscript),
                ("audio", fixture::Action::FailAudio),
                ("disconnect", fixture::Action::Disconnect),
            ] {
                let id = format!("{base}-{suffix}");
                create(app, &id)?;
                start(app, &id)?;
                feed(app, &id, 1)?;
                wait(|| Ok(get(app, &id)?.segments.len() == 1))?;
                control(app, &id, fault)?;
                if suffix != "disconnect" {
                    control(app, &id, fixture::Action::Feed)?;
                }
                wait(|| Ok(status(app, &id)?.state == "error"))?;
                if suffix == "transcript" {
                    meeting::stop(app)?;
                } else {
                    wait(|| Ok(!status(app, &id)?.recording))?;
                }
                let expected = if suffix == "audio" { 1 } else { 2 };
                ensure(
                    status(app, &id)?.mic_samples == expected * 16_000,
                    "last saved position is wrong",
                )?;
                ensure(
                    get(app, &id)?.segments.len() == 1,
                    "failed transcript write committed",
                )?;
                control(app, &id, fixture::Action::Clear)?;
                retry(app, &id)?;
                complete(app, &id, expected as usize)?;
                remove(app, &id)?;
            }
        }
        "crash-write" => {
            create(app, base)?;
            start(app, base)?;
            feed(app, base, 1)?;
            wait(|| Ok(get(app, base)?.segments.len() == 1))?;
            store::update_meeting_notes(
                &app.state::<store::Db>(),
                base,
                "Saved before the crash.",
            )?;
            control(app, base, fixture::Action::StallUncancellable)?;
            feed(app, base, 2)?;
            wait(|| Ok(in_flight(app, base)))?;
            println!("MEETING_FIXTURE_KILL_READY");
            // The launcher kills this process. No graceful stop or destructor runs.
            loop {
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        "crash-read" | "crash-cleanup" => {
            complete(app, base, 2)?;
            let m = get(app, base)?;
            ensure(
                m.notes == "Saved before the crash." && m.ended_at.is_some(),
                "notes or recovered metadata missing",
            )?;
            ensure(
                status(app, base)?
                    .warning
                    .contains("Recording was interrupted"),
                "interruption was hidden",
            )?;
            if case == "crash-cleanup" {
                remove(app, base)?;
            }
        }
        _ => return Err("unknown native scenario".into()),
    }
    ensure(
        !*app.state::<AppState>().meeting.recording.lock().unwrap(),
        "capture remained active",
    )
}
