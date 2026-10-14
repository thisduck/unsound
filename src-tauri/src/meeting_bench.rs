//! Native pipeline benchmark; PCM injection is synthetic, inference is real.
use crate::{audio_store, audiofile, meeting, meeting_fixture, store, AppState};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};

pub fn run(app: &AppHandle, case: &str, id: &str) -> Result<(), String> {
    let minutes = std::env::var("UNSOUND_BENCH_MINUTES")
        .unwrap_or("60".into())
        .parse::<u64>()
        .map_err(|e| e.to_string())?;
    let speed = std::env::var("UNSOUND_BENCH_SPEED")
        .unwrap_or("20".into())
        .parse::<u64>()
        .map_err(|e| e.to_string())?;
    if !(1..=90).contains(&minutes) || !(1..=20).contains(&speed) {
        return Err("minutes must be 1–90; speed must be 1–20".into());
    }
    let expected = minutes * 60 * 16_000;
    let started = Instant::now();
    if case == "soak" {
        let model =
            std::env::var("UNSOUND_BENCH_MODEL").map_err(|_| "set a local Whisper model path")?;
        let wav = std::env::var("UNSOUND_BENCH_AUDIO").map_err(|_| "set a PCM speech clip path")?;
        if !std::path::Path::new(&model).is_file() {
            return Err("benchmark model is not a file".into());
        }
        // Keep input preparation bounded too. Require a short clip, never load a meeting.
        if std::fs::metadata(&wav).map_err(|e| e.to_string())?.len() > 10_000_000 {
            return Err("use a speech clip smaller than 10 MB".into());
        }
        let seed = audiofile::decode_to_samples(std::path::Path::new(&wav))?;
        if seed.len() < 16_000 || seed.len() > 60 * 16_000 {
            return Err("use a 1–60 second speech clip".into());
        }
        store::create_meeting(&app.state::<store::Db>(),&serde_json::from_value(serde_json::json!({"id":id,"title":"Performance fixture","startedAt":crate::util::now_iso()})).map_err(|e|e.to_string())?)?;
        meeting::start_benchmark(app, id, &model)?;
        let source = meeting_fixture::get(app, id);
        let mut sent = 0u64;
        while sent < expected {
            let n = (speed * 16_000).min(expected - sent) as usize;
            let pcm: Vec<_> = (0..n)
                .map(|i| seed[(sent as usize + i) % seed.len()])
                .collect();
            source.feed_pcm(pcm.clone(), pcm)?;
            sent += n as u64;
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                let row = meeting::statuses(app)?
                    .into_iter()
                    .find(|r| r.meeting_id == id)
                    .ok_or("missing benchmark work")?;
                if row.state == "error" {
                    return Err(row.error);
                }
                if row.mic_samples >= sent as i64 && row.system_samples >= sent as i64 {
                    break;
                }
                if Instant::now() > deadline {
                    return Err("capture fell behind the bounded benchmark source".into());
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            // Capture polls once per second. Speed=1 is wall-clock playback;
            // higher values load the same audio timeline faster.
        }
        meeting::stop(app)?;
        let before = app.state::<AppState>().meeting.metrics.report(id);
        println!(
            "MEETING_BENCH_STOP {}",
            serde_json::to_string(&before).map_err(|e| e.to_string())?
        );
        if std::env::var("UNSOUND_BENCH_RESTART").ok().as_deref() == Some("1") {
            println!("MEETING_BENCH_RESTART_READY");
            loop {
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
    let deadline = Instant::now() + Duration::from_secs(15 * 60);
    loop {
        let row = meeting::statuses(app)?
            .into_iter()
            .find(|r| r.meeting_id == id)
            .ok_or("missing benchmark work")?;
        if row.state == "error" {
            return Err(row.error);
        }
        if row.state == "complete" {
            break;
        }
        if Instant::now() > deadline {
            return Err("transcription did not drain within 15 minutes".into());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    for path in [
        audio_store::mic_path(app, id)?,
        audio_store::system_path(app, id)?,
    ] {
        if audio_store::sample_count(&path)? != expected as i64 {
            return Err("benchmark lost saved audio samples".into());
        }
    }
    let m =
        store::get_meeting(&app.state::<store::Db>(), id)?.ok_or("missing benchmark meeting")?;
    if m.segments.is_empty() {
        return Err("benchmark produced no speech; use a spoken audio clip".into());
    }
    let metrics = app.state::<AppState>().meeting.metrics.report(id);
    println!(
        "MEETING_BENCH_REPORT {}",
        serde_json::json!({"minutesPerChannel":minutes,"channels":2,"speed":speed,"wallSeconds":started.elapsed().as_secs_f64(),"transcriptRows":m.segments.len(),"metrics":metrics,"recovered":case=="soak-recover"})
    );
    // Content-free report remains; disposable speech and transcript fixtures do not.
    meeting_fixture::get(app, id)
        .system_audio
        .store(false, Ordering::SeqCst);
    store::delete_meeting(&app.state::<store::Db>(), id)?;
    audio_store::delete(app, id);
    Ok(())
}
