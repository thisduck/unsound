//! Live meeting recording: capture mic ("me") + system audio ("them") and
//! transcribe them *as the meeting happens*, in rolling windows, rather than
//! all at once when it ends. Segments are persisted and emitted live, so when
//! the meeting stops the transcript is already complete and only the summary
//! remains — which is what makes ending feel instant.

use crate::store::{self, Db, Segment};
use crate::{audio, models, settings, stt, AppState};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};

/// How often the loop pulls new audio and transcribes it. A window long enough
/// for Whisper to have context, short enough to feel live.
const WINDOW_SECS: u64 = 8;

#[derive(Default)]
pub struct MeetingState {
    active: Mutex<Option<Handle>>,
    /// Whether a meeting is currently recording — drives the tray label.
    pub recording: Mutex<bool>,
}

struct Handle {
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

/// Begin recording + live transcription. Returns once capture is running; the
/// transcription loop continues on its own thread until `stop`.
pub fn start(
    app: &AppHandle,
    meeting_id: String,
    model_path: PathBuf,
    language: Option<String>,
) -> Result<(), String> {
    let st = app.state::<AppState>();
    let mut active = st.meeting.active.lock().unwrap();
    if active.is_some() {
        return Err("a meeting is already recording".into());
    }

    // Same recognition-biasing vocabulary as the dictation/file paths.
    let vocab: Vec<String> = {
        let mut seen = std::collections::HashSet::new();
        settings::load(app)
            .dictionary
            .iter()
            .map(|e| e.to.trim().to_string())
            .filter(|t| !t.is_empty() && seen.insert(t.to_lowercase()))
            .collect()
    };
    let initial_prompt = (!vocab.is_empty()).then(|| format!("Glossary: {}.", vocab.join(", ")));

    // Start both channels. The mic is required; system audio is best-effort so
    // a meeting still works (mic-only) if it's unsupported or denied.
    let mic = settings::load(app).mic_device;
    audio::start_recording(app.clone(), &st.audio, mic)?;
    #[cfg(target_os = "macos")]
    if let Err(e) = crate::syscapture::start(&st.sys) {
        eprintln!("[meeting] system audio unavailable, mic only: {e}");
    }

    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = stop.clone();
    let app_thread = app.clone();
    let join = std::thread::spawn(move || {
        run_loop(
            app_thread,
            meeting_id,
            model_path,
            language,
            initial_prompt,
            stop_thread,
        );
    });

    *active = Some(Handle {
        stop,
        join: Some(join),
    });
    *st.meeting.recording.lock().unwrap() = true;
    crate::tray::refresh_menu(app);
    Ok(())
}

/// Stop the loop (which does a final flush of the tail audio), then tear down
/// capture. Blocks until the loop's in-flight transcription finishes.
pub fn stop(app: &AppHandle) -> Result<(), String> {
    let st = app.state::<AppState>();
    let handle = st.meeting.active.lock().unwrap().take();
    if let Some(mut h) = handle {
        h.stop.store(true, Ordering::Relaxed);
        if let Some(j) = h.join.take() {
            let _ = j.join();
        }
    }
    let _ = audio::stop_recording(&st.audio);
    #[cfg(target_os = "macos")]
    {
        let _ = crate::syscapture::stop(&st.sys);
    }
    *st.meeting.recording.lock().unwrap() = false;
    crate::tray::refresh_menu(app);
    Ok(())
}

pub fn is_recording(app: &AppHandle) -> bool {
    *app.state::<AppState>().meeting.recording.lock().unwrap()
}

fn run_loop(
    app: AppHandle,
    meeting_id: String,
    model_path: PathBuf,
    language: Option<String>,
    initial_prompt: Option<String>,
    stop: Arc<AtomicBool>,
) {
    let st = app.state::<AppState>();
    let db = app.state::<Db>();
    // Per-channel running offset (ms) so each window's relative timestamps land
    // at the right place on the meeting timeline.
    let mut mic_off_ms = 0i64;
    #[cfg(target_os = "macos")]
    let mut sys_off_ms = 0i64;

    loop {
        // Wait ~WINDOW_SECS, but wake promptly on stop.
        let mut waited = 0;
        while waited < WINDOW_SECS * 2 && !stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(500));
            waited += 1;
        }
        let stopping = stop.load(Ordering::Relaxed);

        let mic = audio::drain_16k(&st.audio);
        process_window(
            &app,
            &st,
            &db,
            &model_path,
            &meeting_id,
            "me",
            "mic",
            &mic,
            language.as_deref(),
            initial_prompt.as_deref(),
            &mut mic_off_ms,
        );

        #[cfg(target_os = "macos")]
        {
            let sys = crate::syscapture::drain(&st.sys);
            process_window(
                &app,
                &st,
                &db,
                &model_path,
                &meeting_id,
                "them",
                "system",
                &sys,
                language.as_deref(),
                initial_prompt.as_deref(),
                &mut sys_off_ms,
            );
        }

        if stopping {
            break;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn process_window(
    app: &AppHandle,
    st: &AppState,
    db: &Db,
    model_path: &PathBuf,
    meeting_id: &str,
    speaker: &str,
    source: &str,
    samples: &[f32],
    language: Option<&str>,
    initial_prompt: Option<&str>,
    off_ms: &mut i64,
) {
    if samples.is_empty() {
        return;
    }
    let window_ms = samples.len() as i64 * 1000 / audio::WHISPER_SAMPLE_RATE as i64;
    let start_ms = *off_ms;
    *off_ms += window_ms;

    let parts = match stt::transcribe_segments(&st.stt, model_path, samples, language, initial_prompt)
    {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[meeting] window transcription failed: {e}");
            return;
        }
    };
    if parts.is_empty() {
        return;
    }
    let segs: Vec<Segment> = parts
        .into_iter()
        .map(|(s, e, text)| Segment {
            id: 0,
            speaker: speaker.to_string(),
            source: source.to_string(),
            start_ms: start_ms + s,
            end_ms: start_ms + e,
            text,
        })
        .collect();
    if let Err(e) = store::add_segments(db, meeting_id, &segs) {
        eprintln!("[meeting] failed to persist segments: {e}");
    }
    let _ = app.emit("meeting-segments", &segs);
}

/// Resolve the model path and start a meeting (used by the command layer).
pub fn start_with_model(
    app: &AppHandle,
    meeting_id: String,
    model_id: String,
    language: Option<String>,
) -> Result<(), String> {
    let model_path = models::downloaded_model_path(app, &model_id)?;
    start(app, meeting_id, model_path, language)
}
