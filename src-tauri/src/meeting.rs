//! Live meeting recording: capture mic ("me") + system audio ("them") and
//! transcribe them *as the meeting happens*, rather than all at once when it
//! ends. Segments are persisted and emitted live, so when the meeting stops the
//! transcript is already complete and only the summary remains.
//!
//! Chunking is driven by voice-activity detection (Silero VAD), not a fixed
//! clock: each channel's audio is cut on natural pauses, so words are never
//! split at an arbitrary window boundary and silence is skipped for free.

use crate::store::{self, Db, Segment};
use crate::{audio, models, settings, stt, AppState};
use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};

/// How often the loop pulls new audio and lets the VADs emit finished speech
/// segments. Shorter than the old fixed window → more responsive live captions.
const POLL_MS: u64 = 1000;

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

/// Begin recording + live transcription. Returns once capture is running.
pub fn start(
    app: &AppHandle,
    meeting_id: String,
    stt_path: PathBuf,
    vad_path: PathBuf,
    language: Option<String>,
) -> Result<(), String> {
    let st = app.state::<AppState>();
    let mut active = st.meeting.active.lock().unwrap();
    if active.is_some() {
        return Err("a meeting is already recording".into());
    }

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
            stt_path,
            vad_path,
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

/// Stop the loop (final VAD flush) then tear down capture.
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

fn make_vad(vad_path: &PathBuf) -> Option<SileroVad> {
    let config = SileroVadConfig {
        model: vad_path.to_string_lossy().to_string(),
        // End a segment after 0.5s of silence; ignore <0.25s blips; cap a single
        // segment at 20s so a long monologue is still fed to Whisper in pieces.
        min_silence_duration: 0.5,
        min_speech_duration: 0.25,
        max_speech_duration: 20.0,
        threshold: 0.5,
        sample_rate: audio::WHISPER_SAMPLE_RATE,
        window_size: 512,
        provider: None,
        num_threads: Some(1),
        debug: false,
    };
    match SileroVad::new(config, 30.0) {
        Ok(v) => Some(v),
        Err(e) => {
            eprintln!("[meeting] failed to init VAD: {e}");
            None
        }
    }
}

fn run_loop(
    app: AppHandle,
    meeting_id: String,
    stt_path: PathBuf,
    vad_path: PathBuf,
    language: Option<String>,
    initial_prompt: Option<String>,
    stop: Arc<AtomicBool>,
) {
    let mut mic_vad = make_vad(&vad_path);
    #[cfg(target_os = "macos")]
    let mut sys_vad = make_vad(&vad_path);

    loop {
        let mut waited = 0;
        while waited < POLL_MS / 100 && !stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(100));
            waited += 1;
        }
        let stopping = stop.load(Ordering::Relaxed);

        {
            let st = app.state::<AppState>();
            let mic = audio::drain_16k(&st.audio);
            if let Some(v) = mic_vad.as_mut() {
                if !mic.is_empty() {
                    v.accept_waveform(mic);
                }
            }
            #[cfg(target_os = "macos")]
            {
                let sys = crate::syscapture::drain(&st.sys);
                if let Some(v) = sys_vad.as_mut() {
                    if !sys.is_empty() {
                        v.accept_waveform(sys);
                    }
                }
            }
        }

        if stopping {
            if let Some(v) = mic_vad.as_mut() {
                v.flush();
            }
            #[cfg(target_os = "macos")]
            if let Some(v) = sys_vad.as_mut() {
                v.flush();
            }
        }

        if let Some(v) = mic_vad.as_mut() {
            drain_vad(
                &app,
                &meeting_id,
                &stt_path,
                language.as_deref(),
                initial_prompt.as_deref(),
                v,
                "me",
                "mic",
            );
        }
        #[cfg(target_os = "macos")]
        if let Some(v) = sys_vad.as_mut() {
            drain_vad(
                &app,
                &meeting_id,
                &stt_path,
                language.as_deref(),
                initial_prompt.as_deref(),
                v,
                "them",
                "system",
            );
        }

        if stopping {
            break;
        }
    }
}

/// Transcribe every finished speech segment the VAD has queued and emit it.
#[allow(clippy::too_many_arguments)]
fn drain_vad(
    app: &AppHandle,
    meeting_id: &str,
    stt_path: &PathBuf,
    language: Option<&str>,
    initial_prompt: Option<&str>,
    vad: &mut SileroVad,
    speaker: &str,
    source: &str,
) {
    while !vad.is_empty() {
        let seg = vad.front();
        vad.pop();
        // seg.start is the sample index into everything fed to this VAD, i.e.
        // the position on the meeting timeline; convert to milliseconds.
        let base_ms = seg.start as i64 * 1000 / audio::WHISPER_SAMPLE_RATE as i64;

        let st = app.state::<AppState>();
        let parts = match stt::transcribe_segments(
            &st.stt,
            stt_path,
            &seg.samples,
            language,
            initial_prompt,
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[meeting] segment transcription failed: {e}");
                continue;
            }
        };
        if parts.is_empty() {
            continue;
        }
        let segs: Vec<Segment> = parts
            .into_iter()
            .map(|(s, e, text)| Segment {
                id: 0,
                speaker: speaker.to_string(),
                source: source.to_string(),
                start_ms: base_ms + s,
                end_ms: base_ms + e,
                text,
            })
            .collect();
        let db = app.state::<Db>();
        if let Err(e) = store::add_segments(&db, meeting_id, &segs) {
            eprintln!("[meeting] failed to persist segments: {e}");
        }
        let _ = app.emit("meeting-segments", &segs);
    }
}

/// Resolve model paths and start a meeting (used by the command layer).
pub fn start_with_model(
    app: &AppHandle,
    meeting_id: String,
    model_id: String,
    language: Option<String>,
) -> Result<(), String> {
    let stt_path = models::downloaded_model_path(app, &model_id)?;
    let vad_path = models::downloaded_model_path(app, "vad-silero")?;
    start(app, meeting_id, stt_path, vad_path, language)
}
