//! Live meeting recording: capture mic ("me") + system audio ("them") and
//! transcribe them *as the meeting happens*, rather than all at once when it
//! ends. Segments are persisted and emitted live, so when the meeting stops the
//! transcript is already complete and only the summary remains.
//!
//! The pipeline is split across two threads connected by a queue:
//!   • the **capture** thread drains audio, runs Silero VAD, and pushes each
//!     finished speech segment onto the queue (cutting on natural pauses so
//!     words are never split mid-word);
//!   • the **transcription** thread drains the queue and transcribes.
//! Decoupling them means a slow transcription never stalls capture/VAD.

use crate::store::{self, Db, Segment};
use crate::{asr, audio, models, settings, stt, AppState};
use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};

/// How often the capture thread pulls new audio and lets the VADs emit finished
/// speech segments.
const POLL_MS: u64 = 1000;

/// Which speech model a meeting uses, resolved from the chosen model id.
pub enum AsrChoice {
    /// whisper.cpp model file.
    Whisper(PathBuf),
    /// Moonshine (sherpa-onnx) bundle directory.
    Moonshine(PathBuf),
    /// Parakeet TDT (sherpa-onnx) bundle directory.
    Parakeet(PathBuf),
}

/// The live transcription engine, constructed on the transcription thread
/// (Moonshine's recognizer is !Send, so it can't be built ahead of time).
enum Engine {
    Whisper(PathBuf),
    Moonshine(asr::Moonshine),
    Parakeet(asr::Parakeet),
    /// Engine failed to initialize — record audio but produce no transcript.
    None,
}

/// A unit of audio to transcribe: a finalized segment (persisted + shown as a
/// real line) or an in-progress partial (shown tentatively, not persisted).
enum Job {
    Final {
        samples: Vec<f32>,
        base_ms: i64,
        speaker: &'static str,
        source: &'static str,
    },
    Partial {
        samples: Vec<f32>,
        speaker: &'static str,
    },
}

/// Minimum in-progress audio before we bother emitting a partial (1s).
const PARTIAL_MIN_SAMPLES: usize = audio::WHISPER_SAMPLE_RATE as usize;

#[derive(Default)]
pub struct MeetingState {
    active: Mutex<Option<Handle>>,
    /// Whether a meeting is currently recording — drives the tray label.
    pub recording: Mutex<bool>,
}

struct Handle {
    stop: Arc<AtomicBool>,
    capture: Option<JoinHandle<()>>,
    transcribe: Option<JoinHandle<()>>,
}

/// Begin recording + live transcription. Returns once capture is running.
pub fn start(
    app: &AppHandle,
    meeting_id: String,
    asr_choice: AsrChoice,
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
    // Outstanding jobs in the queue; used to throttle partials so they only go
    // out when the transcriber is idle (never let partials back up the queue).
    let depth = Arc::new(AtomicI64::new(0));
    let (tx, rx) = mpsc::channel::<Job>();

    // Capture thread (producer): audio → VAD → queue.
    let capture = {
        let app = app.clone();
        let stop = stop.clone();
        let depth = depth.clone();
        std::thread::spawn(move || run_capture(app, vad_path, stop, tx, depth))
    };
    // Transcription thread (consumer): queue → text → persist + emit.
    let transcribe = {
        let app = app.clone();
        std::thread::spawn(move || {
            run_transcribe(app, meeting_id, asr_choice, language, initial_prompt, rx, depth)
        })
    };

    *active = Some(Handle {
        stop,
        capture: Some(capture),
        transcribe: Some(transcribe),
    });
    *st.meeting.recording.lock().unwrap() = true;
    crate::tray::refresh_menu(app);
    Ok(())
}

/// Stop capture (final VAD flush), then wait for the queue to drain so the
/// transcript is complete before we return (and summarize).
pub fn stop(app: &AppHandle) -> Result<(), String> {
    let st = app.state::<AppState>();
    let handle = st.meeting.active.lock().unwrap().take();
    if let Some(mut h) = handle {
        h.stop.store(true, Ordering::Relaxed);
        // Capture finishes first (final flush + drops the queue sender)…
        if let Some(j) = h.capture.take() {
            let _ = j.join();
        }
        // …then the transcription thread drains what's left and exits.
        if let Some(j) = h.transcribe.take() {
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
        // segment at 20s so a long monologue is still fed to the model in pieces.
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

fn make_engine(choice: AsrChoice) -> Engine {
    match choice {
        AsrChoice::Whisper(p) => Engine::Whisper(p),
        AsrChoice::Moonshine(dir) => match asr::Moonshine::new(&dir) {
            Ok(m) => Engine::Moonshine(m),
            Err(e) => {
                eprintln!("[meeting] Moonshine init failed: {e}");
                Engine::None
            }
        },
        AsrChoice::Parakeet(dir) => match asr::Parakeet::new(&dir) {
            Ok(p) => Engine::Parakeet(p),
            Err(e) => {
                eprintln!("[meeting] Parakeet init failed: {e}");
                Engine::None
            }
        },
    }
}

// ---- capture thread (producer) ---------------------------------------------

fn run_capture(
    app: AppHandle,
    vad_path: PathBuf,
    stop: Arc<AtomicBool>,
    tx: mpsc::Sender<Job>,
    depth: Arc<AtomicI64>,
) {
    let mut mic_vad = make_vad(&vad_path);
    #[cfg(target_os = "macos")]
    let mut sys_vad = make_vad(&vad_path);
    // Audio accumulated since the last finalized segment, per channel — the
    // in-progress utterance used to produce partials.
    let mut mic_pending: Vec<f32> = Vec::new();
    #[cfg(target_os = "macos")]
    let mut sys_pending: Vec<f32> = Vec::new();

    loop {
        let mut waited = 0;
        while waited < POLL_MS / 100 && !stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(100));
            waited += 1;
        }
        let stopping = stop.load(Ordering::Relaxed);

        let mic = audio::drain_16k(&app.state::<AppState>().audio);
        if let Some(v) = mic_vad.as_mut() {
            handle_channel(&tx, &depth, v, &mut mic_pending, mic, stopping, "me", "mic");
        }
        #[cfg(target_os = "macos")]
        {
            let sys = crate::syscapture::drain(&app.state::<AppState>().sys);
            if let Some(v) = sys_vad.as_mut() {
                handle_channel(&tx, &depth, v, &mut sys_pending, sys, stopping, "them", "system");
            }
        }

        if stopping {
            break;
        }
    }
    // `tx` drops here, signaling the transcription thread to finish.
}

/// Feed a channel's new audio to its VAD, push any finished segments, and — when
/// the transcriber is idle and we're mid-utterance — push a partial.
#[allow(clippy::too_many_arguments)]
fn handle_channel(
    tx: &mpsc::Sender<Job>,
    depth: &Arc<AtomicI64>,
    vad: &mut SileroVad,
    pending: &mut Vec<f32>,
    new: Vec<f32>,
    stopping: bool,
    speaker: &'static str,
    source: &'static str,
) {
    if !new.is_empty() {
        pending.extend_from_slice(&new);
        vad.accept_waveform(new);
    }
    if stopping {
        vad.flush();
    }

    let mut had_final = false;
    while !vad.is_empty() {
        let seg = vad.front();
        vad.pop();
        // seg.start is the sample index into everything fed to this VAD, i.e.
        // the position on the meeting timeline; convert to milliseconds.
        let base_ms = seg.start as i64 * 1000 / audio::WHISPER_SAMPLE_RATE as i64;
        depth.fetch_add(1, Ordering::Relaxed);
        let _ = tx.send(Job::Final {
            samples: seg.samples,
            base_ms,
            speaker,
            source,
        });
        had_final = true;
    }

    if had_final {
        pending.clear();
        return;
    }

    // Only emit a partial when the transcriber is idle (keeps partials fresh and
    // never lets them pile up behind finals) and we're clearly mid-utterance.
    if depth.load(Ordering::Relaxed) == 0
        && pending.len() >= PARTIAL_MIN_SAMPLES
        && vad.is_speech()
    {
        depth.fetch_add(1, Ordering::Relaxed);
        let _ = tx.send(Job::Partial {
            samples: pending.clone(),
            speaker,
        });
    }
}

// ---- transcription thread (consumer) ---------------------------------------

fn run_transcribe(
    app: AppHandle,
    meeting_id: String,
    asr_choice: AsrChoice,
    language: Option<String>,
    initial_prompt: Option<String>,
    rx: mpsc::Receiver<Job>,
    depth: Arc<AtomicI64>,
) {
    let mut engine = make_engine(asr_choice);
    let lang = language.as_deref();
    let prompt = initial_prompt.as_deref();
    while let Ok(job) = rx.recv() {
        match job {
            Job::Final {
                samples,
                base_ms,
                speaker,
                source,
            } => {
                let parts = transcribe_parts(&app, &mut engine, lang, prompt, &samples);
                if !parts.is_empty() {
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
                    if let Err(e) = store::add_segments(&db, &meeting_id, &segs) {
                        eprintln!("[meeting] failed to persist segments: {e}");
                    }
                    let _ = app.emit("meeting-segments", &segs);
                }
            }
            Job::Partial { samples, speaker } => {
                let text = transcribe_parts(&app, &mut engine, lang, prompt, &samples)
                    .into_iter()
                    .map(|(_, _, t)| t)
                    .collect::<Vec<_>>()
                    .join(" ");
                if !text.trim().is_empty() {
                    let _ = app.emit(
                        "meeting-partial",
                        serde_json::json!({ "speaker": speaker, "text": text }),
                    );
                }
            }
        }
        depth.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Transcribe a chunk to `(start_ms, end_ms, text)` parts using the active engine.
fn transcribe_parts(
    app: &AppHandle,
    engine: &mut Engine,
    language: Option<&str>,
    initial_prompt: Option<&str>,
    samples: &[f32],
) -> Vec<(i64, i64, String)> {
    match engine {
        Engine::Whisper(path) => {
            let st = app.state::<AppState>();
            stt::transcribe_segments(&st.stt, path, samples, language, initial_prompt)
                .unwrap_or_else(|e| {
                    eprintln!("[meeting] transcription failed: {e}");
                    Vec::new()
                })
        }
        Engine::Moonshine(m) => text_part(m.transcribe(samples), samples.len()),
        Engine::Parakeet(p) => text_part(p.transcribe(samples), samples.len()),
        Engine::None => Vec::new(),
    }
}

/// Wrap a whole-chunk transcription (Moonshine/Parakeet return plain text) as a
/// single timestamped part spanning the chunk.
fn text_part(text: String, n_samples: usize) -> Vec<(i64, i64, String)> {
    if text.is_empty() {
        Vec::new()
    } else {
        let dur_ms = n_samples as i64 * 1000 / audio::WHISPER_SAMPLE_RATE as i64;
        vec![(0, dur_ms, text)]
    }
}

/// Resolve model paths and start a meeting (used by the command layer).
pub fn start_with_model(
    app: &AppHandle,
    meeting_id: String,
    model_id: String,
    language: Option<String>,
) -> Result<(), String> {
    let vad_path = models::downloaded_model_path(app, "vad-silero")?;
    let info = models::find_model(app, &model_id)?;
    let path = models::downloaded_model_path(app, &model_id)?;
    // Bundle (multi-file) speech models are sherpa-onnx; single files are whisper.
    let asr_choice = if info.files.is_empty() {
        AsrChoice::Whisper(path)
    } else if model_id.starts_with("parakeet") {
        AsrChoice::Parakeet(path)
    } else {
        AsrChoice::Moonshine(path)
    };
    start(app, meeting_id, asr_choice, vad_path, language)
}
