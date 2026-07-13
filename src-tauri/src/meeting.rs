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
use crate::{asr, audio, models, settings, AppState};
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
const MAX_QUEUED_JOBS: usize = 24;
const CLOUD_TRANSCRIBE_WORKERS: usize = 2;
const CLOUD_TRANSCRIBE_ATTEMPTS: usize = 3;

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
    transcribe: Vec<JoinHandle<()>>,
    failures: Arc<Mutex<Vec<String>>>,
}

/// Begin recording + live transcription. Returns once capture is running.
pub fn start(
    app: &AppHandle,
    meeting_id: String,
    model_id: String,
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
    let cloud = model_id == "cloud";
    // Snapshot provider selection and the key once at start. A meeting uses a
    // stable configuration even if Settings changes while it is recording.
    let cloud_session = cloud
        .then(|| crate::remote::VoiceSession::new(settings::load(app)))
        .transpose()?;

    let mic = settings::load(app).mic_device;
    audio::start_recording(app.clone(), &st.audio, mic)?;
    #[cfg(target_os = "macos")]
    if let Err(e) = crate::syscapture::start(&st.sys) {
        eprintln!("[meeting] system audio unavailable, mic only: {e}");
    }

    let stop = Arc::new(AtomicBool::new(false));
    let failures = Arc::new(Mutex::new(Vec::new()));
    // Outstanding jobs in the queue; used to throttle partials so they only go
    // out when the transcriber is idle (never let partials back up the queue).
    let depth = Arc::new(AtomicI64::new(0));
    let (tx, rx) = mpsc::sync_channel::<Job>(MAX_QUEUED_JOBS);
    let rx = Arc::new(Mutex::new(rx));

    // Capture thread (producer): audio → VAD → queue.
    let capture = {
        let app = app.clone();
        let stop = stop.clone();
        let depth = depth.clone();
        let failures = failures.clone();
        std::thread::spawn(move || run_capture(app, vad_path, stop, tx, depth, failures))
    };
    // Cloud jobs use a small worker pool to keep up with both audio channels.
    // Local engines remain single-threaded because recognizers are !Send.
    let worker_count = if cloud { CLOUD_TRANSCRIBE_WORKERS } else { 1 };
    let mut transcribe = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let app = app.clone();
        let rx = rx.clone();
        let depth = depth.clone();
        let failures = failures.clone();
        let cloud_session = cloud_session.clone();
        transcribe.push(std::thread::spawn({
            let meeting_id = meeting_id.clone();
            let model_id = model_id.clone();
            let language = language.clone();
            let initial_prompt = initial_prompt.clone();
            move || {
                run_transcribe(
                    app,
                    meeting_id,
                    model_id,
                    language,
                    initial_prompt,
                    cloud_session,
                    rx,
                    depth,
                    failures,
                )
            }
        }));
    }

    *active = Some(Handle {
        stop,
        capture: Some(capture),
        transcribe,
        failures,
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
        for j in h.transcribe.drain(..) {
            if j.join().is_err() {
                record_failure(
                    app,
                    &h.failures,
                    "meeting transcription worker stopped unexpectedly".into(),
                );
            }
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

// ---- capture thread (producer) ---------------------------------------------

fn run_capture(
    app: AppHandle,
    vad_path: PathBuf,
    stop: Arc<AtomicBool>,
    tx: mpsc::SyncSender<Job>,
    depth: Arc<AtomicI64>,
    failures: Arc<Mutex<Vec<String>>>,
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
            handle_channel(
                &app,
                &tx,
                &depth,
                &failures,
                v,
                &mut mic_pending,
                mic,
                stopping,
                "me",
                "mic",
            );
        }
        #[cfg(target_os = "macos")]
        {
            let sys = crate::syscapture::drain(&app.state::<AppState>().sys);
            if let Some(v) = sys_vad.as_mut() {
                handle_channel(
                    &app,
                    &tx,
                    &depth,
                    &failures,
                    v,
                    &mut sys_pending,
                    sys,
                    stopping,
                    "them",
                    "system",
                );
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
    app: &AppHandle,
    tx: &mpsc::SyncSender<Job>,
    depth: &Arc<AtomicI64>,
    failures: &Arc<Mutex<Vec<String>>>,
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
        let job = Job::Final {
            samples: seg.samples,
            base_ms,
            speaker,
            source,
        };
        match tx.try_send(job) {
            Ok(()) => {
                depth.fetch_add(1, Ordering::Relaxed);
                had_final = true;
            }
            Err(mpsc::TrySendError::Full(_)) => record_failure(
                app,
                failures,
                "cloud transcription queue is full; a finalized audio segment was not processed"
                    .into(),
            ),
            Err(mpsc::TrySendError::Disconnected(_)) => break,
        }
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
        if tx
            .try_send(Job::Partial {
            samples: pending.clone(),
            speaker,
            })
            .is_ok()
        {
            depth.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// ---- transcription thread (consumer) ---------------------------------------

fn run_transcribe(
    app: AppHandle,
    meeting_id: String,
    model_id: String,
    language: Option<String>,
    initial_prompt: Option<String>,
    cloud_session: Option<crate::remote::VoiceSession>,
    rx: Arc<Mutex<mpsc::Receiver<Job>>>,
    depth: Arc<AtomicI64>,
    failures: Arc<Mutex<Vec<String>>>,
) {
    // Build the engine on this thread (sherpa recognizers are !Send).
    let cloud = model_id == "cloud";
    let mut engine = if cloud {
        asr::Engine::None
    } else {
        asr::resolve(&app, &model_id).unwrap_or_else(|e| {
            record_failure(&app, &failures, format!("speech engine unavailable: {e}"));
            asr::Engine::None
        })
    };
    // Warm the model up front so the first segment of the meeting isn't slow.
    // Moonshine/Parakeet already loaded when constructed; whisper is lazy.
    if let asr::Engine::Whisper(path) = &engine {
        if let Err(e) = app.state::<AppState>().stt.warmup(path) {
            eprintln!("[meeting] warmup failed: {e}");
        }
    }
    let lang = language.as_deref();
    let prompt = initial_prompt.as_deref();
    loop {
        let job = match rx.lock().unwrap().recv() {
            Ok(job) => job,
            Err(_) => break,
        };
        match job {
            Job::Final {
                samples,
                base_ms,
                speaker,
                source,
            } => {
                let parts = if cloud {
                    match cloud_session.as_ref() {
                        Some(session) => match transcribe_cloud(session, &samples, lang) {
                            Ok(text) if !text.is_empty() => vec![(
                                0,
                                samples.len() as i64
                                    * 1000
                                    / audio::WHISPER_SAMPLE_RATE as i64,
                                text,
                            )],
                            Ok(_) => vec![],
                            Err(e) => {
                                record_failure(
                                    &app,
                                    &failures,
                                    format!("cloud transcription failed after retries: {e}"),
                                );
                                vec![]
                            }
                        },
                        None => {
                            record_failure(
                                &app,
                                &failures,
                                "cloud voice session was not initialized".into(),
                            );
                            vec![]
                        }
                    }
                } else {
                    asr::transcribe_parts(&app, &mut engine, &samples, lang, prompt)
                };
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
                // Request-based cloud APIs do not provide a useful partial here;
                // finalized VAD segments arrive shortly after each pause.
                let text = if cloud {
                    String::new()
                } else {
                    asr::transcribe_parts(&app, &mut engine, &samples, lang, prompt)
                    .into_iter()
                    .map(|(_, _, t)| t)
                    .collect::<Vec<_>>()
                    .join(" ")
                };
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

fn transcribe_cloud(
    session: &crate::remote::VoiceSession,
    samples: &[f32],
    language: Option<&str>,
) -> Result<String, String> {
    let mut last_error = String::new();
    for attempt in 1..=CLOUD_TRANSCRIBE_ATTEMPTS {
        match tauri::async_runtime::block_on(crate::remote::transcribe(session, samples, language)) {
            Ok(text) => return Ok(text),
            Err(error) => {
                last_error = error;
                if attempt < CLOUD_TRANSCRIBE_ATTEMPTS {
                    std::thread::sleep(Duration::from_millis(500 * attempt as u64));
                }
            }
        }
    }
    Err(last_error)
}

fn record_failure(app: &AppHandle, failures: &Arc<Mutex<Vec<String>>>, message: String) {
    eprintln!("[meeting] {message}");
    let mut recorded = failures.lock().unwrap();
    if !recorded.contains(&message) {
        recorded.push(message.clone());
        let _ = app.emit("meeting-transcription-error", &message);
    }
}

/// Validate models are present and start a meeting (used by the command layer).
pub fn start_with_model(
    app: &AppHandle,
    meeting_id: String,
    model_id: String,
    language: Option<String>,
) -> Result<(), String> {
    let vad_path = models::downloaded_model_path(app, "vad-silero")?;
    // Surface a missing speech model early; the engine itself is built on the
    // transcription thread (sherpa recognizers are !Send).
    if model_id != "cloud" {
        models::downloaded_model_path(app, &model_id)?;
    } else if !crate::remote::voice_ready(&settings::load(app)) {
        return Err("configure a cloud voice provider and API key in Settings first".into());
    }
    start(app, meeting_id, model_id, vad_path, language)
}
