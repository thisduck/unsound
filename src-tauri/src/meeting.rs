//! Live meeting recording: capture mic ("me") + system audio ("them") and
//! transcribe them *as the meeting happens*, so when it stops the transcript
//! is complete, the notes are mostly extracted, and only the final summary
//! remains.
//!
//! Three threads per meeting:
//!   • **capture** drains both channels, writes them to disk (so nothing is
//!     held in memory and a crash loses seconds, not the meeting), runs Silero
//!     VAD, and queues each finished speech segment — cutting on natural
//!     pauses so words are never split;
//!   • **transcription** drains the queue: spoken commands first, then
//!     segments (batched when it has fallen behind), labels remote speakers
//!     live from voice embeddings, applies the user's dictionary, and hands
//!     "unsound, …" requests to the assistant;
//!   • **notes** extracts each completed five-minute stretch in the
//!     background while the transcriber is idle (see summary.rs).
//! Both channels share one clock — the mic's — so "them" lines interleave
//! correctly with what they answer.

use crate::audio_store::{self, WavWriter};
use crate::store::{self, Db, Segment};
use crate::{agent, asr, audio, models, settings, speakers, stt, summary, AppState};
use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager};

/// How often the capture thread pulls new audio and lets the VADs emit
/// finished speech segments.
const POLL_MS: u64 = 1000;
const RATE: i64 = audio::WHISPER_SAMPLE_RATE as i64;
/// Minimum in-progress audio before we bother emitting a partial (1s).
const PARTIAL_MIN_SAMPLES: usize = audio::WHISPER_SAMPLE_RATE as usize;
/// When behind, consecutive segments of one channel are transcribed together
/// up to this long — one Whisper encoder pass instead of several (its encoder
/// costs the same for 3 s of audio as for 30 s).
const BATCH_MAX_SAMPLES: usize = 25 * audio::WHISPER_SAMPLE_RATE as usize;
/// Silence between batched segments, and before every segment (Whisper drops
/// words that start right at sample zero).
const GAP_SAMPLES: usize = 4_800;
/// After a bare "unsound", the next line within this long is the request.
const ARMED_MS: i64 = 12_000;

/// A unit of work for the transcription thread.
enum Job {
    /// A finished segment: persisted and shown as a real line.
    Final {
        samples: Vec<f32>,
        base_ms: i64,
        speaker: &'static str,
        source: &'static str,
    },
    /// The in-progress utterance, shown tentatively and not persisted.
    Partial { samples: Vec<f32>, speaker: &'static str },
    /// A request spoken while holding the command key.
    Command { samples: Vec<f32>, at_ms: i64 },
}

#[derive(Default)]
pub struct MeetingState {
    active: Mutex<Option<Handle>>,
    /// Whether a meeting is currently recording — drives the tray label.
    pub recording: Mutex<bool>,
}

struct Handle {
    meeting_id: String,
    stop: Arc<AtomicBool>,
    capture: Option<JoinHandle<()>>,
    transcribe: Option<JoinHandle<()>>,
    /// Outstanding jobs in the queue (throttles partials; "is it idle?").
    depth: Arc<AtomicI64>,
    tx: mpsc::Sender<Job>,
    /// Mic-timeline windows (ms) claimed by dictation or the command key,
    /// kept out of the transcript. `i64::MAX` end = still open.
    exclusions: Arc<Mutex<Vec<(i64, i64)>>>,
    /// The command key's tap on the mic: (tap id, start ms).
    command_tap: Option<(u64, i64)>,
}

/// Everything a meeting needs to start.
pub struct StartOptions {
    pub meeting_id: String,
    pub stt_model: String,
    pub language: Option<String>,
    /// Summary model for background note extraction (None: skip).
    pub llm_model: Option<String>,
    /// Speaker-embedding model for live speaker labels (None: default).
    pub embedding_model: Option<String>,
}

/// Validate models are present and start a meeting (used by the command layer).
pub fn start_with_options(app: &AppHandle, opts: StartOptions) -> Result<(), String> {
    let vad_path = models::downloaded_model_path(app, "vad-silero")?;
    // Surface a missing speech model early; the engine itself is built on the
    // transcription thread (sherpa recognizers are !Send).
    models::downloaded_model_path(app, &opts.stt_model)?;
    start(app, opts, vad_path)
}

fn start(app: &AppHandle, opts: StartOptions, vad_path: PathBuf) -> Result<(), String> {
    let st = app.state::<AppState>();
    let mut active = st.meeting.active.lock().unwrap();
    if active.is_some() {
        return Err("a meeting is already recording".into());
    }
    let s = settings::load(app);

    // Bias recognition toward the user's vocabulary and the assistant's name.
    let mut seen = std::collections::HashSet::new();
    let mut vocab: Vec<String> = Vec::new();
    if s.assistant.enabled && s.assistant.listen_for_wake_word && !s.assistant.wake_word.trim().is_empty() {
        vocab.push(s.assistant.wake_word.trim().to_string());
    }
    vocab.extend(s.dictionary.iter().map(|e| e.to.trim().to_string()));
    vocab.retain(|t| !t.is_empty() && seen.insert(t.to_lowercase()));
    let initial_prompt = (!vocab.is_empty()).then(|| format!("Glossary: {}.", vocab.join(", ")));

    let mic_file = audio_store::mic_path(app, &opts.meeting_id)?;
    let sys_file = audio_store::system_path(app, &opts.meeting_id)?;

    // The mic defines the meeting clock; system audio starts a moment later
    // and is offset onto it.
    audio::start_recording(app.clone(), &st.audio, s.mic_device.clone(), audio::Owner::Meeting)?;
    let mic_started = Instant::now();
    #[cfg(target_os = "macos")]
    let sys_ok = match crate::syscapture::start(&st.sys) {
        Ok(()) => true,
        Err(e) => {
            eprintln!("[meeting] system audio unavailable, mic only: {e}");
            let _ = app.emit(
                "meeting-warning",
                format!("System audio unavailable — recording your mic only ({e})."),
            );
            false
        }
    };
    #[cfg(not(target_os = "macos"))]
    let sys_ok = false;
    let sys_offset = (mic_started.elapsed().as_millis() as i64 * RATE / 1000) as u64;

    let stop = Arc::new(AtomicBool::new(false));
    let depth = Arc::new(AtomicI64::new(0));
    let exclusions: Arc<Mutex<Vec<(i64, i64)>>> = Arc::new(Mutex::new(Vec::new()));
    let (tx, rx) = mpsc::channel::<Job>();

    let capture = {
        let app = app.clone();
        let stop = stop.clone();
        let depth = depth.clone();
        let exclusions = exclusions.clone();
        let tx = tx.clone();
        std::thread::spawn(move || {
            run_capture(app, vad_path, stop, tx, depth, exclusions, mic_file, sys_file, sys_ok, sys_offset)
        })
    };
    let transcribe = {
        let app = app.clone();
        let depth = depth.clone();
        let meeting_id = opts.meeting_id.clone();
        let stt_model = opts.stt_model.clone();
        let language = opts.language.clone();
        let emb_model = opts
            .embedding_model
            .clone()
            .filter(|m| !m.is_empty())
            .or_else(|| Some(s.meetings.embedding_model.clone()).filter(|m| !m.is_empty()))
            .unwrap_or_else(|| "diarize-embedding".into());
        std::thread::spawn(move || {
            run_transcribe(app, meeting_id, stt_model, language, initial_prompt, emb_model, rx, depth)
        })
    };
    // Background notes: detached; it exits on `stop` and the final summary
    // waits for any extraction in flight (LLM turn).
    if let Some(llm) = opts.llm_model.clone().filter(|m| !m.is_empty()) {
        if let Ok(path) = models::downloaded_model_path(app, &llm) {
            let app = app.clone();
            let stop = stop.clone();
            let depth = depth.clone();
            let meeting_id = opts.meeting_id.clone();
            std::thread::spawn(move || run_notes(app, meeting_id, path, stop, depth));
        }
    }

    *active = Some(Handle {
        meeting_id: opts.meeting_id,
        stop,
        capture: Some(capture),
        transcribe: Some(transcribe),
        depth,
        tx,
        exclusions,
        command_tap: None,
    });
    *st.meeting.recording.lock().unwrap() = true;
    drop(active);
    crate::tray::refresh_menu(app);
    Ok(())
}

/// Stop capture (final VAD flush), wait for the queue to drain so the
/// transcript is complete, record the end time, and settle speaker labels.
/// Returns the meeting id that was recording.
pub fn stop(app: &AppHandle) -> Result<Option<String>, String> {
    let st = app.state::<AppState>();
    let handle = st.meeting.active.lock().unwrap().take();
    let mut id = None;
    if let Some(mut h) = handle {
        h.stop.store(true, Ordering::Relaxed);
        // Capture finishes first (final flush, stops the devices)…
        if let Some(j) = h.capture.take() {
            let _ = j.join();
        }
        // …then the transcription thread drains what's left and exits once
        // every sender is gone.
        drop(h.tx);
        if let Some(j) = h.transcribe.take() {
            let _ = j.join();
        }
        let db = app.state::<Db>();
        let _ = store::set_meeting_ended(&db, &h.meeting_id, &crate::util::now_iso());
        id = Some(h.meeting_id);
    }
    *st.meeting.recording.lock().unwrap() = false;
    crate::tray::refresh_menu(app);
    Ok(id)
}

pub fn is_recording(app: &AppHandle) -> bool {
    *app.state::<AppState>().meeting.recording.lock().unwrap()
}

/// The id of the meeting being recorded, if any.
pub fn active_meeting(app: &AppHandle) -> Option<String> {
    app.state::<AppState>()
        .meeting
        .active
        .lock()
        .unwrap()
        .as_ref()
        .map(|h| h.meeting_id.clone())
}

/// True when nothing is waiting to be transcribed (also true with no meeting).
pub fn transcriber_idle(app: &AppHandle) -> bool {
    app.state::<AppState>()
        .meeting
        .active
        .lock()
        .unwrap()
        .as_ref()
        .is_none_or(|h| h.depth.load(Ordering::Relaxed) <= 0)
}

/// Where the meeting is now, in ms on its clock.
pub fn position_ms(app: &AppHandle) -> Option<i64> {
    audio::position_ms(&app.state::<AppState>().audio)
}

/// Keep `[start_ms, …)` of the mic out of the transcript (dictation, the
/// command key). Returns false when no meeting is recording.
pub fn exclude_from(app: &AppHandle, start_ms: i64) -> bool {
    let st = app.state::<AppState>();
    let active = st.meeting.active.lock().unwrap();
    let Some(h) = active.as_ref() else { return false };
    h.exclusions.lock().unwrap().push((start_ms, i64::MAX));
    true
}

/// Close the exclusion window that started at `start_ms`.
pub fn exclude_until(app: &AppHandle, start_ms: i64, end_ms: i64) {
    let st = app.state::<AppState>();
    let active = st.meeting.active.lock().unwrap();
    if let Some(h) = active.as_ref() {
        for w in h.exclusions.lock().unwrap().iter_mut() {
            if w.0 == start_ms && w.1 == i64::MAX {
                w.1 = end_ms;
            }
        }
    }
}

/// Command key pressed during a meeting: claim the mic from now on.
pub fn command_begin(app: &AppHandle) -> Result<(), String> {
    let st = app.state::<AppState>();
    let (tap, start) = audio::tap_begin(&st.audio)?;
    {
        let mut active = st.meeting.active.lock().unwrap();
        let Some(h) = active.as_mut() else {
            drop(active);
            let _ = audio::tap_end(&st.audio, tap);
            return Err("no meeting is recording".into());
        };
        h.command_tap = Some((tap, start));
    }
    exclude_from(app, start);
    Ok(())
}

/// Command key released: queue what was said as a request (ahead of any
/// transcription backlog).
pub fn command_end(app: &AppHandle) -> Result<(), String> {
    let st = app.state::<AppState>();
    let (tap, start, tx, depth) = {
        let mut active = st.meeting.active.lock().unwrap();
        let h = active.as_mut().ok_or("no meeting is recording")?;
        let (tap, start) = h.command_tap.take().ok_or("the command key wasn't held")?;
        (tap, start, h.tx.clone(), h.depth.clone())
    };
    let (samples, end) = audio::tap_end(&st.audio, tap)?;
    exclude_until(app, start, end);
    if samples.len() < audio::WHISPER_SAMPLE_RATE as usize / 3 {
        return Err("that was too short to hear a request".into());
    }
    depth.fetch_add(1, Ordering::Relaxed);
    tx.send(Job::Command { samples, at_ms: end })
        .map_err(|_| "the meeting is ending".to_string())
}

pub fn command_active(app: &AppHandle) -> bool {
    app.state::<AppState>()
        .meeting
        .active
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|h| h.command_tap.is_some())
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

/// One channel's VAD plus where its fed audio sits on the meeting clock.
struct Channel {
    vad: Option<SileroVad>,
    /// Audio since the last finalized segment (the in-progress utterance).
    pending: Vec<f32>,
    /// Meeting-clock sample index of the first sample fed to this VAD.
    offset: u64,
    speaker: &'static str,
    source: &'static str,
}

#[allow(clippy::too_many_arguments)]
fn run_capture(
    app: AppHandle,
    vad_path: PathBuf,
    stop: Arc<AtomicBool>,
    tx: mpsc::Sender<Job>,
    depth: Arc<AtomicI64>,
    exclusions: Arc<Mutex<Vec<(i64, i64)>>>,
    mic_file: PathBuf,
    sys_file: PathBuf,
    #[allow(unused_mut)] mut sys_ok: bool,
    sys_offset: u64,
) {
    let st = app.state::<AppState>();
    let open = |p: &PathBuf| {
        WavWriter::create(p)
            .map_err(|e| eprintln!("[meeting] can't save audio to {p:?}: {e}"))
            .ok()
    };
    let mut mic_w = open(&mic_file);
    #[allow(unused_mut)]
    let mut sys_w = if sys_ok { open(&sys_file) } else { None };
    let mut mic = Channel {
        vad: make_vad(&vad_path),
        pending: Vec::new(),
        offset: 0,
        speaker: "me",
        source: "mic",
    };
    #[allow(unused_mut, unused_variables)]
    let mut sys = Channel {
        vad: make_vad(&vad_path),
        pending: Vec::new(),
        offset: sys_offset,
        speaker: "them",
        source: "system",
    };
    let mut mic_pos: u64 = 0;
    #[allow(unused_mut, unused_variables)]
    let mut sys_fed: u64 = 0;
    let mut last_status = Instant::now();

    loop {
        let mut waited = 0;
        while waited < POLL_MS / 100 && !stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(100));
            waited += 1;
        }
        let stopping = stop.load(Ordering::Relaxed);

        // Mic: on the final pass, stopping the device also returns the tail
        // still inside the resampler.
        let new_mic = if stopping {
            audio::stop_meeting_recording(&st.audio)
        } else {
            audio::drain_16k(&st.audio)
        };
        if let Some(w) = mic_w.as_mut() {
            let _ = w.append(&new_mic);
        }
        mic_pos += new_mic.len() as u64;
        let excluded = exclusions.lock().unwrap().clone();
        handle_channel(&tx, &depth, &mut mic, new_mic, stopping, &excluded);

        #[cfg(target_os = "macos")]
        if sys_ok {
            let new_sys = crate::syscapture::drain(&st.sys);
            if let Some(w) = sys_w.as_mut() {
                // Written on the meeting clock: lead-in and gaps as silence.
                let _ = w.pad_to(sys.offset + sys_fed);
                let _ = w.append(&new_sys);
            }
            sys_fed += new_sys.len() as u64;
            handle_channel(&tx, &depth, &mut sys, new_sys, stopping, &[]);

            // macOS can stop the stream on its own (display change, sleep):
            // restart it and carry on at the current point of the meeting.
            if let Some(err) = crate::syscapture::take_stop_error(&st.sys) {
                if !stopping {
                    let _ = crate::syscapture::stop(&st.sys);
                    match crate::syscapture::start(&st.sys) {
                        Ok(()) => {
                            let _ = app.emit(
                                "meeting-warning",
                                format!("System audio was interrupted ({err}) and has resumed."),
                            );
                            sys = Channel {
                                vad: make_vad(&vad_path),
                                pending: Vec::new(),
                                offset: mic_pos,
                                speaker: "them",
                                source: "system",
                            };
                            sys_fed = 0;
                        }
                        Err(e) => {
                            sys_ok = false;
                            let _ = app.emit(
                                "meeting-warning",
                                format!("System audio stopped ({err}) and couldn't restart ({e}) — recording your mic only."),
                            );
                        }
                    }
                }
            }
            if stopping {
                let _ = crate::syscapture::stop(&st.sys);
            }
        }

        if last_status.elapsed() >= Duration::from_secs(2) || stopping {
            last_status = Instant::now();
            let _ = app.emit(
                "meeting-status",
                serde_json::json!({
                    "positionMs": mic_pos as i64 * 1000 / RATE,
                    "pending": depth.load(Ordering::Relaxed).max(0),
                    "systemAudio": sys_ok,
                }),
            );
        }
        if stopping {
            break;
        }
    }
    if let Some(w) = mic_w {
        let _ = w.finish();
    }
    if let Some(w) = sys_w {
        let _ = w.finish();
    }
    // `tx` drops here; the handle's clone goes in `stop`, ending transcription.
}

/// Feed a channel's new audio to its VAD, push any finished segments, and —
/// when the transcriber is idle and we're mid-utterance — push a partial.
/// Mic segments mostly inside an exclusion window (dictation, command key)
/// are dropped.
fn handle_channel(
    tx: &mpsc::Sender<Job>,
    depth: &Arc<AtomicI64>,
    ch: &mut Channel,
    new: Vec<f32>,
    stopping: bool,
    exclusions: &[(i64, i64)],
) {
    let Some(vad) = ch.vad.as_mut() else { return };
    if !new.is_empty() {
        ch.pending.extend_from_slice(&new);
        vad.accept_waveform(new);
    }
    if stopping {
        vad.flush();
    }

    let mut had_final = false;
    while !vad.is_empty() {
        let seg = vad.front();
        vad.pop();
        // seg.start indexes everything fed to this VAD; shift onto the clock.
        let start = ch.offset as i64 + seg.start as i64;
        let base_ms = start * 1000 / RATE;
        let end_ms = base_ms + seg.samples.len() as i64 * 1000 / RATE;
        had_final = true;
        let dur = (end_ms - base_ms).max(1);
        let hidden: i64 = exclusions
            .iter()
            .map(|(a, b)| (end_ms.min(*b) - base_ms.max(*a)).max(0))
            .sum();
        if hidden * 2 >= dur {
            continue;
        }
        depth.fetch_add(1, Ordering::Relaxed);
        let _ = tx.send(Job::Final {
            samples: seg.samples,
            base_ms,
            speaker: ch.speaker,
            source: ch.source,
        });
    }

    if had_final {
        ch.pending.clear();
        return;
    }

    // Only emit a partial when the transcriber is idle (keeps partials fresh
    // and never lets them pile up behind finals), we're clearly mid-utterance,
    // and nobody has claimed the mic.
    let claimed = exclusions.iter().any(|(_, b)| *b == i64::MAX);
    if depth.load(Ordering::Relaxed) == 0
        && ch.pending.len() >= PARTIAL_MIN_SAMPLES
        && vad.is_speech()
        && !claimed
    {
        depth.fetch_add(1, Ordering::Relaxed);
        let _ = tx.send(Job::Partial {
            samples: ch.pending.clone(),
            speaker: ch.speaker,
        });
    }
}

// ---- transcription thread (consumer) ---------------------------------------

/// A finished segment waiting to be transcribed.
struct Piece {
    samples: Vec<f32>,
    base_ms: i64,
    speaker: &'static str,
    source: &'static str,
}

/// Live state the transcription thread keeps.
struct Live {
    meeting_id: String,
    language: Option<String>,
    /// Whisper on auto: detect the language once from the first speech, then
    /// lock it (auto-detect per segment costs a second encoder pass each time
    /// and can flip languages on short lines).
    detect_with: Option<PathBuf>,
    detect_buf: Vec<f32>,
    embedder: Option<speakers::Embedder>,
    tracker: speakers::SpeakerTracker,
    emb_model: String,
    named: std::collections::HashSet<usize>,
    dictionary: Vec<(String, String)>,
    wake_word: Option<String>,
    armed_until: Option<i64>,
}

#[allow(clippy::too_many_arguments)]
fn run_transcribe(
    app: AppHandle,
    meeting_id: String,
    model_id: String,
    language: Option<String>,
    initial_prompt: Option<String>,
    emb_model: String,
    rx: mpsc::Receiver<Job>,
    depth: Arc<AtomicI64>,
) {
    // Build the engine on this thread (sherpa recognizers are !Send).
    let mut engine = asr::resolve(&app, &model_id).unwrap_or_else(|e| {
        eprintln!("[meeting] speech engine unavailable: {e}");
        asr::Engine::None
    });
    // Warm the model up front so the first segment of the meeting isn't slow.
    let mut detect_with = None;
    if let asr::Engine::Whisper(path) = &engine {
        let st = app.state::<AppState>();
        if let Err(e) = st.stt.warmup(path) {
            eprintln!("[meeting] warmup failed: {e}");
        }
        if language.is_none() {
            detect_with = Some(path.clone());
        }
    }
    let s = settings::load(&app);
    let db = app.state::<Db>();
    let voiceprints: Vec<speakers::Voiceprint> = store::voiceprints(&db, &emb_model)
        .unwrap_or_default()
        .into_iter()
        .map(|v| speakers::Voiceprint {
            name: v.name,
            embedding: speakers::decode(&v.embedding),
        })
        .collect();
    let embedder = models::downloaded_model_path(&app, &emb_model)
        .and_then(|p| speakers::Embedder::new(&p))
        .map_err(|e| eprintln!("[meeting] live speaker labels unavailable: {e}"))
        .ok();
    // Each embedding model has its own tuned thresholds.
    let tracker = match embedder.as_ref() {
        Some(e) => speakers::SpeakerTracker::with_params(voiceprints, e.params()),
        None => speakers::SpeakerTracker::new(voiceprints),
    };
    let mut live = Live {
        meeting_id: meeting_id.clone(),
        language,
        detect_with,
        detect_buf: Vec::new(),
        embedder,
        tracker,
        emb_model,
        named: Default::default(),
        dictionary: s
            .dictionary
            .iter()
            .filter(|e| !e.from.trim().is_empty() && !e.to.trim().is_empty())
            .map(|e| (e.from.trim().to_string(), e.to.trim().to_string()))
            .collect(),
        wake_word: (s.assistant.enabled && s.assistant.listen_for_wake_word)
            .then(|| s.assistant.wake_word.trim().to_string())
            .filter(|w| !w.is_empty()),
        armed_until: None,
    };
    let prompt = initial_prompt.as_deref();

    let mut queue: VecDeque<Job> = VecDeque::new();
    loop {
        if queue.is_empty() {
            match rx.recv() {
                Ok(j) => queue.push_back(j),
                Err(_) => break,
            }
        }
        while let Ok(j) = rx.try_recv() {
            queue.push_back(j);
        }
        // Spoken commands jump the queue.
        let job = match queue.iter().position(|j| matches!(j, Job::Command { .. })) {
            Some(i) => queue.remove(i).expect("index from position"),
            None => queue.pop_front().expect("queue is non-empty"),
        };
        match job {
            Job::Command { samples, at_ms } => {
                let text = transcribe_all(&app, &mut engine, &samples, live.language.as_deref(), prompt);
                depth.fetch_sub(1, Ordering::Relaxed);
                let text = apply_dictionary(&text, &live.dictionary);
                if text.trim().is_empty() {
                    crate::overlay_toast(&app, "Didn't catch that — hold the key and try again.");
                } else {
                    agent::on_request(&app, Some(meeting_id.clone()), Some(at_ms), &text, "hotkey");
                }
            }
            Job::Partial { samples, speaker } => {
                depth.fetch_sub(1, Ordering::Relaxed);
                // Stale if a final for that channel is already waiting.
                let superseded = queue
                    .iter()
                    .any(|j| matches!(j, Job::Final { speaker: s, .. } if *s == speaker));
                if superseded {
                    continue;
                }
                let text = transcribe_all(&app, &mut engine, &samples, live.language.as_deref(), prompt);
                if !text.trim().is_empty() {
                    let _ = app.emit(
                        "meeting-partial",
                        serde_json::json!({ "speaker": speaker, "text": text }),
                    );
                }
            }
            Job::Final { samples, base_ms, speaker, source } => {
                let mut batch = vec![Piece { samples, base_ms, speaker, source }];
                // Behind? Take the following segments of the same channel too
                // (Whisper only — the sherpa models are fast enough alone).
                if matches!(engine, asr::Engine::Whisper(_)) {
                    let mut total = batch[0].samples.len();
                    while let Some(Job::Final { samples, source: s, .. }) = queue.front() {
                        if *s != source || total + samples.len() + GAP_SAMPLES > BATCH_MAX_SAMPLES {
                            break;
                        }
                        total += samples.len() + GAP_SAMPLES;
                        if let Some(Job::Final { samples, base_ms, speaker, source }) = queue.pop_front() {
                            batch.push(Piece { samples, base_ms, speaker, source });
                        }
                    }
                }
                let n = batch.len() as i64;
                process_finals(&app, &mut engine, &mut live, batch, prompt);
                depth.fetch_sub(n, Ordering::Relaxed);
            }
        }
    }
    finish_speakers(&app, &mut live);
}

fn transcribe_all(
    app: &AppHandle,
    engine: &mut asr::Engine,
    samples: &[f32],
    lang: Option<&str>,
    prompt: Option<&str>,
) -> String {
    let mut padded = vec![0.0; GAP_SAMPLES];
    padded.extend_from_slice(samples);
    asr::transcribe_parts(app, engine, &padded, lang, prompt)
        .into_iter()
        .map(|(_, _, t)| t)
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

/// Transcribe a batch of segments (one Whisper pass), then label, correct,
/// scan for the wake word, persist and emit each resulting line.
fn process_finals(
    app: &AppHandle,
    engine: &mut asr::Engine,
    live: &mut Live,
    batch: Vec<Piece>,
    prompt: Option<&str>,
) {
    maybe_lock_language(app, live, &batch);

    // Lay the batch out on one buffer: [gap][piece 0][gap][piece 1]…
    let mut buf: Vec<f32> = Vec::new();
    let mut spans: Vec<(i64, i64)> = Vec::new(); // (offset_ms, len_ms) in buf
    for p in &batch {
        buf.extend(std::iter::repeat_n(0.0, GAP_SAMPLES));
        let off = buf.len() as i64 * 1000 / RATE;
        buf.extend_from_slice(&p.samples);
        spans.push((off, p.samples.len() as i64 * 1000 / RATE));
    }
    let parts = asr::transcribe_parts(app, engine, &buf, live.language.as_deref(), prompt);

    // Speaker per piece (remote channel), from its voice.
    let labels: Vec<String> = batch
        .iter()
        .map(|p| {
            if p.source != "system" {
                return p.speaker.to_string();
            }
            let Some(embedder) = live.embedder.as_mut() else {
                return p.speaker.to_string();
            };
            let emb = embedder.embed(&p.samples);
            let end_ms = p.base_ms + p.samples.len() as i64 * 1000 / RATE;
            let key = live.tracker.assign(emb.as_deref(), p.base_ms, end_ms);
            note_voice_name(app, live, key);
            format!("them:{key}")
        })
        .collect();

    let mut segs: Vec<Segment> = Vec::new();
    for (s, e, text) in parts {
        // Which piece this text came from: the one its midpoint falls in.
        let mid = (s + e) / 2;
        let i = spans
            .iter()
            .position(|(off, len)| mid >= *off && mid <= off + len)
            .or_else(|| {
                spans
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, (off, len))| (mid - (off + len / 2)).abs())
                    .map(|(i, _)| i)
            })
            .unwrap_or(0);
        let (off, len) = spans[i];
        let p = &batch[i];
        let start_ms = p.base_ms + (s - off).clamp(0, len);
        let end_ms = p.base_ms + (e - off).clamp(0, len);
        let mut text = apply_dictionary(&text, &live.dictionary);
        if p.source == "mic" {
            match scan_wake(app, live, &text, end_ms) {
                Some(rest) => text = rest,
                None => continue,
            }
            if text.trim().is_empty() {
                continue;
            }
        }
        segs.push(Segment {
            id: 0,
            speaker: labels[i].clone(),
            source: p.source.to_string(),
            start_ms,
            end_ms: end_ms.max(start_ms),
            text,
        });
    }
    if segs.is_empty() {
        return;
    }
    let db = app.state::<Db>();
    if let Err(e) = store::add_segments(&db, &live.meeting_id, &segs) {
        eprintln!("[meeting] failed to persist segments: {e}");
    }
    let _ = app.emit("meeting-segments", &segs);
}

/// Handle the wake word in one of the user's lines. Returns the text to keep
/// in the transcript (the request itself is lifted out), or `None` to drop
/// the line entirely (it was all request).
fn scan_wake(app: &AppHandle, live: &mut Live, text: &str, end_ms: i64) -> Option<String> {
    let Some(wake) = live.wake_word.clone() else {
        return Some(text.to_string());
    };
    // The line right after a bare "unsound" is the request.
    if let Some(until) = live.armed_until.take() {
        if end_ms <= until && !text.trim().is_empty() {
            agent::on_request(app, Some(live.meeting_id.clone()), Some(end_ms), text, "wake");
            return None;
        }
    }
    let Some(hit) = agent::detect_wake(text, &wake) else {
        return Some(text.to_string());
    };
    if hit.request.split_whitespace().count() >= 2 {
        agent::on_request(app, Some(live.meeting_id.clone()), Some(end_ms), &hit.request, "wake");
    } else {
        live.armed_until = Some(end_ms + ARMED_MS);
        crate::overlay_toast(app, "Listening…");
    }
    Some(hit.before)
}

fn maybe_lock_language(app: &AppHandle, live: &mut Live, batch: &[Piece]) {
    let Some(path) = live.detect_with.clone() else { return };
    for p in batch {
        live.detect_buf.extend_from_slice(&p.samples);
    }
    // Enough speech to judge (a few seconds); give up after ~40 s of it.
    if live.detect_buf.len() < 6 * audio::WHISPER_SAMPLE_RATE as usize {
        return;
    }
    let st = app.state::<AppState>();
    match stt::detect_language(&st.stt, &path, &live.detect_buf) {
        Ok(Some(lang)) => {
            eprintln!("[meeting] language locked to {lang}");
            live.language = Some(lang);
            live.detect_with = None;
            live.detect_buf = Vec::new();
        }
        Ok(None) if live.detect_buf.len() < 40 * audio::WHISPER_SAMPLE_RATE as usize => {}
        _ => {
            live.detect_with = None;
            live.detect_buf = Vec::new();
        }
    }
}

/// A live cluster matched a remembered voice: name it for this meeting.
fn note_voice_name(app: &AppHandle, live: &mut Live, key: usize) {
    if live.named.contains(&key) {
        return;
    }
    if let Some(name) = live.tracker.cluster_name(key) {
        live.named.insert(key);
        let speaker = format!("them:{key}");
        let db = app.state::<Db>();
        let has_name = store::get_meeting(&db, &live.meeting_id)
            .ok()
            .flatten()
            .is_some_and(|m| m.speaker_names.contains_key(&speaker));
        if !has_name {
            let _ = store::set_speaker_name(&db, &live.meeting_id, &speaker, &name);
            let _ = app.emit(
                "meeting-speaker-named",
                serde_json::json!({ "speaker": speaker, "name": name }),
            );
        }
    }
}

/// End of meeting: merge phantom/duplicate speaker clusters, relabel the
/// transcript to match, and store each speaker's voice centroid (so naming
/// them later can remember the voice).
fn finish_speakers(app: &AppHandle, live: &mut Live) {
    if live.embedder.is_none() {
        return;
    }
    let remap = live.tracker.consolidate();
    let db = app.state::<Db>();
    if !remap.is_empty() {
        let pairs: Vec<(String, String)> = remap
            .iter()
            .map(|(a, b)| (format!("them:{a}"), format!("them:{b}")))
            .collect();
        let _ = store::remap_speakers(&db, &live.meeting_id, &pairs);
    }
    let rows: Vec<store::SpeakerCentroid> = live
        .tracker
        .centroids()
        .into_iter()
        .map(|(k, c, n, ms)| (format!("them:{k}"), live.emb_model.clone(), speakers::encode(&c), n as i64, ms))
        .collect();
    let _ = store::set_meeting_speakers(&db, &live.meeting_id, &rows);
}

/// Apply the user's learned corrections ("pria" → "Priya") to meeting text:
/// whole words, case-insensitive.
fn apply_dictionary(text: &str, dict: &[(String, String)]) -> String {
    let mut out = text.to_string();
    for (from, to) in dict {
        out = store::replace_words(&out, from, to);
    }
    out
}

// ---- notes thread -----------------------------------------------------------

/// Extract each five-minute stretch soon after it completes, in the gaps
/// between transcription work, so the final summary has little left to do.
fn run_notes(app: AppHandle, meeting_id: String, model: PathBuf, stop: Arc<AtomicBool>, depth: Arc<AtomicI64>) {
    let mut next_check = Instant::now() + Duration::from_secs(60);
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(500));
        if Instant::now() < next_check || depth.load(Ordering::Relaxed) > 0 {
            continue;
        }
        next_check = Instant::now() + Duration::from_secs(30);
        let Some(pos) = position_ms(&app) else { continue };
        if pos < summary::CHUNK_MS {
            continue;
        }
        if let Err(e) = summary::ensure_chunks(&app, &meeting_id, &model, Some(pos)) {
            eprintln!("[meeting] background notes failed: {e}");
            next_check = Instant::now() + Duration::from_secs(120);
        }
    }
}

// ---- after the meeting --------------------------------------------------------

/// Re-detect remote speakers for a finished meeting from its saved audio
/// (optionally with a known count), relabel the transcript, and refresh the
/// stored voice centroids.
pub fn redetect_speakers(
    app: &AppHandle,
    meeting_id: &str,
    emb_model: &str,
    num_speakers: Option<usize>,
) -> Result<(), String> {
    let db = app.state::<Db>();
    let meeting = store::get_meeting(&db, meeting_id)?.ok_or("meeting not found")?;
    let path = audio_store::system_path(app, meeting_id)?;
    if !path.exists() {
        return Err("no system audio was saved for this meeting".into());
    }
    let samples = crate::audiofile::decode_to_samples(&path)?;
    let system: Vec<&Segment> = meeting.segments.iter().filter(|s| s.source == "system").collect();
    if system.is_empty() {
        return Ok(());
    }
    let spans: Vec<(i64, i64)> = system.iter().map(|s| (s.start_ms, s.end_ms)).collect();
    let voiceprints: Vec<speakers::Voiceprint> = store::voiceprints(&db, emb_model)?
        .into_iter()
        .map(|v| speakers::Voiceprint {
            name: v.name,
            embedding: speakers::decode(&v.embedding),
        })
        .collect();
    let mut embedder = speakers::Embedder::new(&models::downloaded_model_path(app, emb_model)?)?;
    let (keys, centroids) = speakers::cluster_offline(&mut embedder, &samples, &spans, num_speakers, &voiceprints);
    let updates: Vec<(i64, String)> = system
        .iter()
        .zip(&keys)
        .filter_map(|(s, k)| k.map(|k| (s.id, format!("them:{k}"))))
        .collect();
    store::update_segment_speakers(&db, &updates)?;
    // Old names referred to old clusters; keep only names matched by voice.
    for sp in meeting.speaker_names.keys().filter(|k| k.starts_with("them")) {
        let _ = store::set_speaker_name(&db, meeting_id, sp, "");
    }
    let rows: Vec<store::SpeakerCentroid> = centroids
        .iter()
        .map(|(k, c, n, ms)| (format!("them:{k}"), emb_model.to_string(), speakers::encode(c), *n as i64, *ms))
        .collect();
    store::set_meeting_speakers(&db, meeting_id, &rows)?;
    for (k, c, _, _) in &centroids {
        if let Some(name) = speakers::match_voiceprint(c, &voiceprints) {
            let _ = store::set_speaker_name(&db, meeting_id, &format!("them:{k}"), &name);
        }
    }
    Ok(())
}

/// Remember a named speaker's voice (from this meeting's centroid), so future
/// meetings label them automatically. False when no voice was captured.
pub fn remember_voice(app: &AppHandle, meeting_id: &str, speaker: &str, name: &str) -> Result<bool, String> {
    let db = app.state::<Db>();
    let Some((_, model, emb, segs, _)) = store::meeting_speaker(&db, meeting_id, speaker)? else {
        return Ok(false);
    };
    let new = speakers::decode(&emb);
    let existing = store::voiceprints(&db, &model)?
        .into_iter()
        .find(|v| v.name.eq_ignore_ascii_case(name));
    let (embedding, samples) = match existing {
        Some(v) => (
            speakers::blend(&speakers::decode(&v.embedding), v.samples as u32, &new),
            v.samples + segs.max(1),
        ),
        None => (new, segs.max(1)),
    };
    store::save_voiceprint(
        &db,
        &store::VoiceprintRow {
            name: name.to_string(),
            model,
            embedding: speakers::encode(&embedding),
            samples,
        },
    )?;
    Ok(true)
}
