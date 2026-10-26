//! Live capture writes checkpointed audio and durable transcription ranges.
//! Capture never waits for inference. A single supervised worker reads bounded
//! ranges from disk; a result and its completion marker commit atomically.
//! Startup requeues saved ranges that were still inside VAD when the app stopped.

use crate::audio_store::{self, WavWriter};
use crate::meeting_work;
use crate::store::{self, Db, Segment};
use crate::{agent, asr, audio, models, settings, speakers, stt, summary, AppState};
use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};
use std::collections::HashMap;
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
    /// The in-progress utterance, shown tentatively and not persisted.
    Partial {
        samples: Vec<f32>,
        speaker: &'static str,
    },
    /// A request spoken while holding the command key.
    Command { samples: Vec<f32>, at_ms: i64 },
}

#[derive(Default)]
pub struct MeetingState {
    active: Mutex<Option<Handle>>,
    /// Whether a meeting is currently recording — drives the tray label.
    pub recording: Mutex<bool>,
    worker: Mutex<Option<Worker>>,
    faults: Mutex<HashMap<String, String>>,
    pub metrics: crate::meeting_metrics::Metrics,
}

struct Worker {
    id: String,
    cancel: Arc<AtomicBool>,
    progress: Arc<Mutex<Instant>>,
    join: JoinHandle<()>,
}

impl MeetingState {
    /// Reserve the recorder until capture has released the devices. The old
    /// transcriber may finish later, but must no longer change recording state.
    fn release_capture(&self) -> Option<Handle> {
        let mut active = self.active.lock().unwrap();
        let mut handle = active.take()?;
        handle.stop.store(true, Ordering::Relaxed);
        if let Some(capture) = handle.capture.take() {
            let _ = capture.join();
        }
        *self.recording.lock().unwrap() = false;
        Some(handle)
    }
}

struct Handle {
    meeting_id: String,
    stop: Arc<AtomicBool>,
    capture: Option<JoinHandle<()>>,
    rx: Option<mpsc::Receiver<Job>>,
    /// Outstanding jobs in the queue (throttles partials; "is it idle?").
    depth: Arc<AtomicI64>,
    tx: mpsc::SyncSender<Job>,
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
    #[cfg(feature = "automation")]
    if crate::meeting_fixture::valid_id(&opts.meeting_id)
        && opts.stt_model == format!("{}{}", crate::meeting_fixture::MODEL_PREFIX, opts.meeting_id)
    {
        return start(app, opts, PathBuf::new());
    }
    let vad_path = models::downloaded_model_path(app, "vad-silero")?;
    // Surface a missing speech model early; the engine itself is built on the
    // transcription thread (sherpa recognizers are !Send).
    models::downloaded_model_path(app, &opts.stt_model)?;
    let id = opts.meeting_id.clone();
    start(app, opts, vad_path).map_err(|e| {
        if active_meeting(app).as_deref() != Some(&id) {
            fail(app, &id, &e);
        }
        e
    })
}

#[cfg(feature="automation")]
pub fn start_benchmark(app:&AppHandle,id:&str,model:&str)->Result<(),String> {
    if app.config().identifier!="com.unsound.e2e" || !crate::meeting_fixture::valid_id(id) {return Err("benchmark requires an isolated fixture".into());}
    crate::meeting_fixture::get(app,id).system_audio.store(true,Ordering::SeqCst);
    start(app,StartOptions{meeting_id:id.into(),stt_model:format!("e2e-real:{model}"),language:Some("en".into()),llm_model:None,embedding_model:Some(String::new())},PathBuf::new())
}

fn start(app: &AppHandle, opts: StartOptions, vad_path: PathBuf) -> Result<(), String> {
    let _permit = crate::updates::capture_permit()?;
    let st = app.state::<AppState>();
    let mut active = st.meeting.active.lock().unwrap();
    if active.is_some() {
        return Err("a meeting is already recording".into());
    }
    let s = settings::load(app);
    st.meeting.metrics.begin(&opts.meeting_id,false);

    let mic_file = audio_store::mic_path(app, &opts.meeting_id)?;
    let sys_file = audio_store::system_path(app, &opts.meeting_id)?;

    meeting_work::begin(
        &app.state::<Db>(),
        &opts.meeting_id,
        &opts.stt_model,
        opts.language.as_deref(),
        opts.embedding_model
            .as_deref()
            .unwrap_or(&s.meetings.embedding_model),
    )?;
    let mic_writer = WavWriter::create(&mic_file)?;
    let sys_writer = WavWriter::create(&sys_file)?;

    #[cfg(feature = "automation")]
    let fixture = if vad_path.as_os_str().is_empty() { Some(crate::meeting_fixture::get(app,&opts.meeting_id)) } else { None };
    #[cfg(feature = "automation")]
    let synthetic = fixture.is_some();
    #[cfg(not(feature = "automation"))]
    let synthetic = false;

    // The mic defines the meeting clock; system audio starts a moment later
    // and is offset onto it.
    if !synthetic {
        audio::start_recording(
            app.clone(),
            &st.audio,
            s.mic_device.clone(),
            audio::Owner::Meeting,
        )?;
    }
    let mic_started = Instant::now();
    #[cfg(target_os = "macos")]
    let sys_ok = if synthetic {
        #[cfg(feature = "automation")]
        { fixture.as_ref().is_some_and(|f|f.system_audio.load(Ordering::SeqCst)) }
        #[cfg(not(feature = "automation"))]
        { false }
    } else {
        match crate::syscapture::start(&st.sys) {
            Ok(()) => true,
            Err(e) => {
                eprintln!("[meeting] system audio unavailable, mic only: {e}");
                let _ = app.emit(
                    "meeting-warning",
                    format!("System audio unavailable — recording your mic only ({e})."),
                );
                false
            }
        }
    };
    #[cfg(not(target_os = "macos"))]
    let sys_ok = false;
    let sys_offset = if synthetic {0} else {(mic_started.elapsed().as_millis() as i64 * RATE / 1000) as u64};

    let stop = Arc::new(AtomicBool::new(false));
    let depth = Arc::new(AtomicI64::new(0));
    let exclusions: Arc<Mutex<Vec<(i64, i64)>>> = Arc::new(Mutex::new(Vec::new()));
    let (tx, rx) = mpsc::sync_channel::<Job>(4);

    let capture = {
        let app = app.clone();
        let stop = stop.clone();
        let depth = depth.clone();
        let exclusions = exclusions.clone();
        let tx = tx.clone();
        let id = opts.meeting_id.clone();
        std::thread::spawn(move || {
            run_capture(
                app, id, vad_path, stop, tx, depth, exclusions, mic_writer, sys_writer, sys_ok,
                sys_offset,
                #[cfg(feature = "automation")]
                fixture,
            )
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
        rx: Some(rx),
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

/// Release capture promptly. Durable work continues in the single background worker.
pub fn stop(app: &AppHandle) -> Result<Option<String>, String> {
    let started=Instant::now();
    let st = app.state::<AppState>();
    let Some(h) = st.meeting.release_capture() else {
        return Ok(None);
    };
    crate::tray::refresh_menu(app);
    let id = h.meeting_id.clone();
    drop(h);
    let db = app.state::<Db>();
    if let Err(e) = store::set_meeting_ended(&db, &id, &crate::util::now_iso())
        .and_then(|_| recover_audio(app, &id))
    {
        fail(
            app,
            &id,
            &format!("Recording stopped, but recovery could not finish: {e}"),
        );
    }
    st.meeting.metrics.stopped(&id,started.elapsed().as_millis() as u64);
    if !st.meeting.faults.lock().unwrap().contains_key(&id) {
        if let Err(e) = meeting_work::state(&db, &id, "finishing", "") {
            fail(app, &id, &e);
        }
    }
    Ok(Some(id))
}

pub fn statuses(app: &AppHandle) -> Result<Vec<meeting_work::Status>, String> {
    let mut rows = meeting_work::statuses(&app.state::<Db>())?;
    let st = app.state::<AppState>();
    let active_id = active_meeting(app);
    let faults = st.meeting.faults.lock().unwrap();
    for row in &mut rows {
        row.recording = active_id.as_deref() == Some(&row.meeting_id);
        if let Some(error) = faults.get(&row.meeting_id) {
            row.error = error.clone();
            row.state = "error".into();
        }
    }
    Ok(rows)
}

fn fail(app: &AppHandle, id: &str, error: &str) {
    app.state::<AppState>()
        .meeting
        .faults
        .lock()
        .unwrap()
        .insert(id.into(), error.into());
    let _ = meeting_work::state(&app.state::<Db>(), id, "error", error);
    let _ = app.emit("meeting-work-changed", id);
    eprintln!("[meeting {id}] {error}");
}

fn recover_audio(app: &AppHandle, id: &str) -> Result<(), String> {
    let mut total = 0;
    for (source, path) in [
        ("mic", audio_store::mic_path(app, id)?),
        ("system", audio_store::system_path(app, id)?),
    ] {
        audio_store::repair(&path)?;
        let samples = audio_store::sample_count(&path)?;
        total = total.max(samples);
        meeting_work::recover_channel(&app.state::<Db>(), id, source, samples)?;
    }
    if let Some(m) = store::get_meeting(&app.state::<Db>(), id)? {
        if let Some(start) = crate::util::unix_ms_from_iso(&m.started_at) {
            let saved_end = start + total / 16;
            let old_end = m
                .ended_at
                .as_deref()
                .and_then(crate::util::unix_ms_from_iso)
                .unwrap_or(start);
            if saved_end > old_end {
                store::set_meeting_ended(
                    &app.state::<Db>(),
                    id,
                    &crate::util::iso_from_unix_ms(saved_end),
                )?;
            }
        }
    }
    Ok(())
}

pub fn pause(app: &AppHandle, id: &str) -> Result<(), String> {
    if active_meeting(app).as_deref() == Some(id) {
        return Err("Stop recording before pausing transcription.".into());
    }
    // The database gate prevents a late result from committing, even if its
    // native model cannot immediately honor cancellation.
    meeting_work::state(&app.state::<Db>(), id, "paused", "")?;
    if let Some(w) = app
        .state::<AppState>()
        .meeting
        .worker
        .lock()
        .unwrap()
        .as_ref()
        .filter(|w| w.id == id)
    {
        w.cancel.store(true, Ordering::Relaxed);
    }
    Ok(())
}

pub fn retry(app: &AppHandle, id: &str) -> Result<(), String> {
    if active_meeting(app).as_deref() == Some(id) {
        return Err("Stop recording before retrying transcription.".into());
    }
    let st = app.state::<AppState>();
    if st
        .meeting
        .worker
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|w| w.id == id && !w.join.is_finished())
    {
        return Err("The previous inference call is still stopping. If it remains stuck, restart the app; saved audio will remain available.".into());
    }
    recover_audio(app, id)?;
    if let Some(row) = statuses(app)?
        .into_iter()
        .find(|r| r.meeting_id == id && !r.error.is_empty())
    {
        meeting_work::warning(
            &app.state::<Db>(),
            id,
            &format!(
                "Earlier interruption: {} Recovery covers only audio saved on disk.",
                row.error
            ),
        )?;
    }
    meeting_work::state(&app.state::<Db>(), id, "finishing", "")?;
    st.meeting.faults.lock().unwrap().remove(id);
    Ok(())
}

fn stall_timeout(_app: &AppHandle, _id: &str) -> Duration {
    #[cfg(feature = "automation")]
    {
        crate::meeting_fixture::timeout(_app, _id)
    }
    #[cfg(not(feature = "automation"))]
    {
        Duration::from_secs(90)
    }
}

/// One meeting transcription worker for the app. A non-cooperative native call can
/// occupy this slot, but cannot spawn an unbounded number of replacement calls.
/// Capture and stop never join this worker.
pub fn start_recovery(app: AppHandle) {
    if let Ok(rows) = meeting_work::statuses(&app.state::<Db>()) {
        for row in rows.into_iter().filter(|r| r.state != "complete") {
            app.state::<AppState>().meeting.metrics.begin(&row.meeting_id,true);
            if let Err(e) = recover_audio(&app, &row.meeting_id) {
                fail(&app, &row.meeting_id, &e);
                continue;
            }
            if row.state == "recording" {
                let _ = meeting_work::warning(&app.state::<Db>(),&row.meeting_id,
                        "Recording was interrupted. Saved audio is being recovered; audio after the last save may be missing.");
                let _ = meeting_work::state(&app.state::<Db>(), &row.meeting_id, "finishing", "");
            }
        }
    }
    std::thread::spawn(move || {
        loop {
            let st = app.state::<AppState>();
            let stopped = st
                .meeting
                .active
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|h| h.capture.as_ref().is_some_and(|t| t.is_finished()));
            if stopped {
                let _ = stop(&app);
            }
            let rows=meeting_work::statuses(&app.state::<Db>()).ok();
            if let Some(rows)=&rows {
                for row in rows.iter().filter(|r|r.state=="recording" || r.state=="finishing") {
                    st.meeting.metrics.backlog(&row.meeting_id,row.pending,row.pending_audio_ms);
                }
            }
            {
                let mut slot = st.meeting.worker.lock().unwrap();
                if slot.as_ref().is_some_and(|w| w.join.is_finished()) {
                    let worker = slot.take().unwrap();
                    if worker.join.join().is_err() {
                        fail(
                            &app,
                            &worker.id,
                            "Transcription crashed. Saved audio can be retried.",
                        );
                    }
                }
                if let Some(w) = slot.as_ref() {
                    if !w.cancel.load(Ordering::Relaxed)
                        && w.progress.lock().unwrap().elapsed() > stall_timeout(&app, &w.id)
                    {
                        w.cancel.store(true, Ordering::Relaxed);
                        fail(&app,&w.id,"Transcription stopped making progress. Recording can continue. Stop recording, then retry; restart the app if the model cannot stop.");
                    }
                } else if let Some(rows) = rows {
                    for row in rows {
                        // If SQLite could not persist the error, the in-memory
                        // fault must still prevent an automatic retry loop.
                        if st.meeting.faults.lock().unwrap().contains_key(&row.meeting_id) { continue; }
                        if row.state != "recording" && row.state != "finishing" {
                            continue;
                        }
                        if row.state == "finishing" && row.pending == 0 {
                            if meeting_work::state(
                                &app.state::<Db>(),
                                &row.meeting_id,
                                "complete",
                                "",
                            )
                            .is_ok()
                            {
                                st.meeting.metrics.complete(&row.meeting_id);
                                if !cfg!(feature = "automation") {
                                    agent::on_meeting_ended(&app, &row.meeting_id);
                                }
                                let _ = app.emit("meeting-work-changed", &row.meeting_id);
                            }
                            continue;
                        }
                        let id = row.meeting_id;
                        let (rx, depth) = {
                            let mut active = st.meeting.active.lock().unwrap();
                            if let Some(h) = active.as_mut().filter(|h| h.meeting_id == id) {
                                if let Some(rx) = h.rx.take() {
                                    (rx, h.depth.clone())
                                } else {
                                    // Only needed if a previous consumer exited while capture remained active.
                                    let (tx, rx) = mpsc::sync_channel(4);
                                    h.tx = tx;
                                    h.depth.store(0, Ordering::Relaxed);
                                    (rx, h.depth.clone())
                                }
                            } else {
                                let (_, rx) = mpsc::sync_channel(4);
                                (rx, Arc::new(AtomicI64::new(0)))
                            }
                        };
                        let cancel = Arc::new(AtomicBool::new(false));
                        let progress = Arc::new(Mutex::new(Instant::now()));
                        let (a, i, c, p) =
                            (app.clone(), id.clone(), cancel.clone(), progress.clone());
                        let join = std::thread::spawn(move || {
                            let result =
                                run_transcribe(a.clone(), i.clone(), rx, depth, c.clone(), p);
                            if let Err(e) = result {
                                if !c.load(Ordering::Relaxed) {
                                    fail(&a, &i, &e);
                                }
                            }
                        });
                        *slot = Some(Worker {
                            id,
                            cancel,
                            progress,
                            join,
                        });
                        break;
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    });
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
    let Some(h) = active.as_ref() else {
        return false;
    };
    if let Err(e) = meeting_work::exclusion(&app.state::<Db>(), &h.meeting_id, start_ms, None) {
        fail(
            app,
            &h.meeting_id,
            &format!("Could not save the dictation exclusion: {e}"),
        );
        h.stop.store(true, Ordering::Relaxed);
        return false;
    }
    h.exclusions.lock().unwrap().push((start_ms, i64::MAX));
    true
}

/// Close the exclusion window that started at `start_ms`.
pub fn exclude_until(app: &AppHandle, start_ms: i64, end_ms: i64) {
    let st = app.state::<AppState>();
    let active = st.meeting.active.lock().unwrap();
    if let Some(h) = active.as_ref() {
        if let Err(e) =
            meeting_work::exclusion(&app.state::<Db>(), &h.meeting_id, start_ms, Some(end_ms))
        {
            fail(
                app,
                &h.meeting_id,
                &format!("Could not save the dictation exclusion: {e}"),
            );
            h.stop.store(true, Ordering::Relaxed);
        }
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
    if samples.len() > BATCH_MAX_SAMPLES {
        depth.fetch_sub(1, Ordering::Relaxed);
        return Err("Keep spoken requests under 25 seconds.".into());
    }
    tx.try_send(Job::Command {
        samples,
        at_ms: end,
    })
    .map_err(|_| {
        depth.fetch_sub(1, Ordering::Relaxed);
        "Transcription is busy; try the request again later.".to_string()
    })
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
    #[cfg(feature = "automation")]
    if vad_path.as_os_str().is_empty() {
        return None;
    }
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
    fed: u64,
}

#[allow(clippy::too_many_arguments)]
fn run_capture(
    app: AppHandle,
    meeting_id: String,
    vad_path: PathBuf,
    stop: Arc<AtomicBool>,
    tx: mpsc::SyncSender<Job>,
    depth: Arc<AtomicI64>,
    exclusions: Arc<Mutex<Vec<(i64, i64)>>>,
    mut mic_writer: WavWriter,
    mut sys_writer: WavWriter,
    #[allow(unused_mut)] mut sys_ok: bool,
    sys_offset: u64,
    #[cfg(feature = "automation")] fixture: Option<Arc<crate::meeting_fixture::Fixture>>,
) {
    let st = app.state::<AppState>();
    let mut storage_failed = false;
    let mut mic = Channel {
        vad: make_vad(&vad_path),
        pending: Vec::new(),
        fed: 0,
        offset: 0,
        speaker: "me",
        source: "mic",
    };
    #[allow(unused_mut, unused_variables)]
    let mut sys = Channel {
        vad: make_vad(&vad_path),
        pending: Vec::new(),
        fed: 0,
        offset: sys_offset,
        speaker: "them",
        source: "system",
    };
    if mic.vad.is_none() || (sys_ok && sys.vad.is_none()) {
        let warning = "Live speech detection is unavailable. Audio is still being saved and will be transcribed after recording stops.";
        let _ = meeting_work::warning(&app.state::<Db>(), &meeting_id, warning);
        let _ = app.emit("meeting-warning", warning);
    }
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
        #[cfg(feature = "automation")]
        let fixture_input = fixture.as_ref().map(|f| f.capture_input());
        #[cfg(feature = "automation")]
        if fixture_input.as_ref().is_some_and(|(disconnected, _, _)| *disconnected) {
            *st.audio.capture_error.lock().unwrap() = Some("injected audio disconnect".into());
        }
        if let Some(error) = st.audio.capture_error.lock().unwrap().take() {
            fail(&app,&meeting_id,&format!("Microphone capture was interrupted: {error}. Recording has stopped; saved audio is available for recovery."));
            stop.store(true, Ordering::Relaxed);
        }
        let stopping = stop.load(Ordering::Relaxed);

        // Mic: on the final pass, stopping the device also returns the tail
        // still inside the resampler.
        #[cfg(feature = "automation")]
        let (synthetic_audio,mut synthetic_system) = match fixture_input {
            Some((_,mic,system))=>(Some(mic),Some(system)),None=>(None,None)
        };
        #[cfg(not(feature = "automation"))]
        let mut synthetic_system:Option<Vec<f32>>=None;
        #[cfg(not(feature = "automation"))]
        let synthetic_audio: Option<Vec<f32>> = None;
        let new_mic = if let Some(samples) = synthetic_audio {
            samples
        } else if stopping {
            audio::stop_meeting_recording(&st.audio)
        } else {
            audio::drain_16k(&st.audio)
        };
        if !storage_failed {
            #[cfg(feature = "automation")]
            let writable = fixture.as_ref()
                .filter(|_| !new_mic.is_empty())
                .map(|f| f.check_save())
                .unwrap_or(Ok(()));
            #[cfg(not(feature = "automation"))]
            let writable: Result<(), String> = Ok(());
            let checkpoint_started=Instant::now();
            let save = writable
                .and_then(|_| mic_writer.append(&new_mic))
                .and_then(|_| mic_writer.checkpoint())
                .and_then(|n| {
                    meeting_work::saved(&app.state::<Db>(), &meeting_id, "mic", n as i64)
                });
            if !new_mic.is_empty() { st.meeting.metrics.checkpoint(&meeting_id,checkpoint_started.elapsed().as_millis() as u64); }
            if let Err(e) = save {
                fail(
                    &app,
                    &meeting_id,
                    &format!("Microphone audio stopped saving: {e}"),
                );
                storage_failed = true;
                stop.store(true, Ordering::Relaxed);
            }
        }
        mic_pos += new_mic.len() as u64;
        let excluded = exclusions.lock().unwrap().clone();
        if !storage_failed {
            #[cfg(feature = "automation")]
            if fixture.is_some() && !new_mic.is_empty() {
                // Deterministic speech boundaries replace VAD only for fixture audio.
                if let Err(e) = meeting_work::enqueue(
                    &app.state::<Db>(), &meeting_id, "mic", "me",
                    (mic_pos - new_mic.len() as u64) as i64, mic_pos as i64, false,
                ) {
                    fail(&app, &meeting_id, &e);
                    storage_failed = true;
                    stop.store(true, Ordering::Relaxed);
                }
            }
            if let Err(e) = handle_channel(
                &app.state::<Db>(),
                &meeting_id,
                &tx,
                &depth,
                &mut mic,
                new_mic,
                stopping,
                &excluded,
            ) {
                fail(
                    &app,
                    &meeting_id,
                    &format!("Transcription work stopped saving: {e}"),
                );
                storage_failed = true;
                stop.store(true, Ordering::Relaxed);
            }
        }

        #[cfg(target_os = "macos")]
        if sys_ok {
            let new_sys = if let Some(samples)=synthetic_system.take() {samples} else if stopping {
                match crate::syscapture::stop(&st.sys) {
                    Ok(_) => std::mem::take(&mut *st.sys.last.lock().unwrap()),
                    Err(e) => {
                        fail(
                            &app,
                            &meeting_id,
                            &format!("System capture did not stop cleanly: {e}"),
                        );
                        Vec::new()
                    }
                }
            } else {
                crate::syscapture::drain(&st.sys)
            };
            if !storage_failed {
                let checkpoint_started=Instant::now();
                let save = sys_writer
                    .pad_to(sys.offset + sys_fed)
                    .and_then(|_| sys_writer.append(&new_sys))
                    .and_then(|_| sys_writer.checkpoint())
                    .and_then(|n| {
                        meeting_work::saved(&app.state::<Db>(), &meeting_id, "system", n as i64)
                    });
                if !new_sys.is_empty() {st.meeting.metrics.checkpoint(&meeting_id,checkpoint_started.elapsed().as_millis() as u64);}
                if let Err(e) = save {
                    fail(
                        &app,
                        &meeting_id,
                        &format!("System audio stopped saving: {e}"),
                    );
                    storage_failed = true;
                    stop.store(true, Ordering::Relaxed);
                }
            }
            sys_fed += new_sys.len() as u64;
            if !storage_failed {
                #[cfg(feature = "automation")]
                if fixture.is_some() && !new_sys.is_empty() {
                    if let Err(e)=meeting_work::enqueue(&app.state::<Db>(),&meeting_id,"system","them",(sys.offset+sys_fed-new_sys.len() as u64) as i64,(sys.offset+sys_fed) as i64,false) {
                        fail(&app,&meeting_id,&e);storage_failed=true;stop.store(true,Ordering::Relaxed);
                    }
                }
                if let Err(e) = handle_channel(
                    &app.state::<Db>(),
                    &meeting_id,
                    &tx,
                    &depth,
                    &mut sys,
                    new_sys,
                    stopping,
                    &[],
                ) {
                    fail(
                        &app,
                        &meeting_id,
                        &format!("Transcription work stopped saving: {e}"),
                    );
                    storage_failed = true;
                    stop.store(true, Ordering::Relaxed);
                }
            }

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
                                fed: 0,
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
        }

        if last_status.elapsed() >= Duration::from_secs(2) || stopping {
            last_status = Instant::now();
            let _ = app.emit(
                "meeting-status",
                serde_json::json!({
                    "positionMs": mic_pos as i64 * 1000 / RATE,
                    "pending": meeting_work::statuses(&app.state::<Db>()).ok().and_then(|rows| rows.into_iter().find(|r| r.meeting_id == meeting_id)).map(|r| r.pending).unwrap_or(0),
                    "systemAudio": sys_ok,
                }),
            );
        }
        if stopping {
            break;
        }
    }
    for (source, writer) in [("microphone", mic_writer), ("system", sys_writer)] {
        if let Err(e) = writer.finish() {
            fail(
                &app,
                &meeting_id,
                &format!("Could not finish saving {source} audio: {e}"),
            );
        }
    }
    // `tx` drops here; the handle's clone goes in `stop`, ending transcription.
}

/// Feed a channel's new audio to its VAD, push any finished segments, and —
/// when the transcriber is idle and we're mid-utterance — push a partial.
/// Mic segments mostly inside an exclusion window (dictation, command key)
/// are dropped.
fn handle_channel(
    db: &Db,
    meeting_id: &str,
    tx: &mpsc::SyncSender<Job>,
    depth: &Arc<AtomicI64>,
    ch: &mut Channel,
    new: Vec<f32>,
    stopping: bool,
    exclusions: &[(i64, i64)],
) -> Result<(), String> {
    let Some(vad) = ch.vad.as_mut() else {
        return Ok(());
    };
    if !new.is_empty() {
        ch.fed += new.len() as u64;
        ch.pending.extend_from_slice(&new);
        // Partials are disposable previews. Keep their memory bounded even
        // during long silence or uninterrupted speech; VAD owns final audio.
        let excess = ch.pending.len().saturating_sub(BATCH_MAX_SAMPLES);
        ch.pending.drain(..excess);
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
        meeting_work::enqueue(
            db,
            meeting_id,
            ch.source,
            ch.speaker,
            start,
            (start + seg.samples.len() as i64).min((ch.offset + ch.fed) as i64),
            hidden * 2 >= dur,
        )?;
    }

    if had_final {
        ch.pending.clear();
        return Ok(());
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
        enqueue_partial(tx, depth, &ch.pending, ch.speaker);
    }
    Ok(())
}

fn enqueue_partial(
    tx: &mpsc::SyncSender<Job>,
    depth: &AtomicI64,
    samples: &[f32],
    speaker: &'static str,
) -> bool {
    if depth
        .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return false;
    }
    let start = samples.len().saturating_sub(BATCH_MAX_SAMPLES);
    if tx
        .try_send(Job::Partial {
            samples: samples[start..].to_vec(),
            speaker,
        })
        .is_err()
    {
        depth.fetch_sub(1, Ordering::Relaxed);
        return false;
    }
    true
}

/// Keep the queue reservation throughout inference, including error unwinding.
fn complete_partial<T>(depth: &AtomicI64, infer: impl FnOnce() -> T) -> T {
    struct Completion<'a>(&'a AtomicI64);
    impl Drop for Completion<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::Relaxed);
        }
    }
    let _completion = Completion(depth);
    infer()
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

fn run_transcribe(
    app: AppHandle,
    meeting_id: String,
    rx: mpsc::Receiver<Job>,
    depth: Arc<AtomicI64>,
    cancel: Arc<AtomicBool>,
    progress: Arc<Mutex<Instant>>,
) -> Result<(), String> {
    let model_started=Instant::now();
    let meeting =
        store::get_meeting(&app.state::<Db>(), &meeting_id)?.ok_or("meeting was deleted")?;
    let model_id = meeting.stt_model;
    let language = meeting.lang;
    let settings = settings::load(&app);
    let emb_model = meeting_work::embedding_model(&app.state::<Db>(), &meeting_id)?;
    let initial_prompt = Some(format!(
        "Glossary: {}",
        settings
            .dictionary
            .iter()
            .map(|e| e.to.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    let recovering = active_meeting(&app).as_deref() != Some(&meeting_id);
    // Build the engine on this thread (sherpa recognizers are !Send).
    let mut engine = asr::resolve(&app, &model_id)?;
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
    let embedder = if recovering {
        None
    } else {
        models::downloaded_model_path(&app, &emb_model)
            .and_then(|p| speakers::Embedder::new(&p))
            .map_err(|e| eprintln!("[meeting] live speaker labels unavailable: {e}"))
            .ok()
    };
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
        wake_word: (!recovering && s.assistant.enabled && s.assistant.listen_for_wake_word)
            .then(|| s.assistant.wake_word.trim().to_string())
            .filter(|w| !w.is_empty()),
        armed_until: None,
    };
    app.state::<AppState>().meeting.metrics.model_ready(&meeting_id,model_started.elapsed().as_millis() as u64);
    let prompt = initial_prompt.as_deref();

    loop {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        *progress.lock().unwrap() = Instant::now();
        let row = meeting_work::statuses(&db)?
            .into_iter()
            .find(|r| r.meeting_id == meeting_id);
        let Some(row) = row else { return Ok(()) };
        if row.state != "recording" && row.state != "finishing" {
            return Ok(());
        }
        // At most four disposable previews/commands are in memory. Durable
        // final work is fetched in bounded batches, never as the whole backlog.
        if let Ok(job) = rx.try_recv() {
            match job {
                Job::Command { samples, at_ms } => {
                    let text = complete_partial(&depth, || {
                        asr::transcribe_preview(
                            &app,
                            &mut engine,
                            &samples,
                            live.language.as_deref(),
                            prompt,
                            cancel.clone(),
                        )
                    })?;
                    let text = apply_dictionary(&text, &live.dictionary);
                    if !text.trim().is_empty()
                        && !cancel.load(Ordering::Relaxed)
                        && active_meeting(&app).as_deref() == Some(&meeting_id)
                    {
                        agent::on_request(
                            &app,
                            Some(meeting_id.clone()),
                            Some(at_ms),
                            &text,
                            "hotkey",
                        );
                    }
                }
                Job::Partial { samples, speaker } => {
                    let text = complete_partial(&depth, || {
                        asr::transcribe_preview(
                            &app,
                            &mut engine,
                            &samples,
                            live.language.as_deref(),
                            prompt,
                            cancel.clone(),
                        )
                    })?;
                    if !cancel.load(Ordering::Relaxed) {
                        let _ = app.emit("meeting-partial",serde_json::json!({"meetingId":meeting_id,"speaker":speaker,"text":text}));
                    }
                }
            }
        }
        let jobs = meeting_work::next_batch(
            &db,
            &meeting_id,
            matches!(engine, asr::Engine::Whisper(_)),
            GAP_SAMPLES as i64,
        )?;
        if !jobs.is_empty() {
            let queued_at=jobs.iter().map(|j|j.queued_at_ms).filter(|t|*t>0).min().unwrap_or(0);
            let audio_ms=jobs.iter().map(|j|(j.end-j.start) as u64/16).sum();
            let batch_started=Instant::now();
            let mut pieces = Vec::new();
            let mut ids = Vec::new();
            for job in jobs {
                let path = if job.source == "mic" {
                    audio_store::mic_path(&app, &meeting_id)?
                } else {
                    audio_store::system_path(&app, &meeting_id)?
                };
                let mut samples = audio_store::read_samples(&path, job.start, job.end)?;
                meeting_work::mask_private_audio(
                    &db,
                    &meeting_id,
                    &job.source,
                    job.start,
                    &mut samples,
                )?;
                let source = if job.source == "mic" { "mic" } else { "system" };
                let speaker = if job.speaker == "me" { "me" } else { "them" };
                pieces.push(Piece {
                    samples,
                    base_ms: job.start / 16,
                    speaker,
                    source,
                });
                ids.push(job.id);
            }
            depth.fetch_add(1, Ordering::Relaxed);
            let committed=complete_partial(&depth, || {
                process_finals(
                    &app,
                    &mut engine,
                    &mut live,
                    pieces,
                    prompt,
                    &ids,
                    cancel.clone(),
                )
            })?;
            if committed { app.state::<AppState>().meeting.metrics.batch(&meeting_id,batch_started.elapsed().as_millis() as u64,audio_ms,queued_at); }
        } else if row.state == "finishing" {
            finish_speakers(&app, &mut live);
            return Ok(()); // scheduler publishes completion on its next pass
        } else {
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

/// Transcribe a batch of segments (one Whisper pass), then label, correct,
/// scan for the wake word, persist and emit each resulting line.
fn process_finals(
    app: &AppHandle,
    engine: &mut asr::Engine,
    live: &mut Live,
    batch: Vec<Piece>,
    prompt: Option<&str>,
    job_ids: &[i64],
    cancel: Arc<AtomicBool>,
) -> Result<bool, String> {
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
    let parts = asr::transcribe_parts_checked(
        app,
        engine,
        &buf,
        live.language.as_deref(),
        prompt,
        cancel.clone(),
    )?;
    if cancel.load(Ordering::Relaxed) {
        return Ok(false);
    }

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
        let text = apply_dictionary(&text, &live.dictionary);
        segs.push(Segment {
            id: 0,
            speaker: labels[i].clone(),
            source: p.source.to_string(),
            start_ms,
            end_ms: end_ms.max(start_ms),
            text,
        });
    }
    if cancel.load(Ordering::Relaxed) {
        return Ok(false);
    }
    let db = app.state::<Db>();
    let committed=meeting_work::complete_batch(&db, &live.meeting_id, job_ids, &segs)?;
    if committed {
        let _ = app.emit(
            "meeting-segments",
            serde_json::json!({"meetingId":live.meeting_id,"segments":segs}),
        );
        // Only a newly committed live result may trigger the assistant. Replay
        // after restart must never repeat an external action.
        if active_meeting(app).as_deref() == Some(&live.meeting_id) {
            for s in &segs {
                if s.source == "mic" {
                    scan_wake(app, live, &s.text, s.end_ms);
                }
            }
        }
    }
    Ok(committed)
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
            agent::on_request(
                app,
                Some(live.meeting_id.clone()),
                Some(end_ms),
                text,
                "wake",
            );
            return None;
        }
    }
    let Some(hit) = agent::detect_wake(text, &wake) else {
        return Some(text.to_string());
    };
    if hit.request.split_whitespace().count() >= 2 {
        agent::on_request(
            app,
            Some(live.meeting_id.clone()),
            Some(end_ms),
            &hit.request,
            "wake",
        );
    } else {
        live.armed_until = Some(end_ms + ARMED_MS);
        crate::overlay_toast(app, "Listening…");
    }
    Some(hit.before)
}

fn maybe_lock_language(app: &AppHandle, live: &mut Live, batch: &[Piece]) {
    let Some(path) = live.detect_with.clone() else {
        return;
    };
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
                serde_json::json!({ "meetingId": live.meeting_id, "speaker": speaker, "name": name }),
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
        .map(|(k, c, n, ms)| {
            (
                format!("them:{k}"),
                live.emb_model.clone(),
                speakers::encode(&c),
                n as i64,
                ms,
            )
        })
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
fn run_notes(
    app: AppHandle,
    meeting_id: String,
    model: PathBuf,
    stop: Arc<AtomicBool>,
    depth: Arc<AtomicI64>,
) {
    let mut next_check = Instant::now() + Duration::from_secs(60);
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(500));
        if Instant::now() < next_check
            || depth.load(Ordering::Relaxed) > 0
            || meeting_work::next(&app.state::<Db>(), &meeting_id)
                .ok()
                .flatten()
                .is_some()
        {
            continue;
        }
        next_check = Instant::now() + Duration::from_secs(30);
        let Some(pos) = position_ms(&app) else {
            continue;
        };
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
    let system: Vec<&Segment> = meeting
        .segments
        .iter()
        .filter(|s| s.source == "system")
        .collect();
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
    let mut embeddings=Vec::with_capacity(spans.len());
    for &(start,end) in &spans {
        // The embedder needs at most ten seconds. Keep long recordings on disk.
        let end=end.min(start+10_000);
        let pcm=audio_store::read_samples(&path,start*16,end*16)?;
        embeddings.push(embedder.embed(&pcm));
    }
    let (keys, centroids) = speakers::cluster_embeddings_with(&embeddings,&spans,num_speakers,&voiceprints,embedder.params());
    let updates: Vec<(i64, String)> = system
        .iter()
        .zip(&keys)
        .filter_map(|(s, k)| k.map(|k| (s.id, format!("them:{k}"))))
        .collect();
    store::update_segment_speakers(&db, &updates)?;
    // Old names referred to old clusters; keep only names matched by voice.
    for sp in meeting
        .speaker_names
        .keys()
        .filter(|k| k.starts_with("them"))
    {
        let _ = store::set_speaker_name(&db, meeting_id, sp, "");
    }
    let rows: Vec<store::SpeakerCentroid> = centroids
        .iter()
        .map(|(k, c, n, ms)| {
            (
                format!("them:{k}"),
                emb_model.to_string(),
                speakers::encode(c),
                *n as i64,
                *ms,
            )
        })
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
pub fn remember_voice(
    app: &AppHandle,
    meeting_id: &str,
    speaker: &str,
    name: &str,
) -> Result<bool, String> {
    let db = app.state::<Db>();
    let row = store::meeting_speaker(&db, meeting_id, speaker)?;
    let (_, model, emb, segs, speech_ms) = match row {
        Some(row) => row,
        None if speaker.starts_with("manual:") => {
            // Rebuild from the corrected utterances, never from the old mixed cluster.
            let m=store::get_meeting(&db,meeting_id)?.ok_or("meeting not found")?;
            let model=meeting_work::embedding_model(&db,meeting_id).unwrap_or_else(|_|settings::load(app).meetings.embedding_model);
            let mut embedder=speakers::Embedder::new(&models::downloaded_model_path(app,&model)?)?;
            let mut centroid=Vec::new(); let mut count=0; let mut speech=0;
            for s in m.segments.iter().filter(|s|s.speaker==speaker && s.end_ms-s.start_ms>=2000).take(15) {
                let end=s.end_ms.min(s.start_ms+10_000);
                let path=if s.source=="mic" { audio_store::mic_path(app,meeting_id)? } else { audio_store::system_path(app,meeting_id)? };
                let pcm=audio_store::read_samples(&path,s.start_ms*16,end*16)?;
                if let Some(embedding)=embedder.embed(&pcm) {
                    centroid=speakers::blend_weighted(&centroid,count,&embedding,1);
                    count+=1; speech+=end-s.start_ms;
                }
            }
            (speaker.to_string(),model,speakers::encode(&centroid),count as i64,speech)
        }
        None => return Ok(false),
    };
    if segs<2 || speech_ms<6000 { return Ok(false); }
    let new = speakers::decode(&emb);
    if new.is_empty() || new.iter().any(|x|!x.is_finite()) { return Ok(false); }
    let existing = store::voiceprints(&db, &model)?.into_iter().find(|v| v.name.eq_ignore_ascii_case(name));
    let (embedding, samples) = match existing {
        Some(v) => {
            let old=speakers::decode(&v.embedding);
            let similarity=speakers::cosine(&old,&new);
            // Re-saving an unchanged name must not give one meeting more weight.
            if similarity>0.9999 { return Ok(true); }
            if old.len()!=new.len() || similarity<0.55 {
                return Err("Name saved, but this voice differs from the remembered voice. Forget the old voice in settings before replacing it.".into());
            }
            (speakers::blend_weighted(&old,v.samples.max(1) as u32,&new,segs as u32),v.samples+segs)
        }
        None => (new,segs),
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

#[cfg(test)]
mod reliability_tests {
    use super::*;

    #[test]
    fn stalled_partial_inference_does_not_block_audio_or_accumulate_previews() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stalled.wav");
        let mut wav = WavWriter::create(&path).unwrap();
        let (tx, rx) = mpsc::sync_channel(4);
        let depth = Arc::new(AtomicI64::new(0));
        let samples = vec![0.25; BATCH_MAX_SAMPLES * 2];
        assert!(enqueue_partial(&tx, &depth, &samples, "me"));
        let (entered, wait_entered) = mpsc::channel();
        let (resume, wait_resume) = mpsc::channel();
        let worker_depth = depth.clone();
        let worker = std::thread::spawn(move || {
            let Job::Partial {
                samples: queued, ..
            } = rx.recv_timeout(Duration::from_secs(1)).unwrap()
            else {
                panic!("expected partial");
            };
            assert_eq!(queued.len(), BATCH_MAX_SAMPLES);
            complete_partial(&worker_depth, || {
                entered.send(()).unwrap();
                wait_resume.recv_timeout(Duration::from_secs(5)).unwrap();
            });
            rx
        });
        wait_entered.recv_timeout(Duration::from_secs(5)).unwrap();
        for _ in 0..120 {
            wav.append(&[0.25; 1600]).unwrap();
            assert!(!enqueue_partial(&tx, &depth, &samples, "me"));
        }
        assert_eq!(depth.load(Ordering::Relaxed), 1);
        assert_eq!(wav.finish().unwrap(), 192_000);
        assert_eq!(
            audio_store::read_range(&path, 0, 12_000).unwrap().len(),
            192_000
        );
        // Model returns; previews can resume.
        resume.send(()).unwrap();
        let rx = worker.join().unwrap();
        assert!(matches!(rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
        assert_eq!(depth.load(Ordering::Relaxed), 0);
        assert!(enqueue_partial(&tx, &depth, &[0.5; 1600], "me"));
    }

    #[test]
    fn disconnected_inference_worker_releases_partial_reservation() {
        let (tx, rx) = mpsc::sync_channel(4);
        let depth = AtomicI64::new(0);
        drop(rx);
        assert!(!enqueue_partial(&tx, &depth, &[0.25; 1600], "me"));
        assert_eq!(depth.load(Ordering::Relaxed), 0);
    }

    fn handle(id: &str) -> Handle {
        let (tx, _) = mpsc::sync_channel(4);
        Handle {
            meeting_id: id.into(),
            stop: Arc::new(AtomicBool::new(false)),
            capture: None,
            rx: None,
            depth: Arc::new(AtomicI64::new(0)),
            tx,
            exclusions: Default::default(),
            command_tap: None,
        }
    }

    #[test]
    fn next_session_waits_for_capture_but_not_previous_transcription() {
        let state = Arc::new(MeetingState::default());
        let mut old = handle("old");
        let stop = old.stop.clone();
        let (capture_done, wait_capture) = mpsc::channel();
        let (inference_done, wait_inference) = mpsc::channel();
        old.capture = Some(std::thread::spawn(move || {
            wait_capture.recv_timeout(Duration::from_secs(5)).unwrap();
        }));
        let inference = std::thread::spawn(move || {
            wait_inference.recv_timeout(Duration::from_secs(5)).unwrap();
        });
        *state.active.lock().unwrap() = Some(old);
        *state.recording.lock().unwrap() = true;
        let (released, wait_released) = mpsc::channel();
        let stopping = state.clone();
        let finalizer = std::thread::spawn(move || {
            let old = stopping.release_capture().unwrap();
            released.send(()).unwrap();
            inference.join().unwrap();
            // Mirrors stop: after release, only the old meeting's metadata
            // can change, never the shared recorder or recording flag.
            old.meeting_id
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        while !stop.load(Ordering::Relaxed) {
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        assert!(
            state.active.try_lock().is_err(),
            "capture still owns the recorder slot"
        );
        capture_done.send(()).unwrap();
        wait_released.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(!*state.recording.lock().unwrap());
        // Install the next session while old inference is deliberately held.
        *state.active.lock().unwrap() = Some(handle("new"));
        *state.recording.lock().unwrap() = true;
        assert!(!finalizer.is_finished());
        inference_done.send(()).unwrap();
        assert_eq!(finalizer.join().unwrap(), "old");
        assert!(*state.recording.lock().unwrap());
        assert_eq!(
            state.active.lock().unwrap().as_ref().unwrap().meeting_id,
            "new"
        );
        state.release_capture();
    }
}
