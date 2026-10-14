mod agent;
mod asr;
pub mod audio;
pub mod audiofile;
pub mod calendar;
mod audio_store;
mod brain;
mod deliver;
pub mod detect;
mod dictation;
mod frontapp;
mod hotkeys;
mod llm;
mod meeting;
mod meeting_work;
#[cfg(any(test, feature = "automation"))]
mod meeting_fixture;
#[cfg(feature = "automation")]
mod meeting_fixture_runner;
mod models;
mod permissions;
#[allow(dead_code)]
mod secrets;
mod settings;
mod speakers;
mod store;
mod stt;
mod summary;
#[cfg(target_os = "macos")]
mod syscapture;
#[allow(dead_code)]
mod tools;
mod tray;
mod util;
#[cfg(test)]
mod reliability_tests;

use audio::{AudioState, RecordingResult};
use llm::LlmState;
use models::{ModelInfo, ModelKind};
use settings::Settings;
use store::{Db, Meeting, Take};
use stt::SttState;
use std::collections::HashMap;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Listener, Manager, State};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

#[derive(Clone, Copy, PartialEq)]
enum ShortcutMode {
    HandsFree,
    PushToTalk,
    /// Hold to speak a request to the assistant.
    Command,
}

#[derive(Default)]
pub struct AppState {
    audio: AudioState,
    stt: SttState,
    llm: LlmState,
    /// System-audio (ScreenCaptureKit) capture — the "them" channel in a
    /// meeting. Separate from the mic so who-said-what stays attributable.
    #[cfg(target_os = "macos")]
    sys: syscapture::SysCaptureState,
    /// Live meeting recording state (streaming transcription loop).
    meeting: meeting::MeetingState,
    /// Live dictation captions state (preview transcription loop).
    live: dictation::LiveState,
    /// Most recent pipeline output, for the tray's "paste last take".
    pub last_output: Mutex<String>,
    /// Registered shortcut id → which mode it drives.
    shortcut_modes: Mutex<HashMap<u32, ShortcutMode>>,
    /// The command key is recording a standalone request (no meeting).
    command_standalone: Mutex<bool>,
    /// The dictation speech model (last warmed), for standalone requests.
    dictation_stt: Mutex<String>,
    /// Where a dictation take tapping a meeting began (ms on its clock).
    dictation_tap_start: Mutex<i64>,
}

#[tauri::command]
fn list_models(app: AppHandle) -> Result<Vec<ModelInfo>, String> {
    models::all_models(&app)
}

#[tauri::command]
async fn download_model(app: AppHandle, id: String) -> Result<(), String> {
    models::download(app, id).await
}

#[tauri::command]
fn delete_model(app: AppHandle, id: String) -> Result<(), String> {
    models::delete_model_file(&app, &id)
}

#[tauri::command]
fn add_custom_model(
    app: AppHandle,
    name: String,
    kind: ModelKind,
    url: String,
) -> Result<ModelInfo, String> {
    models::add_custom(&app, name, kind, url)
}

/// Start a dictation take. During a meeting the mic is already running, so
/// the take taps the meeting's recording (and is kept out of its transcript).
#[tauri::command]
fn start_recording(app: AppHandle, state: State<AppState>) -> Result<(), String> {
    if audio::owner(&state.audio) == Some(audio::Owner::Meeting) {
        let (tap, start) = audio::tap_begin(&state.audio)?;
        *state.audio.dictation_tap.lock().unwrap() = Some(tap);
        *state.dictation_tap_start.lock().unwrap() = start;
        meeting::exclude_from(&app, start);
        return Ok(());
    }
    let mic = settings::load(&app).mic_device;
    audio::start_recording(app, &state.audio, mic, audio::Owner::Dictation)
}

#[tauri::command]
fn list_microphones() -> Vec<String> {
    audio::list_input_devices()
}

#[tauri::command]
fn set_microphone(app: AppHandle, device: String) -> Result<(), String> {
    let mut s = settings::load(&app);
    s.mic_device = device;
    settings::save(&app, &s)?;
    tray::refresh_menu(&app);
    Ok(())
}

#[tauri::command]
fn stop_recording(app: AppHandle, state: State<AppState>) -> Result<RecordingResult, String> {
    let tap = state.audio.dictation_tap.lock().unwrap().take();
    if let Some(tap) = tap {
        let (samples, end) = audio::tap_end(&state.audio, tap)?;
        meeting::exclude_until(&app, *state.dictation_tap_start.lock().unwrap(), end);
        let result = RecordingResult {
            duration_secs: samples.len() as f32 / audio::WHISPER_SAMPLE_RATE as f32,
            sample_count: samples.len(),
        };
        *state.audio.last_recording.lock().unwrap() = samples;
        return Ok(result);
    }
    audio::stop_recording(&state.audio)
}

#[tauri::command]
async fn transcribe(
    app: AppHandle,
    state: State<'_, AppState>,
    model_id: String,
    language: Option<String>,
) -> Result<String, String> {
    let samples = state.audio.last_recording.lock().unwrap().clone();
    let initial_prompt = vocab_prompt(&app);
    // Inference is heavy; keep it off the async runtime. The engine (whisper or
    // a sherpa model) is resolved on the blocking thread.
    let worker_app = app.clone();
    let text = tauri::async_runtime::spawn_blocking(move || {
        // On auto, use the language detected while recording, sparing Whisper
        // its own detection pass.
        let language = language.or_else(|| dictation::finish(&worker_app));
        let mut engine = asr::resolve(&worker_app, &model_id)?;
        asr::transcribe_text(
            &worker_app,
            &mut engine,
            &samples,
            language.as_deref(),
            initial_prompt.as_deref(),
        )
    })
    .await
    .map_err(|e| e.to_string())??;
    *state.last_output.lock().unwrap() = text.clone();
    Ok(text)
}

/// Load the selected speech model and compile its GPU kernels ahead of the
/// first take (at launch, and whenever the selection changes).
#[tauri::command]
async fn warm_stt(app: AppHandle, model_id: String) -> Result<(), String> {
    *app.state::<AppState>().dictation_stt.lock().unwrap() = model_id.clone();
    tauri::async_runtime::spawn_blocking(move || {
        // The sherpa models are built per use, so there's nothing to keep warm.
        match asr::whisper_path(&app, &model_id)? {
            Some(path) => app.state::<AppState>().stt.warmup(&path),
            None => Ok(()),
        }
    })
    .await
    .map_err(|e| e.to_string())?
}

/// The user's corrected vocabulary, as a Whisper `initial_prompt` that biases
/// recognition toward names/jargon.
fn vocab_prompt(app: &AppHandle) -> Option<String> {
    let mut seen = std::collections::HashSet::new();
    let vocab: Vec<String> = settings::load(app)
        .dictionary
        .iter()
        .map(|e| e.to.trim().to_string())
        .filter(|t| !t.is_empty() && seen.insert(t.to_lowercase()))
        .collect();
    (!vocab.is_empty()).then(|| format!("Glossary: {}.", vocab.join(", ")))
}

/// Start a take's background work: live captions, so the words appear (in the
/// window, or the overlay bubble) as you speak, and — with the language on
/// auto — language detection. The final cleaned text still lands at stop.
#[tauri::command]
async fn start_live_dictation(
    app: AppHandle,
    model_id: String,
    language: Option<String>,
) -> Result<(), String> {
    // Validate the model is present; the engine is built on the live thread.
    models::downloaded_model_path(&app, &model_id)?;
    let initial_prompt = vocab_prompt(&app);
    // May wait out a previous take's worker; keep that off the main thread.
    tauri::async_runtime::spawn_blocking(move || {
        dictation::start(&app, model_id, language, initial_prompt)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn stop_live_dictation(app: AppHandle) {
    let _ = tauri::async_runtime::spawn_blocking(move || dictation::stop(&app)).await;
}

/// Decode an uploaded audio file into the recording buffer, then run the
/// same transcription path as a mic take.
#[tauri::command]
async fn transcribe_file(
    app: AppHandle,
    state: State<'_, AppState>,
    path: String,
    model_id: String,
    language: Option<String>,
) -> Result<String, String> {
    let file = std::path::PathBuf::from(&path);
    let size_bytes = std::fs::metadata(&file).map(|m| m.len()).unwrap_or(0);
    let decode_file = file.clone();
    let samples =
        tauri::async_runtime::spawn_blocking(move || audiofile::decode_to_samples(&decode_file))
            .await
            .map_err(|e| e.to_string())??;
    let duration_secs = samples.len() as f32 / audio::WHISPER_SAMPLE_RATE as f32;
    if duration_secs < 0.2 {
        return Err("that file has almost no audio in it".into());
    }
    // Tell the UI the file's size and length now that decode is done, before
    // the slow transcription step.
    let _ = app.emit(
        "file-info",
        serde_json::json!({ "sizeBytes": size_bytes, "durationSecs": duration_secs }),
    );
    *state.audio.last_recording.lock().unwrap() = samples.clone();

    let initial_prompt = vocab_prompt(&app);
    let worker_app = app.clone();
    let text = tauri::async_runtime::spawn_blocking(move || {
        let mut engine = asr::resolve(&worker_app, &model_id)?;
        asr::transcribe_text(
            &worker_app,
            &mut engine,
            &samples,
            language.as_deref(),
            initial_prompt.as_deref(),
        )
    })
    .await
    .map_err(|e| e.to_string())??;
    *state.last_output.lock().unwrap() = text.clone();
    Ok(text)
}

/// The cleanup system prompt — the fixed base, the user's additions on top,
/// and their dictionary — plus the chosen writing style.
fn cleanup_setup(
    app: &AppHandle,
    prompt: Option<String>,
    style_id: Option<String>,
) -> (String, Option<settings::Style>) {
    let settings = settings::load(app);
    let mut system_prompt = llm::DEFAULT_CLEANUP_PROMPT.to_string();
    if let Some(additions) = prompt.filter(|p| !p.trim().is_empty()) {
        system_prompt.push_str(&format!(
            "\n\nAdditional instructions from the user (apply alongside the rules above):\n{additions}"
        ));
    }
    system_prompt.push_str(&llm::dictionary_addendum(&settings.dictionary));
    let style = style_id
        .filter(|id| !id.is_empty())
        .and_then(|id| settings.styles.into_iter().find(|s| s.id == id));
    (system_prompt, style)
}

/// Load the cleanup model and process the prompt every take shares, ahead of
/// the first take (at launch, and when the model, style or language changes).
/// Takes the same settings as `cleanup_text`.
#[tauri::command]
async fn warm_llm(
    app: AppHandle,
    model_id: String,
    prompt: Option<String>,
    style_id: Option<String>,
    target_lang: Option<String>,
    transliterate: bool,
) -> Result<(), String> {
    let model_path = models::downloaded_model_path(&app, &model_id)?;
    let (system_prompt, style) = cleanup_setup(&app, prompt, style_id);
    tauri::async_runtime::spawn_blocking(move || {
        llm::prime_cleanup(
            &app.state::<AppState>().llm,
            &model_path,
            &system_prompt,
            style.as_ref(),
            target_lang.as_deref(),
            transliterate,
        )
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn cleanup_text(
    app: AppHandle,
    model_id: String,
    text: String,
    prompt: Option<String>,
    style_id: Option<String>,
    target_lang: Option<String>,
    transliterate: bool,
) -> Result<String, String> {
    let model_path = models::downloaded_model_path(&app, &model_id)?;
    let (system_prompt, style) = cleanup_setup(&app, prompt, style_id);
    let emitter = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        llm::cleanup_text(
            &app,
            &state.llm,
            &model_path,
            &system_prompt,
            &text,
            style.as_ref(),
            target_lang.as_deref(),
            transliterate,
        )
    })
    .await
    .map_err(|e| e.to_string())?;
    let _ = emitter.emit("llm-done", result.is_ok());
    if let Ok(refined) = &result {
        *emitter.state::<AppState>().last_output.lock().unwrap() = refined.clone();
    }
    result
}

#[tauri::command]
fn default_cleanup_prompt() -> String {
    llm::DEFAULT_CLEANUP_PROMPT.to_string()
}

#[tauri::command]
fn get_settings(app: AppHandle) -> Settings {
    settings::load(&app)
}

#[tauri::command]
fn set_shortcuts(
    app: AppHandle,
    hands_free: Vec<String>,
    push_to_talk: Vec<String>,
) -> Result<(), String> {
    let mut s = settings::load(&app);
    let previous = s.clone();
    s.hands_free = hands_free;
    s.push_to_talk = push_to_talk;
    if let Err(e) = apply_shortcuts(&app, &s) {
        // Restore the working set so a bad combo doesn't kill the others.
        let _ = apply_shortcuts(&app, &previous);
        return Err(e);
    }
    settings::save(&app, &s)
}

#[tauri::command]
fn set_styles(
    app: AppHandle,
    styles: Vec<settings::Style>,
    default_style: String,
) -> Result<(), String> {
    let mut s = settings::load(&app);
    s.styles = styles;
    s.default_style = default_style;
    settings::save(&app, &s)?;
    let _ = app.emit("settings-changed", ());
    Ok(())
}

#[tauri::command]
fn permission_status() -> permissions::PermissionStatus {
    permissions::status()
}

#[tauri::command]
async fn request_accessibility() -> bool {
    tauri::async_runtime::spawn_blocking(permissions::request_accessibility)
        .await
        .unwrap_or(false)
}

#[tauri::command]
async fn request_microphone() -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(permissions::request_microphone)
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn deliver_text(app: AppHandle, text: String) -> Result<String, String> {
    // Capture the target before typing; unsound stays in the background.
    let target = frontapp::frontmost_app().unwrap_or_default();
    tauri::async_runtime::spawn_blocking(move || deliver::deliver_text(&app, &text))
        .await
        .map_err(|e| e.to_string())??;
    Ok(target)
}

#[tauri::command]
fn add_correction(app: AppHandle, from: String, to: String) -> Result<(), String> {
    let (from, to) = (from.trim().to_string(), to.trim().to_string());
    if from.is_empty() || to.is_empty() || from == to {
        return Err("correction needs a word and a different replacement".into());
    }
    let mut s = settings::load(&app);
    s.dictionary.retain(|e| e.from.to_lowercase() != from.to_lowercase());
    s.dictionary.push(settings::DictEntry { from, to });
    settings::save(&app, &s)?;
    let _ = app.emit("settings-changed", ());
    Ok(())
}

#[tauri::command]
fn set_dictionary(app: AppHandle, entries: Vec<settings::DictEntry>) -> Result<(), String> {
    let mut s = settings::load(&app);
    s.dictionary = entries;
    settings::save(&app, &s)?;
    let _ = app.emit("settings-changed", ());
    Ok(())
}

// ---- history (takes) — now backed by SQLite instead of localStorage --------

#[tauri::command]
fn list_takes(db: State<Db>) -> Result<Vec<Take>, String> {
    store::list_takes(&db, 500)
}

#[tauri::command]
fn save_take(db: State<Db>, take: Take) -> Result<(), String> {
    store::save_take(&db, &take)
}

#[tauri::command]
fn delete_take(db: State<Db>, id: String) -> Result<(), String> {
    store::delete_take(&db, &id)
}

#[tauri::command]
fn clear_takes(db: State<Db>) -> Result<(), String> {
    store::clear_takes(&db)
}

/// One-time import of the frontend's old localStorage history; safe to re-run.
#[tauri::command]
fn import_takes(db: State<Db>, takes: Vec<Take>) -> Result<usize, String> {
    store::import_takes(&db, &takes)
}

// ---- meetings --------------------------------------------------------------

/// A meeting with the fields only the command layer can fill in.
fn full_meeting(app: &AppHandle, db: &Db, id: &str) -> Result<Meeting, String> {
    let mut m = store::get_meeting(db, id)?.ok_or_else(|| "meeting not found".to_string())?;
    m.has_audio = audio_store::has_audio(app, id);
    Ok(m)
}

#[tauri::command]
fn create_meeting(db: State<Db>, meeting: Meeting) -> Result<(), String> {
    store::create_meeting(&db, &meeting)
}

#[tauri::command]
fn update_meeting_notes(db: State<Db>, id: String, notes: String) -> Result<(), String> {
    store::update_meeting_notes(&db, &id, &notes)
}

/// Name (or un-name) a speaker in one meeting. With `remember`, their voice is
/// saved too, so future meetings label them automatically. Returns whether a
/// voice was remembered.
#[tauri::command]
fn set_speaker_name(
    app: AppHandle,
    db: State<Db>,
    meeting_id: String,
    speaker: String,
    name: String,
    remember: Option<bool>,
) -> Result<bool, String> {
    store::set_speaker_name(&db, &meeting_id, &speaker, &name)?;
    if remember.unwrap_or(true) && !name.trim().is_empty() && speaker.starts_with("them") {
        return meeting::remember_voice(&app, &meeting_id, &speaker, name.trim());
    }
    Ok(false)
}

#[tauri::command]
fn set_meeting_summary(db: State<Db>, id: String, summary: String) -> Result<(), String> {
    store::set_meeting_summary(&db, &id, &summary)
}

#[tauri::command]
fn update_segment_text(db: State<Db>, segment_id: i64, text: String) -> Result<(), String> {
    store::update_segment_text(&db, segment_id, &text)
}

/// "Fix everywhere": replace a misheard word or name across one meeting's
/// transcript, and optionally teach the dictionary so it's right next time.
#[tauri::command]
fn replace_in_meeting(
    app: AppHandle,
    db: State<Db>,
    meeting_id: String,
    from: String,
    to: String,
    remember: bool,
) -> Result<usize, String> {
    let n = store::replace_in_meeting(&db, &meeting_id, &from, &to)?;
    if remember {
        add_correction(app, from, to)?;
    }
    Ok(n)
}

#[tauri::command]
fn rename_meeting(db: State<Db>, id: String, title: String) -> Result<(), String> {
    store::rename_meeting(&db, &id, &title)
}

#[tauri::command]
fn delete_meeting(app: AppHandle, db: State<Db>, id: String) -> Result<(), String> {
    store::delete_meeting(&db, &id)?;
    audio_store::delete(&app, &id);
    Ok(())
}

#[tauri::command]
fn list_meetings(db: State<Db>) -> Result<Vec<Meeting>, String> {
    store::list_meetings(&db)
}

#[tauri::command]
fn get_meeting(app: AppHandle, db: State<Db>, id: String) -> Result<Option<Meeting>, String> {
    match store::get_meeting(&db, &id)? {
        Some(mut m) => {
            m.has_audio = audio_store::has_audio(&app, &id);
            Ok(Some(m))
        }
        None => Ok(None),
    }
}

/// Write up a meeting: bring its per-stretch notes up to date (most were
/// extracted live), compose the summary (tokens stream on
/// `meeting-summary-token`), extract action items, title it if untitled.
/// Returns the updated meeting.
#[tauri::command]
async fn summarize_meeting(app: AppHandle, meeting_id: String, model_id: String) -> Result<Meeting, String> {
    let model_path = models::downloaded_model_path(&app, &model_id)?;
    let worker = app.clone();
    let id = meeting_id.clone();
    tauri::async_runtime::spawn_blocking(move || summary::summarize(&worker, &id, &model_path))
        .await
        .map_err(|e| e.to_string())??;
    let db = app.state::<Db>();
    full_meeting(&app, &db, &meeting_id)
}

/// Answer a question about one meeting (streams on `meeting-answer-token`).
#[tauri::command]
async fn ask_meeting(app: AppHandle, meeting_id: String, model_id: String, question: String) -> Result<String, String> {
    let model_path = models::downloaded_model_path(&app, &model_id)?;
    tauri::async_runtime::spawn_blocking(move || summary::ask(&app, &meeting_id, &model_path, &question))
        .await
        .map_err(|e| e.to_string())?
}

/// Answer a question across all meetings (streams on `meetings-answer-token`).
#[tauri::command]
async fn ask_meetings(app: AppHandle, model_id: String, question: String) -> Result<String, String> {
    let model_path = models::downloaded_model_path(&app, &model_id)?;
    tauri::async_runtime::spawn_blocking(move || summary::ask_all(&app, &model_path, &question))
        .await
        .map_err(|e| e.to_string())?
}

/// Full-text search across every meeting: transcript lines (with their time)
/// and meeting titles/summaries/notes.
#[tauri::command]
fn search_meetings(db: State<Db>, query: String) -> Result<Vec<store::SearchHit>, String> {
    store::search_meetings(&db, &query)
}

/// Start a live meeting: capture mic + system audio, transcribe as it goes,
/// extract notes in the background. With calendar access on, an untitled
/// meeting takes the current event's title and attendees.
#[tauri::command]
async fn meeting_start(
    app: AppHandle,
    meeting_id: String,
    model_id: String,
    language: Option<String>,
    llm_model_id: Option<String>,
    embedding_model_id: Option<String>,
) -> Result<Option<Meeting>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let s = settings::load(&app);
        let db = app.state::<Db>();
        if s.meetings.use_calendar && calendar::access() == calendar::CalAccess::Granted {
            if let Some(ev) = calendar::current_event() {
                if let Ok(Some(m)) = store::get_meeting(&db, &meeting_id) {
                    if m.title.trim().is_empty() && !ev.title.trim().is_empty() {
                        let _ = store::rename_meeting(&db, &meeting_id, ev.title.trim());
                    }
                }
                let _ = store::set_attendees(&db, &meeting_id, &ev.attendees);
            }
        }
        meeting::start_with_options(
            &app,
            meeting::StartOptions {
                meeting_id: meeting_id.clone(),
                stt_model: model_id,
                language,
                llm_model: llm_model_id,
                embedding_model: embedding_model_id,
            },
        )?;
        Ok(store::get_meeting(&db, &meeting_id)?)
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Stop capture and flush saved audio. Return the current meeting immediately;
/// durable transcription, speaker finalization, and end-of-meeting requests
/// continue in the background. Query meeting_work_status for completion.
#[tauri::command]
async fn meeting_stop(app: AppHandle) -> Result<Option<Meeting>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let Some(id) = meeting::stop(&app)? else {
            return Ok(None);
        };
        let db = app.state::<Db>();
        full_meeting(&app, &db, &id).map(Some)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
fn meeting_work_status(app: AppHandle) -> Result<Vec<meeting_work::Status>, String> {
    meeting::statuses(&app)
}

#[tauri::command]
fn meeting_pause(app: AppHandle, id: String) -> Result<(), String> { meeting::pause(&app,&id) }

#[tauri::command]
async fn meeting_retry(app: AppHandle, id: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || meeting::retry(&app,&id)).await.map_err(|e|e.to_string())?
}

/// Re-detect a meeting's remote speakers from its saved audio, optionally
/// with the known number of people.
#[tauri::command]
async fn diarize_meeting(
    app: AppHandle,
    meeting_id: String,
    embedding_model_id: Option<String>,
    num_speakers: Option<usize>,
) -> Result<Meeting, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let emb = embedding_model_id
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| "diarize-embedding".into());
        meeting::redetect_speakers(&app, &meeting_id, &emb, num_speakers)?;
        let db = app.state::<Db>();
        full_meeting(&app, &db, &meeting_id)
    })
    .await
    .map_err(|e| e.to_string())?
}

/// A playable WAV of part of a meeting (both channels mixed), as raw bytes.
#[tauri::command]
async fn meeting_clip(app: AppHandle, meeting_id: String, start_ms: i64, end_ms: i64) -> Result<tauri::ipc::Response, String> {
    let bytes = tauri::async_runtime::spawn_blocking(move || {
        audio_store::clip(&app, &meeting_id, start_ms, end_ms.min(start_ms + 10 * 60_000))
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(tauri::ipc::Response::new(bytes))
}

/// Flag a moment (now, when recording; else the given time).
#[tauri::command]
fn add_mark(app: AppHandle, db: State<Db>, meeting_id: String, at_ms: Option<i64>, label: String) -> Result<store::Mark, String> {
    let at = at_ms.or_else(|| meeting::position_ms(&app)).unwrap_or(0);
    store::add_mark(&db, &meeting_id, at, label.trim())
}

#[tauri::command]
fn delete_mark(db: State<Db>, id: i64) -> Result<(), String> {
    store::delete_mark(&db, id)
}

#[tauri::command]
fn open_action_items(db: State<Db>) -> Result<Vec<store::ActionItem>, String> {
    store::open_action_items(&db)
}

#[tauri::command]
fn save_action_item(db: State<Db>, item: store::ActionItem) -> Result<store::ActionItem, String> {
    store::save_action_item(&db, &item)
}

#[tauri::command]
fn delete_action_item(db: State<Db>, id: i64) -> Result<(), String> {
    store::delete_action_item(&db, id)
}

// ---- assistant tasks ---------------------------------------------------------

#[tauri::command]
fn list_tasks(db: State<Db>, meeting_id: Option<String>) -> Result<Vec<store::Task>, String> {
    store::list_tasks(&db, meeting_id.as_deref())
}

/// A request typed instead of spoken.
#[tauri::command]
fn add_task(app: AppHandle, meeting_id: Option<String>, text: String) -> Result<Option<store::Task>, String> {
    let at = meeting_id
        .as_ref()
        .filter(|id| meeting::active_meeting(&app).as_ref() == Some(*id))
        .and_then(|_| meeting::position_ms(&app));
    Ok(agent::on_request(&app, meeting_id, at, &text, "typed"))
}

#[tauri::command]
fn approve_task(app: AppHandle, id: String, draft: Option<serde_json::Value>) -> Result<store::Task, String> {
    agent::approve(&app, &id, draft)
}

#[tauri::command]
fn update_task_draft(app: AppHandle, id: String, draft: serde_json::Value, action: Option<String>) -> Result<store::Task, String> {
    agent::update_draft(&app, &id, draft, action)
}

#[tauri::command]
fn retry_task(app: AppHandle, id: String, utterance: Option<String>) -> Result<store::Task, String> {
    agent::retry(&app, &id, utterance)
}

#[tauri::command]
fn redraft_task(app: AppHandle, id: String) -> Result<store::Task, String> {
    agent::redraft(&app, &id)
}

#[tauri::command]
fn dismiss_task(app: AppHandle, id: String) -> Result<store::Task, String> {
    agent::dismiss(&app, &id)
}

#[tauri::command]
fn delete_task(db: State<Db>, id: String) -> Result<(), String> {
    store::delete_task(&db, &id)
}

/// Which model the assistant would use right now (or why it can't run).
#[tauri::command]
fn assistant_status(app: AppHandle) -> serde_json::Value {
    match brain::Brain::resolve(&app) {
        Ok(b) => serde_json::json!({ "ok": true, "model": b.describe(), "cloud": b.is_cloud() }),
        Err(e) => serde_json::json!({ "ok": false, "error": e }),
    }
}

/// A cloud provider's model ids, for the settings picker ("anthropic",
/// "openai", or "compatible" with its base URL).
#[tauri::command]
async fn list_cloud_models(provider: String, base: Option<String>) -> Result<Vec<String>, String> {
    tauri::async_runtime::spawn_blocking(move || brain::list_models(&provider, base.as_deref().unwrap_or("")))
        .await
        .map_err(|e| e.to_string())?
}

/// Which subscription CLIs are installed: (claude, codex) paths.
#[tauri::command]
fn subscription_clis() -> serde_json::Value {
    serde_json::json!({
        "claude": tools::claude_code::find_claude().map(|p| p.to_string_lossy().to_string()),
        "codex": tools::codex::find_codex().map(|p| p.to_string_lossy().to_string()),
    })
}

#[tauri::command]
fn set_assistant_settings(app: AppHandle, assistant: settings::AssistantSettings) -> Result<(), String> {
    let mut s = settings::load(&app);
    s.assistant = assistant;
    settings::save(&app, &s)?;
    let _ = app.emit("settings-changed", ());
    Ok(())
}

#[tauri::command]
fn set_meeting_settings(app: AppHandle, meetings: settings::MeetingSettings) -> Result<(), String> {
    let mut s = settings::load(&app);
    s.meetings = meetings;
    settings::save(&app, &s)?;
    detect::set_enabled(s.meetings.detect_meetings);
    let _ = app.emit("settings-changed", ());
    Ok(())
}

/// Secrets the settings UI may write (never read back — only "is it set?").
const SECRET_KEYS: [&str; 4] = [
    tools::jira::TOKEN_KEY,
    settings::ENDPOINT_KEY,
    settings::OPENAI_KEY,
    settings::ANTHROPIC_KEY,
];

#[tauri::command]
fn set_secret(key: String, value: String) -> Result<(), String> {
    if !SECRET_KEYS.contains(&key.as_str()) {
        return Err(format!("unknown secret '{key}'"));
    }
    secrets::set(&key, value.trim())
}

#[tauri::command]
fn has_secret(key: String) -> bool {
    SECRET_KEYS.contains(&key.as_str()) && secrets::get(&key).is_some_and(|v| !v.is_empty())
}

#[tauri::command]
async fn jira_test(app: AppHandle) -> Result<serde_json::Value, String> {
    let cfg = settings::load(&app).assistant.jira;
    let token = secrets::get(tools::jira::TOKEN_KEY).ok_or("add a Jira API token first")?;
    let who = tools::jira::test(&cfg, &token).await?;
    let projects = tools::jira::list_projects(&cfg, &token).await.unwrap_or_default();
    Ok(serde_json::json!({
        "user": who,
        "projects": projects.into_iter().map(|(k, n)| serde_json::json!({"key": k, "name": n})).collect::<Vec<_>>(),
    }))
}

#[tauri::command]
async fn mcp_probe(server: tools::mcp::McpServer) -> Result<Vec<tools::mcp::ToolInfo>, String> {
    tools::mcp::probe(&server).await
}

#[tauri::command]
fn mcp_parse(json: String) -> Result<Vec<tools::mcp::McpServer>, String> {
    tools::mcp::parse_mcp_json(&json)
}

/// Try the configured browser engine on a harmless, read-only page.
#[tauri::command]
async fn browser_test(app: AppHandle) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || agent::test_browser(&app))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
fn claude_code_path() -> Option<String> {
    tools::claude_code::find_claude().map(|p| p.to_string_lossy().to_string())
}

#[tauri::command]
fn list_voices(db: State<Db>) -> Result<Vec<(String, i64)>, String> {
    store::voiceprint_names(&db)
}

#[tauri::command]
fn delete_voice(db: State<Db>, name: String) -> Result<(), String> {
    store::delete_voiceprint(&db, &name)
}

#[tauri::command]
fn calendar_access() -> calendar::CalAccess {
    calendar::access()
}

#[tauri::command]
async fn calendar_request() -> Result<bool, String> {
    tauri::async_runtime::spawn_blocking(calendar::request_access)
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
fn calendar_current() -> Option<calendar::CalEvent> {
    calendar::current_event()
}

#[tauri::command]
fn set_command_keys(app: AppHandle, keys: Vec<String>) -> Result<(), String> {
    let mut s = settings::load(&app);
    let previous = s.clone();
    s.command_keys = keys;
    if let Err(e) = apply_shortcuts(&app, &s) {
        let _ = apply_shortcuts(&app, &previous);
        return Err(e);
    }
    settings::save(&app, &s)
}

// ---- the command key ----------------------------------------------------------

/// Command key down: during a meeting, claim the meeting's mic; otherwise
/// record a standalone request. The overlay shows it's listening.
fn command_down(app: &AppHandle) {
    let st = app.state::<AppState>();
    let result = if meeting::is_recording(app) {
        meeting::command_begin(app)
    } else {
        let mic = settings::load(app).mic_device;
        audio::start_recording(app.clone(), &st.audio, mic, audio::Owner::Dictation).map(|_| {
            *st.command_standalone.lock().unwrap() = true;
        })
    };
    match result {
        Ok(()) => {
            let _ = set_overlay(app.clone(), true);
            let _ = app.emit("overlay-state", "command");
        }
        Err(e) => overlay_toast(app, &format!("Can't listen right now: {e}")),
    }
}

/// Command key up: hand what was said to the assistant.
fn command_up(app: &AppHandle) {
    let st = app.state::<AppState>();
    let _ = app.emit("overlay-state", "processing");
    if meeting::command_active(app) {
        if let Err(e) = meeting::command_end(app) {
            overlay_toast(app, &e);
        }
        return;
    }
    if !std::mem::take(&mut *st.command_standalone.lock().unwrap()) {
        return;
    }
    let app = app.clone();
    std::thread::spawn(move || {
        let st = app.state::<AppState>();
        if audio::stop_recording(&st.audio).is_err() {
            return;
        }
        let samples = st.audio.last_recording.lock().unwrap().clone();
        let model = st.dictation_stt.lock().unwrap().clone();
        let text = (|| -> Result<String, String> {
            if samples.len() < 8_000 {
                return Err("that was too short to hear a request".into());
            }
            let model = if model.is_empty() {
                models::all_models(&app)?
                    .into_iter()
                    .find(|m| m.kind == models::ModelKind::Stt && m.downloaded)
                    .map(|m| m.id)
                    .ok_or("download a speech model first")?
            } else {
                model
            };
            let mut engine = asr::resolve(&app, &model)?;
            asr::transcribe_text(&app, &mut engine, &samples, None, vocab_prompt(&app).as_deref())
        })();
        match text {
            Ok(t) if !t.trim().is_empty() => {
                agent::on_request(&app, None, None, &t, "hotkey");
            }
            Ok(_) => overlay_toast(&app, "Didn't catch that — hold the key and try again."),
            Err(e) => overlay_toast(&app, &e),
        }
    });
}

/// Hold-to-ask from the meeting window (the same as the command key).
#[tauri::command]
fn command_begin(app: AppHandle) {
    command_down(&app);
}

#[tauri::command]
fn command_end(app: AppHandle) {
    command_up(&app);
}

/// A short message in the overlay bubble (assistant acknowledgements),
/// shown even while the main window is hidden.
pub fn overlay_toast(app: &AppHandle, text: &str) {
    let _ = set_overlay(app.clone(), true);
    let _ = app.emit("assistant-toast", text.to_string());
}

// ---- system-audio capture (ScreenCaptureKit spike) -------------------------

#[tauri::command]
fn system_audio_supported() -> bool {
    #[cfg(target_os = "macos")]
    {
        syscapture::is_supported()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

#[cfg(target_os = "macos")]
#[tauri::command]
fn start_system_capture(state: State<AppState>) -> Result<(), String> {
    syscapture::start(&state.sys)
}

#[cfg(target_os = "macos")]
#[tauri::command]
fn stop_system_capture(state: State<AppState>) -> Result<RecordingResult, String> {
    let r = syscapture::stop(&state.sys)?;
    Ok(RecordingResult {
        duration_secs: r.duration_secs,
        sample_count: r.sample_count,
    })
}

/// Diagnostic: dump the just-captured system audio to a WAV and report its
/// level, so capture can be verified by ear — no transcription involved.
#[cfg(target_os = "macos")]
#[tauri::command]
fn save_system_capture_wav(
    app: AppHandle,
    state: State<AppState>,
) -> Result<serde_json::Value, String> {
    let samples = state.sys.last.lock().unwrap().clone();
    let peak = samples.iter().fold(0f32, |m, &s| m.max(s.abs()));
    let rms = if samples.is_empty() {
        0.0
    } else {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    };
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let path = dir.join("system-audio-test.wav");
    syscapture::write_wav_16(&path, &samples, syscapture::CAPTURE_SAMPLE_RATE)?;
    Ok(serde_json::json!({
        "path": path.to_string_lossy(),
        "sampleCount": samples.len(),
        "durationSecs": samples.len() as f32 / syscapture::CAPTURE_SAMPLE_RATE as f32,
        "peak": peak,
        "rms": rms,
    }))
}

/// Spike helper: transcribe the just-captured system audio with a Whisper
/// model, reusing the same recognition-biasing vocabulary as the mic path.
#[cfg(target_os = "macos")]
#[tauri::command]
async fn transcribe_system_capture(
    app: AppHandle,
    state: State<'_, AppState>,
    model_id: String,
    language: Option<String>,
) -> Result<String, String> {
    let model_path = models::downloaded_model_path(&app, &model_id)?;
    let samples = state.sys.last.lock().unwrap().clone();
    let vocab: Vec<String> = {
        let mut seen = std::collections::HashSet::new();
        settings::load(&app)
            .dictionary
            .iter()
            .map(|e| e.to.trim().to_string())
            .filter(|t| !t.is_empty() && seen.insert(t.to_lowercase()))
            .collect()
    };
    let initial_prompt = (!vocab.is_empty()).then(|| format!("Glossary: {}.", vocab.join(", ")));
    let worker_app = app.clone();
    let text = tauri::async_runtime::spawn_blocking(move || {
        let state = worker_app.state::<AppState>();
        stt::transcribe(
            &state.stt,
            &model_path,
            &samples,
            language.as_deref(),
            initial_prompt.as_deref(),
        )
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(text)
}

/// (Re)register all global shortcuts. Combos involving the fn key go to the
/// low-level event-tap listener; everything else uses the OS hotkey API.
fn apply_shortcuts(app: &AppHandle, s: &Settings) -> Result<(), String> {
    let gs = app.global_shortcut();
    gs.unregister_all().map_err(|e| e.to_string())?;

    let mut modes = HashMap::new();
    let mut fn_bindings: Vec<(String, hotkeys::Mode)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let entries = s
        .hands_free
        .iter()
        .map(|c| (c, ShortcutMode::HandsFree))
        .chain(s.push_to_talk.iter().map(|c| (c, ShortcutMode::PushToTalk)))
        .chain(s.command_keys.iter().map(|c| (c, ShortcutMode::Command)));
    for (combo, mode) in entries {
        if combo.trim().is_empty() {
            continue;
        }
        if !seen.insert(combo.clone()) {
            return Err(format!("'{combo}' is assigned twice"));
        }
        if combo.split('+').any(|p| p == "fn") {
            let mode = match mode {
                ShortcutMode::HandsFree => hotkeys::Mode::HandsFree,
                ShortcutMode::PushToTalk => hotkeys::Mode::PushToTalk,
                ShortcutMode::Command => hotkeys::Mode::Command,
            };
            fn_bindings.push((combo.clone(), mode));
            continue;
        }
        let shortcut: Shortcut = combo
            .parse()
            .map_err(|e| format!("'{combo}' is not a valid shortcut: {e}"))?;
        modes.insert(shortcut.id(), mode);
        gs.register(shortcut)
            .map_err(|e| format!("could not register '{combo}': {e}"))?;
    }
    *app.state::<AppState>().shortcut_modes.lock().unwrap() = modes;

    hotkeys::set_bindings(&fn_bindings)?;
    if !fn_bindings.is_empty() && !hotkeys::ensure_listener(app) {
        return Err(
            "shortcuts using the fn key need the Accessibility permission (settings → permissions), then try again"
                .into(),
        );
    }
    Ok(())
}

// Room for the pill plus a three-line caption above it. The window is
// transparent and click-through, so the unused area is invisible.
const OVERLAY_W: f64 = 460.0;
const OVERLAY_H: f64 = 136.0;

/// Small always-on-top wave shown while dictating into other apps, with the
/// words so far streaming above it.
fn create_overlay(app: &AppHandle) -> tauri::Result<()> {
    let overlay = tauri::WebviewWindowBuilder::new(
        app,
        "overlay",
        tauri::WebviewUrl::App("index.html#overlay".into()),
    )
    .title("unsound")
    .inner_size(OVERLAY_W, OVERLAY_H)
    .decorations(false)
    .transparent(true)
    .always_on_top(true)
    .visible(false)
    .focused(false)
    .resizable(false)
    .skip_taskbar(true)
    .shadow(false)
    .accept_first_mouse(false)
    .visible_on_all_workspaces(true)
    .build()?;
    // Purely an indicator; clicks pass through to whatever is underneath.
    let _ = overlay.set_ignore_cursor_events(true);
    Ok(())
}

/// Park the overlay bottom-center of the monitor the cursor is on.
fn position_overlay(app: &AppHandle, overlay: &tauri::WebviewWindow) {
    let monitor = app
        .cursor_position()
        .ok()
        .and_then(|pos| app.monitor_from_point(pos.x, pos.y).ok().flatten())
        .or_else(|| app.primary_monitor().ok().flatten());
    let Some(monitor) = monitor else { return };
    let scale = monitor.scale_factor();
    let mpos = monitor.position();
    let msize = monitor.size();
    let w = OVERLAY_W * scale;
    let h = OVERLAY_H * scale;
    let x = mpos.x as f64 + (msize.width as f64 - w) / 2.0;
    let y = mpos.y as f64 + msize.height as f64 - h - 96.0 * scale;
    let _ = overlay.set_position(tauri::PhysicalPosition::new(x, y));
}

#[tauri::command]
fn set_overlay(app: AppHandle, visible: bool) -> Result<(), String> {
    let Some(overlay) = app.get_webview_window("overlay") else {
        return Ok(());
    };
    if visible {
        position_overlay(&app, &overlay);
        overlay.show().map_err(|e| e.to_string())?;
    } else {
        overlay.hide().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
fn start_shortcut_capture(app: AppHandle) -> bool {
    hotkeys::start_capture(&app)
}

#[tauri::command]
fn cancel_shortcut_capture() {
    hotkeys::cancel_capture();
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default();
    #[cfg(feature = "automation")]
    let builder = builder.plugin(tauri_plugin_automation::init())
        .manage(meeting_fixture::Fixtures::default());
    builder
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, shortcut, event| {
                    // The frontend owns the pipeline; these events drive it
                    // exactly like clicks on the record control.
                    let mode = app
                        .state::<AppState>()
                        .shortcut_modes
                        .lock()
                        .unwrap()
                        .get(&shortcut.id())
                        .copied();
                    match (mode, event.state()) {
                        (Some(ShortcutMode::HandsFree), ShortcutState::Pressed) => {
                            eprintln!("[shortcut] hands-free pressed");
                            let _ = app.emit("hotkey-toggle", ());
                        }
                        (Some(ShortcutMode::PushToTalk), ShortcutState::Pressed) => {
                            eprintln!("[shortcut] ptt pressed");
                            let _ = app.emit("ptt-down", ());
                        }
                        (Some(ShortcutMode::PushToTalk), ShortcutState::Released) => {
                            eprintln!("[shortcut] ptt released");
                            let _ = app.emit("ptt-up", ());
                        }
                        (Some(ShortcutMode::Command), ShortcutState::Pressed) => {
                            let _ = app.emit("command-down", ());
                        }
                        (Some(ShortcutMode::Command), ShortcutState::Released) => {
                            let _ = app.emit("command-up", ());
                        }
                        _ => {}
                    }
                })
                .build(),
        )
        .manage(AppState::default())
        .setup(|app| {
            #[cfg(feature = "automation")]
            if app.config().identifier != "com.unsound.e2e" {
                return Err("automation builds must use the isolated com.unsound.e2e profile".into());
            }
            // Open the local SQLite store (history + meetings) and manage it.
            app.manage(store::open(app.handle())?);
            let s = settings::load(app.handle());
            if !cfg!(feature = "automation") {
                if let Err(e) = apply_shortcuts(app.handle(), &s) {
                    eprintln!("global shortcuts unavailable: {e}");
                }
                tray::init(app.handle())?;
                create_overlay(app.handle())?;
            }
            let handle = app.handle().clone();
            // Meetings cut off by a crash or quit: close them out and make
            // their audio files consistent.
            match store::recover_open_meetings(&handle.state::<Db>()) {
                Ok(ids) => {
                    for id in ids {
                        for p in [audio_store::system_path(&handle, &id), audio_store::mic_path(&handle, &id)]
                            .into_iter()
                            .flatten()
                        {
                            let _ = audio_store::repair(&p);
                        }
                    }
                }
                Err(e) => eprintln!("[meeting] recovery failed: {e}"),
            }
            // The UI harness uses real IPC and storage, without global shortcuts,
            // background assistants, meeting detection, or microphone capture.
            if cfg!(feature = "automation") {
                meeting::start_recovery(handle.clone());
                #[cfg(feature = "automation")]
                meeting_fixture_runner::launch_if_requested(&handle);
                return Ok(());
            }
            agent::start(&handle);
            meeting::start_recovery(handle.clone());
            // The command key (from either shortcut path) is handled here,
            // not in the frontend: it must work with the window hidden.
            for (event, down) in [("command-down", true), ("command-up", false)] {
                let h = handle.clone();
                app.listen_any(event, move |_| {
                    let h = h.clone();
                    std::thread::spawn(move || if down { command_down(&h) } else { command_up(&h) });
                });
            }
            detect::set_enabled(s.meetings.detect_meetings);
            detect::create_prompt_window(&handle)?;
            detect::start(&handle);
            Ok(())
        })
        .on_window_event(|window, event| {
            // Live in the menu bar: closing the window hides it, the tray
            // (or the global shortcut) keeps working. Quit via the tray.
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let _ = window.hide();
                api.prevent_close();
            }
        })
        .invoke_handler(tauri::generate_handler![
            #[cfg(feature = "automation")]
            meeting_fixture::meeting_fixture,
            list_models,
            download_model,
            delete_model,
            add_custom_model,
            start_recording,
            stop_recording,
            transcribe,
            warm_stt,
            warm_llm,
            transcribe_file,
            start_live_dictation,
            stop_live_dictation,
            cleanup_text,
            default_cleanup_prompt,
            get_settings,
            set_shortcuts,
            set_styles,
            add_correction,
            set_dictionary,
            start_shortcut_capture,
            cancel_shortcut_capture,
            set_overlay,
            deliver_text,
            permission_status,
            request_accessibility,
            request_microphone,
            list_microphones,
            set_microphone,
            list_takes,
            save_take,
            delete_take,
            clear_takes,
            import_takes,
            create_meeting,
            update_meeting_notes,
            set_speaker_name,
            set_meeting_summary,
            update_segment_text,
            replace_in_meeting,
            rename_meeting,
            delete_meeting,
            list_meetings,
            get_meeting,
            summarize_meeting,
            ask_meeting,
            ask_meetings,
            search_meetings,
            meeting_start,
            meeting_stop,
            meeting_work_status,
            meeting_pause,
            meeting_retry,
            diarize_meeting,
            meeting_clip,
            add_mark,
            delete_mark,
            open_action_items,
            save_action_item,
            delete_action_item,
            list_tasks,
            add_task,
            approve_task,
            update_task_draft,
            retry_task,
            redraft_task,
            dismiss_task,
            delete_task,
            assistant_status,
            set_assistant_settings,
            set_meeting_settings,
            set_secret,
            has_secret,
            jira_test,
            mcp_probe,
            mcp_parse,
            claude_code_path,
            browser_test,
            list_cloud_models,
            subscription_clis,
            list_voices,
            delete_voice,
            calendar_access,
            calendar_request,
            calendar_current,
            set_command_keys,
            command_begin,
            command_end,
            detect::hide_meeting_prompt,
            system_audio_supported,
            #[cfg(target_os = "macos")]
            start_system_capture,
            #[cfg(target_os = "macos")]
            stop_system_capture,
            #[cfg(target_os = "macos")]
            save_system_capture_wav,
            #[cfg(target_os = "macos")]
            transcribe_system_capture
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            // Release cached whisper/llama models before process exit;
            // ggml's Metal teardown asserts if their buffers are still alive.
            if let tauri::RunEvent::Exit = event {
                let state = app.state::<AppState>();
                state.stt.clear();
                state.llm.clear();
            }
        });
}
