mod asr;
pub mod audio;
pub mod audiofile;
mod deliver;
mod diarize;
mod dictation;
mod frontapp;
mod hotkeys;
mod llm;
mod meeting;
mod models;
mod permissions;
mod remote;
mod settings;
mod store;
mod stt;
#[cfg(target_os = "macos")]
mod syscapture;
mod tray;

use audio::{AudioState, RecordingResult};
use llm::LlmState;
use models::{ModelInfo, ModelKind};
use settings::Settings;
use store::{Db, Meeting, Segment, Take};
use stt::SttState;
use std::collections::HashMap;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

#[derive(Clone, Copy, PartialEq)]
enum ShortcutMode {
    HandsFree,
    PushToTalk,
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

#[tauri::command]
fn start_recording(app: AppHandle, state: State<AppState>) -> Result<(), String> {
    let mic = settings::load(&app).mic_device;
    audio::start_recording(app, &state.audio, mic)
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
fn stop_recording(state: State<AppState>) -> Result<RecordingResult, String> {
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

/// Start live dictation captions: while recording, the words appear in the
/// window as you speak. The final cleaned text still lands at stop.
#[tauri::command]
fn start_live_dictation(
    app: AppHandle,
    model_id: String,
    language: Option<String>,
) -> Result<(), String> {
    // Validate the model is present; the engine is built on the live thread.
    models::downloaded_model_path(&app, &model_id)?;
    let initial_prompt = vocab_prompt(&app);
    dictation::start(&app, model_id, language, initial_prompt)
}

#[tauri::command]
fn stop_live_dictation(app: AppHandle) {
    dictation::stop(&app);
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
    // The base prompt is fixed; the user contributes additions on top.
    let mut system_prompt = llm::DEFAULT_CLEANUP_PROMPT.to_string();
    if let Some(additions) = prompt.filter(|p| !p.trim().is_empty()) {
        system_prompt.push_str(&format!(
            "\n\nAdditional instructions from the user (apply alongside the rules above):\n{additions}"
        ));
    }
    system_prompt.push_str(&llm::dictionary_addendum(&settings::load(&app).dictionary));
    let style = style_id
        .filter(|id| !id.is_empty())
        .and_then(|id| settings::load(&app).styles.into_iter().find(|s| s.id == id));
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

#[tauri::command]
fn set_cloud_settings(
    app: AppHandle,
    cloud_providers: Vec<settings::CloudProvider>,
    cloud_voice_provider: String,
    cloud_text_provider: String,
) -> Result<(), String> {
    let valid = ["openai", "mistral", "deepgram", "elevenlabs"];
    if cloud_providers.iter().any(|p| !valid.contains(&p.id.as_str())) {
        return Err("unsupported cloud provider".into());
    }
    let mut s = settings::load(&app);
    s.cloud_providers = cloud_providers;
    s.cloud_voice_provider = cloud_voice_provider;
    s.cloud_text_provider = cloud_text_provider;
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

#[tauri::command]
fn create_meeting(db: State<Db>, meeting: Meeting) -> Result<(), String> {
    store::create_meeting(&db, &meeting)
}

#[tauri::command]
fn add_meeting_segments(
    db: State<Db>,
    meeting_id: String,
    segments: Vec<Segment>,
) -> Result<(), String> {
    store::add_segments(&db, &meeting_id, &segments)
}

#[tauri::command]
fn end_meeting(
    db: State<Db>,
    id: String,
    ended_at: String,
    summary: String,
    title: Option<String>,
) -> Result<(), String> {
    store::end_meeting(&db, &id, &ended_at, &summary, title.as_deref())
}

#[tauri::command]
fn update_meeting_notes(db: State<Db>, id: String, notes: String) -> Result<(), String> {
    store::update_meeting_notes(&db, &id, &notes)
}

#[tauri::command]
fn set_speaker_name(
    db: State<Db>,
    meeting_id: String,
    speaker: String,
    name: String,
) -> Result<(), String> {
    store::set_speaker_name(&db, &meeting_id, &speaker, &name)
}

#[tauri::command]
fn set_meeting_summary(db: State<Db>, id: String, summary: String) -> Result<(), String> {
    store::set_meeting_summary(&db, &id, &summary)
}

#[tauri::command]
fn update_segment_text(db: State<Db>, segment_id: i64, text: String) -> Result<(), String> {
    store::update_segment_text(&db, segment_id, &text)
}

#[tauri::command]
fn rename_meeting(db: State<Db>, id: String, title: String) -> Result<(), String> {
    store::rename_meeting(&db, &id, &title)
}

#[tauri::command]
fn delete_meeting(app: AppHandle, db: State<Db>, id: String) -> Result<(), String> {
    store::delete_meeting(&db, &id)?;
    // Best-effort: drop the saved system audio too.
    if let Ok(p) = meeting_audio_path(&app, &id) {
        let _ = std::fs::remove_file(p);
    }
    Ok(())
}

#[tauri::command]
fn list_meetings(db: State<Db>) -> Result<Vec<Meeting>, String> {
    store::list_meetings(&db)
}

#[tauri::command]
fn get_meeting(db: State<Db>, id: String) -> Result<Option<Meeting>, String> {
    store::get_meeting(&db, &id)
}

/// Transcribe a finished meeting's two channels — mic ("me") and system audio
/// ("them") — into timestamped segments, persist them, and return the meeting.
#[tauri::command]
async fn transcribe_meeting(
    app: AppHandle,
    state: State<'_, AppState>,
    db: State<'_, Db>,
    meeting_id: String,
    model_id: String,
    language: Option<String>,
) -> Result<Meeting, String> {
    let model_path = models::downloaded_model_path(&app, &model_id)?;
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

    let mic_samples = state.audio.last_recording.lock().unwrap().clone();
    #[cfg(target_os = "macos")]
    let sys_samples = state.sys.last.lock().unwrap().clone();
    #[cfg(not(target_os = "macos"))]
    let sys_samples: Vec<f32> = Vec::new();

    let worker_app = app.clone();
    let (mic_segs, sys_segs) = tauri::async_runtime::spawn_blocking(
        move || -> Result<(Vec<(i64, i64, String)>, Vec<(i64, i64, String)>), String> {
            let s = worker_app.state::<AppState>();
            let mic = stt::transcribe_segments(
                &s.stt,
                &model_path,
                &mic_samples,
                language.as_deref(),
                initial_prompt.as_deref(),
            )?;
            let sys = stt::transcribe_segments(
                &s.stt,
                &model_path,
                &sys_samples,
                language.as_deref(),
                initial_prompt.as_deref(),
            )?;
            Ok((mic, sys))
        },
    )
    .await
    .map_err(|e| e.to_string())??;

    let mut segs: Vec<Segment> = Vec::new();
    for (start_ms, end_ms, text) in mic_segs {
        segs.push(Segment {
            id: 0,
            speaker: "me".into(),
            source: "mic".into(),
            start_ms,
            end_ms,
            text,
        });
    }
    for (start_ms, end_ms, text) in sys_segs {
        segs.push(Segment {
            id: 0,
            speaker: "them".into(),
            source: "system".into(),
            start_ms,
            end_ms,
            text,
        });
    }
    store::add_segments(&db, &meeting_id, &segs)?;
    store::get_meeting(&db, &meeting_id)?.ok_or_else(|| "meeting not found".to_string())
}

/// Summarize a meeting's stored transcript with the cleanup LLM; persist and
/// return the summary. Tokens stream to the UI on `meeting-summary-token`.
#[tauri::command]
async fn summarize_meeting(
    app: AppHandle,
    db: State<'_, Db>,
    meeting_id: String,
    model_id: String,
) -> Result<String, String> {
    let meeting =
        store::get_meeting(&db, &meeting_id)?.ok_or_else(|| "meeting not found".to_string())?;
    let mut transcript = String::new();
    for s in &meeting.segments {
        transcript.push_str(&format!("{}: {}\n", speaker_label(&s.speaker), s.text));
    }
    if transcript.trim().is_empty() {
        return Ok(String::new());
    }
    let summary = if model_id == "cloud" {
        remote::summarize(&settings::load(&app), &transcript).await?
    } else {
        let model_path = models::downloaded_model_path(&app, &model_id)?;
        let worker_app = app.clone();
        tauri::async_runtime::spawn_blocking(move || {
            let s = worker_app.state::<AppState>();
            llm::summarize_meeting(&worker_app, &s.llm, &model_path, &transcript)
        })
        .await
        .map_err(|e| e.to_string())??
    };
    store::set_meeting_summary(&db, &meeting_id, &summary)?;
    Ok(summary)
}

/// Answer a question about one meeting from its transcript + summary + notes.
/// The answer streams to the UI on `meeting-answer-token`.
#[tauri::command]
async fn ask_meeting(
    app: AppHandle,
    db: State<'_, Db>,
    meeting_id: String,
    model_id: String,
    question: String,
) -> Result<String, String> {
    let meeting =
        store::get_meeting(&db, &meeting_id)?.ok_or_else(|| "meeting not found".to_string())?;
    let mut context = String::new();
    if !meeting.summary.trim().is_empty() {
        context.push_str(&format!("Summary:\n{}\n\n", meeting.summary));
    }
    if !meeting.notes.trim().is_empty() {
        context.push_str(&format!("My notes:\n{}\n\n", meeting.notes));
    }
    context.push_str("Transcript:\n");
    for s in &meeting.segments {
        context.push_str(&format!("{}: {}\n", speaker_label(&s.speaker), s.text));
    }
    let answer = if model_id == "cloud" {
        remote::answer(&settings::load(&app), &context, &question).await?
    } else {
        let model_path = models::downloaded_model_path(&app, &model_id)?;
        let worker_app = app.clone();
        tauri::async_runtime::spawn_blocking(move || {
            let s = worker_app.state::<AppState>();
            llm::answer_meeting_question(&worker_app, &s.llm, &model_path, &context, &question)
        })
        .await
        .map_err(|e| e.to_string())??
    };
    Ok(answer)
}

/// Search across every meeting by title, summary, notes, and transcript text.
#[tauri::command]
fn search_meetings(db: State<Db>, query: String) -> Result<Vec<store::SearchHit>, String> {
    store::search_meetings(&db, &query)
}

/// Render a segment's speaker for the LLM: "Me" for the user, "Speaker N" for a
/// diarized remote participant (`them:0` → Speaker 1), else "Them".
fn speaker_label(speaker: &str) -> String {
    if speaker == "me" {
        return "Me".into();
    }
    if let Some(n) = speaker.strip_prefix("them:") {
        if let Ok(idx) = n.parse::<i32>() {
            return format!("Speaker {}", idx + 1);
        }
    }
    "Them".into()
}

/// Start a live meeting: capture mic + system audio and transcribe in rolling
/// windows, emitting `meeting-segments` as they finalize.
#[tauri::command]
async fn meeting_start(
    app: AppHandle,
    meeting_id: String,
    model_id: String,
    language: Option<String>,
) -> Result<(), String> {
    // Capture startup waits on device/stream readiness — keep it off the async runtime.
    tauri::async_runtime::spawn_blocking(move || {
        meeting::start_with_model(&app, meeting_id, model_id, language)
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Stop the live meeting: flush the tail audio and tear down capture. The
/// transcript is already complete, so the caller can summarize immediately.
#[tauri::command]
async fn meeting_stop(app: AppHandle) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || meeting::stop(&app))
        .await
        .map_err(|e| e.to_string())?
}

/// Where a meeting's system audio is stored, so diarization can be re-run later.
fn meeting_audio_path(app: &AppHandle, id: &str) -> Result<std::path::PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("meeting-audio");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir.join(format!("{id}.wav")))
}

/// Persist the just-captured system audio for a meeting, so its speakers can be
/// re-detected later (with a corrected count) without the original session.
#[cfg(target_os = "macos")]
#[tauri::command]
fn save_meeting_audio(
    app: AppHandle,
    state: State<AppState>,
    meeting_id: String,
) -> Result<(), String> {
    let samples = state.sys.last.lock().unwrap().clone();
    if samples.is_empty() {
        return Ok(());
    }
    let path = meeting_audio_path(&app, &meeting_id)?;
    syscapture::write_wav_16(&path, &samples, syscapture::CAPTURE_SAMPLE_RATE)
}

/// Diarize a meeting's system-audio channel into speakers and relabel its "them"
/// segments as `them:0`, `them:1`, … (shown as Speaker 1/2). Works on the just-
/// finished meeting or, from the saved audio, any past one.
#[cfg(target_os = "macos")]
#[tauri::command]
async fn diarize_meeting(
    app: AppHandle,
    state: State<'_, AppState>,
    db: State<'_, Db>,
    meeting_id: String,
    embedding_model_id: Option<String>,
    num_speakers: Option<i32>,
) -> Result<Meeting, String> {
    let seg_model = models::downloaded_model_path(&app, "diarize-segmentation")?;
    let emb_id = embedding_model_id.unwrap_or_else(|| "diarize-embedding".to_string());
    let emb_model = models::downloaded_model_path(&app, &emb_id)?;
    // Prefer the meeting's saved system audio (lets past meetings be re-run);
    // fall back to the just-captured audio for the meeting that just ended.
    let wav = meeting_audio_path(&app, &meeting_id).ok().filter(|p| p.exists());
    let sys_last = state.sys.last.lock().unwrap().clone();
    let spans = tauri::async_runtime::spawn_blocking(
        move || -> Result<Vec<diarize::Span>, String> {
            let samples = match wav {
                Some(p) => audiofile::decode_to_samples(&p).unwrap_or(sys_last),
                None => sys_last,
            };
            // Nothing meaningful to cluster in under a second of audio.
            if samples.len() < 16_000 {
                return Ok(Vec::new());
            }
            // Higher default threshold than sherpa's 0.5 to curb over-splitting;
            // an explicit speaker count (when set) overrides it.
            diarize::diarize(&seg_model, &emb_model, samples, 0.7, num_speakers)
        },
    )
    .await
    .map_err(|e| e.to_string())??;

    let meeting =
        store::get_meeting(&db, &meeting_id)?.ok_or_else(|| "meeting not found".to_string())?;
    if spans.is_empty() {
        return Ok(meeting);
    }

    // Assign each system-channel segment to the speaker span it overlaps most.
    let mut updates: Vec<(i64, String)> = Vec::new();
    for seg in &meeting.segments {
        if seg.source != "system" {
            continue;
        }
        let (s0, s1) = (seg.start_ms as f32 / 1000.0, seg.end_ms as f32 / 1000.0);
        let mut best: Option<(f32, i32)> = None;
        for sp in &spans {
            let overlap = (s1.min(sp.end) - s0.max(sp.start)).max(0.0);
            if overlap > 0.0 && best.map_or(true, |(b, _)| overlap > b) {
                best = Some((overlap, sp.speaker));
            }
        }
        if let Some((_, speaker)) = best {
            updates.push((seg.id, format!("them:{speaker}")));
        }
    }
    if !updates.is_empty() {
        store::update_segment_speakers(&db, &updates)?;
    }
    store::get_meeting(&db, &meeting_id)?.ok_or_else(|| "meeting not found".to_string())
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
        .chain(s.push_to_talk.iter().map(|c| (c, ShortcutMode::PushToTalk)));
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

const OVERLAY_W: f64 = 200.0;
const OVERLAY_H: f64 = 56.0;

/// Small always-on-top wave shown while dictating into other apps.
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
    tauri::Builder::default()
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
                        _ => {}
                    }
                })
                .build(),
        )
        .manage(AppState::default())
        .setup(|app| {
            // Open the local SQLite store (history + meetings) and manage it.
            app.manage(store::open(app.handle())?);
            let s = settings::load(app.handle());
            if let Err(e) = apply_shortcuts(app.handle(), &s) {
                eprintln!("global shortcuts unavailable: {e}");
            }
            tray::init(app.handle())?;
            create_overlay(app.handle())?;
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
            list_models,
            download_model,
            delete_model,
            add_custom_model,
            start_recording,
            stop_recording,
            transcribe,
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
            set_cloud_settings,
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
            add_meeting_segments,
            end_meeting,
            update_meeting_notes,
            set_speaker_name,
            set_meeting_summary,
            update_segment_text,
            rename_meeting,
            delete_meeting,
            list_meetings,
            get_meeting,
            transcribe_meeting,
            summarize_meeting,
            ask_meeting,
            search_meetings,
            meeting_start,
            meeting_stop,
            #[cfg(target_os = "macos")]
            diarize_meeting,
            #[cfg(target_os = "macos")]
            save_meeting_audio,
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
