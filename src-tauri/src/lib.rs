pub mod audio;
pub mod audiofile;
mod deliver;
mod frontapp;
mod hotkeys;
mod llm;
mod models;
mod permissions;
mod settings;
mod stt;
mod tray;

use audio::{AudioState, RecordingResult};
use llm::LlmState;
use models::{ModelInfo, ModelKind};
use settings::Settings;
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
    let model_path = models::downloaded_model_path(&app, &model_id)?;
    let samples = state.audio.last_recording.lock().unwrap().clone();
    // The user's corrected vocabulary biases recognition.
    let vocab: Vec<String> = {
        let mut seen = std::collections::HashSet::new();
        settings::load(&app)
            .dictionary
            .iter()
            .map(|e| e.to.trim().to_string())
            .filter(|t| !t.is_empty() && seen.insert(t.to_lowercase()))
            .collect()
    };
    let initial_prompt =
        (!vocab.is_empty()).then(|| format!("Glossary: {}.", vocab.join(", ")));
    // Whisper inference is heavy; keep it off the async runtime.
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
    *state.last_output.lock().unwrap() = text.clone();
    Ok(text)
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
    let model_path = models::downloaded_model_path(&app, &model_id)?;
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
            set_microphone
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
