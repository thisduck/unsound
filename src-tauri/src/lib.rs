mod audio;
mod llm;
mod models;
mod stt;

use audio::{AudioState, RecordingResult};
use llm::LlmState;
use models::{ModelInfo, ModelKind};
use stt::SttState;
use tauri::{AppHandle, Emitter, Manager, State};

#[derive(Default)]
struct AppState {
    audio: AudioState,
    stt: SttState,
    llm: LlmState,
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
    audio::start_recording(app, &state.audio)
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
    // Whisper inference is heavy; keep it off the async runtime.
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        stt::transcribe(&state.stt, &model_path, &samples, language.as_deref())
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
) -> Result<String, String> {
    let model_path = models::downloaded_model_path(&app, &model_id)?;
    let system_prompt = prompt
        .filter(|p| !p.trim().is_empty())
        .unwrap_or_else(|| llm::DEFAULT_CLEANUP_PROMPT.to_string());
    let emitter = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        llm::cleanup_text(&app, &state.llm, &model_path, &system_prompt, &text)
    })
    .await
    .map_err(|e| e.to_string())?;
    let _ = emitter.emit("llm-done", result.is_ok());
    result
}

#[tauri::command]
fn default_cleanup_prompt() -> String {
    llm::DEFAULT_CLEANUP_PROMPT.to_string()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            list_models,
            download_model,
            delete_model,
            add_custom_model,
            start_recording,
            stop_recording,
            transcribe,
            cleanup_text,
            default_cleanup_prompt
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
