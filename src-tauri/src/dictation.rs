//! Live captions for dictation. While an in-app dictation take is recording,
//! this periodically re-transcribes the whole take-so-far and emits it on
//! `dictation-live`, so the words appear in the window as you speak. The final,
//! authoritative transcription + cleanup still happens once at stop (this only
//! drives the live preview and, as a bonus, warms the model).

use crate::{audio, stt, AppState};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};

const POLL_MS: u64 = 1200;

#[derive(Default)]
pub struct LiveState {
    active: Mutex<Option<Handle>>,
}

struct Handle {
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

pub fn start(
    app: &AppHandle,
    model_path: PathBuf,
    language: Option<String>,
    initial_prompt: Option<String>,
) -> Result<(), String> {
    let st = app.state::<AppState>();
    let mut active = st.live.active.lock().unwrap();
    if active.is_some() {
        return Ok(());
    }
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = stop.clone();
    let app_thread = app.clone();
    let join = std::thread::spawn(move || {
        run_live(app_thread, model_path, language, initial_prompt, stop_thread)
    });
    *active = Some(Handle {
        stop,
        join: Some(join),
    });
    Ok(())
}

pub fn stop(app: &AppHandle) {
    let handle = app
        .state::<AppState>()
        .live
        .active
        .lock()
        .unwrap()
        .take();
    if let Some(mut h) = handle {
        h.stop.store(true, Ordering::Relaxed);
        if let Some(j) = h.join.take() {
            let _ = j.join();
        }
    }
}

fn run_live(
    app: AppHandle,
    model_path: PathBuf,
    language: Option<String>,
    initial_prompt: Option<String>,
    stop: Arc<AtomicBool>,
) {
    loop {
        let mut waited = 0;
        while waited < POLL_MS / 100 && !stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(100));
            waited += 1;
        }
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let st = app.state::<AppState>();
        let samples = audio::snapshot_16k(&st.audio);
        if samples.len() < 1600 {
            continue;
        }
        if let Ok(text) = stt::transcribe(
            &st.stt,
            &model_path,
            &samples,
            language.as_deref(),
            initial_prompt.as_deref(),
        ) {
            if !text.is_empty() {
                let _ = app.emit("dictation-live", text);
            }
        }
    }
}
