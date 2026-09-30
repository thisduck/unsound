//! Background work while a dictation take is recording:
//!  • **live captions**: periodically re-transcribe the whole take-so-far and
//!    emit it on `dictation-live`, so the words appear as you speak — in the
//!    window, or the overlay bubble for hotkey takes;
//!  • **language detection** (spoken language on auto, Whisper models): name
//!    the language from the first few seconds, so the final transcription can
//!    skip Whisper's own detection — a second full encoder pass that roughly
//!    doubles the time on a short take.
//! The final, authoritative transcription + cleanup still happens once at stop.

use crate::{asr, audio, stt, AppState};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager};

/// Minimum gap between caption passes. Each pass re-reads the whole take, so
/// the gap also stretches to twice the last pass, keeping the GPU at most half
/// busy on long takes.
const CAPTION_GAP_MS: u64 = 600;
/// Voiced audio needed before trying to detect the language — judging a
/// silent lead-in or room noise gives a confident but meaningless guess.
const DETECT_MIN_VOICED_SECS: f32 = 1.5;
/// How often to re-check for enough speech, and how long to wait for more
/// audio before retrying after an unsure guess.
const DETECT_POLL_MS: u64 = 250;
const DETECT_RETRY_MS: u64 = 1000;
/// A 30 ms frame counts as voiced above this RMS (mic noise sits well below).
const VOICED_FRAME: usize = 480;
const VOICED_RMS: f32 = 0.01;
const TICK_MS: u64 = 100;

#[derive(Default)]
pub struct LiveState {
    active: Mutex<Option<Handle>>,
    /// The spoken language detected for the current take, if any.
    detected: Mutex<Option<String>>,
}

struct Handle {
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

pub fn start(
    app: &AppHandle,
    model_id: String,
    language: Option<String>,
    initial_prompt: Option<String>,
) -> Result<(), String> {
    // A fresh take: end any leftover worker and forget its language.
    stop(app);
    let st = app.state::<AppState>();
    *st.live.detected.lock().unwrap() = None;
    let detect_with = match language {
        None => asr::whisper_path(app, &model_id)?,
        Some(_) => None,
    };
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_thread = stop_flag.clone();
    let app_thread = app.clone();
    let join = std::thread::spawn(move || {
        run_live(
            app_thread,
            model_id,
            language,
            initial_prompt,
            detect_with,
            stop_thread,
        )
    });
    *st.live.active.lock().unwrap() = Some(Handle {
        stop: stop_flag,
        join: Some(join),
    });
    Ok(())
}

pub fn stop(app: &AppHandle) {
    // Hold the lock through the join so a concurrent `finish` waits for an
    // in-flight detection to land rather than reading too early.
    let st = app.state::<AppState>();
    let mut active = st.live.active.lock().unwrap();
    if let Some(mut h) = active.take() {
        h.stop.store(true, Ordering::Relaxed);
        if let Some(j) = h.join.take() {
            let _ = j.join();
        }
    }
}

/// Stop the take's background work and hand over the language it detected.
/// Taken, not copied: re-running transcription later detects afresh.
pub fn finish(app: &AppHandle) -> Option<String> {
    stop(app);
    app.state::<AppState>().live.detected.lock().unwrap().take()
}

fn run_live(
    app: AppHandle,
    model_id: String,
    language: Option<String>,
    initial_prompt: Option<String>,
    mut detect_with: Option<PathBuf>,
    stop: Arc<AtomicBool>,
) {
    // Build the engine once on this thread (sherpa recognizers are !Send).
    let mut engine = match asr::resolve(&app, &model_id) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("[dictation] live captions unavailable: {e}");
            None
        }
    };
    let mut language = language;
    let prompt = initial_prompt.as_deref();
    let mut next_detect = Instant::now();
    let mut next_caption = Instant::now() + Duration::from_millis(CAPTION_GAP_MS);
    while !stop.load(Ordering::Relaxed) && (engine.is_some() || detect_with.is_some()) {
        std::thread::sleep(Duration::from_millis(TICK_MS));
        let now = Instant::now();
        if let Some(path) = detect_with.clone().filter(|_| now >= next_detect) {
            let st = app.state::<AppState>();
            let samples = audio::snapshot_16k(&st.audio);
            if voiced_secs(&samples) < DETECT_MIN_VOICED_SECS {
                next_detect = now + Duration::from_millis(DETECT_POLL_MS);
            } else {
                match stt::detect_language(&st.stt, &path, &samples) {
                    Ok(Some(lang)) => {
                        *st.live.detected.lock().unwrap() = Some(lang.clone());
                        language = Some(lang);
                        detect_with = None;
                    }
                    Ok(None) => next_detect = now + Duration::from_millis(DETECT_RETRY_MS),
                    Err(e) => {
                        eprintln!("[dictation] language detection failed: {e}");
                        detect_with = None;
                    }
                }
            }
        }
        // With the language on auto, wait for it: Whisper's own guess on the
        // first second or two is unreliable and would caption another language.
        let caption_due = now >= next_caption && detect_with.is_none();
        if let Some(engine) = engine.as_mut().filter(|_| caption_due) {
            next_caption = now + Duration::from_millis(CAPTION_GAP_MS);
            let samples = audio::snapshot_16k(&app.state::<AppState>().audio);
            if samples.len() < 1600 {
                continue;
            }
            // Abandoned when the take ends, so the final pass never waits on it.
            let lang = language.as_deref();
            let result = asr::transcribe_preview(&app, engine, &samples, lang, prompt, stop.clone());
            next_caption = next_caption.max(Instant::now() + now.elapsed() * 2);
            if let Ok(text) = result {
                if !text.is_empty() && !stop.load(Ordering::Relaxed) {
                    let _ = app.emit("dictation-live", text);
                }
            }
        }
    }
}

/// Seconds of `samples` (16 kHz) that carry speech-level energy.
fn voiced_secs(samples: &[f32]) -> f32 {
    let voiced = samples
        .chunks_exact(VOICED_FRAME)
        .filter(|f| (f.iter().map(|s| s * s).sum::<f32>() / f.len() as f32).sqrt() >= VOICED_RMS)
        .count();
    (voiced * VOICED_FRAME) as f32 / audio::WHISPER_SAMPLE_RATE as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_only_frames_with_speech_energy() {
        let quiet = vec![0.003f32; 16_000];
        let loud: Vec<f32> = (0..16_000).map(|i| if i % 2 == 0 { 0.1 } else { -0.1 }).collect();
        assert_eq!(voiced_secs(&quiet), 0.0);
        let mixed = [quiet.clone(), loud.clone()].concat();
        assert!((voiced_secs(&mixed) - 0.99).abs() < 0.01);
    }
}
