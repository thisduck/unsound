//! Background work while a dictation take is recording:
//!  • **live captions**: periodically re-transcribe the whole take-so-far and
//!    emit it on `dictation-live`, so the words appear as you speak — in the
//!    window, or the overlay bubble for hotkey takes;
//!  • **language detection** (spoken language on auto): name the language
//!    from the first few seconds, so the final transcription can skip
//!    Whisper's own detection — a second full encoder pass that roughly
//!    doubles the time on a short take — and, for a model with only some
//!    languages (Parakeet), so speech in others goes to Whisper.
//! The final, authoritative transcription + cleanup still happens once at stop.

use crate::{asr, audio, stt, AppState};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
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

/// Completed takes own their audio and detected language until their turn in
/// the processing queue. A later recording must never replace either one.
pub struct RecordedTake {
    pub samples: Vec<f32>,
    pub language: Option<String>,
}

#[derive(Default)]
pub struct PendingTakes(Mutex<HashMap<String, RecordedTake>>);

impl PendingTakes {
    pub fn insert(&self, id: String, take: RecordedTake) {
        self.0.lock().unwrap().insert(id, take);
    }

    pub fn take(&self, id: &str) -> Result<RecordedTake, String> {
        self.0.lock().unwrap().remove(id).ok_or_else(|| "recording is no longer available".into())
    }
}

#[derive(Default)]
pub struct LiveState {
    active: Mutex<Option<Handle>>,
}

struct Handle {
    stop: Arc<AtomicBool>,
    /// Owned by this take, including after its worker has been stopped.
    detected: Arc<Mutex<Option<String>>>,
}

impl LiveState {
    fn finish(&self) -> Option<String> {
        let handle = self.active.lock().unwrap().take()?;
        handle.stop.store(true, Ordering::Relaxed);
        // Native inference can take time to notice cancellation. Capture must
        // not join it; a late detection stays with the retired take instead.
        let language = handle.detected.lock().unwrap().take();
        language
    }
}

pub fn start(
    app: &AppHandle,
    take_id: String,
    model_id: String,
    language: Option<String>,
    initial_prompt: Option<String>,
) -> Result<(), String> {
    // A fresh take: retire the previous preview without waiting on inference.
    stop(app);
    let st = app.state::<AppState>();
    let selection = asr::Selection::load(app, &model_id)?;
    // A Whisper model detects with itself; a model with only some languages
    // with the fallback Whisper model, when there is one.
    let detect_with = match language {
        Some(_) => None,
        None if selection.is_limited() => selection.detector().map(PathBuf::from),
        None if selection.info.engine == crate::models::SttEngine::Whisper => {
            Some(crate::models::downloaded_model_path(app, &model_id)?)
        }
        None => None,
    };
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_thread = stop_flag.clone();
    let detected = Arc::new(Mutex::new(None));
    let detected_thread = detected.clone();
    let app_thread = app.clone();
    std::thread::spawn(move || {
        run_live(
            app_thread,
            take_id,
            selection,
            language,
            initial_prompt,
            detect_with,
            stop_thread,
            detected_thread,
        )
    });
    *st.live.active.lock().unwrap() = Some(Handle {
        stop: stop_flag,
        detected,
    });
    Ok(())
}

pub fn stop(app: &AppHandle) {
    finish(app);
}

/// Stop the take's background work and hand over the language it detected.
/// Taken, not copied: re-running transcription later detects afresh.
pub fn finish(app: &AppHandle) -> Option<String> {
    app.state::<AppState>().live.finish()
}

/// The caption engine, picked once the spoken language is settled.
enum Captions {
    Pending,
    Ready(asr::Engine),
    Off,
}

#[allow(clippy::too_many_arguments)]
fn run_live(
    app: AppHandle,
    take_id: String,
    selection: asr::Selection,
    language: Option<String>,
    initial_prompt: Option<String>,
    mut detect_with: Option<PathBuf>,
    stop: Arc<AtomicBool>,
    detected: Arc<Mutex<Option<String>>>,
) {
    if stop.load(Ordering::Relaxed) { return; }
    let could_detect = detect_with.is_some();
    let mut captions = Captions::Pending;
    let mut language = language;
    let prompt = initial_prompt.as_deref();
    let mut next_detect = Instant::now();
    let mut next_caption = Instant::now() + Duration::from_millis(CAPTION_GAP_MS);
    while !stop.load(Ordering::Relaxed) && (!matches!(captions, Captions::Off) || detect_with.is_some()) {
        std::thread::sleep(Duration::from_millis(TICK_MS));
        if stop.load(Ordering::Relaxed) { break; }
        let now = Instant::now();
        if let Some(path) = detect_with.clone().filter(|_| now >= next_detect) {
            let st = app.state::<AppState>();
            let samples = audio::snapshot_16k(&st.audio);
            if voiced_secs(&samples) < DETECT_MIN_VOICED_SECS {
                next_detect = now + Duration::from_millis(DETECT_POLL_MS);
            } else {
                match stt::detect_language(&st.stt, &path, &samples) {
                    Ok(Some(lang)) => {
                        *detected.lock().unwrap() = Some(lang.clone());
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
        if stop.load(Ordering::Relaxed) { break; }
        // With the language on auto, wait for it: Whisper's own guess on the
        // first second or two is unreliable and would caption another language.
        let caption_due = now >= next_caption && detect_with.is_none();
        if caption_due && matches!(captions, Captions::Pending) {
            // Built on this thread (Moonshine is !Send).
            captions = match caption_engine(&app, &selection, language.as_deref(), could_detect) {
                Ok(engine) => Captions::Ready(engine),
                Err(e) => {
                    eprintln!("[dictation] live captions unavailable: {e}");
                    Captions::Off
                }
            };
        }
        let engine = match &mut captions {
            Captions::Ready(engine) if caption_due => engine,
            _ => continue,
        };
        next_caption = now + Duration::from_millis(CAPTION_GAP_MS);
        let samples = audio::snapshot_16k(&app.state::<AppState>().audio);
        if samples.len() < 1600 {
            continue;
        }
        // Abandoned when the take ends, so the final pass never waits on it.
        let lang = language.as_deref();
        let st = app.state::<AppState>();
        let result = asr::transcribe_preview(asr::Caches::of(&st), engine, &samples, lang, prompt, stop.clone());
        next_caption = next_caption.max(Instant::now() + now.elapsed() * 2);
        if let Ok(text) = result {
            if !text.is_empty() && !stop.load(Ordering::Relaxed) {
                let _ = app.emit("dictation-live", serde_json::json!({ "takeId": take_id, "text": text }));
            }
        }
    }
}

/// The engine captions use, given what is known about the language. Speech the
/// selected model can't transcribe gets the fallback Whisper model, or none.
fn caption_engine(
    app: &AppHandle,
    selection: &asr::Selection,
    language: Option<&str>,
    could_detect: bool,
) -> Result<asr::Engine, String> {
    let heard = match language {
        Some(lang) => asr::Heard::Known(lang),
        None if could_detect => asr::Heard::Unsure,
        None => asr::Heard::Unknown,
    };
    match selection.route(heard) {
        asr::Route::Selected => asr::resolve(app, &selection.info.id),
        asr::Route::Fallback => {
            Ok(asr::Engine::Whisper(selection.fallback.clone().map(|(_, p)| p).unwrap_or_default()))
        }
        asr::Route::Refuse => Err(asr::refusal(&selection.info, language)),
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
    fn finishing_does_not_wait_for_or_adopt_an_older_detection() {
        let live = LiveState::default();
        let old_stop = Arc::new(AtomicBool::new(false));
        let old_language = Arc::new(Mutex::new(None));
        *live.active.lock().unwrap() = Some(Handle {
            stop: old_stop.clone(), detected: old_language.clone(),
        });
        // The old native inference has not returned. Finish still releases it.
        assert_eq!(live.finish(), None);
        assert!(old_stop.load(Ordering::Relaxed));
        *live.active.lock().unwrap() = Some(Handle {
            stop: Arc::new(AtomicBool::new(false)),
            detected: Arc::new(Mutex::new(Some("fr".into()))),
        });
        // A late result cannot supply the language of the new take.
        *old_language.lock().unwrap() = Some("en".into());
        assert_eq!(live.finish().as_deref(), Some("fr"));
        assert_eq!(live.finish(), None);
    }

    #[test]
    fn counts_only_frames_with_speech_energy() {
        let quiet = vec![0.003f32; 16_000];
        let loud: Vec<f32> = (0..16_000).map(|i| if i % 2 == 0 { 0.1 } else { -0.1 }).collect();
        assert_eq!(voiced_secs(&quiet), 0.0);
        let mixed = [quiet.clone(), loud.clone()].concat();
        assert!((voiced_secs(&mixed) - 0.99).abs() < 0.01);
    }
}
