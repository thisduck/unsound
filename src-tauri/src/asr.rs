//! Alternative on-device ASR via sherpa-onnx — currently Moonshine, a fast
//! English model for live meeting captions. Whisper stays the default and the
//! only option for non-English. The recognizer holds a raw sherpa pointer, so
//! it is not Send and must live on the thread that uses it (the meeting loop).

use crate::{models, stt, AppState};
use sherpa_rs::moonshine::{MoonshineConfig, MoonshineRecognizer};
use sherpa_rs::transducer::{TransducerConfig, TransducerRecognizer};
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};

pub struct Moonshine {
    rec: MoonshineRecognizer,
}

impl Moonshine {
    /// Build from a downloaded Moonshine bundle directory (the five files the
    /// registry entry pulls in).
    pub fn new(dir: &Path) -> Result<Self, String> {
        let p = |f: &str| dir.join(f).to_string_lossy().to_string();
        let config = MoonshineConfig {
            preprocessor: p("preprocess.onnx"),
            encoder: p("encode.int8.onnx"),
            uncached_decoder: p("uncached_decode.int8.onnx"),
            cached_decoder: p("cached_decode.int8.onnx"),
            tokens: p("tokens.txt"),
            ..Default::default()
        };
        let rec = MoonshineRecognizer::new(config).map_err(|e| e.to_string())?;
        Ok(Self { rec })
    }

    /// Transcribe 16 kHz mono samples to text (empty if nothing recognized).
    pub fn transcribe(&mut self, samples: &[f32]) -> String {
        self.rec
            .transcribe(crate::audio::WHISPER_SAMPLE_RATE, samples)
            .text
            .trim()
            .to_string()
    }
}

/// Parakeet TDT (NeMo transducer) — fast, multilingual on-device ASR.
pub struct Parakeet {
    rec: TransducerRecognizer,
}

impl Parakeet {
    /// Build from a downloaded Parakeet bundle directory (encoder/decoder/
    /// joiner/tokens).
    pub fn new(dir: &Path) -> Result<Self, String> {
        let p = |f: &str| dir.join(f).to_string_lossy().to_string();
        let config = TransducerConfig {
            encoder: p("encoder.int8.onnx"),
            decoder: p("decoder.int8.onnx"),
            joiner: p("joiner.int8.onnx"),
            tokens: p("tokens.txt"),
            num_threads: 2,
            sample_rate: crate::audio::WHISPER_SAMPLE_RATE as i32,
            feature_dim: 80,
            decoding_method: "greedy_search".to_string(),
            model_type: "nemo_transducer".to_string(),
            ..Default::default()
        };
        let rec = TransducerRecognizer::new(config).map_err(|e| e.to_string())?;
        Ok(Self { rec })
    }

    /// Transcribe 16 kHz mono samples to text.
    pub fn transcribe(&mut self, samples: &[f32]) -> String {
        self.rec
            .transcribe(crate::audio::WHISPER_SAMPLE_RATE, samples)
            .trim()
            .to_string()
    }
}

/// The active speech engine. Whisper uses the shared cached context in
/// `AppState.stt`; the sherpa recognizers (Moonshine/Parakeet) own their model
/// and are !Send, so an `Engine` must be built and used on one thread.
pub enum Engine {
    Whisper(PathBuf),
    Moonshine(Moonshine),
    Parakeet(Parakeet),
    /// Failed to initialize — callers produce no transcript.
    None,
}

/// Resolve the selected STT model id into a ready engine. For sherpa models
/// this loads the recognizer, so call it on the thread that will use it.
pub fn resolve(app: &AppHandle, model_id: &str) -> Result<Engine, String> {
    let info = models::find_model(app, model_id)?;
    let path = models::downloaded_model_path(app, model_id)?;
    if info.files.is_empty() {
        Ok(Engine::Whisper(path))
    } else if model_id.starts_with("parakeet") {
        Ok(Engine::Parakeet(Parakeet::new(&path)?))
    } else {
        Ok(Engine::Moonshine(Moonshine::new(&path)?))
    }
}

fn text_part(text: String, n_samples: usize) -> Vec<(i64, i64, String)> {
    if text.is_empty() {
        Vec::new()
    } else {
        let dur_ms = n_samples as i64 * 1000 / crate::audio::WHISPER_SAMPLE_RATE as i64;
        vec![(0, dur_ms, text)]
    }
}

/// Transcribe a chunk to timestamped `(start_ms, end_ms, text)` parts (used by
/// the meeting pipeline, which needs a timeline).
pub fn transcribe_parts(
    app: &AppHandle,
    engine: &mut Engine,
    samples: &[f32],
    language: Option<&str>,
    initial_prompt: Option<&str>,
) -> Vec<(i64, i64, String)> {
    match engine {
        Engine::Whisper(path) => {
            let st = app.state::<AppState>();
            stt::transcribe_segments(&st.stt, path, samples, language, initial_prompt)
                .unwrap_or_else(|e| {
                    eprintln!("[asr] transcription failed: {e}");
                    Vec::new()
                })
        }
        Engine::Moonshine(m) => text_part(m.transcribe(samples), samples.len()),
        Engine::Parakeet(p) => text_part(p.transcribe(samples), samples.len()),
        Engine::None => Vec::new(),
    }
}

/// Transcribe a chunk to a single plain string (used by dictation).
pub fn transcribe_text(
    app: &AppHandle,
    engine: &mut Engine,
    samples: &[f32],
    language: Option<&str>,
    initial_prompt: Option<&str>,
) -> Result<String, String> {
    match engine {
        Engine::Whisper(path) => {
            let st = app.state::<AppState>();
            stt::transcribe(&st.stt, path, samples, language, initial_prompt)
        }
        Engine::Moonshine(m) => Ok(m.transcribe(samples)),
        Engine::Parakeet(p) => Ok(p.transcribe(samples)),
        Engine::None => Ok(String::new()),
    }
}
