//! Alternative on-device ASR via sherpa-onnx — currently Moonshine, a fast
//! English model for live meeting captions. Whisper stays the default and the
//! only option for non-English. The recognizer holds a raw sherpa pointer, so
//! it is not Send and must live on the thread that uses it (the meeting loop).

use sherpa_rs::moonshine::{MoonshineConfig, MoonshineRecognizer};
use std::path::Path;

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
