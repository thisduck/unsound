//! Speech engines for the spike examples (asr_eval, three_engines): Whisper as
//! unsound runs it, the sherpa-onnx Parakeet unsound used to ship, whisper.cpp's
//! own Parakeet TDT, and transcribe.cpp (directly and through the app's
//! `parakeet` module), all on the shared ggml.

use super::harness::{Asr, Output};
use sherpa_rs::transducer::{TransducerConfig, TransducerRecognizer};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
#[cfg(not(no_parakeet))]
use whisper_rs_sys as sys;

fn n_threads() -> i32 {
    std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4).min(8)
}

/// Mirrors `stt::transcribe` (Metal, flash attention, greedy, 500 ms lead-in,
/// suppress_blank, segment timestamps, `[BLANK_AUDIO]` stripped).
pub struct Whisper {
    pub ctx: WhisperContext,
    pub prompt: Option<String>,
}

impl Whisper {
    pub fn load(path: &str) -> Self {
        let mut cp = WhisperContextParameters::default();
        cp.flash_attn(true);
        let ctx = WhisperContext::new_with_params(path, cp).unwrap();
        Whisper { ctx, prompt: std::env::var("WHISPER_PROMPT").ok() }
    }
}

impl Asr for Whisper {
    fn run(&mut self, pcm: &[f32], lang: &str) -> Output {
        let mut ws = self.ctx.create_state().unwrap();
        let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        p.set_n_threads(n_threads());
        p.set_language(Some(lang));
        p.set_translate(false);
        p.set_print_special(false);
        p.set_print_progress(false);
        p.set_print_realtime(false);
        p.set_print_timestamps(false);
        p.set_suppress_blank(true);
        if let Some(pr) = &self.prompt {
            p.set_initial_prompt(pr);
        }
        let mut padded = vec![0.0f32; 8000];
        padded.extend_from_slice(pcm);
        ws.full(p, &padded).unwrap();
        let mut text = String::new();
        let mut spans = vec![];
        for i in 0..ws.full_n_segments() {
            if let Some(s) = ws.get_segment(i) {
                let t = s.to_str_lossy().unwrap().replace("[BLANK_AUDIO]", "");
                // centiseconds, minus the lead-in
                let a = (s.start_timestamp() as f64 / 100.0 - 0.5).max(0.0);
                let b = (s.end_timestamp() as f64 / 100.0 - 0.5).max(0.0);
                text.push_str(&t);
                spans.push((a, b, t));
            }
        }
        Output { text, spans }
    }
}

/// Same config as the removed `asr::Parakeet` (sherpa-onnx, CPU int8, 2 threads).
pub struct SherpaParakeet(pub TransducerRecognizer);

impl SherpaParakeet {
    pub fn load(dir: &str) -> Self {
        let p = |f: &str| std::path::Path::new(dir).join(f).to_string_lossy().to_string();
        let rec = TransducerRecognizer::new(TransducerConfig {
            encoder: p("encoder.int8.onnx"),
            decoder: p("decoder.int8.onnx"),
            joiner: p("joiner.int8.onnx"),
            tokens: p("tokens.txt"),
            num_threads: std::env::var("SHERPA_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(2),
            sample_rate: 16000,
            feature_dim: std::env::var("SHERPA_FEAT_DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(80),
            decoding_method: "greedy_search".to_string(),
            model_type: "nemo_transducer".to_string(),
            ..Default::default()
        })
        .unwrap();
        SherpaParakeet(rec)
    }
}

impl Asr for SherpaParakeet {
    fn run(&mut self, pcm: &[f32], _lang: &str) -> Output {
        // Like the removed asr::Parakeet: one span per chunk.
        let text = self.0.transcribe(16000, pcm);
        let spans = vec![(0.0, pcm.len() as f64 / 16000.0, text.clone())];
        Output { text, spans }
    }
}

/// transcribe.cpp (Handy's engine) through its safe crate, built by the
/// vendored transcribe-cpp-sys against llama.cpp's ggml. Word spans from
/// Parakeet's token timestamps.
#[cfg(not(no_parakeet))]
pub struct Tc {
    _model: transcribe_cpp::Model,
    session: transcribe_cpp::Session,
    ts: transcribe_cpp::TimestampKind,
}

#[cfg(not(no_parakeet))]
impl Tc {
    pub fn load(path: &str) -> Self {
        use transcribe_cpp::TimestampKind;
        let model = transcribe_cpp::Model::load(path).expect("transcribe.cpp load");
        let session = model.session().expect("transcribe.cpp session");
        let max = model.capabilities().expect("caps").max_timestamp_kind;
        let ts = if max == TimestampKind::Token { TimestampKind::Word } else { max };
        eprintln!("transcribe.cpp {} on {}", model.variant(), model.backend());
        Tc { _model: model, session, ts }
    }
}

#[cfg(not(no_parakeet))]
impl Asr for Tc {
    fn run(&mut self, pcm: &[f32], _lang: &str) -> Output {
        let opts = transcribe_cpp::RunOptions { timestamps: self.ts, ..Default::default() };
        match self.session.run(pcm, &opts) {
            Ok(r) => {
                let spans = r
                    .words
                    .iter()
                    .map(|w| (w.t0_ms as f64 / 1000.0, w.t1_ms as f64 / 1000.0, w.text.clone()))
                    .collect();
                Output { text: r.text, spans }
            }
            Err(e) => {
                eprintln!("transcribe.cpp run failed: {e}");
                Output { text: String::new(), spans: vec![] }
            }
        }
    }
}

/// Parakeet as the app runs it (`unsound_lib::parakeet`: cached model and
/// session, word timestamps), unfiltered, for checks against `Tc`/standalone.
#[cfg(not(no_parakeet))]
pub struct AppParakeet {
    state: unsound_lib::parakeet::ParakeetState,
    path: std::path::PathBuf,
}

#[cfg(not(no_parakeet))]
impl AppParakeet {
    pub fn load(path: &str) -> Self {
        let state = unsound_lib::parakeet::ParakeetState::default();
        let path = std::path::PathBuf::from(path);
        state.warmup(&path).expect("parakeet load");
        AppParakeet { state, path }
    }
}

#[cfg(not(no_parakeet))]
impl Drop for AppParakeet {
    fn drop(&mut self) {
        self.state.clear();
    }
}

#[cfg(not(no_parakeet))]
impl Asr for AppParakeet {
    fn run(&mut self, pcm: &[f32], _lang: &str) -> Output {
        match self.state.transcribe(&self.path, pcm, None) {
            Ok(o) => Output {
                spans: o.words.iter().map(|w| (w.start_ms as f64 / 1000.0, w.end_ms as f64 / 1000.0, w.text.clone())).collect(),
                text: o.text,
            },
            Err(e) => {
                eprintln!("parakeet run failed: {e}");
                Output { text: String::new(), spans: vec![] }
            }
        }
    }
}

/// whisper.cpp's own Parakeet TDT (`parakeet.h`, v1.9.4+), on the same ggml and
/// Metal backend as Whisper. Raw FFI through the vendored sys crate. Gated so a
/// baseline build against the unpatched vendor crate (`--cfg no_parakeet`)
/// still compiles.
#[cfg(not(no_parakeet))]
pub struct WcppParakeet {
    ctx: *mut sys::parakeet_context,
    state: *mut sys::parakeet_state,
}

#[cfg(not(no_parakeet))]
impl WcppParakeet {
    pub fn load(path: &str) -> Self {
        let c = std::ffi::CString::new(path).unwrap();
        unsafe {
            let mut cp = sys::parakeet_context_default_params();
            cp.use_gpu = std::env::var("PARAKEET_CPU").is_err();
            let ctx = sys::parakeet_init_from_file_with_params_no_state(c.as_ptr(), cp);
            assert!(!ctx.is_null(), "parakeet load failed");
            let state = sys::parakeet_init_state(ctx);
            assert!(!state.is_null(), "parakeet state failed");
            WcppParakeet { ctx, state }
        }
    }
}

#[cfg(not(no_parakeet))]
impl Drop for WcppParakeet {
    // Free before ggml's static destructors run, or ggml-metal asserts on live
    // residency sets at exit (as stt::SttState::clear does for Whisper).
    fn drop(&mut self) {
        unsafe {
            sys::parakeet_free_state(self.state);
            sys::parakeet_free(self.ctx);
        }
    }
}

#[cfg(not(no_parakeet))]
impl Asr for WcppParakeet {
    fn run(&mut self, pcm: &[f32], _lang: &str) -> Output {
        unsafe {
            let mut p = sys::parakeet_full_default_params(sys::parakeet_sampling_strategy_PARAKEET_SAMPLING_GREEDY);
            p.n_threads = n_threads();
            let rc = sys::parakeet_full_with_state(self.ctx, self.state, p, pcm.as_ptr(), pcm.len() as i32);
            if rc != 0 {
                eprintln!("parakeet_full failed: {rc}");
                return Output { text: String::new(), spans: vec![] };
            }
            let cstr = |p: *const std::os::raw::c_char| std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
            let mut text = String::new();
            // Word spans from token data (t0/t1 in 10 ms units).
            let mut words: Vec<(f64, f64, String)> = vec![];
            for s in 0..sys::parakeet_full_n_segments_from_state(self.state) {
                let st = cstr(sys::parakeet_full_get_segment_text_from_state(self.state, s));
                if !text.is_empty() && !st.starts_with(' ') {
                    text.push(' ');
                }
                text.push_str(&st);
                for k in 0..sys::parakeet_full_n_tokens_from_state(self.state, s) {
                    let d = sys::parakeet_full_get_token_data_from_state(self.state, s, k);
                    let piece = cstr(sys::parakeet_token_to_str(self.ctx, d.id)).replace('\u{2581}', "");
                    let (a, b) = (d.t0 as f64 / 100.0, d.t1 as f64 / 100.0);
                    match words.last_mut() {
                        Some(w) if !d.is_word_start => {
                            w.1 = w.1.max(b);
                            w.2.push_str(&piece);
                        }
                        _ => words.push((a, b, piece)),
                    }
                }
            }
            Output { text, spans: words }
        }
    }
}
