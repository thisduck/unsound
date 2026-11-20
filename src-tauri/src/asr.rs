//! The speech engines behind dictation and meetings, and which one hears a
//! take. Whisper (whisper.cpp) is the default and covers every language;
//! Parakeet (transcribe.cpp) is faster and more accurate where it has the
//! language; Moonshine (sherpa-onnx) is a small English model. Parakeet and
//! Moonshine only transcribe their own languages, so `route` sends other
//! speech to a downloaded Whisper model, or refuses it.

use crate::models::{self, ModelInfo, SttEngine};
use crate::{parakeet, stt, AppState};
use sherpa_rs::moonshine::{MoonshineConfig, MoonshineRecognizer};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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

/// The active speech engine. Whisper and Parakeet use the cached models in
/// `AppState`; Moonshine owns its recognizer and is !Send, so an `Engine` must
/// be built and used on one thread.
pub enum Engine {
    #[cfg(feature = "automation")]
    Fixture(Arc<crate::meeting_fixture::Fixture>),
    Whisper(PathBuf),
    Parakeet(PathBuf),
    Moonshine(Moonshine),
}

impl Engine {
    /// Whether the meeting pipeline lays consecutive pieces on one buffer.
    /// Whisper's encoder costs the same for 3 s as for 30 s, so batching pays.
    /// Parakeet's cost scales with the audio, and joining unrelated pieces
    /// cost it accuracy (AMI headset: 9.0% WER one piece per call vs 12.6%
    /// batched, for 1.4x the speed), so it takes one piece per call.
    pub fn batches(&self) -> bool {
        matches!(self, Engine::Whisper(_))
    }
}

/// The engines' model caches (held in `AppState`).
#[derive(Clone, Copy)]
pub struct Caches<'a> {
    pub stt: &'a stt::SttState,
    pub parakeet: &'a parakeet::ParakeetState,
}

impl<'a> Caches<'a> {
    pub fn of(st: &'a AppState) -> Self {
        Caches { stt: &st.stt, parakeet: &st.parakeet }
    }
}

fn engine_for(info: &ModelInfo, path: PathBuf) -> Result<Engine, String> {
    Ok(match info.engine {
        SttEngine::Whisper => Engine::Whisper(path),
        SttEngine::Parakeet => Engine::Parakeet(path),
        SttEngine::Moonshine => Engine::Moonshine(Moonshine::new(&path)?),
    })
}

/// Resolve the selected STT model id into a ready engine, ignoring language.
/// For Moonshine this loads the recognizer, so call it on the thread that
/// will use it.
pub fn resolve(app: &AppHandle, model_id: &str) -> Result<Engine, String> {
    #[cfg(feature = "automation")]
    if let Some(fixture) = crate::meeting_fixture::for_model(app, model_id) {
        return Ok(Engine::Fixture(fixture));
    }
    #[cfg(feature = "automation")]
    if let Some(path) = model_id.strip_prefix("e2e-real:") {
        return Ok(Engine::Whisper(PathBuf::from(path)));
    }
    let info = models::find_model(app, model_id)?;
    let path = models::downloaded_model_path(app, model_id)?;
    engine_for(&info, path)
}

/// Load the model and compile its GPU kernels ahead of the first take.
pub fn warm(app: &AppHandle, model_id: &str) -> Result<(), String> {
    let info = models::find_model(app, model_id)?;
    let path = models::downloaded_model_path(app, model_id)?;
    let st = app.state::<AppState>();
    match info.engine {
        SttEngine::Whisper => st.stt.warmup(&path),
        SttEngine::Parakeet => st.parakeet.warmup(&path),
        // Built per use; nothing to keep warm.
        SttEngine::Moonshine => Ok(()),
    }
}

// ---- language routing --------------------------------------------------------

/// What is known about the spoken language.
#[derive(Clone, Copy, Debug)]
pub enum Heard<'a> {
    /// Chosen by the user, or detected with confidence.
    Known(&'a str),
    /// Detection ran but could not tell.
    Unsure,
    /// Nothing could detect it (no Whisper model downloaded).
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// The selected model.
    Selected,
    /// The fallback Whisper model.
    Fallback,
    /// Nothing downloaded can transcribe it.
    Refuse,
}

/// Which model should hear speech in `heard`, for a selected model that
/// transcribes `supported` (empty: everything).
///  - A known language the model has: the model. One it lacks: Whisper if
///    downloaded, else refuse rather than emit another language's garbage.
///  - Detection unsure: Whisper, whose own detection covers every language.
///  - Undetectable (no Whisper at all): the model, in its own languages —
///    Parakeet v2 hears English, v3 any of its 25 languages. The UI says so.
pub fn route(supported: &[String], heard: Heard, has_fallback: bool) -> Route {
    if supported.is_empty() {
        return Route::Selected;
    }
    let fallback = if has_fallback { Route::Fallback } else { Route::Refuse };
    match heard {
        Heard::Known(lang) if supported.iter().any(|l| l == base_language(lang)) => Route::Selected,
        Heard::Known(_) => fallback,
        Heard::Unsure if has_fallback => Route::Fallback,
        Heard::Unsure | Heard::Unknown => Route::Selected,
    }
}

/// "pt-BR" → "pt". Whisper and the UI use bare ISO 639-1 codes already.
fn base_language(lang: &str) -> &str {
    lang.split(['-', '_']).next().unwrap_or(lang)
}

/// The language's English name ("Hindi"), or its code.
pub fn language_name(code: &str) -> String {
    let full = whisper_rs::get_lang_id(base_language(code)).and_then(whisper_rs::get_lang_str_full);
    match full {
        Some(name) => {
            let mut c = name.chars();
            c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
        }
        None => code.to_string(),
    }
}

/// Shown when no downloaded model can transcribe the language.
pub fn refusal(model: &ModelInfo, lang: Option<&str>) -> String {
    let what = lang.map_or("this language".to_string(), language_name);
    format!(
        "{} can't transcribe {what}. Download a Whisper model in Settings › Models, or pick another speech model.",
        model.name
    )
}

/// The selected model plus what routing needs: the fallback Whisper model.
pub struct Selection {
    pub info: ModelInfo,
    pub fallback: Option<(String, PathBuf)>,
}

impl Selection {
    pub fn load(app: &AppHandle, model_id: &str) -> Result<Self, String> {
        let all = models::all_models(app)?;
        let id = models::migrate_id(model_id);
        let info = all
            .iter()
            .find(|m| m.id == id)
            .cloned()
            .ok_or_else(|| format!("unknown model id: {model_id}"))?;
        let fallback = models::fallback_whisper(&all)
            .and_then(|m| models::model_path(app, m).ok().map(|p| (m.id.clone(), p)));
        Ok(Selection { info, fallback })
    }

    /// Whether this model only covers some languages.
    pub fn is_limited(&self) -> bool {
        !self.info.supported_languages.is_empty()
    }

    /// The Whisper model to detect the language with, when one is needed.
    pub fn detector(&self) -> Option<&Path> {
        self.fallback.as_ref().filter(|_| self.is_limited()).map(|(_, p)| p.as_path())
    }

    pub fn route(&self, heard: Heard) -> Route {
        route(&self.info.supported_languages, heard, self.fallback.is_some())
    }
}

/// Pick the engine and language for a finished take. On auto, for a model
/// that only knows some languages, the language is detected first with the
/// fallback Whisper model (when one is downloaded).
pub fn resolve_for_take(
    app: &AppHandle,
    model_id: &str,
    language: Option<&str>,
    samples: &[f32],
) -> Result<(Engine, Option<String>), String> {
    #[cfg(feature = "automation")]
    if crate::meeting_fixture::for_model(app, model_id).is_some() || model_id.starts_with("e2e-real:") {
        return Ok((resolve(app, model_id)?, language.map(String::from)));
    }
    let sel = Selection::load(app, model_id)?;
    let path = models::downloaded_model_path(app, &sel.info.id)?;
    take_engine(&sel, path, Caches::of(app.state::<AppState>().inner()), language, samples)
}

/// The engine (and language) for one take, given the selected model's file.
pub fn take_engine(
    sel: &Selection,
    selected_path: PathBuf,
    caches: Caches,
    language: Option<&str>,
    samples: &[f32],
) -> Result<(Engine, Option<String>), String> {
    let (route, language) = route_take(sel, caches, language, samples);
    match route {
        Route::Selected => Ok((engine_for(&sel.info, selected_path)?, language)),
        Route::Fallback => {
            let (id, path) = sel.fallback.clone().expect("fallback route has a model");
            eprintln!("[asr] {} can't take {language:?}; using {id}", sel.info.id);
            Ok((Engine::Whisper(path), language))
        }
        Route::Refuse => Err(refusal(&sel.info, language.as_deref())),
    }
}

/// The routing decision for one take, detecting the language if needed.
/// Returns the language to transcribe with (None: the engine decides).
pub fn route_take(
    sel: &Selection,
    caches: Caches,
    language: Option<&str>,
    samples: &[f32],
) -> (Route, Option<String>) {
    if let Some(lang) = language {
        return (sel.route(Heard::Known(lang)), Some(lang.to_string()));
    }
    let Some(detector) = sel.detector() else {
        return (sel.route(Heard::Unknown), None);
    };
    // Nothing to hear: let the selected model return its empty transcript.
    if stt::is_effectively_silent(samples) {
        return (Route::Selected, None);
    }
    match stt::detect_language(caches.stt, detector, samples) {
        Ok(Some(lang)) => (sel.route(Heard::Known(&lang)), Some(lang)),
        Ok(None) => (sel.route(Heard::Unsure), None),
        Err(e) => {
            eprintln!("[asr] language detection failed: {e}");
            (sel.route(Heard::Unsure), None)
        }
    }
}

// ---- transcription -------------------------------------------------------------

fn text_part(text: String, n_samples: usize) -> Vec<(i64, i64, String)> {
    if text.is_empty() {
        Vec::new()
    } else {
        let dur_ms = n_samples as i64 * 1000 / crate::audio::WHISPER_SAMPLE_RATE as i64;
        vec![(0, dur_ms, text)]
    }
}

/// Parakeet text for a take: the filter Whisper output goes through, minus
/// the parts that need a no-speech probability.
fn parakeet_text(caches: Caches, path: &Path, samples: &[f32], stop: Option<&AtomicBool>) -> Result<String, String> {
    if samples.len() < 1600 || stt::is_effectively_silent(samples) {
        return Ok(String::new());
    }
    let out = caches.parakeet.transcribe(path, samples, stop)?;
    Ok(stt::sanitize_unscored(&out.text).trim().to_string())
}

/// Transcribe a chunk to timestamped `(start_ms, end_ms, text)` parts (used by
/// the meeting pipeline, which needs a timeline). Parakeet's parts are the
/// words, unfiltered (`transcribe_pieces` assigns and cleans them).
/// Durable work must distinguish inference errors from successfully recognized silence.
pub fn transcribe_parts_checked(
    caches: Caches, engine: &mut Engine, samples: &[f32], language: Option<&str>,
    prompt: Option<&str>, cancel: Arc<AtomicBool>,
) -> Result<Vec<(i64, i64, String)>, String> {
    if cancel.load(Ordering::Relaxed) { return Err("transcription paused".into()); }
    match engine {
        #[cfg(feature = "automation")]
        Engine::Fixture(f) => f.transcribe(samples, &cancel).map(|text| text_part(text, samples.len())),
        Engine::Whisper(path) => stt::transcribe_segments_until(caches.stt,
            path, samples, language, prompt, Some(cancel)),
        Engine::Parakeet(path) => {
            if samples.len() < 1600 || stt::is_effectively_silent(samples) {
                return Ok(Vec::new());
            }
            let out = caches.parakeet.transcribe(path, samples, Some(&cancel))?;
            Ok(out.words.into_iter().map(|w| (w.start_ms, w.end_ms, w.text)).collect())
        }
        Engine::Moonshine(m) => Ok(text_part(m.transcribe(samples), samples.len())),
    }
}

/// One line of meeting transcript from a batch: which piece it belongs to and
/// its span in ms from that piece's start.
#[derive(Debug, Clone, PartialEq)]
pub struct PiecePart {
    pub piece: usize,
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

/// Silence between batched pieces, and before every piece (Whisper drops words
/// that start right at sample zero).
pub const PIECE_GAP_SAMPLES: usize = 4_800;

/// Transcribe meeting pieces laid out on one buffer — [gap][piece 0][gap]… —
/// and map the results back to pieces. Whisper segments go to the piece their
/// midpoint falls in; Parakeet words likewise, then each piece's words are
/// joined and filtered into one line spanning its first to last word.
pub fn transcribe_pieces(
    caches: Caches, engine: &mut Engine, pieces: &[&[f32]], language: Option<&str>,
    prompt: Option<&str>, cancel: Arc<AtomicBool>,
) -> Result<Vec<PiecePart>, String> {
    const RATE: i64 = crate::audio::WHISPER_SAMPLE_RATE as i64;
    let mut buf: Vec<f32> = Vec::new();
    let mut spans: Vec<(i64, i64)> = Vec::new(); // (offset_ms, len_ms) in buf
    for p in pieces {
        buf.extend(std::iter::repeat_n(0.0, PIECE_GAP_SAMPLES));
        let off = buf.len() as i64 * 1000 / RATE;
        buf.extend_from_slice(p);
        spans.push((off, p.len() as i64 * 1000 / RATE));
    }
    let word_level = matches!(engine, Engine::Parakeet(_));
    let parts = transcribe_parts_checked(caches, engine, &buf, language, prompt, cancel)?;
    if word_level {
        let words: Vec<parakeet::Word> = parts
            .into_iter()
            .map(|(start_ms, end_ms, text)| parakeet::Word { start_ms, end_ms, text })
            .collect();
        return Ok(parakeet::assign_words(&words, &spans)
            .into_iter()
            .enumerate()
            .filter_map(|(piece, words)| {
                let text = words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
                let text = stt::sanitize_unscored(&text).trim().to_string();
                (!text.is_empty()).then(|| PiecePart {
                    piece,
                    start_ms: words.first().unwrap().start_ms,
                    end_ms: words.last().unwrap().end_ms,
                    text,
                })
            })
            .collect());
    }
    Ok(parts
        .into_iter()
        .map(|(s, e, text)| {
            let mid = (s + e) / 2;
            let piece = spans
                .iter()
                .position(|(off, len)| mid >= *off && mid <= off + len)
                .or_else(|| {
                    spans
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, (off, len))| (mid - (off + len / 2)).abs())
                        .map(|(i, _)| i)
                })
                .unwrap_or(0);
            let (off, len) = spans[piece];
            let start_ms = (s - off).clamp(0, len);
            PiecePart { piece, start_ms, end_ms: (e - off).clamp(0, len).max(start_ms), text }
        })
        .collect())
}

/// Like `transcribe_text`, for a live preview: gives up soon after `stop` is
/// set. (Moonshine is quick enough to let finish.)
pub fn transcribe_preview(
    caches: Caches,
    engine: &mut Engine,
    samples: &[f32],
    language: Option<&str>,
    initial_prompt: Option<&str>,
    stop: Arc<AtomicBool>,
) -> Result<String, String> {
    match engine {
        #[cfg(feature = "automation")]
        Engine::Fixture(f) => f.transcribe(samples, &stop),
        Engine::Whisper(path) => {
            stt::transcribe_until(caches.stt, path, samples, language, initial_prompt, Some(stop))
        }
        Engine::Parakeet(path) => parakeet_text(caches, path, samples, Some(&stop)),
        Engine::Moonshine(_) => transcribe_text(caches, engine, samples, language, initial_prompt),
    }
}

/// Transcribe a chunk to a single plain string (used by dictation).
pub fn transcribe_text(
    caches: Caches,
    engine: &mut Engine,
    samples: &[f32],
    language: Option<&str>,
    initial_prompt: Option<&str>,
) -> Result<String, String> {
    match engine {
        #[cfg(feature = "automation")]
        Engine::Fixture(f) => f.transcribe(samples, &AtomicBool::new(false)),
        Engine::Whisper(path) => stt::transcribe(caches.stt, path, samples, language, initial_prompt),
        Engine::Parakeet(path) => parakeet_text(caches, path, samples, None),
        Engine::Moonshine(m) => Ok(m.transcribe(samples)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn langs(codes: &[&str]) -> Vec<String> {
        codes.iter().map(|c| c.to_string()).collect()
    }

    #[test]
    fn whisper_models_take_every_language() {
        for heard in [Heard::Known("hi"), Heard::Unsure, Heard::Unknown] {
            assert_eq!(route(&[], heard, false), Route::Selected);
        }
    }

    #[test]
    fn a_known_language_goes_to_the_model_that_has_it() {
        let v2 = langs(&["en"]);
        let v3 = langs(models::PARAKEET_V3_LANGUAGES);
        assert_eq!(route(&v2, Heard::Known("en"), false), Route::Selected);
        assert_eq!(route(&v3, Heard::Known("de"), false), Route::Selected);
        assert_eq!(route(&v3, Heard::Known("pt-BR"), false), Route::Selected);
        // Unsupported: Whisper when there is one, else refuse.
        assert_eq!(route(&v2, Heard::Known("de"), true), Route::Fallback);
        assert_eq!(route(&v3, Heard::Known("hi"), true), Route::Fallback);
        assert_eq!(route(&v3, Heard::Known("hi"), false), Route::Refuse);
        assert_eq!(route(&v3, Heard::Known("ja"), false), Route::Refuse);
    }

    #[test]
    fn auto_without_a_confident_detection() {
        let v3 = langs(models::PARAKEET_V3_LANGUAGES);
        // Whisper is downloaded but could not tell: it transcribes.
        assert_eq!(route(&v3, Heard::Unsure, true), Route::Fallback);
        // Nothing can detect: the model hears its own languages.
        assert_eq!(route(&v3, Heard::Unknown, false), Route::Selected);
        assert_eq!(route(&langs(&["en"]), Heard::Unknown, false), Route::Selected);
    }

    #[test]
    fn refusal_names_the_model_and_the_language() {
        let info = ModelInfo {
            name: "Parakeet TDT v3 (European languages)".into(),
            ..serde_json::from_value(serde_json::json!({
                "id": "x", "name": "", "kind": "stt", "url": "", "filename": "", "sizeBytes": 0,
                "description": ""
            }))
            .unwrap()
        };
        assert_eq!(
            refusal(&info, Some("hi")),
            "Parakeet TDT v3 (European languages) can't transcribe Hindi. Download a Whisper model in Settings › Models, or pick another speech model."
        );
        assert!(refusal(&info, None).contains("can't transcribe this language."));
    }
}
