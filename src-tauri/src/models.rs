use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, Manager};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    Stt,
    Llm,
    /// Speaker-diarization models (sherpa-onnx): segmentation + embedding.
    Diarize,
    /// Voice-activity detection (Silero) — chunks meeting audio on silence.
    Vad,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub kind: ModelKind,
    pub url: String,
    pub filename: String,
    /// Approximate download size in bytes, for display before the download starts.
    pub size_bytes: u64,
    pub description: String,
    /// Human-readable language coverage, e.g. "Multilingual · 99 languages".
    #[serde(default)]
    pub languages: String,
    /// Part of the "download recommended" pair in onboarding.
    #[serde(default)]
    pub recommended: bool,
    #[serde(default)]
    pub custom: bool,
    #[serde(default)]
    pub downloaded: bool,
    /// Multi-file models (e.g. sherpa-onnx Moonshine) list their file URLs here;
    /// they download into a per-id directory instead of a single file. Empty for
    /// ordinary single-file models.
    #[serde(default)]
    pub files: Vec<String>,
    /// Which speech engine runs this model (speech models only).
    #[serde(default)]
    pub engine: SttEngine,
    /// Spoken languages the model can transcribe (ISO 639-1); empty = any
    /// language the app offers. Speech is routed elsewhere for the rest.
    #[serde(default)]
    pub supported_languages: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SttEngine {
    /// whisper.cpp (custom models are assumed to be Whisper GGML files).
    #[default]
    Whisper,
    /// Parakeet TDT through transcribe.cpp.
    Parakeet,
    /// Moonshine through sherpa-onnx.
    Moonshine,
}

/// Speech model ids that were removed, and the model that replaces each.
/// `parakeet-tdt-v3` was the sherpa-onnx (CPU, int8) build of Parakeet v3.
const RENAMED: &[(&str, &str)] = &[("parakeet-tdt-v3", "parakeet-tdt-0.6b-v3")];

/// The current id for a possibly removed one.
pub fn migrate_id(id: &str) -> &str {
    RENAMED
        .iter()
        .find(|(old, _)| *old == id)
        .map_or(id, |(_, new)| new)
}

/// Remove the files of models that no longer exist (they are not listed, so
/// the user could never delete them). Called once at launch.
pub fn remove_retired(app: &AppHandle) {
    let Ok(dir) = models_dir(app) else { return };
    for (old, _) in RENAMED {
        let path = dir.join(old);
        if path.is_dir() {
            match fs::remove_dir_all(&path) {
                Ok(()) => eprintln!("[models] removed retired model files {}", path.display()),
                Err(e) => eprintln!("[models] could not remove {}: {e}", path.display()),
            }
        }
    }
}

/// Parakeet TDT v3's languages, from the model card and the GGUF's
/// `general.languages`.
pub const PARAKEET_V3_LANGUAGES: &[&str] = &[
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt",
    "mt", "pl", "pt", "ro", "ru", "sk", "sl", "es", "sv", "uk",
];

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgress {
    pub id: String,
    pub downloaded: u64,
    pub total: u64,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadDone {
    pub id: String,
    pub ok: bool,
    pub error: Option<String>,
}

const GB: u64 = 1_000_000_000;
const MB: u64 = 1_000_000;

pub(crate) fn builtin_registry() -> Vec<ModelInfo> {
    let m = |id: &str,
             name: &str,
             kind: ModelKind,
             url: &str,
             size: u64,
             desc: &str,
             languages: &str,
             recommended: bool| ModelInfo {
        id: id.into(),
        name: name.into(),
        kind,
        url: url.into(),
        filename: url.rsplit('/').next().unwrap_or(id).to_string(),
        size_bytes: size,
        description: desc.into(),
        languages: languages.into(),
        recommended,
        custom: false,
        downloaded: false,
        files: vec![],
        engine: SttEngine::Whisper,
        supported_languages: vec![],
    };
    // Parakeet TDT (NVIDIA, CC-BY-4.0) as a Q8_0 GGUF for transcribe.cpp.
    let parakeet = |id: &str, name: &str, url: &str, size: u64, desc: &str, languages: &str, codes: &[&str]| ModelInfo {
        engine: SttEngine::Parakeet,
        supported_languages: codes.iter().map(|c| c.to_string()).collect(),
        ..m(id, name, ModelKind::Stt, url, size, desc, languages, false)
    };
    // A multi-file (bundle) model — downloads its files into a per-id directory.
    let mb = |id: &str,
              name: &str,
              kind: ModelKind,
              files: &[&str],
              size: u64,
              desc: &str,
              languages: &str| ModelInfo {
        id: id.into(),
        name: name.into(),
        kind,
        url: String::new(),
        filename: String::new(),
        size_bytes: size,
        description: desc.into(),
        languages: languages.into(),
        recommended: false,
        custom: false,
        downloaded: false,
        files: files.iter().map(|s| s.to_string()).collect(),
        engine: SttEngine::Moonshine,
        supported_languages: vec!["en".into()],
    };
    const WHISPER_LANGS: &str = "Multilingual · ~99 languages";
    const EN_ONLY: &str = "English only";
    const QWEN_LANGS: &str = "Multilingual · ~29 languages";
    const LLAMA_LANGS: &str = "English + 7 languages (es, fr, de, it, pt, hi, th)";
    vec![
        // --- Speech to text (whisper.cpp GGML) ---
        m(
            "whisper-tiny",
            "Whisper Tiny",
            ModelKind::Stt,
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
            78 * MB,
            "Fastest, lowest accuracy. Good for quick tests.",
            WHISPER_LANGS,
            false,
        ),
        m(
            "whisper-base",
            "Whisper Base",
            ModelKind::Stt,
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
            148 * MB,
            "Fast with reasonable accuracy.",
            WHISPER_LANGS,
            false,
        ),
        m(
            "whisper-base-en",
            "Whisper Base (English)",
            ModelKind::Stt,
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
            148 * MB,
            "Slightly better than Base when you only speak English.",
            EN_ONLY,
            false,
        ),
        m(
            "whisper-small",
            "Whisper Small",
            ModelKind::Stt,
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
            488 * MB,
            "Good balance of speed and accuracy.",
            WHISPER_LANGS,
            false,
        ),
        m(
            "whisper-small-en",
            "Whisper Small (English)",
            ModelKind::Stt,
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
            488 * MB,
            "Slightly better than Small when you only speak English.",
            EN_ONLY,
            false,
        ),
        m(
            "whisper-medium",
            "Whisper Medium",
            ModelKind::Stt,
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
            1533 * MB,
            "High accuracy, slower.",
            WHISPER_LANGS,
            false,
        ),
        m(
            "whisper-large-v3-turbo",
            "Whisper Large v3 Turbo",
            ModelKind::Stt,
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
            1620 * MB,
            "Recommended default: close to Large v3 accuracy at near-medium speed.",
            WHISPER_LANGS,
            true,
        ),
        m(
            "whisper-large-v3",
            "Whisper Large v3",
            ModelKind::Stt,
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
            3100 * MB,
            "Most accurate; slower. Best for hard audio and non-English (Hindi/Urdu/Punjabi).",
            WHISPER_LANGS,
            false,
        ),
        m(
            "indic-whisper-medium",
            "IndicWhisper Medium",
            ModelKind::Stt,
            "https://huggingface.co/rupind/indic-whisper-medium-gguf/resolve/main/ggml-indic-whisper-medium-q8_0.bin",
            824 * MB,
            "AI4Bharat fine-tune for Indian languages; strongest on Hindi (helps Urdu too).",
            "Hindi + Indic languages",
            false,
        ),
        m(
            "whisper-hindi2hinglish",
            "Whisper Hindi→Hinglish (Apex)",
            ModelKind::Stt,
            "https://huggingface.co/Marquestra/Whisper-Hindi2Hinglish-Apex-GGML/resolve/main/ggml-apex-hinglish-q8_0.bin",
            875 * MB,
            "Hindi dictation that outputs romanized Hinglish; great for Hindi/Urdu in Latin script.",
            "Hindi / Hinglish (romanized)",
            false,
        ),
        // --- Parakeet TDT 0.6B (transcribe.cpp, GPU) ---
        parakeet(
            "parakeet-tdt-0.6b-v2",
            "Parakeet TDT v2 (English)",
            "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q8_0.gguf",
            730 * MB,
            "Fast and accurate English: beat Large v3 Turbo on meeting audio in our tests at several times the speed. No vocabulary hints. NVIDIA, CC-BY-4.0.",
            EN_ONLY,
            &["en"],
        ),
        parakeet(
            "parakeet-tdt-0.6b-v3",
            "Parakeet TDT v3 (European languages)",
            "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q8_0.gguf",
            740 * MB,
            "Fast and accurate in 25 European languages; keeps fillers like \"uh\" and spells out numbers. Other languages need a Whisper model. NVIDIA, CC-BY-4.0.",
            "25 European languages",
            PARAKEET_V3_LANGUAGES,
        ),
        // --- Text cleanup (llama.cpp GGUF) ---
        m(
            "qwen2.5-1.5b",
            "Qwen 2.5 1.5B Instruct (Q4_K_M)",
            ModelKind::Llm,
            "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            1 * GB,
            "Small and fast; fine for light cleanup.",
            QWEN_LANGS,
            false,
        ),
        m(
            "qwen2.5-3b",
            "Qwen 2.5 3B Instruct (Q4_K_M)",
            ModelKind::Llm,
            "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
            2 * GB,
            "Solid mid-size cleanup model.",
            QWEN_LANGS,
            false,
        ),
        m(
            "qwen3-4b",
            "Qwen3 4B Instruct (Q4_K_M)",
            ModelKind::Llm,
            "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            2500 * MB,
            "Recommended default: near-7B quality at close to 3B speed.",
            QWEN_LANGS,
            true,
        ),
        m(
            "gemma-3-4b",
            "Gemma 3 4B Instruct (Q4_K_M)",
            ModelKind::Llm,
            "https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf",
            2490 * MB,
            "Natural prose and broad language coverage.",
            "Multilingual · 140+ languages",
            false,
        ),
        m(
            "qwen2.5-7b",
            "Qwen 2.5 7B Instruct (Q4_K_M)",
            ModelKind::Llm,
            "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            4680 * MB,
            "Noticeably smarter cleanup; best choice for writing styles.",
            QWEN_LANGS,
            false,
        ),
        m(
            "llama-3.1-8b",
            "Llama 3.1 8B Instruct (Q4_K_M)",
            ModelKind::Llm,
            "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            4920 * MB,
            "Strong instruction following and style mimicry.",
            LLAMA_LANGS,
            false,
        ),
        m(
            "llama-3.2-1b",
            "Llama 3.2 1B Instruct (Q4_K_M)",
            ModelKind::Llm,
            "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            808 * MB,
            "Very fast, lighter cleanup quality.",
            LLAMA_LANGS,
            false,
        ),
        m(
            "llama-3.2-3b",
            "Llama 3.2 3B Instruct (Q4_K_M)",
            ModelKind::Llm,
            "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            2020 * MB,
            "Strong quality for its size.",
            LLAMA_LANGS,
            false,
        ),
        m(
            "qwen3-8b",
            "Qwen3 8B Instruct (Q4_K_M)",
            ModelKind::Llm,
            "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
            5030 * MB,
            "Sharper cleanup and summaries than Qwen3 4B; better at context and homophones.",
            QWEN_LANGS,
            false,
        ),
        m(
            "granite-3.3-8b",
            "Granite 3.3 8B Instruct (Q4_K_M)",
            ModelKind::Llm,
            "https://huggingface.co/bartowski/ibm-granite_granite-3.3-8b-instruct-GGUF/resolve/main/ibm-granite_granite-3.3-8b-instruct-Q4_K_M.gguf",
            4950 * MB,
            "IBM model tuned for summarization and extraction with 128K context — well suited to meeting notes.",
            "Multilingual · 12 languages",
            false,
        ),
        m(
            "gemma-3-12b",
            "Gemma 3 12B Instruct (Q4_K_M)",
            ModelKind::Llm,
            "https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q4_K_M.gguf",
            7300 * MB,
            "High-quality cleanup and summaries with the best multilingual coverage. Wants ~16GB RAM.",
            "Multilingual · 140+ languages",
            false,
        ),
        m(
            "phi-4-mini",
            "Phi-4 mini Instruct (Q4_K_M)",
            ModelKind::Llm,
            "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf",
            2500 * MB,
            "Small and fast with strong reasoning; a good quick pick for English cleanup.",
            "English + multilingual",
            false,
        ),
        // --- Assistant-grade models: follow instructions and fill structured
        // drafts well (the meeting assistant's picks; also strong summarizers).
        m(
            "gemma-4-e4b",
            "Gemma 4 E4B Instruct (Q4_K_M)",
            ModelKind::Llm,
            "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf",
            4980 * MB,
            "Assistant for 8-16GB Macs: small and quick, good at drafts and structured output. 128K context.",
            "Multilingual · 140+ languages",
            false,
        ),
        m(
            "gemma-4-12b",
            "Gemma 4 12B Instruct (QAT Q4_0)",
            ModelKind::Llm,
            "https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/resolve/main/gemma-4-12b-it-qat-q4_0.gguf",
            6980 * MB,
            "Recommended assistant for 16-32GB Macs: reliable drafts and requests, strong summaries. 256K context.",
            "Multilingual · 140+ languages",
            false,
        ),
        m(
            "gemma-4-26b-a4b",
            "Gemma 4 26B-A4B Instruct (Q4_K_M)",
            ModelKind::Llm,
            "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
            16950 * MB,
            "Best assistant for 32GB+ Macs: mixture-of-experts, so near-4B speed with much stronger reasoning. 256K context.",
            "Multilingual · 140+ languages",
            false,
        ),
        m(
            "qwen3.6-35b-a3b",
            "Qwen3.6 35B-A3B Instruct (Q4_K_M)",
            ModelKind::Llm,
            "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            22130 * MB,
            "Strong agentic model for 48GB+ Macs (3B active per token, so fast). Great at tool use and long context.",
            QWEN_LANGS,
            false,
        ),
        m(
            "gpt-oss-20b",
            "gpt-oss 20B (MXFP4)",
            ModelKind::Llm,
            "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf",
            12110 * MB,
            "OpenAI's open-weight model, built for tool calling and structured reasoning. Wants 24GB+.",
            "English-first",
            false,
        ),
        // --- Speaker diarization (sherpa-onnx) — meetings "who said what" ---
        // Both are required together; the ids are referenced by name in the
        // diarization command, so keep them stable.
        m(
            "diarize-segmentation",
            "Speaker segmentation (pyannote)",
            ModelKind::Diarize,
            "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx",
            6 * MB,
            "Finds speech and speaker-change points in a meeting.",
            "Language-independent",
            true,
        ),
        m(
            "diarize-embedding",
            "Speaker embeddings (3D-Speaker CAM++)",
            ModelKind::Diarize,
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
            28 * MB,
            "Tells voices apart to label Speaker 1, Speaker 2, … Good all-round default.",
            "Language-independent",
            true,
        ),
        m(
            "diarize-embedding-titanet",
            "Speaker embeddings (NeMo TitaNet)",
            ModelKind::Diarize,
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx",
            40 * MB,
            "Larger, English-tuned. Often separates similar voices better.",
            "English",
            false,
        ),
        m(
            "diarize-embedding-eres2netv2",
            "Speaker embeddings (3D-Speaker ERes2NetV2)",
            ModelKind::Diarize,
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx",
            72 * MB,
            "Most accurate; built for short utterances, so it over-splits less.",
            "Language-independent",
            false,
        ),
        // --- Voice-activity detection (meeting streaming) ---
        m(
            "vad-silero",
            "Voice activity detection (Silero)",
            ModelKind::Vad,
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
            1 * MB,
            "Chunks meeting audio on natural pauses so words aren't cut mid-sentence.",
            "Language-independent",
            true,
        ),
        // Alternative meeting speech model: fast on-device English ASR via
        // sherpa-onnx. Picked in meeting options; keeps whisper as the default
        // (and the only choice for non-English).
        mb(
            "moonshine-base-en",
            "Moonshine Base (English, fast)",
            ModelKind::Stt,
            &[
                "https://huggingface.co/csukuangfj/sherpa-onnx-moonshine-base-en-int8/resolve/main/preprocess.onnx",
                "https://huggingface.co/csukuangfj/sherpa-onnx-moonshine-base-en-int8/resolve/main/encode.int8.onnx",
                "https://huggingface.co/csukuangfj/sherpa-onnx-moonshine-base-en-int8/resolve/main/uncached_decode.int8.onnx",
                "https://huggingface.co/csukuangfj/sherpa-onnx-moonshine-base-en-int8/resolve/main/cached_decode.int8.onnx",
                "https://huggingface.co/csukuangfj/sherpa-onnx-moonshine-base-en-int8/resolve/main/tokens.txt",
            ],
            210 * MB,
            "Low-latency English transcription for live meetings. English only.",
            "English",
        ),
    ]
}

/// Multilingual Whisper models that can stand in for a model without the
/// spoken language (and detect it), most accurate first. Fine-tunes such as
/// IndicWhisper are left out: their language detection is not trustworthy.
const FALLBACK_WHISPER: &[&str] = &[
    "whisper-large-v3-turbo",
    "whisper-large-v3",
    "whisper-medium",
    "whisper-small",
    "whisper-base",
    "whisper-tiny",
];

/// The downloaded Whisper model to use where the selected model cannot
/// transcribe the spoken language, or cannot tell which language it is.
pub fn fallback_whisper(models: &[ModelInfo]) -> Option<&ModelInfo> {
    FALLBACK_WHISPER
        .iter()
        .find_map(|id| models.iter().find(|m| m.id == *id && m.downloaded))
}

pub fn models_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("models");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

fn custom_models_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(models_dir(app)?.join("custom_models.json"))
}

fn load_custom_models(app: &AppHandle) -> Result<Vec<ModelInfo>, String> {
    let path = custom_models_path(app)?;
    if !path.exists() {
        return Ok(vec![]);
    }
    let data = fs::read_to_string(&path).map_err(|e| e.to_string())?;
    serde_json::from_str(&data).map_err(|e| e.to_string())
}

fn save_custom_models(app: &AppHandle, models: &[ModelInfo]) -> Result<(), String> {
    let data = serde_json::to_string_pretty(models).map_err(|e| e.to_string())?;
    fs::write(custom_models_path(app)?, data).map_err(|e| e.to_string())
}

/// The base filenames a bundle model expects on disk (derived from its URLs).
fn bundle_filenames(info: &ModelInfo) -> Vec<String> {
    info.files
        .iter()
        .map(|u| {
            u.split('?')
                .next()
                .unwrap_or(u)
                .rsplit('/')
                .next()
                .unwrap_or("file")
                .to_string()
        })
        .collect()
}

/// On-disk location: a single file for normal models, a per-id directory for
/// multi-file bundles.
pub fn model_path(app: &AppHandle, info: &ModelInfo) -> Result<PathBuf, String> {
    let dir = models_dir(app)?;
    if info.files.is_empty() {
        Ok(dir.join(&info.filename))
    } else {
        Ok(dir.join(&info.id))
    }
}

fn is_downloaded(app: &AppHandle, info: &ModelInfo) -> bool {
    let Ok(base) = model_path(app, info) else {
        return false;
    };
    if info.files.is_empty() {
        base.exists()
    } else {
        bundle_filenames(info).iter().all(|f| base.join(f).exists())
    }
}

pub fn all_models(app: &AppHandle) -> Result<Vec<ModelInfo>, String> {
    let mut models = builtin_registry();
    models.extend(load_custom_models(app)?);
    for m in &mut models {
        m.downloaded = is_downloaded(app, m);
    }
    Ok(models)
}

pub fn find_model(app: &AppHandle, id: &str) -> Result<ModelInfo, String> {
    let id = migrate_id(id);
    all_models(app)?
        .into_iter()
        .find(|m| m.id == id)
        .ok_or_else(|| format!("unknown model id: {id}"))
}

/// Resolve the on-disk path of a downloaded model, erroring if it is missing.
pub fn downloaded_model_path(app: &AppHandle, id: &str) -> Result<PathBuf, String> {
    let info = find_model(app, id)?;
    if !is_downloaded(app, &info) {
        return Err(format!("model '{}' is not downloaded yet", info.name));
    }
    model_path(app, &info)
}

pub fn add_custom(app: &AppHandle, name: String, kind: ModelKind, url: String) -> Result<ModelInfo, String> {
    let filename = url
        .split('?')
        .next()
        .unwrap_or(&url)
        .rsplit('/')
        .next()
        .filter(|f| !f.is_empty())
        .ok_or("could not derive a filename from that URL")?
        .to_string();
    let mut customs = load_custom_models(app)?;
    let id = format!("custom-{}", filename.to_lowercase());
    if customs.iter().any(|m| m.id == id) {
        return Err("a custom model with that filename already exists".into());
    }
    let info = ModelInfo {
        id,
        name,
        kind,
        url,
        filename,
        size_bytes: 0,
        description: "Custom model".into(),
        languages: String::new(),
        recommended: false,
        custom: true,
        downloaded: false,
        files: vec![],
        engine: SttEngine::Whisper,
        supported_languages: vec![],
    };
    customs.push(info.clone());
    save_custom_models(app, &customs)?;
    Ok(info)
}

pub fn delete_model_file(app: &AppHandle, id: &str) -> Result<(), String> {
    let info = find_model(app, id)?;
    let path = model_path(app, &info)?;
    if path.exists() {
        // Bundle models are a directory; single-file models are a file.
        if path.is_dir() {
            fs::remove_dir_all(&path).map_err(|e| e.to_string())?;
        } else {
            fs::remove_file(&path).map_err(|e| e.to_string())?;
        }
    }
    if info.custom {
        let customs: Vec<ModelInfo> = load_custom_models(app)?
            .into_iter()
            .filter(|m| m.id != id)
            .collect();
        save_custom_models(app, &customs)?;
    }
    Ok(())
}

/// Download a model file, emitting `model-download-progress` and
/// `model-download-done` events. Runs on the async runtime.
pub async fn download(app: AppHandle, id: String) -> Result<(), String> {
    let result = download_inner(&app, &id).await;
    let done = DownloadDone {
        id: id.clone(),
        ok: result.is_ok(),
        error: result.as_ref().err().cloned(),
    };
    let _ = app.emit("model-download-done", done);
    result
}

async fn download_inner(app: &AppHandle, id: &str) -> Result<(), String> {
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;

    let info = find_model(app, id)?;

    // Multi-file bundle (e.g. Moonshine): download each file into the per-id
    // directory, reporting cumulative progress against the model's total size.
    if !info.files.is_empty() {
        let dir = model_path(app, &info)?;
        tokio::fs::create_dir_all(&dir).await.map_err(|e| e.to_string())?;
        let total = info.size_bytes.max(1);
        let mut downloaded: u64 = 0;
        let mut last_emit: u64 = 0;
        for (url, name) in info.files.iter().zip(bundle_filenames(&info)) {
            let resp = reqwest::get(url).await.map_err(|e| e.to_string())?;
            if !resp.status().is_success() {
                return Err(format!("download failed: HTTP {}", resp.status()));
            }
            let dest = dir.join(&name);
            let part = dest.with_extension("part");
            let mut file = tokio::fs::File::create(&part).await.map_err(|e| e.to_string())?;
            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| e.to_string())?;
                file.write_all(&chunk).await.map_err(|e| e.to_string())?;
                downloaded += chunk.len() as u64;
                if downloaded - last_emit > 2 * MB {
                    last_emit = downloaded;
                    let _ = app.emit(
                        "model-download-progress",
                        DownloadProgress {
                            id: id.to_string(),
                            downloaded,
                            total,
                        },
                    );
                }
            }
            file.flush().await.map_err(|e| e.to_string())?;
            drop(file);
            tokio::fs::rename(&part, &dest).await.map_err(|e| e.to_string())?;
        }
        return Ok(());
    }

    let final_path = model_path(app, &info)?;
    let part_path = final_path.with_extension("part");

    let resp = reqwest::get(&info.url).await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("download failed: HTTP {}", resp.status()));
    }
    let total = resp.content_length().unwrap_or(info.size_bytes);

    let mut file = tokio::fs::File::create(&part_path)
        .await
        .map_err(|e| e.to_string())?;
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut last_emit: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| e.to_string())?;
        file.write_all(&chunk).await.map_err(|e| e.to_string())?;
        downloaded += chunk.len() as u64;
        // Throttle progress events to roughly every 2 MB.
        if downloaded - last_emit > 2 * MB {
            last_emit = downloaded;
            let _ = app.emit(
                "model-download-progress",
                DownloadProgress {
                    id: id.to_string(),
                    downloaded,
                    total,
                },
            );
        }
    }
    file.flush().await.map_err(|e| e.to_string())?;
    drop(file);
    tokio::fs::rename(&part_path, &final_path)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn installed(ids: &[&str]) -> Vec<ModelInfo> {
        builtin_registry()
            .into_iter()
            .map(|mut m| {
                m.downloaded = ids.contains(&m.id.as_str());
                m
            })
            .collect()
    }

    #[test]
    fn removed_sherpa_parakeet_maps_to_the_new_v3() {
        assert_eq!(migrate_id("parakeet-tdt-v3"), "parakeet-tdt-0.6b-v3");
        assert_eq!(migrate_id("whisper-large-v3-turbo"), "whisper-large-v3-turbo");
        let reg = builtin_registry();
        assert!(reg.iter().all(|m| m.id != "parakeet-tdt-v3"));
        // Every replacement exists, so a migrated id never dangles.
        for (_, new) in RENAMED {
            assert!(reg.iter().any(|m| m.id == *new), "{new}");
        }
    }

    #[test]
    fn parakeet_entries_are_single_gguf_files_with_their_languages() {
        let reg = builtin_registry();
        let v2 = reg.iter().find(|m| m.id == "parakeet-tdt-0.6b-v2").unwrap();
        let v3 = reg.iter().find(|m| m.id == "parakeet-tdt-0.6b-v3").unwrap();
        for m in [v2, v3] {
            assert_eq!(m.engine, SttEngine::Parakeet);
            assert!(m.files.is_empty() && m.filename.ends_with("-Q8_0.gguf"));
            assert!(!m.recommended);
        }
        assert_eq!(v2.supported_languages, ["en"]);
        assert_eq!(v3.supported_languages.len(), 25);
        assert!(v3.supported_languages.iter().any(|l| l == "de"));
        assert!(!v3.supported_languages.iter().any(|l| l == "hi"));
    }

    #[test]
    fn whisper_turbo_stays_the_recommended_speech_model() {
        let rec: Vec<_> = builtin_registry()
            .into_iter()
            .filter(|m| m.kind == ModelKind::Stt && m.recommended)
            .map(|m| m.id)
            .collect();
        assert_eq!(rec, ["whisper-large-v3-turbo"]);
    }

    #[test]
    fn fallback_prefers_the_most_accurate_multilingual_whisper() {
        let pick = |ids: &[&str]| {
            let models = installed(ids);
            fallback_whisper(&models).map(|m| m.id.clone())
        };
        assert_eq!(pick(&["parakeet-tdt-0.6b-v3"]), None);
        assert_eq!(pick(&["whisper-small-en", "indic-whisper-medium"]), None);
        assert_eq!(pick(&["whisper-tiny", "whisper-large-v3-turbo"]).as_deref(), Some("whisper-large-v3-turbo"));
        assert_eq!(pick(&["whisper-base", "whisper-small"]).as_deref(), Some("whisper-small"));
    }
}
