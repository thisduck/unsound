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
}

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

fn builtin_registry() -> Vec<ModelInfo> {
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
            true,
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
            "Best accuracy at near-medium speed.",
            WHISPER_LANGS,
            false,
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
    ]
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

pub fn model_path(app: &AppHandle, info: &ModelInfo) -> Result<PathBuf, String> {
    Ok(models_dir(app)?.join(&info.filename))
}

pub fn all_models(app: &AppHandle) -> Result<Vec<ModelInfo>, String> {
    let mut models = builtin_registry();
    models.extend(load_custom_models(app)?);
    for m in &mut models {
        m.downloaded = model_path(app, m)?.exists();
    }
    Ok(models)
}

pub fn find_model(app: &AppHandle, id: &str) -> Result<ModelInfo, String> {
    all_models(app)?
        .into_iter()
        .find(|m| m.id == id)
        .ok_or_else(|| format!("unknown model id: {id}"))
}

/// Resolve the on-disk path of a downloaded model, erroring if it is missing.
pub fn downloaded_model_path(app: &AppHandle, id: &str) -> Result<PathBuf, String> {
    let info = find_model(app, id)?;
    let path = model_path(app, &info)?;
    if !path.exists() {
        return Err(format!("model '{}' is not downloaded yet", info.name));
    }
    Ok(path)
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
    };
    customs.push(info.clone());
    save_custom_models(app, &customs)?;
    Ok(info)
}

pub fn delete_model_file(app: &AppHandle, id: &str) -> Result<(), String> {
    let info = find_model(app, id)?;
    let path = model_path(app, &info)?;
    if path.exists() {
        fs::remove_file(&path).map_err(|e| e.to_string())?;
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
