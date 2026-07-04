use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::TokenToStringError;
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use tauri::{AppHandle, Emitter};

pub const DEFAULT_CLEANUP_PROMPT: &str = "You clean up raw speech-to-text transcripts. Fix punctuation, capitalization and obvious transcription errors, remove filler words (um, uh, you know), and break the text into paragraphs where natural. Preserve the speaker's wording and meaning; do not summarize or add anything. Output only the cleaned text.";

const MAX_NEW_TOKENS: usize = 4096;

static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();

fn backend() -> Result<&'static LlamaBackend, String> {
    BACKEND
        .get_or_init(|| LlamaBackend::init().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| e.clone())
}

/// Caches the most recently used LLM so repeated cleanups are fast.
#[derive(Default)]
pub struct LlmState {
    loaded: Mutex<Option<(String, Arc<LlamaModel>)>>,
}

impl LlmState {
    fn model_for(&self, model_path: &Path) -> Result<Arc<LlamaModel>, String> {
        let key = model_path.to_string_lossy().to_string();
        let mut loaded = self.loaded.lock().unwrap();
        if let Some((path, model)) = loaded.as_ref() {
            if *path == key {
                return Ok(model.clone());
            }
        }
        let backend = backend()?;
        // Offload everything to the GPU where one exists (Metal on macOS);
        // harmless on CPU-only builds.
        let params = LlamaModelParams::default().with_n_gpu_layers(1_000_000);
        let model = LlamaModel::load_from_file(backend, model_path, &params)
            .map_err(|e| format!("failed to load LLM: {e}"))?;
        let model = Arc::new(model);
        *loaded = Some((key, model.clone()));
        Ok(model)
    }
}

/// Incrementally converts token bytes to valid UTF-8 chunks, holding back
/// incomplete multi-byte sequences until the next token completes them.
struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    fn new() -> Self {
        Self { pending: Vec::new() }
    }
    fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_string();
                self.pending.clear();
                out
            }
            Err(e) => {
                let valid = e.valid_up_to();
                let out = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                self.pending.drain(..valid);
                out
            }
        }
    }
}

pub fn cleanup_text(
    app: &AppHandle,
    state: &LlmState,
    model_path: &Path,
    system_prompt: &str,
    transcript: &str,
) -> Result<String, String> {
    let backend = backend()?;
    let model = state.model_for(model_path)?;

    let messages = vec![
        LlamaChatMessage::new("system".into(), system_prompt.into()).map_err(|e| e.to_string())?,
        LlamaChatMessage::new("user".into(), transcript.into()).map_err(|e| e.to_string())?,
    ];
    let template = model
        .chat_template(None)
        .map_err(|e| format!("model has no usable chat template: {e}"))?;
    let prompt = model
        .apply_chat_template(&template, &messages, true)
        .map_err(|e| format!("failed to apply chat template: {e}"))?;

    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| format!("tokenization failed: {e}"))?;

    let n_ctx = (tokens.len() + MAX_NEW_TOKENS + 16).max(2048) as u32;
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| format!("failed to create LLM context: {e}"))?;

    let mut batch = LlamaBatch::new(tokens.len().max(512), 1);
    let last = tokens.len() - 1;
    for (i, token) in tokens.iter().enumerate() {
        batch
            .add(*token, i as i32, &[0], i == last)
            .map_err(|e| e.to_string())?;
    }
    ctx.decode(&mut batch).map_err(|e| format!("prompt decode failed: {e}"))?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(0.2),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::dist(42),
    ]);

    let mut utf8 = Utf8Stream::new();
    let mut output = String::new();
    let mut n_cur = batch.n_tokens();
    let mut generated = 0usize;

    loop {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        let bytes = match model.token_to_piece_bytes(token, 64, true, None) {
            Err(TokenToStringError::InsufficientBufferSpace(i)) => model
                .token_to_piece_bytes(token, (-i).max(1) as usize, true, None)
                .map_err(|e| e.to_string())?,
            other => other.map_err(|e| e.to_string())?,
        };
        let chunk = utf8.push(&bytes);
        if !chunk.is_empty() {
            output.push_str(&chunk);
            let _ = app.emit("llm-token", &chunk);
        }
        generated += 1;
        if generated >= MAX_NEW_TOKENS {
            break;
        }
        batch.clear();
        batch.add(token, n_cur, &[0], true).map_err(|e| e.to_string())?;
        n_cur += 1;
        ctx.decode(&mut batch)
            .map_err(|e| format!("decode failed: {e}"))?;
    }

    Ok(output.trim().to_string())
}
