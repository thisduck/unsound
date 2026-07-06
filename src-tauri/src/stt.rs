use std::path::Path;
use std::sync::{Arc, Mutex};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Caches the most recently used whisper model so switching between
/// recordings doesn't reload multi-hundred-MB weights every time.
#[derive(Default)]
pub struct SttState {
    loaded: Mutex<Option<(String, Arc<WhisperContext>)>>,
}

impl SttState {
    /// Drop the cached model so its Metal buffers are released before
    /// ggml's static destructors run at process exit.
    pub fn clear(&self) {
        *self.loaded.lock().unwrap() = None;
    }

    fn context_for(&self, model_path: &Path) -> Result<Arc<WhisperContext>, String> {
        let key = model_path.to_string_lossy().to_string();
        let mut loaded = self.loaded.lock().unwrap();
        if let Some((path, ctx)) = loaded.as_ref() {
            if *path == key {
                return Ok(ctx.clone());
            }
        }
        let ctx = WhisperContext::new_with_params(&key, WhisperContextParameters::default())
            .map_err(|e| format!("failed to load whisper model: {e}"))?;
        let ctx = Arc::new(ctx);
        *loaded = Some((key, ctx.clone()));
        Ok(ctx)
    }
}

pub fn transcribe(
    state: &SttState,
    model_path: &Path,
    samples: &[f32],
    language: Option<&str>,
    initial_prompt: Option<&str>,
) -> Result<String, String> {
    if samples.len() < 1600 {
        return Err("recording is too short to transcribe".into());
    }
    let ctx = state.context_for(model_path)?;
    let mut ws = ctx
        .create_state()
        .map_err(|e| format!("failed to create whisper state: {e}"))?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .min(8);
    params.set_n_threads(threads);
    params.set_language(language.or(Some("auto")));
    params.set_translate(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(true);
    // Bias recognition toward the user's vocabulary (names, jargon).
    if let Some(p) = initial_prompt {
        params.set_initial_prompt(p);
    }

    ws.full(params, samples)
        .map_err(|e| format!("transcription failed: {e}"))?;

    let n = ws.full_n_segments();
    let mut text = String::new();
    for i in 0..n {
        if let Some(segment) = ws.get_segment(i) {
            let piece = segment
                .to_str_lossy()
                .map_err(|e| format!("failed to read segment {i}: {e}"))?;
            text.push_str(&piece);
        }
    }
    Ok(text.trim().to_string())
}

/// Like `transcribe`, but keeps Whisper's per-segment timestamps — the shape a
/// meeting transcript needs so the two channels (mic/system) can be interleaved
/// on a timeline. Returns `(start_ms, end_ms, text)` per non-empty segment. An
/// empty/near-silent channel yields an empty vec rather than an error.
pub fn transcribe_segments(
    state: &SttState,
    model_path: &Path,
    samples: &[f32],
    language: Option<&str>,
    initial_prompt: Option<&str>,
) -> Result<Vec<(i64, i64, String)>, String> {
    if samples.len() < 1600 {
        return Ok(Vec::new());
    }
    let ctx = state.context_for(model_path)?;
    let mut ws = ctx
        .create_state()
        .map_err(|e| format!("failed to create whisper state: {e}"))?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .min(8);
    params.set_n_threads(threads);
    params.set_language(language.or(Some("auto")));
    params.set_translate(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(true);
    if let Some(p) = initial_prompt {
        params.set_initial_prompt(p);
    }

    ws.full(params, samples)
        .map_err(|e| format!("transcription failed: {e}"))?;

    let n = ws.full_n_segments();
    let mut out = Vec::new();
    for i in 0..n {
        if let Some(segment) = ws.get_segment(i) {
            let text = segment
                .to_str_lossy()
                .map_err(|e| format!("failed to read segment {i}: {e}"))?
                .trim()
                .to_string();
            if text.is_empty() {
                continue;
            }
            // Whisper timestamps are in centiseconds (1/100 s).
            let start_ms = segment.start_timestamp() * 10;
            let end_ms = segment.end_timestamp() * 10;
            out.push((start_ms, end_ms, text));
        }
    }
    Ok(out)
}
