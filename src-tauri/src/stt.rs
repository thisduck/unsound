use std::path::Path;
use std::sync::{Arc, Mutex};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Whisper is prone to dropping the first word when speech starts at sample
/// zero. A short quiet lead-in gives its speech detector enough context without
/// delaying capture or changing the duration reported to the UI.
const LEADING_SILENCE_SAMPLES: usize = 8_000; // 500 ms at 16 kHz

fn with_leading_silence(samples: &[f32]) -> Vec<f32> {
    let mut padded = Vec::with_capacity(LEADING_SILENCE_SAMPLES + samples.len());
    padded.resize(LEADING_SILENCE_SAMPLES, 0.0);
    padded.extend_from_slice(samples);
    padded
}

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

    /// Preload the model and compile its Metal kernels by running one throwaway
    /// decode, so the first real transcription of a meeting isn't slow.
    pub fn warmup(&self, model_path: &Path) -> Result<(), String> {
        let ctx = self.context_for(model_path)?;
        let mut ws = ctx
            .create_state()
            .map_err(|e| format!("failed to create whisper state: {e}"))?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(1);
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        // ~0.3s of silence is enough to build the graph and compile kernels.
        let warm = vec![0.0f32; 4800];
        let _ = ws.full(params, &warm);
        Ok(())
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

    // Dictation often begins immediately after push-to-talk is pressed. Keep
    // the captured speech intact, but move it away from the decoder's absolute
    // start boundary so the opening phonemes are not mistaken for noise.
    let padded = with_leading_silence(samples);
    ws.full(params, &padded)
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
    // Whisper hallucinates a repeated phrase (often in a wrong auto-detected
    // language) when fed near-silence. Skip a channel that carries no real
    // signal — e.g. a system-audio capture that didn't actually receive the
    // call — rather than emit that garbage. Real speech peaks well above this.
    let peak = samples.iter().fold(0f32, |m, &s| m.max(s.abs()));
    eprintln!("[stt] segments: {} samples, peak {peak:.4}", samples.len());
    if peak < 0.005 {
        eprintln!("[stt] channel is effectively silent — skipping to avoid hallucination");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictation_padding_preserves_every_captured_sample() {
        let speech = [0.25, -0.5, 0.75];
        let padded = with_leading_silence(&speech);

        assert_eq!(padded.len(), LEADING_SILENCE_SAMPLES + speech.len());
        assert!(padded[..LEADING_SILENCE_SAMPLES]
            .iter()
            .all(|sample| *sample == 0.0));
        assert_eq!(&padded[LEADING_SILENCE_SAMPLES..], speech);
    }
}
