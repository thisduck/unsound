use std::path::Path;
use std::sync::{Arc, Mutex};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const SILENCE_PEAK_THRESHOLD: f32 = 0.005;
const NO_SPEECH_PROBABILITY_THRESHOLD: f32 = 0.6;

/// Whisper is prone to dropping the first word when speech starts at sample
/// zero. A short quiet lead-in gives its speech detector enough context without
/// delaying capture or changing the duration reported to the UI.
const LEADING_SILENCE_SAMPLES: usize = 8_000; // 500 ms at 16 kHz

/// Whisper can emit special markers, or a short stock phrase such as "Thank
/// you", when it believes the input contains no speech. Keep that model output
/// from becoming user text while preserving the same words when they were
/// actually spoken.
fn sanitize_segment(text: &str, no_speech_probability: f32) -> String {
    let mut cleaned = text.to_string();
    let marker = "[BLANK_AUDIO]";
    let lowercase_marker = marker.to_ascii_lowercase();
    while let Some(start) = cleaned.to_ascii_lowercase().find(&lowercase_marker) {
        cleaned.replace_range(start..start + marker.len(), "");
    }
    let words = cleaned
        .trim()
        .chars()
        .filter(|c| c.is_alphabetic() || c.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    let words = words.split_whitespace().collect::<Vec<_>>().join(" ");
    if no_speech_probability >= NO_SPEECH_PROBABILITY_THRESHOLD && words == "thank you" {
        String::new()
    } else {
        cleaned
    }
}

fn is_effectively_silent(samples: &[f32]) -> bool {
    samples
        .iter()
        .all(|sample| sample.abs() < SILENCE_PEAK_THRESHOLD)
}

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
    if is_effectively_silent(samples) {
        return Ok(String::new());
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
            text.push_str(&sanitize_segment(&piece, segment.no_speech_probability()));
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
    if is_effectively_silent(samples) {
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
            let text = sanitize_segment(
                &segment
                    .to_str_lossy()
                    .map_err(|e| format!("failed to read segment {i}: {e}"))?,
                segment.no_speech_probability(),
            )
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
    fn strips_blank_audio_markers() {
        assert_eq!(sanitize_segment(" [BLANK_AUDIO] ", 0.0), "  ");
        assert_eq!(
            sanitize_segment("hello [blank_audio] world", 0.0),
            "hello  world"
        );
    }

    #[test]
    fn drops_thank_you_only_when_model_detects_no_speech() {
        assert_eq!(sanitize_segment("Thank you.", 0.9), "");
        assert_eq!(sanitize_segment("Thank you.", 0.1), "Thank you.");
    }

    #[test]
    fn preserves_word_boundaries_between_segments() {
        let mut transcript = sanitize_segment(" Hello", 0.0);
        transcript.push_str(&sanitize_segment(" world", 0.0));
        assert_eq!(transcript, " Hello world");
    }

    #[test]
    fn detects_effectively_silent_audio() {
        assert!(is_effectively_silent(&[0.0, 0.001, -0.0049]));
        assert!(!is_effectively_silent(&[0.0, 0.005, -0.0049]));
    }

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
