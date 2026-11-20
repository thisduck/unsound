use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

mod filter;
use filter::{collapse_repetitions, sanitize_segment};

const SILENCE_PEAK_THRESHOLD: f32 = 0.005;
/// Below this, a background language guess is discarded and the final
/// transcription detects the language itself, as it would without one.
const LANGUAGE_MIN_PROBABILITY: f32 = 0.5;

/// Whisper is prone to dropping the first word when speech starts at sample
/// zero. A short quiet lead-in gives its speech detector enough context without
/// delaying capture or changing the duration reported to the UI.
const LEADING_SILENCE_SAMPLES: usize = 8_000; // 500 ms at 16 kHz

fn is_effectively_silent(samples: &[f32]) -> bool {
    samples
        .iter()
        .all(|sample| sample.abs() < SILENCE_PEAK_THRESHOLD)
}

/// Whisper's abort hook: give up once the `AtomicBool` behind `flag` is set.
unsafe extern "C" fn abort_when_set(flag: *mut std::ffi::c_void) -> bool {
    (*(flag as *const AtomicBool)).load(Ordering::Relaxed)
}

fn n_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .min(8)
}

/// Decoding parameters shared by every Whisper call (and `examples/stt_bench`),
/// so they cannot drift apart. Greedy, no conditioning on previous text.
pub fn decode_params<'a, 'b>(n_threads: i32, language: Option<&'a str>) -> FullParams<'a, 'b> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(n_threads);
    params.set_language(language);
    params.set_translate(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(true);
    // Suppress symbol and annotation tokens (♪, [, (...), as openai-whisper
    // does by default; cuts "[Music]"-style output on noise.
    params.set_suppress_nst(true);
    params
}

/// Abort `full` soon after `stop` is set. Wired directly: whisper-rs 0.16's
/// set_abort_callback_safe hands its trampoline the wrong pointer type, so the
/// closure reads garbage and aborts at once.
///
/// SAFETY: the caller keeps `stop` alive until `full` returns.
unsafe fn set_abort(params: &mut FullParams, stop: &Arc<AtomicBool>) {
    params.set_abort_callback(Some(abort_when_set));
    params.set_abort_callback_user_data(Arc::as_ptr(stop) as *mut std::ffi::c_void);
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
        // whisper-rs defaults flash attention off (whisper.cpp itself defaults
        // it on); with Metal it cuts transcription time by 15-40%.
        let mut params = WhisperContextParameters::default();
        params.flash_attn(true);
        let ctx = WhisperContext::new_with_params(&key, params)
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
        let params = decode_params(1, Some("en"));
        // ~0.3s of silence is enough to build the graph and compile kernels.
        let warm = vec![0.0f32; 4800];
        let _ = ws.full(params, &warm);
        Ok(())
    }
}

/// Identify the spoken language of `samples` (the first 30 s are used), for
/// passing to `transcribe` in place of auto-detection, which otherwise costs
/// Whisper a second full encoder pass on every take. `None` when the guess
/// isn't confident enough to trust.
pub fn detect_language(
    state: &SttState,
    model_path: &Path,
    samples: &[f32],
) -> Result<Option<String>, String> {
    let ctx = state.context_for(model_path)?;
    // English-only models ignore the language anyway.
    if !ctx.is_multilingual() {
        return Ok(Some("en".into()));
    }
    let mut ws = ctx
        .create_state()
        .map_err(|e| format!("failed to create whisper state: {e}"))?;
    // Match what `transcribe` feeds Whisper, whose own detection this replaces.
    let window = &samples[..samples.len().min(30 * 16_000)];
    let threads = n_threads() as usize;
    ws.pcm_to_mel(&with_leading_silence(window), threads)
        .map_err(|e| format!("failed to compute mel: {e}"))?;
    let (id, probs) = ws
        .lang_detect(0, threads)
        .map_err(|e| format!("language detection failed: {e}"))?;
    let p = probs.get(id as usize).copied().unwrap_or(0.0);
    let lang = whisper_rs::get_lang_str(id).map(str::to_string);
    eprintln!("[stt] detected language {lang:?} (p = {p:.2})");
    Ok(lang.filter(|_| p >= LANGUAGE_MIN_PROBABILITY))
}

pub fn transcribe(
    state: &SttState,
    model_path: &Path,
    samples: &[f32],
    language: Option<&str>,
    initial_prompt: Option<&str>,
) -> Result<String, String> {
    transcribe_until(state, model_path, samples, language, initial_prompt, None)
}

/// `transcribe`, giving up with an error soon after `stop` is set — for live
/// previews, which mustn't hold up the final pass when a take ends.
pub fn transcribe_until(
    state: &SttState,
    model_path: &Path,
    samples: &[f32],
    language: Option<&str>,
    initial_prompt: Option<&str>,
    stop: Option<Arc<AtomicBool>>,
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

    let mut params = decode_params(n_threads(), language.or(Some("auto")));
    // Bias recognition toward the user's vocabulary (names, jargon).
    if let Some(p) = initial_prompt {
        params.set_initial_prompt(p);
    }
    if let Some(stop) = stop.as_ref() {
        // SAFETY: `stop` is alive until this function returns, after `full`.
        unsafe { set_abort(&mut params, stop) };
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
    // A loop can also span Whisper's segment boundaries.
    Ok(collapse_repetitions(text.trim()))
}

/// Like `transcribe`, but keeps Whisper's per-segment timestamps — the shape a
/// meeting transcript needs so the two channels (mic/system) can be interleaved
/// on a timeline. Returns `(start_ms, end_ms, text)` per non-empty segment. An
/// empty/near-silent channel yields an empty vec rather than an error.
pub fn transcribe_segments_until(
    state: &SttState, model_path: &Path, samples: &[f32], language: Option<&str>,
    initial_prompt: Option<&str>, stop: Option<Arc<AtomicBool>>,
) -> Result<Vec<(i64, i64, String)>, String> {
    if stop.as_ref().is_some_and(|s| s.load(Ordering::Relaxed)) { return Err("transcription paused".into()); }
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

    let mut params = decode_params(n_threads(), language.or(Some("auto")));
    if let Some(p) = initial_prompt {
        params.set_initial_prompt(p);
    }

    if let Some(stop) = stop.as_ref() {
        // SAFETY: `stop` is alive until this function returns, after `full`.
        unsafe { set_abort(&mut params, stop) };
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
