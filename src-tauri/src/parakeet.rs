//! Parakeet TDT 0.6B (NVIDIA; weights CC-BY-4.0) through transcribe.cpp, on the
//! GPU. transcribe.cpp is built on the ggml llama-cpp-sys-2 provides, shared
//! with whisper.cpp and llama.cpp (vendor/transcribe-cpp-sys/README.md).
//!
//! No prompt or vocabulary biasing; word timestamps at the encoder's 80 ms
//! frame rate. The model card's languages are the only ones it can transcribe
//! (`asr::route` keeps it off the rest).

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use transcribe_cpp::{CancelToken, Model, RunOptions, Session, TimestampKind};

/// One recognized word, in ms from the start of the buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

/// Raw engine output: no filtering applied.
#[derive(Debug, Default)]
pub struct Output {
    pub text: String,
    pub words: Vec<Word>,
}

struct Loaded {
    key: String,
    // Declared before the model: a session must be freed first.
    session: Session,
    _model: Model,
}

/// Caches the most recently used Parakeet model and its session, as
/// `stt::SttState` does for Whisper. Runs are serialized by the lock.
#[derive(Default)]
pub struct ParakeetState {
    loaded: Mutex<Option<Loaded>>,
}

impl ParakeetState {
    /// Free the model so its Metal buffers are released before ggml's static
    /// destructors run at process exit.
    pub fn clear(&self) {
        *self.loaded.lock().unwrap() = None;
    }

    fn with_session<R>(&self, model_path: &Path, f: impl FnOnce(&mut Session) -> R) -> Result<R, String> {
        let key = model_path.to_string_lossy().to_string();
        let mut loaded = self.loaded.lock().unwrap_or_else(|e| e.into_inner());
        if loaded.as_ref().is_none_or(|l| l.key != key) {
            // Drop the previous model before loading another.
            *loaded = None;
            let model = Model::load(model_path).map_err(|e| format!("failed to load Parakeet model: {e}"))?;
            let session = model
                .session()
                .map_err(|e| format!("failed to start Parakeet session: {e}"))?;
            eprintln!("[parakeet] loaded {} on {}", model.variant(), model.backend());
            *loaded = Some(Loaded { key, session, _model: model });
        }
        Ok(f(&mut loaded.as_mut().unwrap().session))
    }

    /// Load the model and build its Metal pipelines with a throwaway run.
    pub fn warmup(&self, model_path: &Path) -> Result<(), String> {
        self.with_session(model_path, |s| {
            let _ = s.run(&[0.0; 4800], &run_options());
        })
    }

    /// Transcribe 16 kHz mono samples. `stop` aborts the run between decode
    /// steps (transcribe.cpp polls it); an aborted run is an error.
    pub fn transcribe(&self, model_path: &Path, samples: &[f32], stop: Option<&AtomicBool>) -> Result<Output, String> {
        let stopped = || stop.is_some_and(|s| s.load(Ordering::Relaxed));
        if stopped() {
            return Err("transcription paused".into());
        }
        let result = self.with_session(model_path, |session| match stop {
            None => session.run(samples, &run_options()),
            Some(stop) => {
                // transcribe.cpp takes its own cancel token; mirror `stop` into it.
                let token = CancelToken::new();
                session.set_cancel_token(&token);
                let done = AtomicBool::new(false);
                let result = std::thread::scope(|scope| {
                    scope.spawn(|| {
                        while !done.load(Ordering::Relaxed) {
                            if stop.load(Ordering::Relaxed) {
                                token.cancel();
                                return;
                            }
                            std::thread::sleep(Duration::from_millis(10));
                        }
                    });
                    let r = session.run(samples, &run_options());
                    done.store(true, Ordering::Relaxed);
                    r
                });
                session.clear_cancel_token();
                result
            }
        })?;
        if stopped() {
            return Err("transcription paused".into());
        }
        match result {
            Ok(t) => Ok(Output {
                text: t.text.trim().to_string(),
                words: t
                    .words
                    .into_iter()
                    .map(|w| Word { start_ms: w.t0_ms, end_ms: w.t1_ms, text: w.text })
                    .filter(|w| !w.text.trim().is_empty())
                    .collect(),
            }),
            Err(e) => Err(format!("transcription failed: {e}")),
        }
    }
}

/// Greedy TDT decode with word timestamps. No language hint: the TDT models
/// do not condition on one (transcribe.cpp would only validate it).
fn run_options() -> RunOptions {
    RunOptions { timestamps: TimestampKind::Word, ..Default::default() }
}

/// Assign words to the pieces of a batched buffer by each word's midpoint.
/// `spans` are `(offset_ms, len_ms)` of each piece in the buffer; a word in a
/// gap goes to the nearest piece. Returns one word list per piece, with times
/// made relative to that piece and clamped to it.
pub fn assign_words(words: &[Word], spans: &[(i64, i64)]) -> Vec<Vec<Word>> {
    let mut out = vec![Vec::new(); spans.len()];
    if spans.is_empty() {
        return out;
    }
    for w in words {
        let mid = (w.start_ms + w.end_ms) / 2;
        let distance = |&(off, len): &(i64, i64)| {
            if mid < off {
                off - mid
            } else {
                (mid - (off + len)).max(0)
            }
        };
        let (i, &(off, len)) = spans
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| distance(s))
            .unwrap();
        let start_ms = (w.start_ms - off).clamp(0, len);
        out[i].push(Word {
            start_ms,
            end_ms: (w.end_ms - off).clamp(start_ms, len),
            text: w.text.clone(),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(start_ms: i64, end_ms: i64, text: &str) -> Word {
        Word { start_ms, end_ms, text: text.into() }
    }

    #[test]
    fn words_go_to_the_piece_their_midpoint_falls_in() {
        // [300 ms gap][piece 0: 2 s][300 ms gap][piece 1: 1.5 s]
        let spans = [(300, 2000), (2600, 1500)];
        let words = [
            w(320, 640, "So"),
            w(640, 1200, "that's"),
            // Straddles the end of piece 0; its midpoint is inside it.
            w(2000, 2400, "done."),
            // Starts in the gap, midpoint in piece 1.
            w(2480, 2880, "Next"),
            w(2880, 3520, "item."),
        ];
        let got = assign_words(&words, &spans);
        let texts: Vec<Vec<&str>> = got.iter().map(|p| p.iter().map(|w| w.text.as_str()).collect()).collect();
        assert_eq!(texts, [vec!["So", "that's", "done."], vec!["Next", "item."]]);
        // Times are relative to the piece and clamped to it.
        assert_eq!((got[0][0].start_ms, got[0][0].end_ms), (20, 340));
        assert_eq!((got[0][2].start_ms, got[0][2].end_ms), (1700, 2000));
        assert_eq!((got[1][0].start_ms, got[1][0].end_ms), (0, 280));
    }

    #[test]
    fn a_word_in_a_gap_goes_to_the_nearest_piece() {
        let spans = [(300, 1000), (2300, 1000)];
        // Midpoint 1400: 100 ms after piece 0, 900 ms before piece 1.
        let got = assign_words(&[w(1360, 1440, "uh")], &spans);
        assert_eq!(got[0].len(), 1);
        assert!(got[1].is_empty());
        assert_eq!((got[0][0].start_ms, got[0][0].end_ms), (1000, 1000));
        // Before the first piece.
        let got = assign_words(&[w(0, 80, "um")], &spans);
        assert_eq!((got[0][0].start_ms, got[0][0].end_ms), (0, 0));
    }

    #[test]
    fn no_pieces_or_no_words() {
        assert!(assign_words(&[w(0, 80, "a")], &[]).is_empty());
        assert_eq!(assign_words(&[], &[(0, 100)]), vec![Vec::<Word>::new()]);
    }
}
