//! Accuracy/latency eval of speech engines over a JSONL manifest of reference
//! clips (see spikes/parakeet-eval/README.md).
//!
//!   cargo run --release --example asr_eval -- whisper <ggml.bin> eval <manifest> <out.jsonl>
//!   cargo run --release --example asr_eval -- sherpa-parakeet <dir> eval <manifest> <out.jsonl>
//!   cargo run --release --example asr_eval -- wcpp-parakeet <ggml-parakeet.bin> eval <manifest> <out.jsonl>
//!   cargo run --release --example asr_eval -- tc <model.gguf> eval <manifest> <out.jsonl>
//!   ... probe <lang> <wav>...
//!
//! Whisper mirrors `stt::transcribe` with the clip's language passed in, as
//! when background language detection is confident. `WHISPER_PROMPT` sets an
//! `initial_prompt` glossary.

#[allow(dead_code)]
#[path = "../spikes/parakeet-eval/src/harness.rs"]
mod harness;
#[allow(dead_code)]
#[path = "support/engines.rs"]
mod engines;

// Links llama-cpp-sys-2's ggml, which whisper.cpp is built against, even in a
// baseline build against the unpatched vendor crate (see vendor README).
extern crate llama_cpp_2;

use harness::Asr;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (kind, model) = (args[0].as_str(), &args[1]);
    let t = Instant::now();
    let mut eng: Box<dyn Asr> = match kind {
        #[cfg(not(no_parakeet))]
        "wcpp-parakeet" => Box::new(engines::WcppParakeet::load(model)),
        #[cfg(not(no_parakeet))]
        "tc" => Box::new(engines::Tc::load(model)),
        "whisper" => Box::new(engines::Whisper::load(model)),
        "sherpa-parakeet" => Box::new(engines::SherpaParakeet::load(model)),
        k => panic!("unknown engine {k}"),
    };
    let load_ms = harness::ms(t);
    eprintln!("{kind} load {load_ms:.0}ms");
    match args[2].as_str() {
        "eval" => harness::eval(eng.as_mut(), kind, load_ms, &args[3], &args[4]),
        "probe" => harness::probe(eng.as_mut(), &args[3], &args[4..]),
        m => panic!("unknown mode {m}"),
    }
}
