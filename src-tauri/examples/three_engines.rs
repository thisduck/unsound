//! Same-process check: Whisper (whisper.cpp), Parakeet (whisper.cpp's
//! parakeet.h) and an LLM (llama.cpp) loaded together on the one shared ggml,
//! interleaved, twice. Prints JSON with every transcript and generation so the
//! outputs can be diffed against standalone runs (spikes/parakeet-eval/README.md).
//!
//!   cargo run --release --example three_engines -- <whisper.bin> <parakeet.bin> <llm.gguf> <manifest.jsonl> [max_clips]

#[allow(dead_code)]
#[path = "../spikes/parakeet-eval/src/harness.rs"]
mod harness;
#[allow(dead_code)]
#[path = "support/engines.rs"]
mod engines;

use harness::Asr;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use std::io::BufRead;
use std::num::NonZeroU32;
use std::time::Instant;

fn generate(backend: &LlamaBackend, model: &LlamaModel, user: &str) -> String {
    let msgs = vec![LlamaChatMessage::new("user".into(), user.into()).unwrap()];
    let prompt = model
        .chat_template(None)
        .ok()
        .and_then(|t| model.apply_chat_template(&t, &msgs, true).ok())
        .unwrap_or_else(|| format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"));
    let toks = model.str_to_token(&prompt, AddBos::Always).unwrap();
    let params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(2048));
    let mut ctx = model.new_context(backend, params).unwrap();
    let mut batch = LlamaBatch::new(2048, 1);
    for (i, t) in toks.iter().enumerate() {
        batch.add(*t, i as i32, &[0], i == toks.len() - 1).unwrap();
    }
    ctx.decode(&mut batch).unwrap();
    let mut n = toks.len() as i32;
    let mut sampler = LlamaSampler::greedy();
    let mut out = Vec::new();
    for _ in 0..64 {
        let tok = sampler.sample(&ctx, batch.n_tokens() - 1);
        if model.is_eog_token(tok) {
            break;
        }
        out.extend(model.token_to_piece_bytes(tok, 128, true, None).unwrap());
        batch.clear();
        batch.add(tok, n, &[0], true).unwrap();
        n += 1;
        ctx.decode(&mut batch).unwrap();
    }
    String::from_utf8_lossy(&out).trim().to_string()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let max: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let clips: Vec<serde_json::Value> = std::io::BufReader::new(std::fs::File::open(&a[3]).unwrap())
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .take(max)
        .collect();

    // LLM_ONLY=<earlier three_engines output>: load only the LLM and replay the
    // round-0 prompts, for an LLM baseline with no speech engine in the process.
    if let Ok(prev) = std::env::var("LLM_ONLY") {
        let prev: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(prev).unwrap()).unwrap();
        let backend = LlamaBackend::init().unwrap();
        let llm = LlamaModel::load_from_file(&backend, &a[2], &LlamaModelParams::default().with_n_gpu_layers(1_000_000))
            .unwrap();
        let rows: Vec<_> = prev
            .iter()
            .filter(|r| r["round"] == 0)
            .map(|r| {
                let w = r["whisper"].as_str().unwrap();
                let g = generate(&backend, &llm, &format!("Summarise in five words: {w}"));
                serde_json::json!({"round": 0, "id": r["id"], "whisper": w, "parakeet": r["parakeet"], "llm": g})
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows).unwrap());
        return;
    }

    let t = Instant::now();
    let mut whisper = engines::Whisper::load(&a[0]);
    let whisper_ms = harness::ms(t);
    let t = Instant::now();
    let backend = LlamaBackend::init().unwrap();
    let llm = LlamaModel::load_from_file(&backend, &a[2], &LlamaModelParams::default().with_n_gpu_layers(1_000_000))
        .unwrap();
    let llm_ms = harness::ms(t);
    // PARAKEET=tc runs Parakeet through transcribe.cpp (a GGUF) instead of
    // whisper.cpp's parakeet.h (a ggml .bin); both on the same ggml.
    let t = Instant::now();
    let mut parakeet: Box<dyn Asr> = match std::env::var("PARAKEET").as_deref() {
        Ok("tc") => Box::new(engines::Tc::load(&a[1])),
        _ => Box::new(engines::WcppParakeet::load(&a[1])),
    };
    let parakeet_ms = harness::ms(t);
    eprintln!("loaded whisper {whisper_ms:.0}ms, llm {llm_ms:.0}ms, parakeet {parakeet_ms:.0}ms");

    let mut rows = vec![];
    for round in 0..2 {
        for c in &clips {
            let pcm = harness::read_wav(c["path"].as_str().unwrap());
            let lang = c["lang"].as_str().unwrap_or("en");
            // Round 0: whisper -> parakeet -> llm; round 1 the reverse order.
            let (w, p, g);
            if round == 0 {
                w = whisper.run(&pcm, lang).text.trim().to_string();
                p = parakeet.run(&pcm, lang).text.trim().to_string();
                g = generate(&backend, &llm, &format!("Summarise in five words: {w}"));
            } else {
                let w0 = rows.iter().find(|r: &&serde_json::Value| r["id"] == c["id"]).unwrap()["whisper"]
                    .as_str()
                    .unwrap()
                    .to_string();
                g = generate(&backend, &llm, &format!("Summarise in five words: {w0}"));
                p = parakeet.run(&pcm, lang).text.trim().to_string();
                w = whisper.run(&pcm, lang).text.trim().to_string();
            }
            rows.push(serde_json::json!({"round": round, "id": c["id"], "whisper": w, "parakeet": p, "llm": g}));
        }
    }
    println!("{}", serde_json::to_string_pretty(&rows).unwrap());
    drop(parakeet);
    drop(whisper);
}
