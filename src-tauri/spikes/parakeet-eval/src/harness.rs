//! Engine-agnostic eval loop shared by the standalone Parakeet bench and the
//! `asr_eval` example in the unsound crate (included there with `#[path]`).
//!
//! Manifest: JSONL with `id`, `path` (16 kHz mono WAV), `ref`, `dur`, `lang`.
//! Output: JSONL, first line `{"meta":..}` (load / first-run timings), then
//! one line per clip with the hypothesis, wall time and timestamp spans.

use std::io::{BufRead, Write};
use std::time::Instant;

pub struct Output {
    pub text: String,
    /// (start s, end s, text); empty when the engine gives no timestamps.
    pub spans: Vec<(f64, f64, String)>,
}

pub trait Asr {
    fn run(&mut self, pcm: &[f32], lang: &str) -> Output;
}

/// Minimal WAV reader (no extra dependency): 16 kHz mono, PCM16 or float32.
pub fn read_wav(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert!(&b[0..4] == b"RIFF" && &b[8..12] == b"WAVE", "{path}: not a WAV");
    let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    let (mut fmt, mut ch, mut rate, mut bits) = (0, 0, 0, 0);
    let mut i = 12;
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let len = u32_at(i + 4) as usize;
        let body = i + 8;
        if id == b"fmt " {
            fmt = u16_at(body);
            ch = u16_at(body + 2);
            rate = u32_at(body + 4);
            bits = u16_at(body + 14);
        } else if id == b"data" {
            assert_eq!(rate, 16000, "{path}: need 16 kHz");
            assert_eq!(ch, 1, "{path}: need mono");
            let d = &b[body..(body + len).min(b.len())];
            return match (fmt, bits) {
                (1, 16) => d.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect(),
                (3, 32) | (0xFFFE, 32) => d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                _ => panic!("{path}: unsupported format {fmt}/{bits}"),
            };
        }
        i = body + len + (len & 1);
    }
    panic!("{path}: no data chunk")
}

pub fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Run every clip in `manifest` once (after a warm-up on the first clip) and
/// write results to `out`. `load_ms` is the caller's model-load time.
pub fn eval(engine: &mut dyn Asr, name: &str, load_ms: f64, manifest: &str, out: &str) {
    let lines: Vec<serde_json::Value> = std::io::BufReader::new(std::fs::File::open(manifest).unwrap())
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect();
    let mut f = std::fs::File::create(out).unwrap();
    let first = read_wav(lines[0]["path"].as_str().unwrap());
    let lang0 = lines[0]["lang"].as_str().unwrap_or("en").to_string();
    let t = Instant::now();
    let _ = engine.run(&first, &lang0);
    let first_ms = ms(t);
    writeln!(
        f,
        "{}",
        serde_json::json!({"meta": true, "engine": name, "load_ms": load_ms, "first_run_ms": first_ms,
                           "first_clip_s": first.len() as f64 / 16000.0})
    )
    .unwrap();
    for (i, l) in lines.iter().enumerate() {
        let pcm = read_wav(l["path"].as_str().unwrap());
        let lang = l["lang"].as_str().unwrap_or("en");
        let t = Instant::now();
        let o = engine.run(&pcm, lang);
        let wall = ms(t);
        let spans: Vec<_> = o
            .spans
            .iter()
            .map(|(a, b, s)| serde_json::json!([(a * 100.0).round() / 100.0, (b * 100.0).round() / 100.0, s]))
            .collect();
        writeln!(
            f,
            "{}",
            serde_json::json!({"id": l["id"], "dur": pcm.len() as f64 / 16000.0, "ms": wall,
                               "hyp": o.text.trim(), "spans": spans})
        )
        .unwrap();
        if i % 20 == 0 {
            eprintln!("[{name}] {i}/{} {:.0}ms {}", lines.len(), wall, o.text.trim().chars().take(60).collect::<String>());
        }
    }
}

/// Print the transcript and spans for each WAV (for qualitative checks).
pub fn probe(engine: &mut dyn Asr, lang: &str, wavs: &[String]) {
    for w in wavs {
        let pcm = read_wav(w);
        let t = Instant::now();
        let o = engine.run(&pcm, lang);
        println!("== {w} ({:.1}s) {:.0}ms\n{}", pcm.len() as f64 / 16000.0, ms(t), o.text.trim());
        for (a, b, s) in o.spans.iter().take(40) {
            println!("   [{a:7.2} -> {b:7.2}] {s}");
        }
    }
}
