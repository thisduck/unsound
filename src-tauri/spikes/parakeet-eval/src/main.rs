//! Parakeet (or any transcribe.cpp GGUF) through transcribe.cpp on Metal.
//!
//!   parakeet-eval <model.gguf> eval <manifest.jsonl> <out.jsonl>
//!   parakeet-eval <model.gguf> probe <lang> <wav>...
//!
//! Spans are word timestamps when the model gives them (Parakeet: token
//! timestamps grouped into words), else segments.

mod harness;

use harness::{Asr, Output};
use std::time::Instant;
use transcribe_cpp::{Model, RunOptions, Session, TimestampKind};

struct Tc {
    session: Session,
    ts: TimestampKind,
    multilingual_hint: bool,
}

impl Asr for Tc {
    fn run(&mut self, pcm: &[f32], lang: &str) -> Output {
        let opts = RunOptions {
            timestamps: self.ts,
            // Parakeet v3 auto-detects and ignores the hint; whisper uses it.
            language: if self.multilingual_hint { Some(lang.to_string()) } else { None },
            ..Default::default()
        };
        let r = match self.session.run(pcm, &opts) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("run failed: {e}");
                return Output { text: String::new(), spans: vec![] };
            }
        };
        let spans = if !r.words.is_empty() {
            r.words.iter().map(|w| (w.t0_ms as f64 / 1000.0, w.t1_ms as f64 / 1000.0, w.text.clone())).collect()
        } else {
            r.segments.iter().map(|s| (s.t0_ms as f64 / 1000.0, s.t1_ms as f64 / 1000.0, s.text.clone())).collect()
        };
        Output { text: r.text, spans }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_path = &args[0];
    let t = Instant::now();
    let model = Model::load(model_path).expect("load");
    let session = model.session().expect("session");
    let load_ms = harness::ms(t);
    let caps = model.capabilities().expect("caps");
    eprintln!(
        "model {} variant {} on {} load {:.0}ms max_ts {:?}",
        model.arch(),
        model.variant(),
        model.backend(),
        load_ms,
        caps.max_timestamp_kind
    );
    let ts = if caps.max_timestamp_kind == TimestampKind::Token { TimestampKind::Word } else { caps.max_timestamp_kind };
    let mut eng = Tc { session, ts, multilingual_hint: model.arch() == "whisper" };
    let name = format!("tc:{}", model.variant());
    match args[1].as_str() {
        "eval" => harness::eval(&mut eng, &name, load_ms, &args[2], &args[3]),
        "probe" => harness::probe(&mut eng, &args[2], &args[3..]),
        m => panic!("unknown mode {m}"),
    }
}
