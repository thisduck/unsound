//! Times the speech models on this machine, to find where transcription
//! latency goes. Mirrors the app's `stt::transcribe` params.
//! Run: cargo run --release --example stt_bench -- <models_dir> <wav>...

use std::path::{Path, PathBuf};
use std::time::Instant;

use sherpa_rs::moonshine::{MoonshineConfig, MoonshineRecognizer};
use sherpa_rs::transducer::{TransducerConfig, TransducerRecognizer};
use unsound_lib::audiofile::decode_to_samples;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

struct Run<'a> {
    lang: &'a str,
    audio_ctx: Option<i32>,
}

fn whisper_full(ctx: &WhisperContext, samples: &[f32], run: &Run) -> (f64, f64, String) {
    let t = Instant::now();
    let mut ws = ctx.create_state().unwrap();
    let state_ms = ms(t);
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .min(8);
    params.set_n_threads(threads);
    params.set_language(Some(run.lang));
    params.set_translate(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(true);
    if let Some(a) = run.audio_ctx {
        params.set_audio_ctx(a);
    }
    let mut padded = vec![0.0f32; 8000];
    padded.extend_from_slice(samples);
    let t = Instant::now();
    ws.full(params, &padded).unwrap();
    let full_ms = ms(t);
    let mut text = String::new();
    for i in 0..ws.full_n_segments() {
        if let Some(s) = ws.get_segment(i) {
            text.push_str(&s.to_str_lossy().unwrap());
        }
    }
    (state_ms, full_ms, text.trim().to_string())
}

fn bench_whisper(file: &Path, clips: &[(String, Vec<f32>)]) {
    let name = file.file_name().unwrap().to_string_lossy();
    for flash in [false, true] {
        let mut cp = WhisperContextParameters::default();
        cp.flash_attn(flash);
        let t = Instant::now();
        let ctx = WhisperContext::new_with_params(file, cp).unwrap();
        let load_ms = ms(t);
        let t = Instant::now();
        let _ = whisper_full(&ctx, &vec![0.0; 16000], &Run { lang: "en", audio_ctx: None });
        let first_ms = ms(t);
        println!("\n## {name}  flash_attn={flash}  load={load_ms:.0}ms  first-run={first_ms:.0}ms");
        for (clip, samples) in clips {
            let secs = samples.len() as f64 / 16000.0;
            let mut runs = vec![("en", None), ("auto", None)];
            if secs < 12.0 {
                // Encoder context proportional to the clip (+ padding), 1500 = 30s.
                let ctx_frames = (((secs + 1.5) / 30.0) * 1500.0).ceil() as i32;
                runs.push(("en", Some(ctx_frames.max(128))));
            }
            for (lang, audio_ctx) in runs {
                let r = Run { lang, audio_ctx };
                let mut states = vec![];
                let mut fulls = vec![];
                let mut text = String::new();
                for _ in 0..3 {
                    let (s, f, tx) = whisper_full(&ctx, samples, &r);
                    states.push(s);
                    fulls.push(f);
                    text = tx;
                }
                let label = match audio_ctx {
                    Some(a) => format!("{lang} audio_ctx={a}"),
                    None => lang.to_string(),
                };
                println!(
                    "  {clip:>6} ({secs:>4.1}s) {label:<18} state={:>5.0}ms full={:>6.0}ms  | {}",
                    median(states),
                    median(fulls),
                    text.chars().take(70).collect::<String>()
                );
            }
        }
    }
}

fn bench_parakeet(dir: &Path, clips: &[(String, Vec<f32>)]) {
    for threads in [2, 4, 8] {
        let p = |f: &str| dir.join(f).to_string_lossy().to_string();
        let t = Instant::now();
        let mut rec = TransducerRecognizer::new(TransducerConfig {
            encoder: p("encoder.int8.onnx"),
            decoder: p("decoder.int8.onnx"),
            joiner: p("joiner.int8.onnx"),
            tokens: p("tokens.txt"),
            num_threads: threads,
            sample_rate: 16000,
            feature_dim: 80,
            decoding_method: "greedy_search".to_string(),
            model_type: "nemo_transducer".to_string(),
            ..Default::default()
        })
        .unwrap();
        println!("\n## parakeet-tdt-v3  threads={threads}  load={:.0}ms", ms(t));
        for (clip, samples) in clips {
            let mut times = vec![];
            let mut text = String::new();
            for _ in 0..3 {
                let t = Instant::now();
                text = rec.transcribe(16000, samples);
                times.push(ms(t));
            }
            println!(
                "  {clip:>6} ({:>4.1}s) full={:>6.0}ms  | {}",
                samples.len() as f64 / 16000.0,
                median(times),
                text.trim().chars().take(70).collect::<String>()
            );
        }
    }
}

fn bench_moonshine(dir: &Path, clips: &[(String, Vec<f32>)]) {
    let p = |f: &str| dir.join(f).to_string_lossy().to_string();
    let t = Instant::now();
    let mut rec = MoonshineRecognizer::new(MoonshineConfig {
        preprocessor: p("preprocess.onnx"),
        encoder: p("encode.int8.onnx"),
        uncached_decoder: p("uncached_decode.int8.onnx"),
        cached_decoder: p("cached_decode.int8.onnx"),
        tokens: p("tokens.txt"),
        ..Default::default()
    })
    .unwrap();
    println!("\n## moonshine-base-en  load={:.0}ms", ms(t));
    for (clip, samples) in clips {
        let mut times = vec![];
        let mut text = String::new();
        for _ in 0..3 {
            let t = Instant::now();
            text = rec.transcribe(16000, samples).text;
            times.push(ms(t));
        }
        println!(
            "  {clip:>6} ({:>4.1}s) full={:>6.0}ms  | {}",
            samples.len() as f64 / 16000.0,
            median(times),
            text.trim().chars().take(70).collect::<String>()
        );
    }
}

/// Accuracy check for the trimmed encoder context: full vs trimmed transcript
/// of each clip, flagged when they differ.
fn compare_audio_ctx(file: &Path, clips: &[(String, Vec<f32>)]) {
    let mut cp = WhisperContextParameters::default();
    cp.flash_attn(true);
    let ctx = WhisperContext::new_with_params(file, cp).unwrap();
    let _ = whisper_full(&ctx, &vec![0.0; 16000], &Run { lang: "en", audio_ctx: None });
    let (mut t_full, mut t_trim, mut same) = (0.0, 0.0, 0);
    for (clip, samples) in clips {
        let secs = samples.len() as f64 / 16000.0;
        let a = (((secs + 1.5) / 30.0) * 1500.0).ceil() as i32;
        let (_, f1, full) = whisper_full(&ctx, samples, &Run { lang: "en", audio_ctx: None });
        let (_, f2, trim) = whisper_full(&ctx, samples, &Run { lang: "en", audio_ctx: Some(a.max(128)) });
        t_full += f1;
        t_trim += f2;
        let eq = full == trim;
        same += eq as usize;
        println!("{clip} ({secs:.1}s) full={f1:.0}ms trim={f2:.0}ms {}", if eq { "SAME" } else { "DIFF" });
        if !eq {
            println!("  full: {full}\n  trim: {trim}");
        }
    }
    println!("\n{same}/{} identical; total full={t_full:.0}ms trim={t_trim:.0}ms", clips.len());
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let models = PathBuf::from(&args[0]);
    let only = std::env::var("ONLY").unwrap_or_default();
    if std::env::var("DUMP").is_ok() {
        // Full transcripts (flash attention on, as the app runs), for diffing builds.
        let mut cp = WhisperContextParameters::default();
        cp.flash_attn(true);
        let ctx = WhisperContext::new_with_params(models.join("ggml-large-v3-turbo.bin"), cp).unwrap();
        for p in &args[1..] {
            let samples = decode_to_samples(Path::new(p)).unwrap();
            let name = Path::new(p).file_stem().unwrap().to_string_lossy().to_string();
            for lang in ["en", "auto"] {
                let (_, ms, text) = whisper_full(&ctx, &samples, &Run { lang, audio_ctx: None });
                println!("{name}\t{lang}\t{ms:.0}\t{text}");
            }
        }
        return;
    }
    if std::env::var("FLASHCMP").is_ok() {
        let file = models.join("ggml-large-v3-turbo.bin");
        let mk = |flash: bool| {
            let mut cp = WhisperContextParameters::default();
            cp.flash_attn(flash);
            WhisperContext::new_with_params(&file, cp).unwrap()
        };
        let (off, on) = (mk(false), mk(true));
        let mut same = 0;
        for p in &args[1..] {
            let samples = decode_to_samples(Path::new(p)).unwrap();
            for lang in ["en", "auto"] {
                let r = Run { lang, audio_ctx: None };
                let (_, _, a) = whisper_full(&off, &samples, &r);
                let (_, _, b) = whisper_full(&on, &samples, &r);
                if a == b { same += 1 } else { println!("DIFF {p} {lang}\n  off: {a}\n  on:  {b}") }
            }
        }
        println!("{same}/{} identical", (args.len() - 1) * 2);
        return;
    }
    if std::env::var("CMP").is_ok() {
        let clips: Vec<(String, Vec<f32>)> = args[1..]
            .iter()
            .map(|p| {
                let path = Path::new(p);
                (path.file_stem().unwrap().to_string_lossy().to_string(), decode_to_samples(path).unwrap())
            })
            .collect();
        compare_audio_ctx(&models.join("ggml-large-v3-turbo.bin"), &clips);
        return;
    }
    let clips: Vec<(String, Vec<f32>)> = args[1..]
        .iter()
        .map(|p| {
            let path = Path::new(p);
            let name = path.file_stem().unwrap().to_string_lossy().to_string();
            (name, decode_to_samples(path).unwrap())
        })
        .collect();
    let want = |id: &str| only.is_empty() || only.split(',').any(|o| o == id);
    for (id, file) in [
        ("turbo", "ggml-large-v3-turbo.bin"),
        ("large", "ggml-large-v3.bin"),
        ("small", "ggml-small.bin"),
    ] {
        let path = models.join(file);
        if want(id) && path.exists() {
            bench_whisper(&path, &clips);
        }
    }
    if want("parakeet") && models.join("parakeet-tdt-v3").exists() {
        bench_parakeet(&models.join("parakeet-tdt-v3"), &clips);
    }
    if want("moonshine") && models.join("moonshine-base-en").exists() {
        bench_moonshine(&models.join("moonshine-base-en"), &clips);
    }
}
