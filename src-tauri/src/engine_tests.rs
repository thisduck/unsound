//! Real speech models through the app's dictation and meeting code paths:
//! language routing (`asr::take_engine`), dictation text (`transcribe_text`),
//! meeting pieces (`transcribe_pieces`, then the dictionary and the piece's
//! timeline as `meeting::process_finals` applies them). Ignored: needs local
//! models and clips.
//!
//!   UNSOUND_WHISPER=…/ggml-large-v3-turbo.bin UNSOUND_PARAKEET_V2=…-v2-Q8_0.gguf \
//!   UNSOUND_PARAKEET_V3=…-v3-Q8_0.gguf UNSOUND_ASR_DATA=<spike data dir> \
//!   cargo test --release --lib engine_tests -- --ignored --nocapture --test-threads=1

use crate::asr::{self, Caches, Selection};
use crate::models::{builtin_registry, ModelInfo};
use crate::{parakeet::ParakeetState, stt::SttState};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

fn env(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
}

fn clip(rel: &str) -> Vec<f32> {
    crate::audiofile::decode_to_samples(&env("UNSOUND_ASR_DATA").join(rel)).unwrap()
}

fn info(id: &str) -> ModelInfo {
    builtin_registry().into_iter().find(|m| m.id == id).unwrap()
}

fn selection(id: &str, with_whisper: bool) -> Selection {
    Selection {
        info: info(id),
        fallback: with_whisper.then(|| ("whisper-large-v3-turbo".to_string(), env("UNSOUND_WHISPER"))),
    }
}

fn engine_name(e: &asr::Engine) -> &'static str {
    match e {
        asr::Engine::Whisper(_) => "whisper",
        asr::Engine::Parakeet(_) => "parakeet",
        _ => "other",
    }
}

struct Run<'a> {
    caches: Caches<'a>,
}

impl Run<'_> {
    /// Dictation: route, then one plain-text transcription of the take.
    fn dictate(&self, sel: &Selection, model: &PathBuf, language: Option<&str>, samples: &[f32]) -> Result<(String, String), String> {
        let (mut engine, lang) = asr::take_engine(sel, model.clone(), self.caches, language, samples)?;
        let t = Instant::now();
        let text = asr::transcribe_text(self.caches, &mut engine, samples, lang.as_deref(), None)?;
        Ok((
            format!("{} lang={:?} {} ms", engine_name(&engine), lang, t.elapsed().as_millis()),
            text,
        ))
    }

    /// Meeting: one VAD piece at `base_ms`, as `process_finals` stores it.
    fn meeting(
        &self, sel: &Selection, model: &PathBuf, language: Option<&str>, samples: &[f32], base_ms: i64,
        dictionary: &[(String, String)],
    ) -> Result<Vec<(i64, i64, String)>, String> {
        let (mut engine, lang) = asr::take_engine(sel, model.clone(), self.caches, language, samples)?;
        let parts = asr::transcribe_pieces(
            self.caches, &mut engine, &[samples], lang.as_deref(), None, Arc::new(AtomicBool::new(false)),
        )?;
        Ok(parts
            .into_iter()
            .map(|p| {
                (base_ms + p.start_ms, base_ms + p.end_ms, crate::meeting::apply_dictionary(&p.text, dictionary))
            })
            .collect())
    }
}

#[test]
#[ignore]
fn real_engines_through_app_paths() {
    let stt = SttState::default();
    let parakeet = ParakeetState::default();
    let run = Run { caches: Caches { stt: &stt, parakeet: &parakeet } };
    let v2 = env("UNSOUND_PARAKEET_V2");
    let v3 = env("UNSOUND_PARAKEET_V3");
    let t = Instant::now();
    parakeet.warmup(&v3).unwrap();
    println!("parakeet v3 load+warmup {} ms", t.elapsed().as_millis());

    let ami = clip("ami_ihm/001.wav");
    let chunk = clip("ami_sdm_chunks/000.wav");
    let german = clip("fleurs_de_de/000.wav");
    let hindi = clip("fleurs_hi_in/000.wav");
    let silence = vec![0.0f32; 3 * 16_000];
    let music = clip("nonspeech/music_voices_break_10s.wav");
    println!("chunk is {:.1} s", chunk.len() as f32 / 16_000.0);

    let sel3 = selection("parakeet-tdt-0.6b-v3", true);
    let sel2 = selection("parakeet-tdt-0.6b-v2", true);
    let sel3_alone = selection("parakeet-tdt-0.6b-v3", false);
    let sel2_alone = selection("parakeet-tdt-0.6b-v2", false);
    let show = |what: &str, r: Result<(String, String), String>| match r {
        Ok((how, text)) => println!("[dictation] {what}: {how}\n    {text:?}"),
        Err(e) => println!("[dictation] {what}: REFUSED/ERR {e}"),
    };

    println!("\n== dictation path ==");
    show("v3, en, AMI headset 6.6 s", run.dictate(&sel3, &v3, Some("en"), &ami));
    show("v2, en, AMI headset 6.6 s", run.dictate(&sel2, &v2, Some("en"), &ami));
    show("v3, en, AMI far-field chunk", run.dictate(&sel3, &v3, Some("en"), &chunk));
    show("v3, en, 3 s silence", run.dictate(&sel3, &v3, Some("en"), &silence));
    show("v3, en, music with voices", run.dictate(&sel3, &v3, Some("en"), &music));
    show("v3, auto (Whisper detects), German", run.dictate(&sel3, &v3, None, &german));
    show("v2, auto (Whisper detects), German", run.dictate(&sel2, &v2, None, &german));
    show("v3, auto (Whisper detects), Hindi", run.dictate(&sel3, &v3, None, &hindi));
    show("v3, hi chosen, Whisper installed", run.dictate(&sel3, &v3, Some("hi"), &hindi));
    show("v3, hi chosen, no Whisper", run.dictate(&sel3_alone, &v3, Some("hi"), &hindi));
    show("v3, auto, no Whisper, German", run.dictate(&sel3_alone, &v3, None, &german));
    show("v2, auto, no Whisper, English", run.dictate(&sel2_alone, &v2, None, &ami));

    println!("\n== meeting path ==");
    let dict = vec![("sense".to_string(), "SENSE".to_string())];
    let show_m = |what: &str, r: Result<Vec<(i64, i64, String)>, String>, len_ms: i64, base: i64| match r {
        Ok(segs) => {
            println!("[meeting] {what} (piece {base}..{} ms):", base + len_ms);
            for (s, e, t) in &segs {
                assert!(base <= *s && s <= e && *e <= base + len_ms, "segment outside its piece");
                println!("    {s}..{e} {t:?}");
            }
            if segs.is_empty() {
                println!("    (no segments)");
            }
        }
        Err(e) => println!("[meeting] {what}: REFUSED/ERR {e}"),
    };
    let ms = |s: &[f32]| s.len() as i64 * 1000 / 16_000;
    show_m("v3, en, AMI headset, dictionary sense→SENSE", run.meeting(&sel3, &v3, Some("en"), &ami, 61_000, &dict), ms(&ami), 61_000);
    show_m("v3, en, far-field chunk", run.meeting(&sel3, &v3, Some("en"), &chunk, 120_000, &[]), ms(&chunk), 120_000);
    show_m("v3, en, silence", run.meeting(&sel3, &v3, Some("en"), &silence, 0, &[]), ms(&silence), 0);
    show_m("v3, en, music with voices", run.meeting(&sel3, &v3, Some("en"), &music, 0, &[]), ms(&music), 0);
    show_m("v3, de", run.meeting(&sel3, &v3, Some("de"), &german, 5_000, &[]), ms(&german), 5_000);
    show_m("v3, hi locked, Whisper installed", run.meeting(&sel3, &v3, Some("hi"), &hindi, 0, &[]), ms(&hindi), 0);
    show_m("v3, hi locked, no Whisper", run.meeting(&sel3_alone, &v3, Some("hi"), &hindi, 0, &[]), ms(&hindi), 0);
    show_m("whisper turbo, en, AMI headset (reference)", run.meeting(&Selection { info: info("whisper-large-v3-turbo"), fallback: None }, &env("UNSOUND_WHISPER"), Some("en"), &ami, 61_000, &dict), ms(&ami), 61_000);

    println!("\n== cancellation ==");
    let stop = Arc::new(AtomicBool::new(true));
    println!("preset stop: {:?}", parakeet.transcribe(&v3, &chunk, Some(&stop)).map(|o| o.text));
    let stop = Arc::new(AtomicBool::new(false));
    let s2 = stop.clone();
    let long: Vec<f32> = chunk.iter().chain(chunk.iter()).chain(chunk.iter()).copied().collect();
    let h = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(15));
        s2.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    let t = Instant::now();
    let r = parakeet.transcribe(&v3, &long, Some(&stop));
    h.join().unwrap();
    println!("stop after 15 ms on {:.0} s: {:?} after {} ms", long.len() as f32 / 16_000.0, r.map(|o| o.text.len()), t.elapsed().as_millis());
    let t = Instant::now();
    let full = parakeet.transcribe(&v3, &long, None).unwrap();
    println!("same audio uncancelled: {} chars in {} ms", full.text.len(), t.elapsed().as_millis());

    println!("\n== long file (dictation path, every AMI headset clip back to back) ==");
    let long: Vec<f32> = (0..80).flat_map(|i| clip(&format!("ami_ihm/{i:03}.wav"))).collect();
    let t = Instant::now();
    match run.dictate(&sel3, &v3, Some("en"), &long) {
        Ok((how, text)) => println!(
            "{:.0} s of audio: {how}, {} words, total {} ms",
            long.len() as f32 / 16_000.0,
            text.split_whitespace().count(),
            t.elapsed().as_millis()
        ),
        Err(e) => println!("long file failed: {e}"),
    }

    parakeet.clear();
    stt.clear();
}
