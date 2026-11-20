//! End-to-end check of the meeting echo canceller on a synthetic laptop-
//! speaker call: two macOS `say` voices, the remote one leaking into the mic
//! through a delay, a small room and attenuation. Runs the app's Silero VAD
//! settings and Whisper on the "me" channel with and without cancellation,
//! then applies the leak guard, and prints the transcripts.
//!
//!   cargo run --release --example echo_bench [-- <whisper model file>]
//!
//! Uses models already downloaded by the app (never downloads).

use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};
use std::path::{Path, PathBuf};
use std::time::Instant;
use unsound_lib::echo::{guard, Canceller};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const SR: usize = 16_000;

fn models() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap()).join("Library/Application Support/com.unsound.app/models")
}

fn say(voice: &str, text: &str) -> Vec<f32> {
    let dir = std::env::temp_dir().join("unsound-echo-bench");
    std::fs::create_dir_all(&dir).unwrap();
    let key: u64 = text
        .bytes()
        .chain(voice.bytes())
        .fold(1469598103934665603u64, |h, b| (h ^ b as u64).wrapping_mul(1099511628211));
    let out = dir.join(format!("{voice}-{key:016x}.wav"));
    if !out.exists() {
        let ok = std::process::Command::new("say")
            .args(["-v", voice, "--data-format=LEI16@16000", "-o"])
            .arg(&out)
            .arg(text)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(ok, "`say` failed");
    }
    unsound_lib::audiofile::decode_to_samples(&out).unwrap()
}

fn place(track: &mut [f32], at_s: f32, clip: &[f32]) -> (f32, f32) {
    let a = (at_s * SR as f32) as usize;
    for (i, s) in clip.iter().enumerate() {
        if a + i < track.len() {
            track[a + i] += s;
        }
    }
    (at_s, at_s + clip.len() as f32 / SR as f32)
}

fn room_echo(far: &[f32], delay_ms: f32, gain_db: f32) -> Vec<f32> {
    // Direct path, two early reflections, 150 ms exponential tail.
    let mut seed = 7u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let len = SR * 150 / 1000;
    let mut h: Vec<f32> = (0..len).map(|n| 0.35 * rnd() * (-(n as f32) / (0.03 * SR as f32)).exp()).collect();
    h[0] = 1.0;
    h[SR * 3 / 1000] += 0.5;
    h[SR * 7 / 1000] -= 0.3;
    let norm: f32 = h.iter().map(|v| v * v).sum::<f32>().sqrt();
    let gain = 10f32.powf(gain_db / 20.0) / norm;
    let d = (delay_ms * SR as f32 / 1000.0) as usize;
    let mut out = vec![0.0; far.len()];
    for (i, &x) in far.iter().enumerate() {
        if x == 0.0 {
            continue;
        }
        for (k, &hk) in h.iter().enumerate() {
            if let Some(o) = out.get_mut(i + d + k) {
                *o += x * hk * gain;
            }
        }
    }
    out
}

fn segments(vad_model: &Path, audio: &[f32]) -> Vec<(usize, usize)> {
    // The app's meeting VAD settings (meeting.rs make_vad).
    let mut vad = SileroVad::new(
        SileroVadConfig {
            model: vad_model.to_string_lossy().to_string(),
            min_silence_duration: 0.5,
            min_speech_duration: 0.25,
            max_speech_duration: 20.0,
            threshold: 0.5,
            sample_rate: SR as u32,
            window_size: 512,
            provider: None,
            num_threads: Some(1),
            debug: false,
        },
        30.0,
    )
    .unwrap();
    let mut out = Vec::new();
    for chunk in audio.chunks(SR) {
        vad.accept_waveform(chunk.to_vec());
        while !vad.is_empty() {
            let s = vad.front();
            out.push((s.start as usize, s.start as usize + s.samples.len()));
            vad.pop();
        }
    }
    vad.flush();
    while !vad.is_empty() {
        let s = vad.front();
        out.push((s.start as usize, s.start as usize + s.samples.len()));
        vad.pop();
    }
    out
}

/// Whisper's segments for a VAD piece, with their times (the app keeps
/// these as separate transcript lines).
fn transcribe(ctx: &WhisperContext, samples: &[f32]) -> Vec<(usize, usize, String)> {
    let mut st = ctx.create_state().unwrap();
    let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    p.set_language(Some("en"));
    p.set_print_special(false);
    p.set_print_progress(false);
    p.set_print_realtime(false);
    p.set_print_timestamps(false);
    p.set_suppress_blank(true);
    p.set_no_context(true);
    let pad = 4800;
    let mut padded = vec![0.0f32; pad];
    padded.extend_from_slice(samples);
    st.full(p, &padded).unwrap();
    let mut out = Vec::new();
    for i in 0..st.full_n_segments() {
        if let Some(seg) = st.get_segment(i) {
            let text = seg.to_str_lossy().unwrap().trim().to_string();
            let a = (seg.start_timestamp() as usize * 160).saturating_sub(pad).min(samples.len());
            let b = (seg.end_timestamp() as usize * 160).saturating_sub(pad).min(samples.len());
            if !text.is_empty() && !text.starts_with('[') && b > a {
                out.push((a, b, text));
            }
        }
    }
    out
}

struct Line {
    start: usize,
    end: usize,
    text: String,
}

fn lines(ctx: &WhisperContext, vad: &Path, audio: &[f32]) -> Vec<Line> {
    segments(vad, audio)
        .into_iter()
        .flat_map(|(a, b)| {
            transcribe(ctx, &audio[a..b.min(audio.len())])
                .into_iter()
                .map(move |(s, e, text)| Line { start: a + s, end: a + e, text })
        })
        .collect()
}

fn show(title: &str, lines: &[Line]) {
    println!("\n{title}");
    if lines.is_empty() {
        println!("  (nothing)");
    }
    for l in lines {
        println!("  [{:>5.1}-{:>5.1}s] {}", l.start as f32 / SR as f32, l.end as f32 / SR as f32, l.text);
    }
}

fn cancel(mic: &[f32], far: &[f32]) -> (Vec<f32>, f64) {
    let mut aec = Canceller::new();
    let mut out = Vec::new();
    let t = Instant::now();
    let mut fed = 0;
    for start in (0..mic.len()).step_by(SR) {
        let end = (start + SR).min(mic.len());
        // System audio arrives a little behind the mic, as from SCK.
        let upto = end.saturating_sub(480).max(fed);
        aec.push_reference(fed as u64, &far[fed..upto]);
        fed = upto;
        if end == mic.len() {
            aec.push_reference(fed as u64, &far[fed..]);
            out.extend(aec.finish(&mic[start..end]));
        } else {
            out.extend(aec.process(&mic[start..end]));
        }
    }
    let rtf = t.elapsed().as_secs_f64() / (mic.len() as f64 / SR as f64);
    (out, rtf)
}

fn guard_lines(me: &[Line], them: &[Line], cleaned: &[f32], far: &[f32]) -> Vec<Line> {
    let lead = guard::LEAD_MS as usize * SR / 1000;
    let trail = guard::TRAIL_MS as usize * SR / 1000;
    let window = guard::TEXT_WINDOW_MS as usize * SR / 1000;
    me.iter()
        .filter_map(|l| {
            let theirs: Vec<&str> = them
                .iter()
                .filter(|t| t.end + window >= l.start && t.start <= l.end + window)
                .map(|t| t.text.as_str())
                .collect();
            let mut sys = vec![0.0; lead.saturating_sub(l.start)];
            sys.extend_from_slice(&far[l.start.saturating_sub(lead)..(l.end + trail).min(far.len())]);
            let m = guard::echo_match(&cleaned[l.start..l.end.min(cleaned.len())], &sys);
            let r = guard::words_repeated(&l.text, &theirs.join(" "));
            let leak = guard::is_leak(m, r, &l.text, ((l.end - l.start) / 16) as i64);
            println!(
                "  guard [{:>5.1}s] match {m:.2} repeated {r:.2} -> {}",
                l.start as f32 / SR as f32,
                if leak { "drop (leak)" } else { "keep" }
            );
            (!leak).then(|| Line { start: l.start, end: l.end, text: l.text.clone() })
        })
        .collect()
}

fn main() {
    let model = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| models().join("ggml-small.bin"));
    let vad = models().join("silero_vad.onnx");
    if !model.exists() || !vad.exists() {
        println!("needs {} and {} (not downloading)", model.display(), vad.display());
        return;
    }
    let total = 30.0;
    let n = (total * SR as f32) as usize;
    let mut far = vec![0.0; n];
    let mut near = vec![0.0; n];
    let remote = "Samantha";
    let user = "Daniel";
    let script: [(&str, f32, &str); 7] = [
        (remote, 0.5, "Okay so let's get started with the quarterly planning review."),
        (remote, 5.0, "I pulled the numbers from last week and they look better than expected."),
        (user, 10.5, "That matches what support has been telling me."),
        (remote, 14.0, "Churn is down about two points, mostly from the onboarding changes."),
        // The user talks over the remote side.
        (user, 15.5, "Sorry, can I jump in on that?"),
        (remote, 20.5, "Let's make sure legal signs off on the new terms before the launch."),
        (user, 26.0, "Yeah."),
    ];
    for (voice, at, text) in script {
        let clip = say(voice, text);
        let (a, b) = place(if voice == remote { &mut far } else { &mut near }, at, &clip);
        println!("script [{a:>5.1}-{b:>5.1}s] {} {text}", if voice == remote { "them" } else { "me  " });
    }
    // Remote saying "yeah" at the same moment as the user's "yeah".
    place(&mut far, 26.1, &say(remote, "Yeah."));

    let mut seed = 42u64;
    let mut noise = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * 3e-4
    };
    let echo = room_echo(&far, 80.0, -6.0);
    let speakers: Vec<f32> = (0..n).map(|i| near[i] + echo[i] + noise()).collect();
    let headphones: Vec<f32> = (0..n).map(|i| near[i] + noise()).collect();

    let ctx = WhisperContext::new_with_params(&model, WhisperContextParameters::default()).unwrap();
    println!("\nwhisper model: {}", model.file_name().unwrap().to_string_lossy());
    let them = lines(&ctx, &vad, &far);
    show("them (system audio)", &them);

    for (name, mic) in [("laptop speakers (80 ms, room, -6 dB)", &speakers), ("headphones (no leak)", &headphones)] {
        println!("\n===== {name}");
        let (cleaned, rtf) = cancel(mic, &far);
        let identical = cleaned == *mic;
        println!("canceller: {:.4}x real time; output identical to input: {identical}", rtf);
        show("me — mic as recorded today (no cancellation)", &lines(&ctx, &vad, mic));
        let me = lines(&ctx, &vad, &cleaned);
        show("me — after the canceller", &me);
        let kept = guard_lines(&me, &them, &cleaned, &far);
        show("me — after the canceller and the leak guard", &kept);
    }
}
