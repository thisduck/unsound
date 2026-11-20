//! Synthetic speaker-leak harness. Two distinct macOS `say` voices stand in
//! for the remote side (far end, what system capture hears) and the user
//! (near end). The mic is near + far through a delay (optionally drifting),
//! a small-room impulse response and attenuation, plus noise. The canceller
//! is fed the way the capture loop feeds it: 1 s polls, with the reference
//! arriving a little behind the mic.
//!
//! Run with `--nocapture` to see the measured numbers.

use super::guard::{echo_match, is_leak, words_repeated};
use super::*;
use std::path::PathBuf;
use std::sync::OnceLock;

const SR: usize = 16_000;
const FAR_VOICE: &str = "Samantha";
const NEAR_VOICE: &str = "Daniel";
const FAR_LINES: &[&str] = &[
    "Okay so let's get started with the quarterly planning review.",
    "I pulled the numbers from last week and they look better than expected.",
    "Churn is down about two points, mostly from the onboarding changes.",
    "Two deals moved to next quarter, but the rest of the pipeline is on track.",
    "Let's make sure legal signs off on the new terms before the launch.",
    "One more thing, the mobile release needs another round of testing.",
    "Can someone send the notes afterwards so everyone has the action items?",
    "We should revisit the pricing page and the onboarding emails next week.",
];
const NEAR_LINES: &[&str] = &[
    "That matches what support has been telling me for a while now.",
    "I can own that, I'll set up time with the legal team on Thursday.",
    "Give it until Friday and then we decide whether to ship.",
    "I'll write them up and share them this afternoon.",
];

fn cache_dir() -> PathBuf {
    let d = std::env::temp_dir().join("unsound-echo-test");
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// One line of speech from a macOS voice, 16 kHz mono (cached).
pub(super) fn say(voice: &str, text: &str) -> Vec<f32> {
    let key: u64 = text
        .bytes()
        .chain(voice.bytes())
        .fold(1469598103934665603u64, |h, b| (h ^ b as u64).wrapping_mul(1099511628211));
    let path = cache_dir().join(format!("{voice}-{key:016x}.wav"));
    if !path.exists() {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = path.with_extension(format!("{}-{n}.tmp.wav", std::process::id()));
        let ok = std::process::Command::new("say")
            .args(["-v", voice, "--data-format=LEI16@16000", "-o"])
            .arg(&tmp)
            .arg(text)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(ok, "macOS `say` failed for voice {voice}");
        std::fs::rename(&tmp, &path).unwrap();
    }
    crate::audiofile::decode_to_samples(&path).expect("decode say output")
}

/// Speech by `voice` filling `[from, to)` seconds of a `total`-second track.
fn fill(voice: &str, lines: &[&str], from: f32, to: f32, total: f32) -> Vec<f32> {
    let mut out = vec![0.0; (total * SR as f32) as usize];
    let end = (to * SR as f32) as usize;
    let mut pos = (from * SR as f32) as usize;
    let mut i = 0;
    while pos < end {
        let clip = say(voice, lines[i % lines.len()]);
        for (k, s) in clip.iter().enumerate() {
            if pos + k >= end {
                break;
            }
            out[pos + k] = *s;
        }
        pos += clip.len() + SR / 4;
        i += 1;
    }
    out
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Small room: direct path, early reflections, 150 ms exponential tail.
fn room() -> Vec<f32> {
    let mut rng = Lcg(7);
    let len = SR * 150 / 1000;
    let mut h: Vec<f32> = (0..len)
        .map(|n| 0.35 * rng.next() * (-(n as f32) / (0.03 * SR as f32)).exp())
        .collect();
    h[0] = 1.0;
    h[SR * 3 / 1000] += 0.5;
    h[SR * 7 / 1000] -= 0.3;
    let e: f32 = h.iter().map(|v| v * v).sum::<f32>().sqrt();
    h.iter_mut().for_each(|v| *v /= e);
    h
}

fn convolve(x: &[f32], h: &[f32]) -> Vec<f32> {
    use realfft::RealFftPlanner;
    let n = (x.len() + h.len()).next_power_of_two();
    let mut planner = RealFftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(n);
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    a[..x.len()].copy_from_slice(x);
    b[..h.len()].copy_from_slice(h);
    let mut sa = fwd.make_output_vec();
    let mut sb = fwd.make_output_vec();
    fwd.process(&mut a, &mut sa).unwrap();
    fwd.process(&mut b, &mut sb).unwrap();
    for (p, q) in sa.iter_mut().zip(&sb) {
        *p = *p * q / n as f32;
    }
    inv.process(&mut sa, &mut a).unwrap();
    a.truncate(x.len());
    a
}

/// `x` delayed by `d(t)` samples: band-limited (32-tap windowed sinc), so a
/// drifting delay behaves like a clock offset, not a wobbling filter.
fn delay_by(x: &[f32], d: impl Fn(usize) -> f32) -> Vec<f32> {
    const HALF: i64 = 16;
    let pi = std::f64::consts::PI;
    (0..x.len())
        .map(|t| {
            let p = t as f64 - d(t) as f64;
            let i = p.floor() as i64;
            let f = p - i as f64;
            if f < 1e-6 {
                return if i >= 0 { x.get(i as usize).copied().unwrap_or(0.0) } else { 0.0 };
            }
            let mut acc = 0.0f64;
            for m in -HALF + 1..=HALF {
                let k = i + m;
                if k < 0 || k as usize >= x.len() {
                    continue;
                }
                let u = m as f64 - f;
                let w = 0.42 + 0.5 * (pi * u / HALF as f64).cos() + 0.08 * (2.0 * pi * u / HALF as f64).cos();
                acc += x[k as usize] as f64 * (pi * u).sin() / (pi * u) * w;
            }
            acc as f32
        })
        .collect()
}

fn db(x: f64) -> f64 {
    10.0 * x.max(1e-30).log10()
}

fn energy(x: &[f32], from: f32, to: f32) -> f64 {
    let (a, b) = ((from * SR as f32) as usize, (to * SR as f32) as usize);
    x[a..b.min(x.len())].iter().map(|v| (*v as f64) * (*v as f64)).sum()
}

struct Scene {
    near: Vec<f32>,
    far: Vec<f32>,
    echo: Vec<f32>,
    noise: Vec<f32>,
    mic: Vec<f32>,
}

/// Far only 0-10 s, near only 10-15 s, double talk 15-22 s, far only 22-30 s.
const T: f32 = 30.0;

fn scene(delay_ms: f32, drift_ms: f32, gain_db: f32) -> Scene {
    let mut far = fill(FAR_VOICE, FAR_LINES, 0.3, 10.0, T);
    let far2 = fill(FAR_VOICE, &FAR_LINES[3..], 15.0, T, T);
    for (a, b) in far.iter_mut().zip(&far2) {
        *a += b;
    }
    let near = fill(NEAR_VOICE, NEAR_LINES, 10.0, 22.0, T);
    let gain = 10f32.powf(gain_db / 20.0);
    let total = far.len() as f32;
    let reverberant = convolve(&far, &room());
    let echo: Vec<f32> = delay_by(&reverberant, |t| {
        (delay_ms + drift_ms * t as f32 / total) * SR as f32 / 1000.0
    })
    .into_iter()
    .map(|v| v * gain)
    .collect();
    let mut rng = Lcg(42);
    let noise: Vec<f32> = (0..far.len()).map(|_| rng.next() * 3e-4).collect();
    let mic = (0..far.len()).map(|i| near[i] + echo[i] + noise[i]).collect();
    Scene {
        near,
        far,
        echo,
        noise,
        mic,
    }
}

/// Feed like the capture loop: 1 s polls, reference `lag_ms` behind.
fn run(aec: &mut Canceller, mic: &[f32], reference: Option<&[f32]>, ref_late_ms: usize) -> Vec<f32> {
    let mut out = Vec::new();
    let mut fed_ref = 0;
    let mut pos = 0;
    while pos < mic.len() {
        let end = (pos + SR).min(mic.len());
        if let Some(r) = reference {
            let upto = end.saturating_sub(ref_late_ms * SR / 1000).max(fed_ref);
            aec.push_reference(fed_ref as u64, &r[fed_ref..upto]);
            fed_ref = upto;
        }
        if end == mic.len() {
            if let Some(r) = reference {
                aec.push_reference(fed_ref as u64, &r[fed_ref..]);
            }
            out.extend(aec.finish(&mic[pos..end]));
        } else {
            out.extend(aec.process(&mic[pos..end]));
        }
        pos = end;
    }
    out
}

struct Metrics {
    erle_start: f64,
    erle_first: f64,
    erle_late: f64,
    near_sdr: f64,
    near_loss: f64,
    dt_sdr: f64,
    dt_loss: f64,
}

fn measure(s: &Scene, out: &[f32]) -> Metrics {
    assert_eq!(out.len(), s.mic.len(), "the timeline must keep every sample");
    let target: Vec<f32> = (0..out.len()).map(|i| s.near[i] + s.noise[i]).collect();
    let err: Vec<f32> = (0..out.len()).map(|i| out[i] - target[i]).collect();
    let loss = |from: f32, to: f32| {
        let (a, b) = ((from * SR as f32) as usize, (to * SR as f32) as usize);
        let num: f64 = (a..b).map(|i| out[i] as f64 * s.near[i] as f64).sum();
        let den: f64 = (a..b).map(|i| s.near[i] as f64 * s.near[i] as f64).sum();
        -20.0 * (num / den).abs().max(1e-9).log10()
    };
    Metrics {
        erle_start: db(energy(&s.mic, 1.0, 5.0) / energy(out, 1.0, 5.0)),
        erle_first: db(energy(&s.mic, 5.0, 10.0) / energy(out, 5.0, 10.0)),
        erle_late: db(energy(&s.mic, 23.0, 30.0) / energy(out, 23.0, 30.0)),
        near_sdr: db(energy(&s.near, 10.5, 15.0) / energy(&err, 10.5, 15.0)),
        near_loss: loss(10.5, 15.0),
        dt_sdr: db(energy(&s.near, 15.0, 22.0) / energy(&err, 15.0, 22.0)),
        dt_loss: loss(15.0, 22.0),
    }
}

fn report(name: &str, s: &Scene, m: &Metrics, aec: &Canceller) {
    let echo_vs_near = db(energy(&s.echo, 15.0, 22.0) / energy(&s.near, 15.0, 22.0));
    println!(
        "[echo] {name}: ERLE far-only 1-5s {:.1} dB, 5-10s {:.1} dB, 23-30s {:.1} dB | near-only SDR {:.1} dB, \
         loss {:.2} dB | double-talk SDR {:.1} dB, loss {:.2} dB (echo/near {:.1} dB) | lag {:?}",
        m.erle_start, m.erle_first, m.erle_late, m.near_sdr, m.near_loss, m.dt_sdr, m.dt_loss, echo_vs_near, aec.lag()
    );
}

fn check(name: &str, delay_ms: f32, drift_ms: f32, gain_db: f32) -> Metrics {
    let s = scene(delay_ms, drift_ms, gain_db);
    let mut aec = Canceller::new();
    let out = run(&mut aec, &s.mic, Some(&s.far), 30);
    let m = measure(&s, &out);
    report(name, &s, &m, &aec);
    // The user's lines, after cancellation, must not look like leak — even
    // if their words happened to repeat the remote side's.
    for (what, a, b) in [("near-only", 10.5, 15.0), ("double-talk", 15.0, 22.0)] {
        let (a, b) = ((a * SR as f32) as usize, (b * SR as f32) as usize);
        let c = echo_match(&out[a..b], &sys_for(&s.far, a, b));
        println!("[echo] {name}: guard on cleaned {what} {c:.2}");
        assert!(!is_leak(c, 1.0, "a line of several words", ((b - a) / 16) as i64), "{what} {c:.2}");
    }
    m
}

fn assert_good(m: &Metrics) {
    assert!(m.erle_first >= 20.0, "far-only ERLE (first) {:.1} dB", m.erle_first);
    assert!(m.erle_late >= 25.0, "far-only ERLE (late) {:.1} dB", m.erle_late);
    assert!(m.near_sdr >= 20.0, "near-only SDR {:.1} dB", m.near_sdr);
    assert!(m.near_loss.abs() <= 0.5, "near-only loss {:.2} dB", m.near_loss);
    assert!(m.dt_loss.abs() <= 1.0, "double-talk near loss {:.2} dB", m.dt_loss);
    assert!(m.dt_sdr >= 12.0, "double-talk SDR {:.1} dB", m.dt_sdr);
}

#[test]
fn cancels_20ms() {
    assert_good(&check("20 ms", 20.0, 0.0, -6.0));
}

#[test]
fn cancels_80ms() {
    assert_good(&check("80 ms", 80.0, 0.0, -6.0));
}

#[test]
fn cancels_150ms() {
    assert_good(&check("150 ms", 150.0, 0.0, -6.0));
}

#[test]
fn cancels_250ms() {
    assert_good(&check("250 ms", 250.0, 0.0, -6.0));
}

#[test]
fn cancels_quiet_leak() {
    // Speakers turned down: echo 20 dB under the reference.
    let m = check("80 ms, -20 dB", 80.0, 0.0, -20.0);
    assert!(m.erle_late >= 15.0, "far-only ERLE {:.1} dB", m.erle_late);
    assert!(m.near_loss.abs() <= 0.5 && m.dt_loss.abs() <= 1.0);
}

#[test]
fn tracks_drift() {
    // Mic and output on different clocks: 100 -> 103 ms over 30 s (100 ppm,
    // the high end for separate USB/Bluetooth/HDMI devices). Until the rate
    // is measured (~7 s in) the moving path keeps the filter from settling,
    // so only the later stretches are held to the usual bar.
    let m = check("100->103 ms drift (100 ppm)", 100.0, 3.0, -6.0);
    assert!(m.erle_late >= 25.0, "far-only ERLE (late) {:.1} dB", m.erle_late);
    assert!(m.near_loss.abs() <= 0.5 && m.dt_loss.abs() <= 1.0);
    let m = check("100->110 ms drift (333 ppm)", 100.0, 10.0, -6.0);
    assert!(m.erle_late >= 12.0, "far-only ERLE (late) {:.1} dB", m.erle_late);
    assert!(m.near_loss.abs() <= 0.5 && m.dt_loss.abs() <= 1.0);
}

#[test]
fn reference_slightly_behind_its_echo() {
    // System audio placed 30 ms late on the mic clock (echo "leads").
    let s = scene(0.0, 0.0, -6.0);
    let late: Vec<f32> = delay_by(&s.far, |_| 30.0 * SR as f32 / 1000.0);
    let mut aec = Canceller::new();
    let out = run(&mut aec, &s.mic, Some(&late), 30);
    let m = measure(&s, &out);
    report("reference 30 ms late", &s, &m, &aec);
    // Measured ~25 dB in both windows (below the ~35 dB of a leading
    // reference late on); held to 20 dB.
    assert!(m.erle_first >= 20.0 && m.erle_late >= 20.0, "{:.1} / {:.1} dB", m.erle_first, m.erle_late);
    assert!(m.near_loss.abs() <= 0.5 && m.dt_loss.abs() <= 1.0 && m.dt_sdr >= 12.0);
}

#[test]
fn silent_reference_is_bit_identical() {
    let s = scene(80.0, 0.0, -6.0);
    let mic: Vec<f32> = (0..s.mic.len()).map(|i| s.near[i] + s.noise[i]).collect();
    let silence = vec![0.0; mic.len()];
    let mut aec = Canceller::new();
    assert_eq!(run(&mut aec, &mic, Some(&silence), 30), mic);
    let mut aec = Canceller::new();
    assert_eq!(run(&mut aec, &mic, None, 0), mic, "no system audio at all");
}

#[test]
fn headphones_are_bit_identical() {
    // Remote speech in the reference, none of it in the mic.
    let s = scene(80.0, 0.0, -6.0);
    let mic: Vec<f32> = (0..s.mic.len()).map(|i| s.near[i] + s.noise[i]).collect();
    let mut aec = Canceller::new();
    let out = run(&mut aec, &mic, Some(&s.far), 30);
    assert_eq!(aec.lag(), None, "locked onto uncorrelated audio");
    assert_eq!(out, mic);
}

#[test]
fn passes_through_while_the_reference_is_gone_then_reconverges() {
    // Leak throughout; system capture drops at 12 s and comes back at 16 s
    // with its timeline 60 ms later than before.
    let total = 40.0;
    let far = fill(FAR_VOICE, FAR_LINES, 0.3, total, total);
    let echo: Vec<f32> = delay_by(&convolve(&far, &room()), |_| 0.1 * SR as f32)
        .into_iter()
        .map(|v| v * 0.5)
        .collect();
    let mut rng = Lcg(3);
    let mic: Vec<f32> = echo.iter().map(|e| e + rng.next() * 3e-4).collect();
    let shifted = delay_by(&far, |_| 0.06 * SR as f32);
    let mut aec = Canceller::new();
    let mut out = Vec::new();
    for sec in 0..total as usize {
        let (a, b) = (sec * SR, (sec + 1) * SR);
        if sec == 12 {
            aec.reference_restarted();
        }
        if sec < 12 {
            aec.push_reference(a as u64, &far[a..b - 480]);
            if a >= 480 {
                aec.push_reference((a - 480) as u64, &far[a - 480..a]);
            }
        } else if sec >= 16 {
            aec.push_reference(a as u64, &shifted[a..b]);
        }
        let chunk = aec.process(&mic[a..b]);
        if (13..16).contains(&sec) {
            // Untouched while there is no reference.
            assert_eq!(aec.held(), 0);
            assert_eq!(chunk, mic[a..b].to_vec(), "second {sec}");
        }
        out.extend(chunk);
    }
    out.extend(aec.finish(&[]));
    assert_eq!(out.len(), mic.len());
    let before = db(energy(&mic, 6.0, 12.0) / energy(&out, 6.0, 12.0));
    let after = db(energy(&mic, 26.0, 40.0) / energy(&out, 26.0, 40.0));
    println!("[echo] restart: ERLE before drop {before:.1} dB, after restart 26-40s {after:.1} dB, lag {:?}", aec.lag());
    assert!(before >= 20.0, "before {before:.1}");
    assert!(after >= 20.0, "after {after:.1}");
}

#[test]
fn tap_is_cleaned_with_the_meetings_model() {
    let s = scene(80.0, 0.0, -6.0);
    let shared = Arc::new(Shared::default());
    let mut cleaner = MicCleaner::new(shared.clone());
    cleaner.system_started(0);
    // Meeting runs 0-24 s; a dictation tap covers 23-27 s and the capture
    // thread has drained system audio only up to 24 s.
    for sec in 0..24 {
        let (a, b) = (sec * SR, (sec + 1) * SR);
        cleaner.reference(a as u64, &s.far[a..b]);
        cleaner.process(&s.mic[a..b], false);
    }
    let (a, b) = (23 * SR, 27 * SR);
    let tail = s.far[24 * SR..b].to_vec();
    let cleaned = clean_tap(&shared, || Some((24 * SR, tail)), 23_000, s.mic[a..b].to_vec());
    assert_eq!(cleaned.len(), b - a);
    let erle = db(energy(&s.mic[a..b], 0.0, 4.0) / energy(&cleaned, 0.0, 4.0));
    println!("[echo] dictation tap during far-end speech: ERLE {erle:.1} dB");
    assert!(erle >= 20.0, "tap ERLE {erle:.1}");
    drop(cleaner);
    let raw = s.mic[a..b].to_vec();
    assert_eq!(clean_tap(&shared, || None, 23_000, raw.clone()), raw, "no meeting: untouched");
}

// ---- decision layer ----

/// System audio for a line `[a, b)` of the mic, starting LEAD_MS early.
fn sys_for(far: &[f32], a: usize, b: usize) -> Vec<f32> {
    let lead = guard::LEAD_MS as usize * SR / 1000;
    let trail = guard::TRAIL_MS as usize * SR / 1000;
    let mut v = vec![0.0; lead.saturating_sub(a)];
    v.extend_from_slice(&far[a.saturating_sub(lead)..(b + trail).min(far.len())]);
    v
}

static LEAK_SCENE: OnceLock<(Vec<f32>, Vec<f32>, Vec<f32>)> = OnceLock::new();

/// Far line at 1 s; the user answers at 6 s (no overlap); then both talk.
fn leak_scene() -> &'static (Vec<f32>, Vec<f32>, Vec<f32>) {
    LEAK_SCENE.get_or_init(|| {
        let total = 14.0;
        let mut far = fill(FAR_VOICE, &FAR_LINES[..1], 1.0, 5.0, total);
        let far2 = fill(FAR_VOICE, &FAR_LINES[1..2], 10.0, 14.0, total);
        far.iter_mut().zip(&far2).for_each(|(a, b)| *a += b);
        let near = fill(NEAR_VOICE, &NEAR_LINES[..1], 6.0, 9.5, total);
        let near2 = fill(NEAR_VOICE, &NEAR_LINES[1..2], 10.2, 14.0, total);
        let near: Vec<f32> = near.iter().zip(&near2).map(|(a, b)| a + b).collect();
        let echo: Vec<f32> = delay_by(&convolve(&far, &room()), |_| 0.12 * SR as f32)
            .into_iter()
            .map(|v| v * 0.5)
            .collect();
        let mut rng = Lcg(11);
        let mic = (0..far.len()).map(|i| near[i] + echo[i] + rng.next() * 3e-4).collect();
        (far, near, mic)
    })
}

#[test]
fn raw_leak_matches_the_system_audio() {
    let (far, _, mic) = leak_scene();
    // The unconverged case: the echo of the 1-5 s line, uncancelled.
    let (a, b) = (SR + SR / 10, 5 * SR);
    let m = echo_match(&mic[a..b], &sys_for(far, a, b));
    println!("[echo] guard: raw leak match {m:.2}");
    assert!(m >= 0.7, "{m:.2}");
    assert!(is_leak(m, 1.0, FAR_LINES[0], 3_900));
    assert!(is_leak(m, 0.0, FAR_LINES[0], 3_900), "long and overwhelming: no text needed");
}

#[test]
fn own_speech_does_not_match() {
    let (far, _, mic) = leak_scene();
    // The user alone (far end silent), then the user over the far end.
    for (name, a, b, text, theirs) in [
        ("near only", 6 * SR, 9 * SR + SR / 2, NEAR_LINES[0], ""),
        ("double talk", 10 * SR + SR / 5, 14 * SR, NEAR_LINES[1], FAR_LINES[1]),
    ] {
        let m = echo_match(&mic[a..b], &sys_for(far, a, b));
        let r = words_repeated(text, theirs);
        println!("[echo] guard: {name} match {m:.2}, repeated {r:.2}");
        assert!(m < 0.4, "{name}: {m:.2}");
        assert!(!is_leak(m, r, text, ((b - a) / 16) as i64), "{name}");
    }
}

#[test]
fn shared_short_words_are_kept_but_their_echo_is_not() {
    // Both sides say "Yeah." 150 ms apart; the user's own is kept.
    let theirs = say(FAR_VOICE, "Yeah.");
    let mine = say(NEAR_VOICE, "Yeah.");
    let len = SR * 2;
    let mut far = vec![0.0; len];
    far[SR / 2..SR / 2 + theirs.len()].copy_from_slice(&theirs);
    let mut mic = vec![0.0; len];
    let at = SR / 2 + SR * 15 / 100;
    mic[at..at + mine.len()].copy_from_slice(&mine);
    let mut rng = Lcg(5);
    mic.iter_mut().for_each(|v| *v += rng.next() * 3e-4);
    let (a, b) = (SR / 2, SR / 2 + SR);
    let m = echo_match(&mic[a..b], &sys_for(&far, a, b));
    println!("[echo] guard: two people saying yeah, match {m:.2}");
    assert!(!is_leak(m, 1.0, "Yeah.", 1000), "{m:.2}");

    // The same remote "Yeah." leaking through the speakers.
    let echo = delay_by(&convolve(&far, &room()), |_| 0.1 * SR as f32);
    let leak: Vec<f32> = echo.iter().map(|v| v * 0.5 + rng.next() * 3e-4).collect();
    let m = echo_match(&leak[a..b], &sys_for(&far, a, b));
    println!("[echo] guard: leaked yeah, match {m:.2}");
    assert!(is_leak(m, 1.0, "Yeah.", 1000), "{m:.2}");
}

#[test]
fn own_voice_returned_by_the_remote_side_is_kept() {
    // A remote participant without echo cancellation plays the user back:
    // system audio carries the user's words 300 ms after the mic.
    let (_, near, mic) = leak_scene();
    let returned = delay_by(near, |_| 0.3 * SR as f32);
    let (a, b) = (6 * SR, 9 * SR + SR / 2);
    let m = echo_match(&mic[a..b], &sys_for(&returned, a, b));
    println!("[echo] guard: own voice returned by the far end, match {m:.2}");
    assert!(!is_leak(m, 1.0, NEAR_LINES[0], 3_500), "{m:.2}");
    // The canceller must not cancel the user's voice along that path either.
    let mut aec = Canceller::new();
    let out = run(&mut aec, &mic[..], Some(&returned[..]), 30);
    let kept = db(energy(&out, 6.0, 9.5) / energy(mic, 6.0, 9.5));
    println!("[echo] own voice returned: mic level change {kept:.2} dB, lag {:?}", aec.lag());
    assert!(kept.abs() < 0.5, "{kept:.2} dB");
}

#[test]
fn saved_mic_is_what_live_transcription_saw() {
    // The capture loop writes each cleaned chunk to <id>.mic.wav and feeds
    // the same chunk to the VAD; recovery and re-transcription read ranges
    // back from that file, so they hear the cancelled audio too.
    let s = scene(80.0, 0.0, -6.0);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.mic.wav");
    let mut writer = crate::audio_store::WavWriter::create(&path).unwrap();
    let mut aec = Canceller::new();
    let mut live = Vec::new();
    for start in (0..s.mic.len()).step_by(SR) {
        let end = (start + SR).min(s.mic.len());
        aec.push_reference(start as u64, &s.far[start..end]);
        let chunk = if end == s.mic.len() {
            aec.finish(&s.mic[start..end])
        } else {
            aec.process(&s.mic[start..end])
        };
        writer.append(&chunk).unwrap();
        writer.checkpoint().unwrap();
        live.extend(chunk);
    }
    assert_eq!(writer.finish().unwrap() as usize, s.mic.len());
    let (a, b) = (6 * SR as i64, 10 * SR as i64);
    let saved = crate::audio_store::read_samples(&path, a, b).unwrap();
    let (a, b) = (a as usize, b as usize);
    assert!(saved.iter().zip(&live[a..b]).all(|(x, y)| (x - y).abs() <= 1.0 / 16384.0));
    let erle = db(energy(&s.mic[a..b], 0.0, 4.0) / energy(&saved, 0.0, 4.0));
    println!("[echo] saved far-only stretch read back as recovery does: ERLE {erle:.1} dB");
    assert!(erle >= 20.0, "{erle:.1}");
}

#[test]
fn word_repetition() {
    assert_eq!(words_repeated("Okay, so let's start.", "okay so let's start the review"), 1.0);
    assert!(words_repeated("I can own that", "okay so let's start") < 0.3);
    assert_eq!(words_repeated("", "anything"), 0.0);
}
