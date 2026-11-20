//! Last check on a transcribed "me" line: is it the remote side leaking in
//! through the speakers? Works from the saved audio and committed lines only,
//! so recovery and re-transcription reach the same verdict as the live pass.
//!
//! Evidence: the line's mic audio is largely explained by the system audio
//! at a lag where the system audio leads (echo), and its words
//! repeat an overlapping "them" line. Matching words alone are never enough —
//! two people both say "yeah" — and a match on sound alone must be strong
//! and long. Anything short of that keeps the line.

use realfft::num_complex::Complex32 as C;
use realfft::RealFftPlanner;

const RATE: usize = 16_000;
/// System audio may lead the mic by up to this much…
pub const LEAD_MS: i64 = 1500;
/// …or trail it by this much.
pub const TRAIL_MS: i64 = 60;
/// Window either side of a line in which "them" text counts as overlapping.
pub const TEXT_WINDOW_MS: i64 = 2500;

/// Coherent share needed alongside repeated words.
const MATCH_WITH_TEXT: f32 = 0.45;
/// Short lines (one or two words) need more.
const MATCH_SHORT: f32 = 0.55;
/// Coherent share that is leak on its own, for a line this long.
const MATCH_ALONE: f32 = 0.7;
const ALONE_MIN_MS: i64 = 2000;
/// Share of the line's words found, in order, in the overlapping "them" text.
const WORDS_REPEATED: f32 = 0.6;

/// Share of the mic's in-band energy that the system audio explains
/// linearly once aligned (magnitude-squared coherence, weighted by the mic's
/// spectrum). An echo is a filtered copy of the system audio, so nearly all
/// of it is explained, however quiet; another voice — even saying the same
/// word at the same moment — is not, because its phase follows its own
/// pitch and articulation. `sys` starts `LEAD_MS` before `mic`; only lags at
/// which the system audio leads (or trails by up to `TRAIL_MS`) count.
pub fn echo_match(mic: &[f32], sys: &[f32]) -> f32 {
    const FRAME: usize = 1024;
    const HOP: usize = 256;
    let lead = LEAD_MS as usize * RATE / 1000;
    let trail = TRAIL_MS as usize * RATE / 1000;
    if mic.len() < FRAME * 2 || sys.len() < mic.len() {
        return 0.0;
    }
    let mut planner = RealFftPlanner::<f32>::new();

    // Alignment: band-limited GCC-PHAT over the allowed offsets.
    let n = (mic.len() + sys.len()).next_power_of_two();
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(n);
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    a[..mic.len()].copy_from_slice(mic);
    b[..sys.len()].copy_from_slice(sys);
    let mut sa: Vec<C> = fwd.make_output_vec();
    let mut sb: Vec<C> = fwd.make_output_vec();
    let _ = fwd.process(&mut a, &mut sa);
    let _ = fwd.process(&mut b, &mut sb);
    let (lo, hi) = (200 * n / RATE, 6000 * n / RATE);
    for (k, (x, y)) in sa.iter_mut().zip(&sb).enumerate() {
        let c = x.conj() * y;
        *x = if k < lo || k >= hi { C::default() } else { c / (c.norm() + 1e-12) };
    }
    let _ = inv.process(&mut sa, &mut a);
    // a[off] = sum_t mic[t] * sys[t + off]
    let max_off = (lead + trail).min(sys.len() - mic.len());
    let off = (0..=max_off)
        .max_by(|x, y| a[*x].abs().total_cmp(&a[*y].abs()))
        .unwrap_or(lead);
    let aligned = &sys[off..off + mic.len()];

    // Welch estimate of the coherent share.
    let fft = planner.plan_fft_forward(FRAME);
    let window: Vec<f32> = (0..FRAME)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / FRAME as f32).cos())
        .collect();
    let bins = FRAME / 2 + 1;
    let (lo, hi) = (200 * FRAME / RATE, 6000 * FRAME / RATE);
    let mut smm = vec![0.0f64; bins];
    let mut sss = vec![0.0f64; bins];
    let mut sms = vec![C::default(); bins];
    let mut fa = fft.make_input_vec();
    let mut fb = fft.make_input_vec();
    let mut xa: Vec<C> = fft.make_output_vec();
    let mut xb: Vec<C> = fft.make_output_vec();
    let mut pos = 0;
    while pos + FRAME <= mic.len() {
        for i in 0..FRAME {
            fa[i] = mic[pos + i] * window[i];
            fb[i] = aligned[pos + i] * window[i];
        }
        let _ = fft.process(&mut fa, &mut xa);
        let _ = fft.process(&mut fb, &mut xb);
        for k in lo..hi {
            smm[k] += xa[k].norm_sqr() as f64;
            sss[k] += xb[k].norm_sqr() as f64;
            sms[k] += xa[k] * xb[k].conj();
        }
        pos += HOP;
    }
    let total: f64 = smm[lo..hi].iter().sum();
    if total <= 1e-12 {
        return 0.0;
    }
    let explained: f64 = (lo..hi)
        .filter(|&k| sss[k] > 1e-12)
        .map(|k| (sms[k].norm_sqr() as f64 / sss[k]).min(smm[k]))
        .sum();
    (explained / total) as f32
}

fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(|c| c.to_lowercase())
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Share of `mine`'s words that appear, in order, in `theirs`.
pub fn words_repeated(mine: &str, theirs: &str) -> f32 {
    let a = words(mine);
    let b = words(theirs);
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut prev = vec![0usize; b.len() + 1];
    for x in &a {
        let mut cur = vec![0usize; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = if x == y {
                prev[j] + 1
            } else {
                cur[j].max(prev[j + 1])
            };
        }
        prev = cur;
    }
    prev[b.len()] as f32 / a.len() as f32
}

/// The verdict, from the measurements above.
pub fn is_leak(matched: f32, repeated: f32, text: &str, duration_ms: i64) -> bool {
    let short = words(text).len() < 3;
    let needed = if short { MATCH_SHORT } else { MATCH_WITH_TEXT };
    let full = if short { 0.99 } else { WORDS_REPEATED };
    (repeated >= full && matched >= needed)
        || (matched >= MATCH_ALONE && duration_ms >= ALONE_MIN_MS)
}

/// Drop "me" lines that are the remote side leaking into the mic.
pub(crate) fn drop_leaked_lines(
    app: &tauri::AppHandle,
    meeting_id: &str,
    segs: &mut Vec<crate::store::Segment>,
) {
    use crate::{audio_store, store::Db};
    use tauri::Manager;
    if !segs.iter().any(|s| s.source == "mic") {
        return;
    }
    let (Ok(mic_path), Ok(sys_path)) = (
        audio_store::mic_path(app, meeting_id),
        audio_store::system_path(app, meeting_id),
    ) else {
        return;
    };
    if audio_store::sample_count(&sys_path).unwrap_or(0) == 0 {
        return;
    }
    let db = app.state::<Db>();
    segs.retain(|s| {
        if s.source != "mic" || s.text.trim().is_empty() {
            return true;
        }
        let them = overlapping_them(&db, meeting_id, s.start_ms, s.end_ms);
        let Ok(mic) = audio_store::read_range(&mic_path, s.start_ms, s.end_ms) else {
            return true;
        };
        let Ok(sys) =
            audio_store::read_range(&sys_path, s.start_ms - LEAD_MS, s.end_ms + TRAIL_MS)
        else {
            return true;
        };
        // read_range clamps at 0: keep the system audio's start aligned.
        let missing = ((LEAD_MS - s.start_ms).max(0) as usize * RATE) / 1000;
        let mut sys_aligned = vec![0.0; missing];
        sys_aligned.extend(sys);
        // Live, the system file can trail the mic by up to a poll: judge the
        // part of the line it covers.
        let covered = sys_aligned.len().saturating_sub(LEAD_MS as usize * RATE / 1000);
        let matched = echo_match(&mic[..mic.len().min(covered)], &sys_aligned);
        let repeated = words_repeated(&s.text, &them);
        let leak = is_leak(matched, repeated, &s.text, s.end_ms - s.start_ms);
        if leak {
            eprintln!(
                "[echo] dropped a mic line at {} ms as speaker leak (match {matched:.2}, repeated {repeated:.2})",
                s.start_ms
            );
        }
        !leak
    });
}

fn overlapping_them(db: &crate::store::Db, meeting_id: &str, start_ms: i64, end_ms: i64) -> String {
    let conn = db.0.lock().unwrap();
    let Ok(mut q) = conn.prepare(
        "SELECT text FROM segments WHERE meeting_id=?1 AND source='system'
         AND end_ms>=?2 AND start_ms<=?3 ORDER BY start_ms",
    ) else {
        return String::new();
    };
    let rows = q.query_map(
        rusqlite::params![meeting_id, start_ms - TEXT_WINDOW_MS, end_ms + TEXT_WINDOW_MS],
        |r| r.get::<_, String>(0),
    );
    match rows {
        Ok(rows) => rows.flatten().collect::<Vec<_>>().join(" "),
        Err(_) => String::new(),
    }
}
