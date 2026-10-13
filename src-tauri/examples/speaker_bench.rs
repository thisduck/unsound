//! Tune/validate live speaker tracking (src/speakers.rs) on real meeting audio
//! and on a synthetic multi-voice conversation with known ground truth.
//!
//!   cargo run --release --example speaker_bench                 # real meeting, all models
//!   cargo run --release --example speaker_bench -- --synthetic  # macOS `say` voices
//!   cargo run --release --example speaker_bench -- <meeting-id> [model-file ...]
//!
//! Prints numbers only — never transcript text.

#[path = "../src/speakers.rs"]
#[allow(dead_code)]
mod speakers;

use speakers::{cluster_embeddings_with, cosine, decode, encode, match_voiceprint, Embedder, Params, SpeakerTracker, Voiceprint};
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_MEETING: &str = "03d4a137-7833-4659-b28b-d44fc55ee11b";
const MODELS: &[&str] = &[
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
    "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx",
    "nemo_en_titanet_small.onnx",
];

fn app_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap()).join("Library/Application Support/com.unsound.app")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let synthetic = args.iter().any(|a| a == "--synthetic");
    let sweep = args.iter().any(|a| a == "--sweep");
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let models: Vec<String> = if positional.len() > 1 {
        positional[1..].iter().map(|s| s.to_string()).collect()
    } else {
        MODELS.iter().map(|s| s.to_string()).collect()
    };

    let id = if synthetic {
        "synthetic".to_string()
    } else {
        positional.first().map(|s| s.to_string()).unwrap_or(DEFAULT_MEETING.into())
    };
    let (samples, spans, truth) = if synthetic {
        synth_conversation()
    } else {
        let (s, sp) = load_meeting(&id);
        (s, sp, None)
    };
    let total_ms: i64 = spans.iter().map(|(a, b)| b - a).sum();
    println!(
        "audio {:.1} min, {} spans, {:.1} min of speech, median span {:.1}s",
        samples.len() as f64 / 16000.0 / 60.0,
        spans.len(),
        total_ms as f64 / 60000.0,
        median(spans.iter().map(|(a, b)| (b - a) as f64 / 1000.0).collect())
    );

    for model in &models {
        let path = app_dir().join("models").join(model);
        println!("\n=== {model}");
        let mut emb = match Embedder::new(&path) {
            Ok(e) => e,
            Err(e) => {
                println!("  load failed: {e}");
                continue;
            }
        };
        let cache = std::env::temp_dir()
            .join("unsound-speaker-bench")
            .join(format!("{id}-{model}.emb"));
        let t0 = Instant::now();
        let cached = (!synthetic).then(|| load_cache(&cache, spans.len())).flatten();
        let from_cache = cached.is_some();
        let embeddings: Vec<Option<Vec<f32>>> = cached.unwrap_or_else(|| {
            spans
                .iter()
                .map(|&(s, e)| {
                    let a = ((s.max(0) as usize) * 16).min(samples.len());
                    let b = ((e.max(0) as usize) * 16).min(samples.len());
                    if b > a {
                        emb.embed(&samples[a..b])
                    } else {
                        None
                    }
                })
                .collect()
        });
        if !from_cache && !synthetic {
            save_cache(&cache, &embeddings);
        }
        let secs = t0.elapsed().as_secs_f64();
        let n_emb = embeddings.iter().filter(|e| e.is_some()).count();
        if from_cache {
            println!("  embedded {}/{} spans (cached)", n_emb, spans.len());
        } else {
            println!(
                "  embedded {}/{} spans in {:.1}s ({:.0} ms/span, {:.3}x realtime of speech)",
                n_emb,
                spans.len(),
                secs,
                secs * 1000.0 / spans.len() as f64,
                secs / (total_ms as f64 / 1000.0)
            );
        }
        similarity_diagnostics(&embeddings, &spans, truth.as_deref());
        if sweep {
            run_sweep(&embeddings, &spans, total_ms, truth.as_deref());
            continue;
        }

        // Live, in timeline order.
        let params = Params::for_model(&path);
        println!("  params {params:?}");
        let mut tracker = SpeakerTracker::with_params(vec![], params);
        let mut keys: Vec<usize> = spans
            .iter()
            .zip(&embeddings)
            .map(|(&(s, e), emb)| tracker.assign(emb.as_deref(), s, e))
            .collect();
        println!("  live (before consolidate): {}", fmt_clusters(&tracker.centroids(), total_ms));
        let remaps = tracker.consolidate();
        for k in keys.iter_mut() {
            if let Some((_, n)) = remaps.iter().find(|(o, _)| o == k) {
                *k = *n;
            }
        }
        let cents = tracker.centroids();
        println!("  live (consolidated):       {}", fmt_clusters(&cents, total_ms));
        print_centroid_sims(&cents);
        if let Some(t) = &truth {
            let ks: Vec<Option<usize>> = keys.iter().map(|&k| Some(k)).collect();
            score("live", &ks, t);
            confusion(&ks, t);
        }

        voiceprint_check(&embeddings, &spans, truth.as_deref(), params);

        for forced in [None, Some(3), Some(4)] {
            let (keys, cents) = cluster_embeddings_with(&embeddings, &spans, forced, &[], params);
            let label = match forced {
                None => "offline auto".to_string(),
                Some(n) => format!("offline n={n}"),
            };
            println!("  {label:<26} {}", fmt_clusters(&cents, total_ms));
            if let Some(t) = &truth {
                score(&label, &keys, t);
            } else if forced.is_none() {
                timeline(&keys, &spans, &cents);
            }
        }
    }
}

fn load_meeting(id: &str) -> (Vec<f32>, Vec<(i64, i64)>) {
    let db = rusqlite::Connection::open_with_flags(
        app_dir().join("unsound.db"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .expect("open db");
    let mut stmt = db
        .prepare(
            "SELECT start_ms, end_ms FROM segments
             WHERE meeting_id = ?1 AND source = 'system' ORDER BY start_ms, id",
        )
        .unwrap();
    let spans: Vec<(i64, i64)> = stmt
        .query_map([id], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .map(|r| r.unwrap())
        .filter(|(a, b)| b > a)
        .collect();
    let wav = app_dir().join("meeting-audio").join(format!("{id}.wav"));
    let samples = unsound_lib::audiofile::decode_to_samples(&wav).expect("decode wav");
    (samples, spans)
}

/// A scripted conversation between macOS voices, with pauses, as one 16 kHz
/// timeline plus the true speaker of each span.
fn synth_conversation() -> (Vec<f32>, Vec<(i64, i64)>, Option<Vec<usize>>) {
    let voices = ["Samantha", "Daniel", "Karen", "Moira", "Rishi"];
    let lines = [
        "Okay so let's get started with the quarterly planning review.",
        "Sure. I pulled the numbers from last week and they look better than expected.",
        "Great, can you walk us through the churn figures first?",
        "Churn is down about two points, mostly from the onboarding changes.",
        "That matches what support has been telling me.",
        "Right.",
        "What about the enterprise pipeline, is anything slipping?",
        "Two deals moved to next quarter, but the rest are on track.",
        "Mm hmm.",
        "I think we should revisit the pricing page before the launch.",
        "Agreed, and let's make sure legal signs off on the new terms.",
        "I can own that, I'll set up time with them on Thursday.",
        "Yes.",
        "One more thing, the mobile release needs another round of testing.",
        "Let's give it until Friday and then decide.",
        "Sounds good to me.",
        "Can someone send the notes afterwards?",
        "I'll write them up and share them this afternoon.",
        "Perfect, thanks everyone.",
        "Thanks, talk soon.",
    ];
    // Speaker for each line: an uneven, realistic turn-taking pattern.
    let base = [0, 1, 0, 1, 2, 0, 3, 1, 0, 2, 3, 4, 3, 1, 0, 2, 4, 3, 0, 1];
    // Three rounds of the script with the cast rotated, so every voice says
    // lines of every length (60 turns).
    let order: Vec<usize> = (0..3).flat_map(|r| base.iter().map(move |&s| (s + r * 2) % 5)).collect();
    let lines: Vec<&str> = (0..3).flat_map(|_| lines.iter().copied()).collect();
    let dir = std::env::temp_dir().join("unsound-speaker-bench");
    std::fs::create_dir_all(&dir).unwrap();
    let mut samples: Vec<f32> = vec![0.0; 16000];
    let mut spans = Vec::new();
    let mut truth = Vec::new();
    for (i, (&spk, line)) in order.iter().zip(lines.iter()).enumerate() {
        let out = dir.join(format!("line{i}.wav"));
        let ok = std::process::Command::new("say")
            .args(["-v", voices[spk], "--data-format=LEI16@16000", "-o"])
            .arg(&out)
            .arg(line)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(ok, "`say` failed for voice {}", voices[spk]);
        let clip = read_wav16(&out);
        let start = samples.len() as i64 / 16;
        samples.extend_from_slice(&clip);
        let end = samples.len() as i64 / 16;
        spans.push((start, end));
        truth.push(spk);
        samples.extend(std::iter::repeat(0.0).take(16000 * 6 / 10));
    }
    println!("synthetic: {} voices {:?}", voices.len(), voices);
    (samples, spans, Some(truth))
}

fn read_wav16(p: &Path) -> Vec<f32> {
    unsound_lib::audiofile::decode_to_samples(p).expect("decode synth wav")
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn fmt_clusters(c: &[(usize, Vec<f32>, usize, i64)], total_ms: i64) -> String {
    let mut v: Vec<&(usize, Vec<f32>, usize, i64)> = c.iter().collect();
    v.sort_by_key(|x| -x.3);
    let parts: Vec<String> = v
        .iter()
        .map(|(_, _, n, ms)| format!("{n}seg/{:.1}m({:.0}%)", *ms as f64 / 60000.0, 100.0 * *ms as f64 / total_ms.max(1) as f64))
        .collect();
    format!("{} clusters: {}", c.len(), parts.join(", "))
}

fn print_centroid_sims(c: &[(usize, Vec<f32>, usize, i64)]) {
    let mut v: Vec<&(usize, Vec<f32>, usize, i64)> = c.iter().filter(|x| !x.1.is_empty()).collect();
    v.sort_by_key(|x| -x.3);
    let v: Vec<_> = v.into_iter().take(6).collect();
    let mut rows = Vec::new();
    for a in &v {
        let r: Vec<String> = v.iter().map(|b| format!("{:.2}", cosine(&a.1, &b.1))).collect();
        rows.push(r.join(" "));
    }
    println!("  centroid sims (top clusters by speech): [{}]", rows.join(" | "));
}

/// Adjacent-span vs random-pair similarity (and true same/different speaker
/// similarity when ground truth is known) — where the thresholds should sit.
fn similarity_diagnostics(e: &[Option<Vec<f32>>], spans: &[(i64, i64)], truth: Option<&[usize]>) {
    let idx: Vec<usize> = (0..e.len()).filter(|&i| e[i].is_some()).collect();
    let mut adj = Vec::new();
    for w in idx.windows(2) {
        let (a, b) = (w[0], w[1]);
        if spans[b].0 - spans[a].1 < 1500 {
            adj.push(cosine(e[a].as_ref().unwrap(), e[b].as_ref().unwrap()) as f64);
        }
    }
    let mut rnd = Vec::new();
    let mut x: u64 = 88172645463325252;
    for _ in 0..4000.min(idx.len() * idx.len()) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let a = idx[(x % idx.len() as u64) as usize];
        let b = idx[((x >> 20) % idx.len() as u64) as usize];
        if a != b {
            rnd.push(cosine(e[a].as_ref().unwrap(), e[b].as_ref().unwrap()) as f64);
        }
    }
    let q = |mut v: Vec<f64>| -> String {
        if v.is_empty() {
            return "-".into();
        }
        v.sort_by(|a, b| a.total_cmp(b));
        let p = |f: f64| v[((v.len() - 1) as f64 * f) as usize];
        format!("p10 {:.2} p50 {:.2} p90 {:.2}", p(0.1), p(0.5), p(0.9))
    };
    println!("  sim adjacent(<1.5s gap): {}   random pairs: {}", q(adj), q(rnd));
    if let Some(t) = truth {
        let (mut same, mut diff) = (Vec::new(), Vec::new());
        for (i, &a) in idx.iter().enumerate() {
            for &b in &idx[i + 1..] {
                let s = cosine(e[a].as_ref().unwrap(), e[b].as_ref().unwrap()) as f64;
                if t[a] == t[b] {
                    same.push(s)
                } else {
                    diff.push(s)
                }
            }
        }
        println!("  sim same-speaker: {}   different: {}", q(same), q(diff));
    }
}

/// Purity + adjusted Rand index against ground truth.
fn score(label: &str, keys: &[Option<usize>], truth: &[usize]) {
    let n = truth.len();
    let k: Vec<usize> = keys.iter().map(|k| k.unwrap_or(usize::MAX)).collect();
    let mut pairs_same_both = 0f64;
    let (mut same_t, mut same_k) = (0f64, 0f64);
    for i in 0..n {
        for j in i + 1..n {
            let st = truth[i] == truth[j];
            let sk = k[i] == k[j];
            same_t += st as u8 as f64;
            same_k += sk as u8 as f64;
            pairs_same_both += (st && sk) as u8 as f64;
        }
    }
    let total = (n * (n - 1) / 2) as f64;
    let expected = same_t * same_k / total;
    let max = (same_t + same_k) / 2.0;
    let ari = if (max - expected).abs() < 1e-9 { 1.0 } else { (pairs_same_both - expected) / (max - expected) };
    // Purity: each predicted cluster votes its majority true speaker.
    let mut by: std::collections::HashMap<usize, std::collections::HashMap<usize, usize>> = Default::default();
    for i in 0..n {
        *by.entry(k[i]).or_default().entry(truth[i]).or_default() += 1;
    }
    let pure: usize = by.values().map(|m| *m.values().max().unwrap()).sum();
    println!(
        "    {label}: ARI {:.2}, purity {:.2}, {} predicted vs {} true speakers",
        ari,
        pure as f64 / n as f64,
        by.len(),
        truth.iter().collect::<std::collections::HashSet<_>>().len()
    );
}

fn load_cache(p: &Path, n: usize) -> Option<Vec<Option<Vec<f32>>>> {
    let data = std::fs::read(p).ok()?;
    let mut out = Vec::new();
    let mut i = 0;
    while i + 4 <= data.len() {
        let len = u32::from_le_bytes(data[i..i + 4].try_into().unwrap()) as usize;
        i += 4;
        if len == 0 {
            out.push(None);
        } else {
            out.push(Some(decode(&data[i..i + len * 4])));
            i += len * 4;
        }
    }
    (out.len() == n).then_some(out)
}

fn save_cache(p: &Path, e: &[Option<Vec<f32>>]) {
    let _ = std::fs::create_dir_all(p.parent().unwrap());
    let mut data = Vec::new();
    for v in e {
        match v {
            None => data.extend_from_slice(&0u32.to_le_bytes()),
            Some(v) => {
                data.extend_from_slice(&(v.len() as u32).to_le_bytes());
                data.extend_from_slice(&encode(v));
            }
        }
    }
    let _ = std::fs::write(p, data);
}

/// Live-tracker cluster counts (before/after consolidate) across a grid.
fn run_sweep(e: &[Option<Vec<f32>>], spans: &[(i64, i64)], total_ms: i64, truth: Option<&[usize]>) {
    for assign in [0.40f32, 0.45, 0.50, 0.55, 0.60] {
        for sticky in [0.0f32, 0.05, 0.10] {
            for new_min in [1000i64, 2000, 3000] {
                for merge in [0.55f32, 0.60, 0.65, 0.70] {
                    let p = Params { assign, sticky, new_cluster_min_ms: new_min, merge, ..Params::default() };
                    let mut t = SpeakerTracker::with_params(vec![], p);
                    let mut keys: Vec<usize> =
                        spans.iter().zip(e).map(|(&(s, en), v)| t.assign(v.as_deref(), s, en)).collect();
                    let live = t.centroids().len();
                    let remaps = t.consolidate();
                    for k in keys.iter_mut() {
                        if let Some((_, n)) = remaps.iter().find(|(o, _)| o == k) {
                            *k = *n;
                        }
                    }
                    let c = t.centroids();
                    let mut shares: Vec<i64> = c.iter().map(|x| x.3).collect();
                    shares.sort_by_key(|x| -x);
                    let sh: Vec<String> =
                        shares.iter().map(|m| format!("{:.0}", 100.0 * *m as f64 / total_ms as f64)).collect();
                    let q = truth.map(|t| {
                        let ks: Vec<Option<usize>> = keys.iter().map(|&k| Some(k)).collect();
                        let (ari, pur, _) = ari_purity(&ks, t);
                        format!(" ARI {ari:.2} pur {pur:.2}")
                    });
                    println!(
                        "    assign {assign:.2} sticky {sticky:.2} newmin {new_min} merge {merge:.2}: live {live:>2} -> {} [{}]{}",
                        c.len(),
                        sh.join("/"),
                        q.unwrap_or_default()
                    );
                }
            }
        }
    }
}

fn ari_purity(keys: &[Option<usize>], truth: &[usize]) -> (f64, f64, usize) {
    let n = truth.len();
    let k: Vec<usize> = keys.iter().map(|k| k.unwrap_or(usize::MAX)).collect();
    let (mut both, mut st_n, mut sk_n) = (0f64, 0f64, 0f64);
    for i in 0..n {
        for j in i + 1..n {
            let st = truth[i] == truth[j];
            let sk = k[i] == k[j];
            st_n += st as u8 as f64;
            sk_n += sk as u8 as f64;
            both += (st && sk) as u8 as f64;
        }
    }
    let total = (n * (n - 1) / 2) as f64;
    let expected = st_n * sk_n / total;
    let max = (st_n + sk_n) / 2.0;
    let ari = if (max - expected).abs() < 1e-9 { 1.0 } else { (both - expected) / (max - expected) };
    let mut by: std::collections::HashMap<usize, std::collections::HashMap<usize, usize>> = Default::default();
    for i in 0..n {
        *by.entry(k[i]).or_default().entry(truth[i]).or_default() += 1;
    }
    let pure: usize = by.values().map(|m| *m.values().max().unwrap()).sum();
    (ari, pure as f64 / n as f64, by.len())
}

/// Rows = true speaker, columns = predicted cluster.
fn confusion(keys: &[Option<usize>], truth: &[usize]) {
    let nt = truth.iter().max().map_or(0, |m| m + 1);
    let nk = keys.iter().flatten().max().map_or(0, |m| m + 1);
    for t in 0..nt {
        let row: Vec<String> = (0..nk)
            .map(|k| {
                let c = keys.iter().zip(truth).filter(|(kk, tt)| **kk == Some(k) && **tt == t).count();
                format!("{c:>2}")
            })
            .collect();
        println!("      true {t}: {}", row.join(" "));
    }
}

/// Where each cluster's speech falls in the meeting, as counts per 10-minute
/// bucket — tells a person who joined briefly from noise scattered throughout.
fn timeline(keys: &[Option<usize>], spans: &[(i64, i64)], cents: &[(usize, Vec<f32>, usize, i64)]) {
    let end = spans.iter().map(|s| s.1).max().unwrap_or(0);
    let buckets = (end / 600_000 + 1) as usize;
    for (k, _, n, ms) in cents {
        let mut row = vec![0usize; buckets];
        for (i, kk) in keys.iter().enumerate() {
            if *kk == Some(*k) {
                row[(spans[i].0 / 600_000) as usize] += 1;
            }
        }
        let r: Vec<String> = row.iter().map(|c| format!("{c:>3}")).collect();
        println!("      cluster {k} ({n} seg, {:.1}m) per 10min: {}", *ms as f64 / 60000.0, r.join(""));
    }
}

fn mean_dir(vs: &[&[f32]]) -> Vec<f32> {
    let d = vs.first().map_or(0, |v| v.len());
    let mut m = vec![0f32; d];
    for v in vs {
        for (a, b) in m.iter_mut().zip(v.iter()) {
            *a += b;
        }
    }
    let n = m.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    m.iter().map(|x| x / n).collect()
}

/// Voiceprint sanity: a speaker's centroid from one part of the recording vs
/// the same speaker elsewhere (should clear VOICEPRINT_THRESHOLD) and vs other
/// speakers (should not). Real data uses offline clusters as pseudo-labels,
/// split at the midpoint of each cluster's speech; synthetic builds voiceprints
/// from round 1 and runs the live tracker on rounds 2–3.
fn voiceprint_check(e: &[Option<Vec<f32>>], spans: &[(i64, i64)], truth: Option<&[usize]>, p: Params) {
    let labels: Vec<Option<usize>> = match truth {
        Some(t) => t.iter().map(|&x| Some(x)).collect(),
        None => cluster_embeddings_with(e, spans, None, &[], p).0,
    };
    let nl = labels.iter().flatten().max().map_or(0, |m| m + 1);
    if truth.is_none() {
        // Split each label's embedded members in half by time.
        let mut halves: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        for l in 0..nl {
            let members: Vec<&[f32]> = (0..e.len())
                .filter(|&i| labels[i] == Some(l) && e[i].is_some())
                .map(|i| e[i].as_deref().unwrap())
                .collect();
            if members.len() < 8 {
                continue;
            }
            let (a, b) = members.split_at(members.len() / 2);
            halves.push((mean_dir(a), mean_dir(b)));
        }
        let mut same = Vec::new();
        let mut diff = Vec::new();
        for (i, (a, _)) in halves.iter().enumerate() {
            for (j, (_, b)) in halves.iter().enumerate() {
                let s = cosine(a, b);
                if i == j { same.push(format!("{s:.2}")) } else { diff.push(format!("{s:.2}")) }
            }
        }
        println!("  voiceprint (1st half vs 2nd half centroids): same speaker [{}], different [{}]", same.join(" "), diff.join(" "));
        return;
    }
    let t = truth.unwrap();
    let third = spans.len() / 3;
    let names = ["S0", "S1", "S2", "S3", "S4"];
    let vps: Vec<Voiceprint> = (0..nl)
        .filter_map(|l| {
            let m: Vec<&[f32]> = (0..third).filter(|&i| t[i] == l && e[i].is_some()).map(|i| e[i].as_deref().unwrap()).collect();
            (!m.is_empty()).then(|| Voiceprint { name: names[l].into(), embedding: mean_dir(&m) })
        })
        .collect();
    let mut tr = SpeakerTracker::with_params(vps.clone(), p);
    let keys: Vec<usize> = (third..spans.len()).map(|i| tr.assign(e[i].as_deref(), spans[i].0, spans[i].1)).collect();
    tr.consolidate();
    let (mut right, mut wrong, mut unnamed) = (0, 0, 0);
    for (j, i) in (third..spans.len()).enumerate() {
        let _ = keys[j];
        match tr.cluster_name(keys[j]) {
            Some(n) if n == names[t[i]] => right += 1,
            Some(_) => wrong += 1,
            None => unnamed += 1,
        }
    }
    // Per-true-speaker centroid (rounds 2–3) vs voiceprints.
    let mut rows = Vec::new();
    for l in 0..nl {
        let m: Vec<&[f32]> = (third..spans.len()).filter(|&i| t[i] == l && e[i].is_some()).map(|i| e[i].as_deref().unwrap()).collect();
        if m.is_empty() { continue; }
        let c = mean_dir(&m);
        let sims: Vec<String> = vps.iter().map(|v| format!("{:.2}", cosine(&c, &v.embedding))).collect();
        rows.push(format!("{}→[{}] {:?}", names[l], sims.join(" "), match_voiceprint(&c, &vps)));
    }
    println!("  voiceprints: segments named right {right}, wrong {wrong}, unnamed {unnamed}");
    println!("  true-speaker centroid vs voiceprints: {}", rows.join("  "));
}
