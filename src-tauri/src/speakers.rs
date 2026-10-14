//! Live speaker tracking for meetings: who is talking on the system-audio
//! ("them") channel, decided segment by segment *while the meeting runs*
//! instead of in one offline diarization pass after it ends.
//!
//! Each finalized speech segment gets a speaker embedding (a fixed-size voice
//! fingerprint from a small sherpa-onnx model — the same CAM++ / ERes2Net /
//! TitaNet files the offline diarizer uses), and an online clusterer assigns
//! it to the closest known voice or opens a new one. At the end of the meeting
//! `consolidate` merges voices that turned out to be the same person and folds
//! tiny phantom clusters into their nearest real speaker.
//!
//! Voiceprints: when the user names a speaker, that cluster's centroid is kept
//! (see `blend`) so the same voice is labeled automatically next time.
//!
//! Everything here is plain vector math except `Embedder`, which wraps the
//! sherpa extractor. Embeddings are always L2-normalized, so the dot product
//! is the cosine similarity.

use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig};
use std::path::Path;

const SAMPLE_RATE: u32 = 16_000;

/// Shorter speech than this gives an unreliable embedding (mostly room tone
/// and one or two phonemes); such segments inherit the previous speaker.
pub const MIN_EMBED_MS: i64 = 1_000;
/// Longer segments are embedded from their middle this-long window: the
/// fingerprint is stable well before this, and it bounds per-segment cost.
pub const MAX_EMBED_MS: i64 = 10_000;

/// Cosine similarity to a cluster centroid needed to join it; below this a
/// segment opens a new cluster.
pub const ASSIGN_THRESHOLD: f32 = 0.50;
/// Added to the most recent speaker's score: consecutive segments are far
/// more often the same person continuing than a change of speaker.
pub const STICKY_BONUS: f32 = 0.05;
/// Hard ceiling on live clusters (a runaway guard, not a real limit).
pub const MAX_CLUSTERS: usize = 16;
/// Centroids at least this similar are the same voice split early in the
/// meeting, and are merged by `consolidate`.
pub const MERGE_THRESHOLD: f32 = 0.60;
/// A cluster holding less than this share of the channel's speech, with few
/// embedded segments, is a phantom and is folded into its nearest neighbor.
pub const MIN_SPEECH_SHARE: f64 = 0.03;
pub const MIN_CLUSTER_SEGMENTS: usize = 6;
/// Below this share a cluster is folded however many segments it has — the
/// scattered one-word-at-a-time tail the old offline diarizer produced.
pub const MIN_SPEECH_SHARE_ANY: f64 = 0.01;
/// Offline (re-detect) agglomerative clustering stops merging below this
/// average-linkage similarity.
pub const OFFLINE_THRESHOLD: f32 = 0.45;
/// A stored voiceprint names a cluster only when its centroid is at least this
/// similar, and clearly closer than any other voiceprint.
pub const VOICEPRINT_THRESHOLD: f32 = 0.65;
pub const VOICEPRINT_MARGIN: f32 = 0.10;
/// A single segment this close to a known voiceprint (with the same margin)
/// goes to that person's cluster even when another cluster scores higher —
/// it keeps known voices that sound alike from merging.
pub const SEGMENT_VOICEPRINT_THRESHOLD: f32 = 0.60;
/// Embedded segments a cluster needs before it's matched against voiceprints.
const VOICEPRINT_MIN_SEGMENTS: usize = 3;
/// A segment unlike every known voice opens a new cluster only if it's at
/// least this long; shorter outliers (a laugh, a cough, a one-word "yeah"
/// over a bad line) join the closest cluster without moving its centroid.
pub const NEW_CLUSTER_MIN_MS: i64 = 2_000;

/// Clustering knobs, defaulting to the constants above (overridable for
/// tuning — see examples/speaker_bench.rs).
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub assign: f32,
    pub sticky: f32,
    pub merge: f32,
    pub new_cluster_min_ms: i64,
    pub min_share: f64,
    pub min_segments: usize,
    pub offline: f32,
}

impl Params {
    /// Defaults tuned per embedding model: their similarity scales differ
    /// (ERes2NetV2 scores everyone higher, TitaNet a little lower spread than
    /// CAM++). Unknown models get the CAM++ values.
    pub fn for_model(model_path: &Path) -> Self {
        let name = model_path
            .file_name()
            .map(|n| n.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let base = Self::default();
        if name.contains("eres2net") {
            Self { assign: 0.55, merge: 0.65, offline: 0.55, ..base }
        } else if name.contains("titanet") {
            Self { assign: 0.55, merge: 0.60, offline: 0.45, ..base }
        } else {
            base
        }
    }
}

impl Default for Params {
    fn default() -> Self {
        Self {
            assign: ASSIGN_THRESHOLD,
            sticky: STICKY_BONUS,
            merge: MERGE_THRESHOLD,
            new_cluster_min_ms: NEW_CLUSTER_MIN_MS,
            min_share: MIN_SPEECH_SHARE,
            min_segments: MIN_CLUSTER_SEGMENTS,
            offline: OFFLINE_THRESHOLD,
        }
    }
}

// ---- embeddings ------------------------------------------------------------

/// A speaker-embedding model. The underlying sherpa extractor is marked
/// `Send`/`Sync` by sherpa-rs, but it isn't safe to use from two threads at
/// once; keep one `Embedder` per thread (the meeting's transcription thread).
pub struct Embedder {
    ext: EmbeddingExtractor,
    params: Params,
}

impl Embedder {
    pub fn new(model_path: &Path) -> Result<Self, String> {
        let config = ExtractorConfig {
            model: model_path.to_string_lossy().to_string(),
            provider: None,
            num_threads: Some(2),
            debug: false,
        };
        let ext = EmbeddingExtractor::new(config).map_err(|e| e.to_string())?;
        Ok(Self { ext, params: Params::for_model(model_path) })
    }

    /// Clustering thresholds tuned for this embedding model — pass to
    /// `SpeakerTracker::with_params` or `cluster_embeddings_with`.
    pub fn params(&self) -> Params {
        self.params
    }

    /// L2-normalized embedding of 16 kHz mono speech, or `None` when the audio
    /// is too short (< `MIN_EMBED_MS`) or the model returns nothing usable.
    pub fn embed(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        let min = ms_to_samples(MIN_EMBED_MS);
        if samples.len() < min {
            return None;
        }
        let max = ms_to_samples(MAX_EMBED_MS);
        let window = if samples.len() > max {
            let start = (samples.len() - max) / 2;
            &samples[start..start + max]
        } else {
            samples
        };
        let v = self
            .ext
            .compute_speaker_embedding(window.to_vec(), SAMPLE_RATE)
            .ok()?;
        normalize(v)
    }
}

fn ms_to_samples(ms: i64) -> usize {
    (ms.max(0) as usize) * SAMPLE_RATE as usize / 1000
}

/// Unit-length copy of `v`; `None` for an all-zero or non-finite vector.
fn normalize(mut v: Vec<f32>) -> Option<Vec<f32>> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_finite() || norm < 1e-6 {
        return None;
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    Some(v)
}

/// Cosine similarity. Safe on unnormalized input; 0 for mismatched/empty.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let d = (na * nb).sqrt();
    if d < 1e-12 {
        0.0
    } else {
        dot / d
    }
}

/// Running average of a stored voiceprint (`existing` built from
/// `existing_count` meetings) with a new centroid, re-normalized.
#[cfg(test)]
pub fn blend(existing: &[f32], existing_count: u32, new: &[f32]) -> Vec<f32> {
    blend_weighted(existing, existing_count, new, 1)
}

pub fn blend_weighted(existing: &[f32], existing_count: u32, new: &[f32], new_count: u32) -> Vec<f32> {
    if existing.len() != new.len() || existing_count == 0 {
        return normalize(new.to_vec()).unwrap_or_else(|| new.to_vec());
    }
    if new_count == 0 { return existing.to_vec(); }
    let (a,b)=(existing_count as f32,new_count as f32);
    let mixed: Vec<f32> = existing.iter().zip(new).map(|(e,x)| (e*a+x*b)/(a+b)).collect();
    normalize(mixed).unwrap_or_else(||existing.to_vec())
}

/// f32 little-endian blob, for SQLite.
pub fn encode(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn decode(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---- voiceprints -----------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Voiceprint {
    pub name: String,
    pub embedding: Vec<f32>,
}

/// The voiceprint a centroid belongs to, if one is clearly it: similarity at
/// least `VOICEPRINT_THRESHOLD` and `VOICEPRINT_MARGIN` above the runner-up
/// (a different name). Conservative on purpose — a wrong name is worse than
/// "Speaker 2".
pub fn match_voiceprint(centroid: &[f32], voiceprints: &[Voiceprint]) -> Option<String> {
    match_voiceprint_at(centroid, voiceprints, VOICEPRINT_THRESHOLD)
}

/// `match_voiceprint` with an explicit similarity floor (the margin rule is
/// the same). Single segments are noisier than centroids, so the tracker
/// uses `SEGMENT_VOICEPRINT_THRESHOLD` for them.
pub fn match_voiceprint_at(v: &[f32], voiceprints: &[Voiceprint], threshold: f32) -> Option<String> {
    // Score people, not individual enrollment rows: duplicate samples for a
    // person must never become their own runner-up and suppress a good match.
    let mut people: std::collections::HashMap<String, (&str,f32)> = std::collections::HashMap::new();
    for vp in voiceprints {
        if vp.name.trim().is_empty() || vp.embedding.len()!=v.len() { continue; }
        let score=cosine(v,&vp.embedding);
        if !score.is_finite() { continue; }
        let entry=people.entry(vp.name.trim().to_lowercase()).or_insert((vp.name.trim(),score));
        if score > entry.1 { *entry=(vp.name.trim(),score); }
    }
    let mut ranked: Vec<_>=people.values().copied().collect();
    ranked.sort_by(|a,b|b.1.total_cmp(&a.1));
    let &(name,score)=ranked.first()?;
    let second=ranked.get(1).map(|(_,s)|*s).unwrap_or(-1.0);
    (score>=threshold && score-second>=VOICEPRINT_MARGIN).then(||name.to_string())
}

// ---- online tracker ----------------------------------------------------------

#[derive(Clone, Debug)]
struct Cluster {
    /// Sum of member embeddings (each unit length); the centroid is its
    /// normalized direction.
    sum: Vec<f32>,
    centroid: Vec<f32>,
    /// Embedded members (what the centroid is built from).
    n_embedded: usize,
    /// All members, including too-short segments that inherited this cluster.
    n_segments: usize,
    speech_ms: i64,
    first_ms: i64,
    /// `Some(k)` once merged into cluster `k` (then it's no longer alive).
    merged_into: Option<usize>,
    name: Option<String>,
    /// Opened for a voiceprint match: keeps that name and only receives
    /// segments matched to it.
    pinned: bool,
}

impl Cluster {
    fn empty(first_ms: i64) -> Self {
        Self {
            sum: Vec::new(),
            centroid: Vec::new(),
            n_embedded: 0,
            n_segments: 0,
            speech_ms: 0,
            first_ms,
            merged_into: None,
            name: None,
            pinned: false,
        }
    }

    fn alive(&self) -> bool {
        self.merged_into.is_none()
    }

    fn add_embedding(&mut self, e: &[f32]) {
        if self.sum.is_empty() {
            self.sum = e.to_vec();
        } else {
            for (s, x) in self.sum.iter_mut().zip(e) {
                *s += x;
            }
        }
        self.n_embedded += 1;
        self.centroid = normalize(self.sum.clone()).unwrap_or_default();
    }

    fn absorb(&mut self, other: &Cluster) {
        if !other.sum.is_empty() {
            if self.sum.is_empty() {
                self.sum = other.sum.clone();
            } else {
                for (s, x) in self.sum.iter_mut().zip(&other.sum) {
                    *s += x;
                }
            }
            self.centroid = normalize(self.sum.clone()).unwrap_or_default();
        }
        self.n_embedded += other.n_embedded;
        self.n_segments += other.n_segments;
        self.speech_ms += other.speech_ms;
        self.first_ms = self.first_ms.min(other.first_ms);
        if self.name.is_none() {
            self.name = other.name.clone();
            self.pinned |= other.pinned;
        }
    }
}

/// Online clustering of one channel's speakers during a live meeting. Keys
/// are indices into an append-only list, so a key handed out stays valid;
/// after `consolidate` the survivors are renumbered 0..k in order of first
/// appearance and the returned remaps say how to relabel stored segments.
pub struct SpeakerTracker {
    clusters: Vec<Cluster>,
    voiceprints: Vec<Voiceprint>,
    last: Option<usize>,
    params: Params,
}

impl SpeakerTracker {
    pub fn new(voiceprints: Vec<Voiceprint>) -> Self {
        Self::with_params(voiceprints, Params::default())
    }

    pub fn with_params(voiceprints: Vec<Voiceprint>, params: Params) -> Self {
        Self {
            clusters: Vec::new(),
            voiceprints,
            last: None,
            params,
        }
    }

    /// Assign a finalized segment to a cluster key. `embedding` is `None` for a
    /// segment too short to embed: it goes to the most recent speaker (or opens
    /// cluster 0 when nobody has spoken yet).
    pub fn assign(&mut self, embedding: Option<&[f32]>, start_ms: i64, end_ms: i64) -> usize {
        let dur = (end_ms - start_ms).max(0);
        let p = self.params;
        // Whether this segment's embedding should shape the cluster it joins.
        let mut learn = embedding.is_some();
        let key = match embedding {
            None => match self.last {
                Some(k) => k,
                None => self.open(start_ms),
            },
            Some(e) if self.voiced(e).is_some() => {
                // A known voice: its own cluster, opened on first sight.
                let name = self.voiced(e).unwrap();
                match self
                    .clusters
                    .iter()
                    .position(|c| c.alive() && c.name.as_deref() == Some(name.as_str()))
                {
                    Some(k) => k,
                    None => {
                        let k = self.open(start_ms);
                        self.clusters[k].name = Some(name);
                        self.clusters[k].pinned = true;
                        k
                    }
                }
            }
            Some(e) => {
                let mut best: Option<(usize, f32)> = None;
                for (k, c) in self.clusters.iter().enumerate() {
                    if !c.alive() || c.centroid.is_empty() {
                        continue;
                    }
                    // Named clusters only take segments the voiceprint rule sent.
                    if c.pinned {
                        continue;
                    }
                    let mut s = cosine(e, &c.centroid);
                    if Some(k) == self.last {
                        s += p.sticky;
                    }
                    if best.map_or(true, |(_, b)| s > b) {
                        best = Some((k, s));
                    }
                }
                let alive = self.clusters.iter().filter(|c| c.alive()).count();
                match best {
                    Some((k, s)) if s >= p.assign => k,
                    Some((k, _)) if alive >= MAX_CLUSTERS => {
                        learn = false;
                        k
                    }
                    // Too short to trust as a new voice: park it with the
                    // closest one, without letting it pull that centroid.
                    Some((k, _)) if dur < p.new_cluster_min_ms => {
                        learn = false;
                        k
                    }
                    _ => {
                        // Adopt an unembedded placeholder (short segments seen
                        // before any real voice) rather than opening another.
                        match self
                            .clusters
                            .iter()
                            .position(|c| c.alive() && c.centroid.is_empty())
                        {
                            Some(k) => k,
                            None => self.open(start_ms),
                        }
                    }
                }
            }
        };
        let c = &mut self.clusters[key];
        if let Some(e) = embedding.filter(|_| learn) {
            c.add_embedding(e);
            if !c.pinned && c.n_embedded >= VOICEPRINT_MIN_SEGMENTS {
                c.name = match_voiceprint(&c.centroid, &self.voiceprints);
            }
        }
        c.n_segments += 1;
        c.speech_ms += dur;
        c.first_ms = c.first_ms.min(start_ms);
        self.last = Some(key);
        key
    }

    /// The voiceprint a single segment embedding clearly belongs to.
    fn voiced(&self, e: &[f32]) -> Option<String> {
        if self.voiceprints.is_empty() {
            return None;
        }
        match_voiceprint_at(e, &self.voiceprints, SEGMENT_VOICEPRINT_THRESHOLD)
    }

    fn open(&mut self, first_ms: i64) -> usize {
        self.clusters.push(Cluster::empty(first_ms));
        self.clusters.len() - 1
    }

    /// Voiceprint name matched for a cluster, if its centroid matches a known
    /// voice. Follows merges, so an old key still answers after `consolidate`.
    pub fn cluster_name(&self, key: usize) -> Option<String> {
        let k = self.resolve(key)?;
        self.clusters[k].name.clone()
    }

    fn resolve(&self, mut key: usize) -> Option<usize> {
        for _ in 0..self.clusters.len() + 1 {
            let c = self.clusters.get(key)?;
            match c.merged_into {
                None => return Some(key),
                Some(k) => key = k,
            }
        }
        None
    }

    /// End-of-meeting cleanup: merge near-duplicate clusters, fold phantom
    /// clusters into their nearest real speaker, then renumber survivors 0..k
    /// by first appearance. Returns `(old_key, new_key)` for every key whose
    /// label changes (apply all of them to the stored segments at once).
    pub fn consolidate(&mut self) -> Vec<(usize, usize)> {
        merge_similar(&mut self.clusters, self.params.merge);
        fold_tiny(&mut self.clusters, self.params.min_share, self.params.min_segments);
        let order = renumber(&mut self.clusters);
        for c in self.clusters.iter_mut().filter(|c| c.alive() && !c.pinned) {
            c.name = if c.n_embedded >= VOICEPRINT_MIN_SEGMENTS {
                match_voiceprint(&c.centroid, &self.voiceprints)
            } else {
                None
            };
        }
        self.last = self.last.and_then(|k| order.get(k).copied());
        (0..order.len())
            .filter(|&old| order[old] != old)
            .map(|old| (old, order[old]))
            .collect()
    }

    /// `(key, centroid, n_segments, speech_ms)` for every surviving cluster.
    /// Clusters that never got an embedding report an empty centroid.
    pub fn centroids(&self) -> Vec<(usize, Vec<f32>, usize, i64)> {
        self.clusters
            .iter()
            .enumerate()
            .filter(|(_, c)| c.alive())
            .map(|(k, c)| (k, c.centroid.clone(), c.n_segments, c.speech_ms))
            .collect()
    }
}

/// Repeatedly merge the most similar pair of alive clusters while their
/// centroids are at least `threshold` alike. The one with more speech keeps
/// its identity.
fn merge_similar(clusters: &mut [Cluster], threshold: f32) {
    loop {
        let alive: Vec<usize> = (0..clusters.len())
            .filter(|&k| clusters[k].alive() && !clusters[k].centroid.is_empty())
            .collect();
        let mut best: Option<(usize, usize, f32)> = None;
        for (i, &a) in alive.iter().enumerate() {
            for &b in &alive[i + 1..] {
                if let (Some(na), Some(nb)) = (&clusters[a].name, &clusters[b].name) {
                    if na != nb {
                        continue; // two different known people
                    }
                }
                let s = cosine(&clusters[a].centroid, &clusters[b].centroid);
                if s >= threshold && best.map_or(true, |(_, _, bs)| s > bs) {
                    best = Some((a, b, s));
                }
            }
        }
        let Some((a, b, _)) = best else { break };
        let (keep, drop) = if clusters[a].speech_ms >= clusters[b].speech_ms {
            (a, b)
        } else {
            (b, a)
        };
        merge_into(clusters, drop, keep);
    }
}

fn merge_into(clusters: &mut [Cluster], drop: usize, keep: usize) {
    let other = clusters[drop].clone();
    clusters[keep].absorb(&other);
    clusters[drop].merged_into = Some(keep);
    // Anything previously merged into `drop` now points through it to `keep`
    // (resolve follows chains, but flattening keeps remaps one hop).
    for c in clusters.iter_mut() {
        if c.merged_into == Some(drop) {
            c.merged_into = Some(keep);
        }
    }
    clusters[drop].merged_into = Some(keep);
}

/// Fold phantom clusters — little speech *and* few segments — into the most
/// similar substantial cluster (or the biggest, when the phantom has no
/// centroid). Never folds the last remaining cluster.
fn fold_tiny(clusters: &mut [Cluster], min_share: f64, min_segments: usize) {
    let total: i64 = clusters.iter().filter(|c| c.alive()).map(|c| c.speech_ms).sum();
    if total <= 0 {
        return;
    }
    let is_tiny = |c: &Cluster| {
        let share = c.speech_ms as f64 / total as f64;
        !c.pinned && ((share < min_share && c.n_segments < min_segments) || share < MIN_SPEECH_SHARE_ANY)
    };
    let big: Vec<usize> = (0..clusters.len())
        .filter(|&k| clusters[k].alive() && !is_tiny(&clusters[k]))
        .collect();
    if big.is_empty() {
        return;
    }
    let tiny: Vec<usize> = (0..clusters.len())
        .filter(|&k| clusters[k].alive() && is_tiny(&clusters[k]))
        .collect();
    for t in tiny {
        let target = if clusters[t].centroid.is_empty() {
            *big.iter().max_by_key(|&&k| clusters[k].speech_ms).unwrap()
        } else {
            *big
                .iter()
                .max_by(|&&a, &&b| {
                    cosine(&clusters[t].centroid, &clusters[a].centroid)
                        .total_cmp(&cosine(&clusters[t].centroid, &clusters[b].centroid))
                })
                .unwrap()
        };
        merge_into(clusters, t, target);
    }
}

/// Compact alive clusters to keys 0..k by first appearance, rewriting the list
/// in place. Returns `order[old_key] = new_key` for every old key (merged-away
/// keys map to their survivor's new key).
fn renumber(clusters: &mut Vec<Cluster>) -> Vec<usize> {
    let resolve = |mut k: usize, cs: &[Cluster]| {
        while let Some(n) = cs[k].merged_into {
            if n == k {
                break;
            }
            k = n;
        }
        k
    };
    let mut alive: Vec<usize> = (0..clusters.len()).filter(|&k| clusters[k].alive()).collect();
    alive.sort_by_key(|&k| (clusters[k].first_ms, k));
    let mut new_of = vec![usize::MAX; clusters.len()];
    for (new, &old) in alive.iter().enumerate() {
        new_of[old] = new;
    }
    let order: Vec<usize> = (0..clusters.len())
        .map(|k| new_of[resolve(k, clusters)])
        .collect();
    let survivors: Vec<Cluster> = alive.iter().map(|&k| clusters[k].clone()).collect();
    *clusters = survivors;
    order
}

// ---- offline (re-detect) -----------------------------------------------------

/// Re-detect speakers for a finished meeting: embed each `(start_ms, end_ms)`
/// span of `samples`, cluster them with average-linkage agglomeration (to
/// exactly `num_speakers` when given, else until `OFFLINE_THRESHOLD`), fold
/// phantoms (auto mode only) and name clusters from voiceprints. Returns a key
/// per span — spans too short to embed take their nearest-in-time embedded
/// neighbor's key, `None` only when nothing in the meeting could be embedded —
/// plus `(key, centroid, n_segments, speech_ms)` per cluster. Keys are 0..k
/// by first appearance.
/// Cluster precomputed embeddings with the default model thresholds.
#[cfg(test)]
pub fn cluster_embeddings(
    embeddings: &[Option<Vec<f32>>],
    spans: &[(i64, i64)],
    num_speakers: Option<usize>,
    voiceprints: &[Voiceprint],
) -> (Vec<Option<usize>>, Vec<(usize, Vec<f32>, usize, i64)>) {
    cluster_embeddings_with(embeddings, spans, num_speakers, voiceprints, Params::default())
}

pub fn cluster_embeddings_with(
    embeddings: &[Option<Vec<f32>>],
    spans: &[(i64, i64)],
    num_speakers: Option<usize>,
    voiceprints: &[Voiceprint],
    p: Params,
) -> (Vec<Option<usize>>, Vec<(usize, Vec<f32>, usize, i64)>) {
    let idx: Vec<usize> = (0..embeddings.len())
        .filter(|&i| embeddings[i].is_some())
        .collect();
    if idx.is_empty() {
        return (vec![None; spans.len()], Vec::new());
    }
    let vecs: Vec<&[f32]> = idx.iter().map(|&i| embeddings[i].as_deref().unwrap()).collect();
    let sim = similarity_matrix(&vecs);

    // Auto: one pass at the configured threshold. Forced count: plain
    // agglomeration to n lets a few outlier segments survive as "speakers"
    // while real voices get merged, so instead cluster naturally, fold the
    // phantoms, then merge the closest survivors down to n — trying stricter
    // thresholds (finer clusters) until there are at least n real ones.
    let (mut clusters, labels) = match num_speakers.filter(|&n| n > 0) {
        None => {
            let labels = agglomerate(&sim, vecs.len(), None, p.offline);
            let mut cs = build_clusters(&labels, &vecs, &idx, spans);
            merge_similar(&mut cs, p.merge);
            fold_tiny(&mut cs, p.min_share, p.min_segments);
            (cs, labels)
        }
        Some(n) => {
            let mut result = None;
            let mut thr = p.offline;
            while thr <= 0.9 {
                let labels = agglomerate(&sim, vecs.len(), None, thr);
                let mut cs = build_clusters(&labels, &vecs, &idx, spans);
                fold_tiny(&mut cs, p.min_share, p.min_segments);
                let alive = cs.iter().filter(|c| c.alive()).count();
                let enough = alive >= n;
                result = Some((cs, labels));
                if enough {
                    break;
                }
                thr += 0.05;
            }
            let (mut cs, labels) = result.unwrap();
            while cs.iter().filter(|c| c.alive()).count() > n {
                merge_similar_pair(&mut cs);
            }
            (cs, labels)
        }
    };

    // Label per span: embedded spans by their agglomeration label, the rest by
    // their nearest-in-time embedded neighbor; then follow merges/renumbering.
    let mut span_label: Vec<Option<usize>> = vec![None; spans.len()];
    for (j, &i) in idx.iter().enumerate() {
        span_label[i] = Some(labels[j]);
    }
    for i in 0..spans.len() {
        if span_label[i].is_none() {
            span_label[i] = nearest_embedded(i, &idx, spans).and_then(|j| span_label[j]);
        }
    }
    let order = renumber(&mut clusters);
    for c in clusters.iter_mut() {
        c.name = match_voiceprint(&c.centroid, voiceprints);
    }
    let keys = span_label.iter().map(|l| l.map(|l| order[l])).collect();
    let cents = clusters
        .iter()
        .enumerate()
        .map(|(k, c)| (k, c.centroid.clone(), c.n_segments, c.speech_ms))
        .collect();
    (keys, cents)
}

fn nearest_embedded(i: usize, idx: &[usize], spans: &[(i64, i64)]) -> Option<usize> {
    let mid = (spans[i].0 + spans[i].1) / 2;
    idx.iter()
        .min_by_key(|&&j| ((spans[j].0 + spans[j].1) / 2 - mid).abs())
        .copied()
}

/// Clusters from agglomeration labels: centroids from the embedded spans,
/// segment counts / speech time from every span (unembeddable ones counted
/// with their nearest embedded neighbor).
fn build_clusters(labels: &[usize], vecs: &[&[f32]], idx: &[usize], spans: &[(i64, i64)]) -> Vec<Cluster> {
    let n_labels = labels.iter().copied().max().map_or(0, |m| m + 1);
    let mut clusters: Vec<Cluster> = (0..n_labels).map(|_| Cluster::empty(i64::MAX)).collect();
    let mut label_of_span: Vec<Option<usize>> = vec![None; spans.len()];
    for (j, &i) in idx.iter().enumerate() {
        clusters[labels[j]].add_embedding(vecs[j]);
        label_of_span[i] = Some(labels[j]);
    }
    for i in 0..spans.len() {
        let l = label_of_span[i]
            .or_else(|| nearest_embedded(i, idx, spans).and_then(|j| label_of_span[j]));
        if let Some(l) = l {
            let c = &mut clusters[l];
            c.n_segments += 1;
            c.speech_ms += (spans[i].1 - spans[i].0).max(0);
            c.first_ms = c.first_ms.min(spans[i].0);
        }
    }
    clusters
}

/// Merge the single most similar pair of alive clusters (by centroid).
fn merge_similar_pair(clusters: &mut [Cluster]) {
    let alive: Vec<usize> = (0..clusters.len()).filter(|&k| clusters[k].alive()).collect();
    let mut best: Option<(usize, usize, f32)> = None;
    for (i, &a) in alive.iter().enumerate() {
        for &b in &alive[i + 1..] {
            let s = if clusters[a].centroid.is_empty() || clusters[b].centroid.is_empty() {
                -1.0
            } else {
                cosine(&clusters[a].centroid, &clusters[b].centroid)
            };
            if best.map_or(true, |(_, _, bs)| s > bs) {
                best = Some((a, b, s));
            }
        }
    }
    if let Some((a, b, _)) = best {
        let (keep, drop) = if clusters[a].speech_ms >= clusters[b].speech_ms { (a, b) } else { (b, a) };
        merge_into(clusters, drop, keep);
    }
}

fn similarity_matrix(vecs: &[&[f32]]) -> Vec<f32> {
    let n = vecs.len();
    let mut sim = vec![0f32; n * n];
    for i in 0..n {
        for j in i + 1..n {
            let s = cosine(vecs[i], vecs[j]);
            sim[i * n + j] = s;
            sim[j * n + i] = s;
        }
    }
    sim
}

/// Average-linkage agglomerative clustering over a precomputed cosine
/// similarity matrix (`n`×`n`, row-major). Merges the most similar pair until
/// `target` clusters remain (when given) or no pair reaches `threshold`.
/// Returns a dense label per input.
fn agglomerate(sim0: &[f32], n: usize, target: Option<usize>, threshold: f32) -> Vec<usize> {
    let mut sim = sim0.to_vec();
    let mut size = vec![1usize; n];
    let mut active: Vec<bool> = vec![true; n];
    let mut parent: Vec<usize> = (0..n).collect();
    let mut count = n;
    loop {
        if let Some(t) = target {
            if count <= t.max(1) {
                break;
            }
        }
        let mut best: Option<(usize, usize, f32)> = None;
        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in i + 1..n {
                if active[j] {
                    let s = sim[i * n + j];
                    if best.map_or(true, |(_, _, b)| s > b) {
                        best = Some((i, j, s));
                    }
                }
            }
        }
        let Some((a, b, s)) = best else { break };
        if target.is_none() && s < threshold {
            break;
        }
        // Merge b into a.
        let (sa, sb) = (size[a] as f32, size[b] as f32);
        for k in 0..n {
            if active[k] && k != a && k != b {
                let v = (sa * sim[a * n + k] + sb * sim[b * n + k]) / (sa + sb);
                sim[a * n + k] = v;
                sim[k * n + a] = v;
            }
        }
        size[a] += size[b];
        active[b] = false;
        parent[b] = a;
        count -= 1;
    }
    // Dense labels from the union structure.
    let root = |mut i: usize| {
        while parent[i] != i {
            i = parent[i];
        }
        i
    };
    let mut label_of = std::collections::HashMap::new();
    (0..n)
        .map(|i| {
            let r = root(i);
            let next = label_of.len();
            *label_of.entry(r).or_insert(next)
        })
        .collect()
}

// ---- tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 32;

    /// Unit vector along axis `i`.
    fn axis(i: usize) -> Vec<f32> {
        let mut v = vec![0f32; DIM];
        v[i] = 1.0;
        v
    }

    /// `base` nudged by a deterministic pseudo-random perturbation of size
    /// `noise`, re-normalized.
    fn perturb(base: &[f32], seed: u64, noise: f32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v: Vec<f32> = base
            .iter()
            .map(|b| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let r = ((x >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
                b + noise * r
            })
            .collect();
        normalize(v).unwrap()
    }

    #[test]
    fn repeated_enrollments_are_grouped_by_person_and_weighted_by_samples() {
        let vps=vec![Voiceprint{name:"Priya".into(),embedding:axis(0)},Voiceprint{name:"priya".into(),embedding:axis(0)},Voiceprint{name:"Sam".into(),embedding:axis(1)}];
        assert_eq!(match_voiceprint(&axis(0),&vps).unwrap().to_lowercase(),"priya");
        let mixed=blend_weighted(&axis(0),2,&axis(1),8);
        assert!(cosine(&mixed,&axis(1))>cosine(&mixed,&axis(0)));
        let ambiguous=vec![Voiceprint{name:"Priya".into(),embedding:axis(0)},Voiceprint{name:"Sam".into(),embedding:axis(0)}];
        assert!(match_voiceprint(&axis(0),&ambiguous).is_none());
    }

    #[test]
    fn cosine_basics() {
        assert!((cosine(&axis(0), &axis(0)) - 1.0).abs() < 1e-6);
        assert!(cosine(&axis(0), &axis(1)).abs() < 1e-6);
        assert_eq!(cosine(&[], &[]), 0.0);
        assert_eq!(cosine(&[1.0, 0.0], &[1.0]), 0.0);
    }

    #[test]
    fn encode_decode_round_trip() {
        let v = vec![0.25f32, -1.5, 3.0e-7, f32::MAX];
        assert_eq!(decode(&encode(&v)), v);
        assert!(decode(&[1, 2, 3]).is_empty());
    }

    #[test]
    fn blend_moves_toward_new_and_stays_unit() {
        let b = blend(&axis(0), 1, &axis(1));
        let norm: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        assert!((cosine(&b, &axis(0)) - cosine(&b, &axis(1))).abs() < 1e-5);
        // A voiceprint built from many meetings moves less.
        let heavy = blend(&axis(0), 9, &axis(1));
        assert!(cosine(&heavy, &axis(0)) > cosine(&heavy, &axis(1)));
        // Zero history just adopts the new centroid.
        assert_eq!(blend(&axis(0), 0, &axis(2)), axis(2));
    }

    #[test]
    fn tracker_separates_distinct_voices_and_reuses_keys() {
        let mut t = SpeakerTracker::new(vec![]);
        let mut keys = Vec::new();
        // A B A B C A… with noisy embeddings.
        let seq = [0, 1, 0, 1, 2, 0, 2, 1, 0, 2];
        for (i, &spk) in seq.iter().enumerate() {
            let e = perturb(&axis(spk), i as u64, 0.15);
            keys.push(t.assign(Some(&e), i as i64 * 5000, i as i64 * 5000 + 4000));
        }
        // Same true speaker → same key, different speakers → different keys.
        for i in 0..seq.len() {
            for j in 0..seq.len() {
                assert_eq!(seq[i] == seq[j], keys[i] == keys[j], "segments {i} and {j}");
            }
        }
        assert_eq!(t.centroids().len(), 3);
    }

    #[test]
    fn short_segments_inherit_the_previous_speaker() {
        let mut t = SpeakerTracker::new(vec![]);
        // Nobody yet: a placeholder cluster 0 is opened…
        assert_eq!(t.assign(None, 0, 400), 0);
        // …and the first real voice adopts it instead of opening another.
        assert_eq!(t.assign(Some(&axis(3)), 500, 3000), 0);
        let b = t.assign(Some(&axis(4)), 3000, 6000);
        assert_ne!(b, 0);
        assert_eq!(t.assign(None, 6000, 6300), b);
        let c = t.centroids();
        let (_, _, n, ms) = c.iter().find(|(k, ..)| *k == b).unwrap();
        assert_eq!((*n, *ms), (2, 3300));
    }

    #[test]
    fn consolidate_merges_split_voice_and_folds_phantoms() {
        let mut t = SpeakerTracker::new(vec![]);
        let mut ms = 0;
        let mut push = |t: &mut SpeakerTracker, e: &[f32], dur: i64| {
            let k = t.assign(Some(e), ms, ms + dur);
            ms += dur + 500;
            k
        };
        let k_a = push(&mut t, &axis(0), 5000);
        for i in 0..20 {
            push(&mut t, &perturb(&axis(0), i, 0.2), 5000);
        }
        let k_b = push(&mut t, &axis(1), 5000);
        for i in 0..20 {
            push(&mut t, &perturb(&axis(1), 100 + i, 0.2), 5000);
        }
        // Phantom: one 2.5 s blip on its own axis (long enough to open a
        // cluster live, too little speech to survive consolidation).
        let k_p = push(&mut t, &axis(7), 2500);
        assert!(k_p != k_a && k_p != k_b);
        // A second, substantial cluster for speaker A (what online clustering
        // produces when A's early segments were noisy): same direction as A.
        let mut dup = Cluster::empty(ms);
        for i in 0..10 {
            dup.add_embedding(&perturb(&axis(0), 500 + i, 0.2));
        }
        dup.n_segments = 10;
        dup.speech_ms = 50_000;
        t.clusters.push(dup);
        let k_dup = t.clusters.len() - 1;

        assert_eq!(t.centroids().len(), 4);
        let remaps = t.consolidate();
        let after = t.centroids();
        assert_eq!(after.len(), 2, "A and B survive; duplicate merged, phantom folded");
        let map: std::collections::HashMap<usize, usize> = remaps.iter().copied().collect();
        let new_of = |k: usize| *map.get(&k).unwrap_or(&k);
        // A first appeared first → key 0; B → key 1.
        assert_eq!((new_of(k_a), new_of(k_b)), (0, 1));
        assert_eq!(new_of(k_dup), 0);
        assert!(new_of(k_p) < 2);
        // Only changed keys are reported.
        assert!(remaps.iter().all(|(o, n)| o != n));
        // Speech is conserved.
        let total: i64 = after.iter().map(|c| c.3).sum();
        assert_eq!(total, 42 * 5000 + 2500 + 50_000);
    }

    #[test]
    fn short_outliers_join_without_moving_the_centroid() {
        let mut t = SpeakerTracker::new(vec![]);
        let a = t.assign(Some(&axis(0)), 0, 4000);
        let before = t.centroids()[0].1.clone();
        // A 1.5 s segment unlike anyone: no new cluster, centroid unchanged.
        let k = t.assign(Some(&axis(6)), 4000, 5500);
        assert_eq!(k, a);
        assert_eq!(t.centroids().len(), 1);
        assert_eq!(t.centroids()[0].1, before);
        assert_eq!(t.centroids()[0].2, 2);
        // The same voice for 3 s does open one.
        assert_ne!(t.assign(Some(&axis(6)), 6000, 9000), a);
    }

    #[test]
    fn voiceprints_name_clusters_conservatively() {
        let vps = vec![
            Voiceprint { name: "Priya".into(), embedding: axis(0) },
            Voiceprint { name: "Sam".into(), embedding: axis(1) },
        ];
        assert_eq!(match_voiceprint(&perturb(&axis(0), 1, 0.2), &vps).as_deref(), Some("Priya"));
        // Halfway between two known voices: ambiguous → no name.
        let mid = normalize(vec![1.0, 1.0].into_iter().chain(vec![0.0; DIM - 2]).collect()).unwrap();
        assert_eq!(match_voiceprint(&mid, &vps), None);
        // Unknown voice → no name.
        assert_eq!(match_voiceprint(&axis(9), &vps), None);

        let mut t = SpeakerTracker::new(vps);
        let mut k = 0;
        for i in 0..4 {
            k = t.assign(Some(&perturb(&axis(1), i, 0.15)), i as i64 * 3000, i as i64 * 3000 + 2500);
        }
        assert_eq!(t.cluster_name(k).as_deref(), Some("Sam"));
    }

    #[test]
    fn voiceprints_keep_similar_known_voices_apart() {
        // Two people whose voices are 0.7 alike: without voiceprints the
        // second joins the first; with them, each gets their own named cluster.
        let a = axis(0);
        let b = normalize(a.iter().zip(axis(1)).map(|(x, y)| 0.7 * x + 0.714 * y).collect()).unwrap();
        let run = |vps: Vec<Voiceprint>| {
            let mut t = SpeakerTracker::new(vps);
            let mut keys = Vec::new();
            for i in 0..12u64 {
                let base = if i % 2 == 0 { &a } else { &b };
                let e = perturb(base, i, 0.03);
                keys.push(t.assign(Some(&e), i as i64 * 4000, i as i64 * 4000 + 3000));
            }
            (t, keys)
        };
        let (t, _) = run(vec![]);
        assert_eq!(t.centroids().len(), 1);
        let vps = vec![
            Voiceprint { name: "Ana".into(), embedding: a.clone() },
            Voiceprint { name: "Bea".into(), embedding: b.clone() },
        ];
        let (mut t, keys) = run(vps);
        assert_eq!(t.centroids().len(), 2);
        assert_eq!(t.cluster_name(keys[0]).as_deref(), Some("Ana"));
        assert_eq!(t.cluster_name(keys[1]).as_deref(), Some("Bea"));
        // Consolidation never merges two different known people.
        let remaps = t.consolidate();
        assert_eq!(t.centroids().len(), 2);
        let new_of = |k: usize| remaps.iter().find(|(o, _)| *o == k).map_or(k, |r| r.1);
        assert_eq!(t.cluster_name(new_of(keys[0])).as_deref(), Some("Ana"));
    }

    #[test]
    fn offline_clusters_by_threshold_or_forced_count() {
        let spkr = [0, 0, 1, 1, 2, 0, 2, 1, 0, 2, 1, 0];
        let mut embs: Vec<Option<Vec<f32>>> = spkr
            .iter()
            .enumerate()
            .map(|(i, &s)| Some(perturb(&axis(s), i as u64 + 50, 0.25)))
            .collect();
        // One span too short to embed, sitting next to a speaker-1 span.
        embs[3] = None;
        let spans: Vec<(i64, i64)> = (0..spkr.len() as i64).map(|i| (i * 4000, i * 4000 + 3000)).collect();

        let (keys, cents) = cluster_embeddings(&embs, &spans, None, &[]);
        assert_eq!(cents.len(), 3);
        for i in 0..spkr.len() {
            for j in 0..spkr.len() {
                if i == 3 || j == 3 {
                    continue;
                }
                assert_eq!(spkr[i] == spkr[j], keys[i] == keys[j]);
            }
        }
        // The unembeddable span took its nearest neighbor's speaker.
        assert!(keys[3] == keys[2] || keys[3] == keys[4]);
        // Keys are by first appearance.
        assert_eq!(keys[0], Some(0));
        assert_eq!(keys[2], Some(1));

        let (_, forced) = cluster_embeddings(&embs, &spans, Some(2), &[]);
        assert_eq!(forced.len(), 2);

        let (none, empty) = cluster_embeddings(&[None, None], &[(0, 100), (200, 300)], None, &[]);
        assert_eq!(none, vec![None, None]);
        assert!(empty.is_empty());
    }
}
