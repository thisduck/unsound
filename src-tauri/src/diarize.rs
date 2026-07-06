//! Offline speaker diarization via sherpa-onnx — figuring out who-said-what
//! within the system-audio ("them") channel, so remote participants become
//! Speaker 1, Speaker 2, … instead of one undifferentiated "Them".
//!
//! Needs two models (downloaded on demand, like the whisper/llama ones): a
//! pyannote segmentation model and a speaker-embedding model. Everything runs
//! locally — same privacy contract as the rest of the app.

use sherpa_rs::diarize::{Diarize, DiarizeConfig};
use std::path::Path;

/// A diarized span: `[start, end)` seconds attributed to `speaker` (a small
/// integer cluster id, 0-based).
pub struct Span {
    pub start: f32,
    pub end: f32,
    pub speaker: i32,
}

/// Cluster 16 kHz mono `samples` into per-speaker spans. When `num_speakers` is
/// `Some(n)` the count is forced (far more reliable when the user knows it);
/// otherwise it's auto-detected by `threshold` (larger → fewer speakers).
pub fn diarize(
    segmentation: &Path,
    embedding: &Path,
    samples: Vec<f32>,
    threshold: f32,
    num_speakers: Option<i32>,
) -> Result<Vec<Span>, String> {
    let config = DiarizeConfig {
        // sherpa uses num_clusters when > 0, else falls back to threshold.
        num_clusters: Some(num_speakers.filter(|&n| n > 0).unwrap_or(-1)),
        threshold: Some(threshold),
        // Over-splitting (many phantom speakers) comes from short, unstable
        // segments. Drop sub-0.5s blips and bridge sub-0.5s gaps so a single
        // speaker's turn stays one segment instead of fragmenting into many.
        min_duration_on: Some(0.5),
        min_duration_off: Some(0.5),
        provider: None,
        debug: false,
    };
    let mut d = Diarize::new(segmentation, embedding, config).map_err(|e| e.to_string())?;
    let segs = d.compute(samples, None).map_err(|e| e.to_string())?;
    Ok(segs
        .into_iter()
        .map(|s| Span {
            start: s.start,
            end: s.end,
            speaker: s.speaker,
        })
        .collect())
}
