//! Content-free, bounded diagnostics. Quantiles cover the latest 512 batches;
//! counters/maxima cover this process. Restart recovery starts a new measurement.
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub fn unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
#[derive(Default)]
struct Distribution {
    recent: VecDeque<u64>,
    count: u64,
    sum: u64,
    max: u64,
}
impl Distribution {
    fn add(&mut self, value: u64) {
        if self.recent.len() == 512 {
            self.recent.pop_front();
        }
        self.recent.push_back(value);
        self.count += 1;
        self.sum = self.sum.saturating_add(value);
        self.max = self.max.max(value);
    }
    fn report(&self) -> Timing {
        let mut v: Vec<_> = self.recent.iter().copied().collect();
        v.sort_unstable();
        let percentile = |p: usize| {
            v.get((v.len() * p).div_ceil(100).saturating_sub(1))
                .copied()
        };
        Timing {
            count: self.count,
            p50_ms: percentile(50),
            p95_ms: percentile(95),
            max_ms: self.max,
            total_ms: self.sum,
        }
    }
}
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Timing {
    count: u64,
    p50_ms: Option<u64>,
    p95_ms: Option<u64>,
    max_ms: u64,
    total_ms: u64,
}
#[derive(Default)]
struct Run {
    inference: Distribution,
    latency: Distribution,
    checkpoint: Distribution,
    audio_ms: u64,
    peak_jobs: i64,
    peak_audio_ms: i64,
    model_ready_ms: Option<u64>,
    stop_ms: Option<u64>,
    drain_start: Option<Instant>,
    drain_ms: Option<u64>,
    recovery_start: Option<Instant>,
    recovery_ms: Option<u64>,
}
#[derive(Default)]
pub struct Metrics(Mutex<VecDeque<(String, Run)>>);
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    pub inference: Timing,
    pub queue_to_commit: Timing,
    pub checkpoint: Timing,
    pub processed_audio_ms: u64,
    pub real_time_factor: Option<f64>,
    pub peak_backlog_jobs: i64,
    pub peak_backlog_audio_ms: i64,
    pub model_ready_ms: Option<u64>,
    pub stop_ms: Option<u64>,
    pub drain_ms: Option<u64>,
    pub recovery_ms: Option<u64>,
}
impl Metrics {
    fn with(&self, id: &str, apply: impl FnOnce(&mut Run)) {
        let mut runs = self.0.lock().unwrap();
        if !runs.iter().any(|(key, _)| key == id) {
            if runs.len() == 32 {
                runs.pop_front();
            }
            runs.push_back((id.into(), Run::default()));
        }
        apply(&mut runs.iter_mut().find(|(key, _)| key == id).unwrap().1);
    }
    pub fn begin(&self, id: &str, recovering: bool) {
        self.with(id, |r| {
            if recovering {
                r.recovery_start = Some(Instant::now());
            }
        });
    }
    pub fn batch(&self, id: &str, elapsed: u64, audio_ms: u64, queued_at: i64) {
        self.with(id, |r| {
            r.inference.add(elapsed);
            r.audio_ms += audio_ms;
            if queued_at > 0 {
                r.latency
                    .add(unix_ms().saturating_sub(queued_at).max(0) as u64);
            }
        });
    }
    pub fn model_ready(&self, id: &str, elapsed: u64) {
        self.with(id, |r| {
            r.model_ready_ms = Some(r.model_ready_ms.unwrap_or(0).max(elapsed))
        });
    }
    pub fn checkpoint(&self, id: &str, elapsed: u64) {
        self.with(id, |r| r.checkpoint.add(elapsed));
    }
    pub fn backlog(&self, id: &str, jobs: i64, audio_ms: i64) {
        self.with(id, |r| {
            r.peak_jobs = r.peak_jobs.max(jobs);
            r.peak_audio_ms = r.peak_audio_ms.max(audio_ms);
        });
    }
    pub fn stopped(&self, id: &str, elapsed: u64) {
        self.with(id, |r| {
            r.stop_ms = Some(elapsed);
            r.drain_start = Some(Instant::now());
        });
    }
    pub fn complete(&self, id: &str) {
        self.with(id, |r| {
            r.drain_ms = r
                .drain_start
                .take()
                .map(|t| t.elapsed().as_millis() as u64)
                .or(r.drain_ms);
            r.recovery_ms = r
                .recovery_start
                .take()
                .map(|t| t.elapsed().as_millis() as u64)
                .or(r.recovery_ms);
        });
    }
    pub fn report(&self, id: &str) -> Option<Report> {
        let runs = self.0.lock().unwrap();
        let r = &runs.iter().find(|(key, _)| key == id)?.1;
        Some(Report {
            inference: r.inference.report(),
            queue_to_commit: r.latency.report(),
            checkpoint: r.checkpoint.report(),
            processed_audio_ms: r.audio_ms,
            real_time_factor: (r.audio_ms > 0).then(|| r.inference.sum as f64 / r.audio_ms as f64),
            peak_backlog_jobs: r.peak_jobs,
            peak_backlog_audio_ms: r.peak_audio_ms,
            model_ready_ms: r.model_ready_ms,
            stop_ms: r.stop_ms,
            drain_ms: r.drain_ms,
            recovery_ms: r.recovery_ms,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagnostics_are_bounded_and_old_samples_do_not_distort_recent_percentiles() {
        let mut d = Distribution::default();
        d.add(50_000);
        for _ in 0..512 {
            d.add(20);
        }
        assert_eq!(d.recent.len(), 512);
        assert_eq!(d.report().p95_ms, Some(20));
        assert_eq!(d.report().max_ms, 50_000);
        let metrics = Metrics::default();
        for n in 0..100 {
            metrics.batch(&n.to_string(), 100, 1000, 0);
        }
        assert_eq!(metrics.0.lock().unwrap().len(), 32);
        assert!(metrics.report("0").is_none());
        assert_eq!(metrics.report("99").unwrap().real_time_factor, Some(0.1));
    }
}
