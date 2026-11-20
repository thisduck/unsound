//! Where the system-audio echo sits in the mic: band-limited GCC-PHAT over a
//! few seconds of both streams. Only lags at which the reference leads (or
//! trails by a few ms) are searched — a mic that leads the system audio by
//! a network round trip is the remote side playing *this* user back, and
//! cancelling along that path would remove the user's own voice.

use realfft::num_complex::Complex32 as C;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

const RATE: usize = 16_000;
/// Longest analysis window (4.1 s).
pub const WINDOW: usize = 1 << 16;
/// Shortest window worth estimating from.
pub const MIN_WINDOW: usize = RATE * 3 / 2;
const FFT: usize = WINDOW * 2;
/// Search range: the reference may trail the echo by this much…
pub const MAX_TRAIL: i64 = RATE as i64 * 60 / 1000;
/// …or lead it by this much.
pub const MAX_LEAD: i64 = RATE as i64 * 3 / 2;
const BAND: (f32, f32) = (200.0, 6000.0);
/// Frame and level for "the far end is talking" (-60 dBFS mean square).
const FRAME: usize = 320;
const ACTIVE: f32 = 1e-6;

pub struct Estimate {
    /// Samples by which the reference leads its echo in the mic
    /// (sub-sample, from a parabola through the peak).
    pub lag: f64,
    /// Where in the window (samples from its start) the far end was
    /// talking, on average: the estimate describes that moment.
    pub center: f64,
    /// Peak height in standard deviations of the correlation over the
    /// searched range; uncorrelated audio stays near 4-5.
    pub z: f32,
}

pub struct DelayEstimator {
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    a: Vec<f32>,
    b: Vec<f32>,
    sa: Vec<C>,
    sb: Vec<C>,
}

impl DelayEstimator {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(FFT);
        let inv = planner.plan_fft_inverse(FFT);
        Self {
            a: vec![0.0; FFT],
            b: vec![0.0; FFT],
            sa: fwd.make_output_vec(),
            sb: fwd.make_output_vec(),
            fwd,
            inv,
        }
    }

    /// `mic` and `reference` cover the same stretch of the meeting clock.
    /// None when the far end was too quiet to say anything.
    pub fn estimate(&mut self, mic: &[f32], reference: &[f32]) -> Option<Estimate> {
        let n = mic.len().min(reference.len()).min(WINDOW);
        if n < MIN_WINDOW {
            return None;
        }
        let (mic, reference) = (&mic[mic.len() - n..], &reference[reference.len() - n..]);
        let frames = n / FRAME;
        let active_at: Vec<usize> = reference
            .chunks_exact(FRAME)
            .enumerate()
            .filter(|(_, f)| f.iter().map(|v| v * v).sum::<f32>() / FRAME as f32 > ACTIVE)
            .map(|(i, _)| i)
            .collect();
        let active = active_at.len();
        let mic_ms = mic.iter().map(|v| v * v).sum::<f32>() / n as f32;
        if active * 5 < frames || mic_ms < 1e-9 {
            return None;
        }
        self.a.iter_mut().for_each(|v| *v = 0.0);
        self.b.iter_mut().for_each(|v| *v = 0.0);
        self.a[..n].copy_from_slice(mic);
        self.b[..n].copy_from_slice(reference);
        let _ = self.fwd.process(&mut self.a, &mut self.sa);
        let _ = self.fwd.process(&mut self.b, &mut self.sb);
        let lo = (BAND.0 * FFT as f32 / RATE as f32) as usize;
        let hi = (BAND.1 * FFT as f32 / RATE as f32) as usize;
        let mut mean = 0.0f32;
        for k in lo..hi {
            self.sa[k] = self.sa[k] * self.sb[k].conj();
            mean += self.sa[k].norm();
        }
        mean /= (hi - lo) as f32;
        let eps = mean * 1e-2 + 1e-20;
        for (k, c) in self.sa.iter_mut().enumerate() {
            if k < lo || k >= hi {
                *c = C::default();
            } else {
                *c /= c.norm() + eps;
            }
        }
        let _ = self.inv.process(&mut self.sa, &mut self.a);
        // a[L] = sum_t mic[t] * ref[t - L]; negative lags wrap to the end.
        let at = |l: i64| -> f32 {
            let i = if l >= 0 { l as usize } else { FFT - (-l) as usize };
            self.a[i].abs()
        };
        let max_lead = MAX_LEAD.min(n as i64 / 2);
        let mut best = (0i64, 0.0f32);
        let mut sum = 0.0f64;
        let mut sum2 = 0.0f64;
        let mut count = 0.0f64;
        for l in -MAX_TRAIL..=max_lead {
            let v = at(l);
            sum += v as f64;
            sum2 += (v as f64) * (v as f64);
            count += 1.0;
            if v > best.1 {
                best = (l, v);
            }
        }
        let mean = sum / count;
        let sd = (sum2 / count - mean * mean).max(1e-30).sqrt();
        // Reflections a few ms behind the direct path can peak as high once
        // whitened; take the earliest peak close to the top, so successive
        // estimates follow the same arrival.
        for l in (-MAX_TRAIL + 1)..best.0 {
            let v = at(l);
            if v >= 0.6 * best.1 && v >= at(l - 1) && v >= at(l + 1) {
                best = (l, v);
                break;
            }
        }
        let (l, c, r) = (at(best.0 - 1) as f64, best.1 as f64, at(best.0 + 1) as f64);
        let curve = l - 2.0 * c + r;
        let frac = if curve < 0.0 { (0.5 * (l - r) / curve).clamp(-0.5, 0.5) } else { 0.0 };
        let center = (active_at.iter().sum::<usize>() as f64 / active as f64 + 0.5) * FRAME as f64
            + (reference.len() - n) as f64;
        Some(Estimate {
            center,
            lag: best.0 as f64 + frac,
            z: ((best.1 as f64 - mean) / sd) as f32,
        })
    }
}
