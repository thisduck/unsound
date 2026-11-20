//! Partitioned-block frequency-domain adaptive filter (MDF) with two-path
//! control: a background filter adapts every block, a foreground copy makes
//! the output and only takes the background's weights when they do better,
//! so double-talk that throws the adaptation off never reaches the output.
//! The step size follows the estimated residual-echo ratio rather than a
//! fixed rate, which is what keeps near-end speech from being learned as
//! echo. After J.-M. Valin, "On adjusting the learning rate in frequency
//! domain echo cancellation with double-talk" (the Speex MDF design).
//!
//! Signals are handled at 16-bit scale (x 32768) with a forward FFT
//! normalized by 1/(2N), so the regularizers below are in those units.

use realfft::num_complex::Complex32 as C;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// Block (hop) size: 16 ms at 16 kHz.
pub const N: usize = 256;
/// Partitions: N * P taps = 320 ms of echo path.
pub const P: usize = 20;
pub const TAPS: usize = N * P;
const F: usize = 2 * N;
const BINS: usize = N + 1;
pub const SCALE: f32 = 32768.0;

const MIN_LEAK: f64 = 0.005;
const VAR1_UPDATE: f64 = 0.5;
const VAR2_UPDATE: f64 = 0.25;
const VAR_BACKTRACK: f64 = 4.0;
/// Step size before the filter has adapted (Speex uses 0.25; speech at 16 kHz
/// with a 320 ms filter converges markedly faster at 0.5 in the harness).
const INITIAL_RATE: f64 = 0.5;

pub struct Fft {
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    time: Vec<f32>,
    spec: Vec<C>,
}

impl Fft {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        Self {
            fwd: planner.plan_fft_forward(F),
            inv: planner.plan_fft_inverse(F),
            time: vec![0.0; F],
            spec: vec![C::default(); BINS],
        }
    }

    /// Spectrum of `x` (2N samples), normalized by 1/(2N).
    fn forward(&mut self, x: &[f32], out: &mut [C]) {
        self.time.copy_from_slice(x);
        let _ = self.fwd.process(&mut self.time, out);
        let k = 1.0 / F as f32;
        for c in out.iter_mut() {
            *c *= k;
        }
    }

    /// Time signal (2N samples) of a spectrum made by `forward`.
    fn inverse(&mut self, x: &[C], out: &mut [f32]) {
        self.spec.copy_from_slice(x);
        self.spec[0].im = 0.0;
        self.spec[N].im = 0.0;
        let _ = self.inv.process(&mut self.spec, out);
    }
}

pub struct Mdf {
    fft: Fft,
    /// Reference spectra, newest first (ring indexed from `head`).
    x: Vec<Vec<C>>,
    head: usize,
    w_bg: Vec<Vec<C>>,
    w_fg: Vec<Vec<C>>,
    power: Vec<f32>,
    eh: Vec<f32>,
    yh: Vec<f32>,
    pey: f64,
    pyy: f64,
    davg1: f64,
    davg2: f64,
    dvar1: f64,
    dvar2: f64,
    /// Linear-regression estimate of how much echo the filter misses.
    pub leak: f64,
    adapted: bool,
    sum_adapt: f64,
    /// Energy of the newest N reference samples.
    sxx: f64,
    /// Consecutive blocks in which the output was much louder than the mic.
    screwed: u32,
    prop: Vec<f32>,
    // scratch
    t: Vec<f32>,
    y_fg: Vec<f32>,
    y_bg: Vec<f32>,
    e_bg: Vec<f32>,
    acc: Vec<C>,
    e_spec: Vec<C>,
    y_spec: Vec<C>,
    step: Vec<f32>,
}

/// What one block did.
pub struct Block {
    /// Energy of the foreground echo estimate (16-bit scale).
    pub echo: f64,
}

impl Mdf {
    pub fn new() -> Self {
        let zeros = || vec![vec![C::default(); BINS]; P];
        Self {
            fft: Fft::new(),
            x: zeros(),
            head: 0,
            w_bg: zeros(),
            w_fg: zeros(),
            power: vec![0.0; BINS],
            eh: vec![0.0; BINS],
            yh: vec![0.0; BINS],
            pey: 1.0,
            pyy: 1.0,
            davg1: 0.0,
            davg2: 0.0,
            dvar1: 0.0,
            dvar2: 0.0,
            leak: 0.0,
            adapted: false,
            sum_adapt: 0.0,
            sxx: 0.0,
            screwed: 0,
            prop: vec![0.99 / P as f32; P],
            t: vec![0.0; F],
            y_fg: vec![0.0; N],
            y_bg: vec![0.0; N],
            e_bg: vec![0.0; N],
            acc: vec![C::default(); BINS],
            e_spec: vec![C::default(); BINS],
            y_spec: vec![C::default(); BINS],
            step: vec![0.0; BINS],
        }
    }

    /// Forget everything learned (weights and statistics).
    pub fn reset(&mut self) {
        let fft = std::mem::replace(&mut self.fft, Fft::new());
        *self = Self::new();
        self.fft = fft;
    }

    pub fn adapted(&self) -> bool {
        self.adapted
    }

    /// Shift in the newest reference frame: the previous N samples followed
    /// by the N aligned with the next mic block (16-bit scale).
    pub fn push_reference(&mut self, frame: &[f32]) {
        self.head = (self.head + P - 1) % P;
        self.sxx = frame[N..].iter().map(|v| (v * v) as f64).sum();
        let mut spec = std::mem::take(&mut self.x[self.head]);
        self.fft.forward(frame, &mut spec);
        self.x[self.head] = spec;
    }

    fn xp(&self, p: usize) -> &[C] {
        &self.x[(self.head + p) % P]
    }

    fn respond(&mut self, w: Foreground, out: &mut [f32]) {
        self.acc.iter_mut().for_each(|c| *c = C::default());
        for p in 0..P {
            let x = &self.x[(self.head + p) % P];
            let wp = match w {
                Foreground::Yes => &self.w_fg[p],
                Foreground::No => &self.w_bg[p],
            };
            for k in 0..BINS {
                self.acc[k] += wp[k] * x[k];
            }
        }
        let acc = std::mem::take(&mut self.acc);
        self.fft.inverse(&acc, &mut self.t);
        self.acc = acc;
        out.copy_from_slice(&self.t[N..]);
    }

    /// Cancel one block: `y` is the mic (16-bit scale), `out` receives
    /// mic minus the foreground echo estimate. With `adapt` false the
    /// weights are left alone (reference not trustworthy for this block).
    pub fn cancel(&mut self, y: &[f32], out: &mut [f32], adapt: bool) -> Block {
        let mut y_fg = std::mem::take(&mut self.y_fg);
        let mut y_bg = std::mem::take(&mut self.y_bg);
        self.respond(Foreground::Yes, &mut y_fg);
        self.respond(Foreground::No, &mut y_bg);

        let mut sff = 0.0f64;
        let mut see = 0.0f64;
        let mut dbf = 10.0f64;
        let mut sdd = 0.0f64;
        for i in 0..N {
            let ef = y[i] - y_fg[i];
            let eb = y[i] - y_bg[i];
            out[i] = ef;
            self.e_bg[i] = eb;
            sff += (ef * ef) as f64;
            see += (eb * eb) as f64;
            dbf += ((ef - eb) * (ef - eb)) as f64;
            sdd += (y[i] * y[i]) as f64;
        }

        // Output much louder than the input for ~0.8 s: the filters are
        // adding echo (the path changed, e.g. headphones plugged in). Start over.
        if sff / 4.0 > sdd + N as f64 * 10_000.0 / 64.0 {
            self.screwed += 1;
        } else {
            self.screwed = 0;
        }
        if self.screwed >= 50 || !sff.is_finite() || !see.is_finite() {
            self.screwed = 0;
            self.reset_weights();
            out.copy_from_slice(y);
            self.y_fg = y_fg;
            self.y_bg = y_bg;
            return Block { echo: 0.0 };
        }

        // Two-path logic: promote the background when it is reliably better,
        // fall back to the foreground when the background went astray.
        let d = sff - see;
        self.davg1 = 0.6 * self.davg1 + 0.4 * d;
        self.davg2 = 0.85 * self.davg2 + 0.15 * d;
        self.dvar1 = 0.36 * self.dvar1 + 0.16 * sff * dbf;
        self.dvar2 = 0.7225 * self.dvar2 + 0.0225 * sff * dbf;
        let update = d * d.abs() > sff * dbf
            || self.davg1 * self.davg1.abs() > VAR1_UPDATE * self.dvar1
            || self.davg2 * self.davg2.abs() > VAR2_UPDATE * self.dvar2;
        if update {
            self.davg1 = 0.0;
            self.davg2 = 0.0;
            self.dvar1 = 0.0;
            self.dvar2 = 0.0;
            for p in 0..P {
                self.w_fg[p].copy_from_slice(&self.w_bg[p]);
            }
            // Crossfade so the switch doesn't click.
            for i in 0..N {
                let a = (i + 1) as f32 / N as f32;
                out[i] = (1.0 - a) * out[i] + a * self.e_bg[i];
            }
            y_fg.copy_from_slice(&y_bg);
        } else {
            let reset = -d * d.abs() > VAR_BACKTRACK * sff * dbf
                || -self.davg1 * self.davg1.abs() > VAR_BACKTRACK * self.dvar1
                || -self.davg2 * self.davg2.abs() > VAR_BACKTRACK * self.dvar2;
            if reset {
                for p in 0..P {
                    self.w_bg[p].copy_from_slice(&self.w_fg[p]);
                }
                y_bg.copy_from_slice(&y_fg);
                for i in 0..N {
                    self.e_bg[i] = y[i] - y_bg[i];
                }
                see = sff;
                self.davg1 = 0.0;
                self.davg2 = 0.0;
                self.dvar1 = 0.0;
                self.dvar2 = 0.0;
            }
        }

        let echo: f64 = y_fg.iter().map(|v| (v * v) as f64).sum();
        if adapt {
            self.adapt(&y_bg, see);
        }
        self.y_fg = y_fg;
        self.y_bg = y_bg;
        Block { echo }
    }

    fn adapt(&mut self, y_bg: &[f32], see: f64) {
        let see = see.max(N as f64 * 100.0 / 64.0);
        // Spectra of the background error and echo estimate.
        self.t[..N].iter_mut().for_each(|v| *v = 0.0);
        self.t[N..].copy_from_slice(&self.e_bg);
        let t = std::mem::take(&mut self.t);
        let mut e_spec = std::mem::take(&mut self.e_spec);
        self.fft.forward(&t, &mut e_spec);
        let mut t = t;
        t[N..].copy_from_slice(y_bg);
        let mut y_spec = std::mem::take(&mut self.y_spec);
        self.fft.forward(&t, &mut y_spec);
        self.t = t;

        let sxx = self.sxx;
        let mut syy = 0.0f64;
        let mut sey = 0.0f64;
        for i in 0..N {
            syy += (y_bg[i] * y_bg[i]) as f64;
            sey += (self.e_bg[i] * y_bg[i]) as f64;
        }

        let ss = 0.35 / P as f32;
        let spec_avg = N as f32 / 16_000.0;
        let mut pey = 0.0f64;
        let mut pyy = 0.0f64;
        for k in 0..BINS {
            let xf = self.xp(0)[k].norm_sqr();
            self.power[k] = (1.0 - ss) * self.power[k] + 1.0 + ss * xf;
            let rf = e_spec[k].norm_sqr();
            let yf = y_spec[k].norm_sqr();
            let eh = (rf - self.eh[k]) as f64;
            let yh = (yf - self.yh[k]) as f64;
            pey += eh * yh;
            pyy += yh * yh;
            self.eh[k] = (1.0 - spec_avg) * self.eh[k] + spec_avg * rf;
            self.yh[k] = (1.0 - spec_avg) * self.yh[k] + spec_avg * yf;
        }
        let pyy = pyy.sqrt().max(1e-12);
        let pey = pey / pyy;
        let beta0 = 2.0 * N as f64 / 16_000.0;
        let beta_max = 0.5 * N as f64 / 16_000.0;
        let alpha = (beta0 * syy).min(beta_max * see) / see;
        self.pey = (1.0 - alpha) * self.pey + alpha * pey;
        self.pyy = (1.0 - alpha) * self.pyy + alpha * pyy;
        if self.pyy < 1.0 {
            self.pyy = 1.0;
        }
        self.pey = self.pey.clamp(MIN_LEAK * self.pyy, self.pyy);
        self.leak = self.pey / self.pyy;
        if self.leak > 0.5 {
            self.leak = 1.0;
        }

        let mut rer = (0.0001 * sxx + 3.0 * self.leak * syy) / see;
        let floor = sey * sey / (1.0 + see * syy);
        if rer < floor {
            rer = floor;
        }
        rer = rer.min(0.5);

        if !self.adapted && self.sum_adapt > 6.0 * P as f64 {
            self.adapted = true;
        }
        if self.adapted {
            for k in 0..BINS {
                let e = (e_spec[k].norm_sqr() + 1.0) as f64;
                let mut r = self.leak * y_spec[k].norm_sqr() as f64;
                if r > 0.5 * e {
                    r = 0.5 * e;
                }
                r = 0.7 * r + 0.3 * rer * e;
                self.step[k] = (r / (e * (self.power[k] as f64 + 10.0))) as f32;
            }
        } else {
            let mut rate = 0.0;
            if sxx > N as f64 * 1000.0 / 64.0 {
                rate = (INITIAL_RATE * sxx).min(INITIAL_RATE * see) / see;
            }
            for k in 0..BINS {
                self.step[k] = (rate / (self.power[k] as f64 + 10.0)) as f32;
            }
            self.sum_adapt += rate;
        }

        self.adjust_prop();
        for p in 0..P {
            let x = &self.x[(self.head + p) % P];
            let g = self.prop[p];
            for k in 0..BINS {
                self.acc[k] = x[k].conj() * e_spec[k] * (self.step[k] * g);
            }
            // Constrain to N taps (no circular wrap-around).
            let acc = std::mem::take(&mut self.acc);
            self.fft.inverse(&acc, &mut self.t);
            self.t[N..].iter_mut().for_each(|v| *v = 0.0);
            let t = std::mem::take(&mut self.t);
            let mut acc = acc;
            self.fft.forward(&t, &mut acc);
            self.t = t;
            for k in 0..BINS {
                self.w_bg[p][k] += acc[k];
            }
            self.acc = acc;
        }
        self.e_spec = e_spec;
        self.y_spec = y_spec;
    }

    /// Step share per partition, proportional to its weight energy
    /// (converges faster on a sparse path such as a laptop's direct echo).
    fn adjust_prop(&mut self) {
        let mut max = 1.0f32;
        for p in 0..P {
            let s: f32 = 1.0 + self.w_bg[p].iter().map(|c| c.norm_sqr()).sum::<f32>();
            self.prop[p] = s.sqrt();
            max = max.max(self.prop[p]);
        }
        let mut sum = 0.0;
        for p in 0..P {
            self.prop[p] += 0.1 * max;
            sum += self.prop[p];
        }
        for p in 0..P {
            self.prop[p] = 0.99 * self.prop[p] / sum;
        }
    }

    fn reset_weights(&mut self) {
        for p in 0..P {
            self.w_bg[p].iter_mut().for_each(|c| *c = C::default());
            self.w_fg[p].iter_mut().for_each(|c| *c = C::default());
        }
        self.davg1 = 0.0;
        self.davg2 = 0.0;
        self.dvar1 = 0.0;
        self.dvar2 = 0.0;
    }

    fn path_taps(&mut self, w: &[Vec<C>]) -> Vec<f32> {
        let mut h = vec![0.0; TAPS];
        for (p, wp) in w.iter().enumerate() {
            self.fft.inverse(wp, &mut self.t);
            h[p * N..(p + 1) * N].copy_from_slice(&self.t[..N]);
        }
        h
    }

    fn path_weights(&mut self, h: &[f32]) -> Vec<Vec<C>> {
        let mut w = vec![vec![C::default(); BINS]; P];
        for (p, wp) in w.iter_mut().enumerate() {
            self.t[..N].copy_from_slice(&h[p * N..(p + 1) * N]);
            self.t[N..].iter_mut().for_each(|v| *v = 0.0);
            let t = std::mem::take(&mut self.t);
            self.fft.forward(&t, wp);
            self.t = t;
        }
        w
    }

    /// Foreground taps in the time domain (TAPS long).
    pub fn taps(&mut self) -> Vec<f32> {
        let w = std::mem::take(&mut self.w_fg);
        let h = self.path_taps(&w);
        self.w_fg = w;
        h
    }

    /// Load the same taps into both paths.
    pub fn set_taps(&mut self, h: &[f32]) {
        self.w_fg = self.path_weights(h);
        self.w_bg = self.w_fg.clone();
    }

    /// The alignment grew by `delta` samples (the reference now leads by
    /// that much more): move both paths' taps `delta` earlier, which leaves
    /// the echo estimate unchanged.
    pub fn shift(&mut self, delta: i64) {
        let moved = |h: Vec<f32>| -> Vec<f32> {
            (0..TAPS as i64)
                .map(|j| {
                    let src = j + delta;
                    if (0..TAPS as i64).contains(&src) {
                        h[src as usize]
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        for fg in [true, false] {
            let w = std::mem::take(if fg { &mut self.w_fg } else { &mut self.w_bg });
            let h = moved(self.path_taps(&w));
            let w = self.path_weights(&h);
            *(if fg { &mut self.w_fg } else { &mut self.w_bg }) = w;
        }
    }
}

#[derive(Clone, Copy)]
enum Foreground {
    Yes,
    No,
}
