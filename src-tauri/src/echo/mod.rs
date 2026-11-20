//! Echo cancellation for meetings on laptop speakers: the remote voices
//! played through the speakers leak into the mic and would be transcribed a
//! second time as "me". System audio is the far-end reference.
//!
//! Signal layer (`Canceller`): a GCC-PHAT delay estimate aligns the reference
//! with its echo (and the filter's own taps then track clock drift), a
//! two-path MDF adaptive filter removes the echo, and a block gate silences
//! what remains only where the filter accounts for nearly all of the mic's
//! energy. Until the delay locks (no system audio, headphones, a fresh
//! restart) the mic passes through bit-for-bit.
//!
//! Decision layer (`guard`): a transcribed "me" line is dropped only when its
//! saved audio is largely a filtered copy of the system audio and, unless
//! that is overwhelming, its words repeat an overlapping "them" line.

mod delay;
mod filter;
pub mod guard;
#[cfg(test)]
mod tests;

use delay::DelayEstimator;
use filter::{Mdf, N, SCALE, TAPS};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

const RATE: u64 = 16_000;
/// Taps kept ahead of the estimated echo onset, so estimation error and
/// a reference that trails slightly still land inside the filter.
const MARGIN: i64 = RATE as i64 * 40 / 1000;
/// Mic held back at most this long waiting for its reference.
const MAX_HOLD: u64 = RATE / 2;
/// From lock until the filter converges, output is held back (at most this
/// long) and then re-run through the converged filter.
const SETTLE_MAX: usize = RATE as usize * 8;
/// Re-estimate the delay after this much new mic audio.
const ESTIMATE_EVERY: u64 = RATE;
/// A peak this many deviations above the correlation floor is an echo
/// candidate; two agreeing candidates (or one this strong) lock.
const Z_CANDIDATE: f32 = 9.0;
const Z_SURE: f32 = 18.0;
/// Agreement between two delay estimates (2 ms).
const AGREE: f64 = 32.0;
/// A locked canceller re-aligns only when the echo moved further than this
/// (reflections a few ms from the direct path also produce peaks).
const REALIGN: f64 = RATE as f64 * 0.012;
/// Tap positions kept for the drift fit, and the span needed before fitting.
const TRACK_SPAN: f64 = 30.0 * RATE as f64;
const FIT_SPAN: f64 = 5.0 * RATE as f64;
/// Echo reduction (dB) at which the filter's taps are trusted for drift.
const CONVERGED: f64 = 8.0;
/// Drift (samples per second) worth correcting…
const MIN_DRIFT: f64 = 0.02;
/// …provided the path moved this many samples across the fitted span.
const MIN_MOVE: f64 = 3.0;
/// Drift beyond 2000 ppm is not believed.
const MAX_DRIFT: f64 = 2e-3;
/// Gate: a block is echo when the filter removed at least this share of its
/// energy (12 dB) and the echo estimate explains it.
const GATE_RATIO: f64 = 15.85;
const GATE_FLOOR: f32 = 0.05;
/// Blocks the gate stays open after anything that looked like near-end.
const GATE_HOLD: u32 = 4;
/// Mean-square level below which a block is just room noise (-70 dBFS).
const NOISE: f64 = 1e-7;
/// Reference kept for re-transcribing dictation/command taps.
const TAP_HISTORY: u64 = RATE * 180;

/// Samples on the meeting clock, starting at `base`.
#[derive(Default)]
struct Ring {
    data: VecDeque<f32>,
    base: u64,
}

impl Ring {
    fn end(&self) -> u64 {
        self.base + self.data.len() as u64
    }

    /// Place `samples` at `at`: a gap before it is silence, an overlap is
    /// overwritten.
    fn put(&mut self, at: u64, samples: &[f32]) {
        if self.data.is_empty() && at > self.base {
            self.base = at;
        }
        if at < self.base {
            let skip = (self.base - at) as usize;
            if skip >= samples.len() {
                return;
            }
            return self.put(self.base, &samples[skip..]);
        }
        let end = self.end();
        if at > end {
            self.data.extend(std::iter::repeat_n(0.0, (at - end) as usize));
        }
        let off = (at - self.base) as usize;
        for (i, &s) in samples.iter().enumerate() {
            if off + i < self.data.len() {
                self.data[off + i] = s;
            } else {
                self.data.push_back(s);
            }
        }
    }

    /// Copy `[start, start + out.len())`, zeros outside what is held.
    fn read(&self, start: i64, out: &mut [f32]) {
        for (i, o) in out.iter_mut().enumerate() {
            let p = start + i as i64;
            *o = if p >= self.base as i64 && p < self.end() as i64 {
                self.data[(p - self.base as i64) as usize]
            } else {
                0.0
            };
        }
    }

    /// Like `read`, at a fractional start (windowed-sinc interpolation).
    fn read_frac(&self, start: f64, out: &mut [f32]) {
        const HALF: i64 = 8;
        let base = start.floor();
        let frac = start - base;
        if frac < 1e-4 || frac > 1.0 - 1e-4 {
            return self.read(start.round() as i64, out);
        }
        let mut k = [0.0f32; 2 * HALF as usize];
        let mut sum = 0.0;
        for (j, kj) in k.iter_mut().enumerate() {
            let u = (j as i64 - HALF + 1) as f64 - frac;
            let sinc = (std::f64::consts::PI * u).sin() / (std::f64::consts::PI * u);
            let w = 0.42
                + 0.5 * (std::f64::consts::PI * u / HALF as f64).cos()
                + 0.08 * (2.0 * std::f64::consts::PI * u / HALF as f64).cos();
            *kj = (sinc * w) as f32;
            sum += *kj;
        }
        k.iter_mut().for_each(|v| *v /= sum);
        let mut raw = vec![0.0; out.len() + 2 * HALF as usize];
        self.read(base as i64 - HALF + 1, &mut raw);
        for (i, o) in out.iter_mut().enumerate() {
            *o = k.iter().zip(&raw[i..]).map(|(a, b)| a * b).sum();
        }
    }

    fn keep_from(&mut self, from: u64) {
        if from > self.base {
            let n = ((from - self.base) as usize).min(self.data.len());
            self.data.drain(..n);
            self.base += n as u64;
        }
    }
}

/// Where the echo sits: the reference leads it by `lag + rate * (t - at)`
/// samples at meeting-clock sample `t` (`rate` is the clock drift).
#[derive(Clone, Copy)]
struct Track {
    at: f64,
    lag: f64,
    rate: f64,
}

impl Track {
    fn lag_at(&self, t: f64) -> f64 {
        self.lag + self.rate * (t - self.at)
    }
}

/// What a tap needs to be cleaned the way the meeting's mic is.
#[derive(Clone)]
pub struct Model {
    track: Track,
    taps: Vec<f32>,
}

pub struct Canceller {
    mdf: Mdf,
    estimator: DelayEstimator,
    reference: Ring,
    mic: Ring,
    /// Raw mic not yet processed; its first sample is at `proc_pos`.
    pending: Vec<f32>,
    proc_pos: u64,
    /// Processed but held back (`settle`) ends at `proc_pos`; output at `out_pos`.
    settle: Option<Settle>,
    out_pos: u64,
    in_pos: u64,
    track: Option<Track>,
    /// Main tap position over time, for the drift fit…
    peaks: VecDeque<(f64, f64)>,
    /// …and delay estimates relative to the track, before the filter has.
    offsets: VecDeque<(f64, f64)>,
    candidate: Option<f64>,
    next_estimate: u64,
    /// The filter's reference history continues at `proc_pos`.
    contiguous: bool,
    estimate: bool,
    gate: f32,
    hold: u32,
    /// Smoothed echo reduction (dB) over echo-dominated blocks.
    erle: f64,
    frame: Vec<f32>,
    block: Vec<f32>,
    out: Vec<f32>,
}

impl Default for Canceller {
    fn default() -> Self {
        Self::new()
    }
}

impl Canceller {
    pub fn new() -> Self {
        Self {
            mdf: Mdf::new(),
            estimator: DelayEstimator::new(),
            reference: Ring::default(),
            mic: Ring::default(),
            pending: Vec::new(),
            proc_pos: 0,
            settle: None,
            out_pos: 0,
            in_pos: 0,
            track: None,
            peaks: VecDeque::new(),
            offsets: VecDeque::new(),
            candidate: None,
            next_estimate: delay::MIN_WINDOW as u64,
            contiguous: false,
            estimate: true,
            gate: 1.0,
            hold: 0,
            erle: 0.0,
            frame: vec![0.0; 2 * N],
            block: vec![0.0; N],
            out: vec![0.0; N],
        }
    }

    /// A canceller already converged (from the meeting) whose mic input
    /// starts at `start` on the meeting clock.
    pub fn from_model(model: &Model, start: u64) -> Self {
        let mut c = Self::new();
        c.mdf.set_taps(&model.taps);
        c.track = Some(model.track);
        c.out_pos = start;
        c.proc_pos = start;
        c.in_pos = start;
        c.estimate = false;
        c
    }

    pub fn model(&mut self) -> Option<Model> {
        let track = self.track?;
        self.mdf.adapted().then(|| Model {
            track,
            taps: self.mdf.taps(),
        })
    }

    /// Current alignment (samples the reference leads its echo), if locked.
    pub fn lag(&self) -> Option<i64> {
        self.track.map(|t| t.lag_at(self.in_pos as f64).round() as i64)
    }

    /// Mic samples taken in but not yet returned.
    pub fn held(&self) -> u64 {
        self.in_pos - self.out_pos
    }

    /// System audio starting at `at` on the meeting clock.
    pub fn push_reference(&mut self, at: u64, samples: &[f32]) {
        self.reference.put(at, samples);
    }

    /// The system stream was restarted (or lost): its timing relative to the
    /// mic is unknown again, so pass the mic through until it re-locks. The
    /// filter keeps its taps — the room hasn't changed — and they apply
    /// again once the new alignment is found.
    pub fn reference_restarted(&mut self) {
        self.track = None;
        self.peaks.clear();
        self.offsets.clear();
        self.candidate = None;
        self.contiguous = false;
        self.next_estimate = self.in_pos + delay::MIN_WINDOW as u64;
        let end = self.reference.end();
        self.reference.keep_from(end);
    }

    /// Take in raw mic audio (continuing the meeting clock) and return the
    /// cleaned audio that is ready — possibly less than was given while the
    /// reference catches up; `finish` returns the rest.
    pub fn process(&mut self, mic: &[f32]) -> Vec<f32> {
        self.mic.put(self.in_pos, mic);
        self.in_pos += mic.len() as u64;
        self.pending.extend_from_slice(mic);
        if self.estimate && self.in_pos >= self.next_estimate {
            self.next_estimate = self.in_pos + ESTIMATE_EVERY;
            self.update_alignment();
            self.follow_drift();
        }
        let out = self.run(false);
        self.trim();
        out
    }

    /// Everything still held, processed with whatever reference exists.
    pub fn finish(&mut self, mic: &[f32]) -> Vec<f32> {
        let mut out = self.process(mic);
        out.extend(self.run(true));
        out
    }

    fn run(&mut self, last: bool) -> Vec<f32> {
        let Some(track) = self.track else {
            // Unlocked (or lost the reference mid-settling): pass through.
            let mut out = self.settle.take().map(|s| s.out).unwrap_or_default();
            out.append(&mut self.pending);
            self.out_pos += out.len() as u64;
            self.proc_pos = self.out_pos;
            self.contiguous = false;
            return out;
        };
        let mut out = Vec::with_capacity(self.pending.len());
        let mut used = 0;
        while self.pending.len() - used >= N || (last && used < self.pending.len()) {
            let pos = self.proc_pos;
            let needed = (frame_start(&track, pos as f64) + 2.0 * N as f64).ceil() as i64 + 8;
            let waited = self.in_pos as i64 - (pos + N as u64) as i64 >= MAX_HOLD as i64;
            let assumed = needed > self.reference.end() as i64;
            if assumed && !waited && !last {
                break;
            }
            let n = (self.pending.len() - used).min(N);
            let block = self.pending[used..used + n].to_vec();
            let cleaned = self.cancel_block(&track, pos, &block, !assumed && n == N);
            match self.settle.as_mut() {
                Some(st) => {
                    st.raw.extend_from_slice(&block);
                    st.out.extend(cleaned);
                }
                None => {
                    out.extend(cleaned);
                    self.out_pos += n as u64;
                }
            }
            used += n;
            self.proc_pos += n as u64;
        }
        self.pending.drain(..used);
        if let Some(st) = self.settle.as_ref() {
            let converged = self.mdf.adapted() && self.erle >= CONVERGED;
            if converged || st.raw.len() >= SETTLE_MAX || last {
                let st = self.settle.take().unwrap();
                self.out_pos += st.raw.len() as u64;
                out = if self.mdf.adapted() { self.replay(&track, st) } else { st.out };
            }
        }
        out
    }

    /// One block through the live filter and gate (mic at `pos`).
    fn cancel_block(&mut self, track: &Track, pos: u64, mic: &[f32], adapt: bool) -> Vec<f32> {
        if !self.contiguous {
            for p in (1..filter::P).rev() {
                let at = pos as f64 - (p * N) as f64;
                self.reference.read_frac(frame_start(track, at), &mut self.frame);
                scale(&mut self.frame);
                self.mdf.push_reference(&self.frame);
            }
            self.contiguous = true;
        }
        self.reference.read_frac(frame_start(track, pos as f64), &mut self.frame);
        scale(&mut self.frame);
        self.mdf.push_reference(&self.frame);
        let n = mic.len();
        self.block[..n].copy_from_slice(mic);
        self.block[n..].iter_mut().for_each(|v| *v = 0.0);
        scale(&mut self.block);
        let block = self.mdf.cancel(&self.block, &mut self.out, adapt);
        self.gate_block(block.echo);
        self.out[..n].iter().map(|v| v / SCALE).collect()
    }

    /// The filter converged while its first output was held back: run that
    /// stretch again through a copy of the converged filter, so the start of
    /// a leak is cancelled as well as the rest.
    fn replay(&mut self, track: &Track, st: Settle) -> Vec<f32> {
        let mut copy = Mdf::new();
        copy.set_taps(&self.mdf.taps());
        let live = std::mem::replace(&mut self.mdf, copy);
        let contiguous = std::mem::replace(&mut self.contiguous, false);
        let mut out = Vec::with_capacity(st.raw.len());
        for (i, chunk) in st.raw.chunks(N).enumerate() {
            let pos = st.start + (i * N) as u64;
            out.extend(self.cancel_block(track, pos, chunk, false));
        }
        self.mdf = live;
        self.contiguous = contiguous;
        out
    }

    /// Silence a block the filter has explained as echo; ramp, never step.
    fn gate_block(&mut self, echo: f64) {
        let scale2 = (SCALE as f64) * (SCALE as f64);
        let raw: f64 = self.block.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let left: f64 = self.out.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let noise = NOISE * N as f64 * scale2;
        let echo_only = raw > noise && left * GATE_RATIO < raw && echo * 2.0 > raw;
        if raw > noise * 100.0 && echo * 2.0 > raw {
            // Blocks dominated by echo: how well the filter is doing.
            self.erle = 0.95 * self.erle + 0.05 * (raw / left.max(1.0)).log10() * 10.0;
        }
        if !echo_only && left > noise {
            self.hold = GATE_HOLD;
        }
        let target = if echo_only && self.hold == 0 {
            GATE_FLOOR
        } else {
            if echo_only {
                self.hold -= 1;
            }
            1.0
        };
        if target == 1.0 && self.gate == 1.0 {
            return;
        }
        let ramp = if target > self.gate { N / 8 } else { N };
        let from = self.gate;
        for (i, v) in self.out.iter_mut().enumerate() {
            let a = ((i + 1) as f32 / ramp as f32).min(1.0);
            *v *= from + (target - from) * a;
        }
        self.gate = target;
    }

    fn update_alignment(&mut self) {
        let end = self.in_pos.min(self.reference.end());
        let start = end
            .saturating_sub(delay::WINDOW as u64)
            .max(self.mic.base)
            .max(self.reference.base);
        if end <= start {
            return;
        }
        let n = (end - start) as usize;
        let mut mic = vec![0.0; n];
        let mut reference = vec![0.0; n];
        self.mic.read(start as i64, &mut mic);
        self.reference.read(start as i64, &mut reference);
        let Some(e) = self.estimator.estimate(&mic, &reference) else {
            return;
        };
        if e.z < Z_CANDIDATE {
            return;
        }
        let t = start as f64 + e.center;
        // Locked: the filter spans reflections a few ms either side and
        // follows drift itself (`follow_drift`); only a route change that
        // moves the echo by more than REALIGN is acted on here.
        if let Some(track) = self.track.filter(|tr| (tr.lag_at(t) - e.lag).abs() <= REALIGN) {
            self.candidate = None;
            // Until the filter has converged its taps can't show drift, so
            // the estimates near the locked peak do (a drifting path never
            // lets the filter converge in the first place).
            if self.erle < CONVERGED {
                self.offsets.push_back((t, e.lag - track.lag_at(t)));
                self.correct_drift(t, Series::Estimates);
            }
            return;
        }
        let agreed = self.candidate.is_some_and(|c| (c - e.lag).abs() <= AGREE);
        self.candidate = Some(e.lag);
        if !(agreed || e.z >= Z_SURE) {
            return;
        }
        self.candidate = None;
        self.peaks.clear();
        self.offsets.clear();
        self.contiguous = false;
        match self.track {
            // Move the learned path along in whole samples: the echo
            // estimate is unchanged by that, so nothing has to re-converge.
            Some(old) => {
                let delta = (e.lag - old.lag_at(t)).round();
                if delta.abs() as usize >= TAPS / 2 {
                    self.mdf.reset();
                } else {
                    self.mdf.shift(delta as i64);
                }
                self.track = Some(Track { lag: old.lag + delta, ..old });
            }
            None => {
                self.track = Some(Track { at: t, lag: e.lag, rate: 0.0 });
                if !(self.mdf.adapted() && self.erle >= CONVERGED) {
                    self.settle = Some(Settle { start: self.proc_pos, raw: Vec::new(), out: Vec::new() });
                }
            }
        }
    }

    /// Clock drift between the mic and the output device slides the echo
    /// along the filter. Watch the main tap's position (sub-sample) and,
    /// once it has clearly moved, fold the slope into the track's rate so the
    /// reference is resampled to match and the path holds still.
    fn follow_drift(&mut self) {
        if self.track.is_none() || self.erle < CONVERGED {
            return;
        }
        let h = self.mdf.taps();
        let energy: f32 = h.iter().map(|v| v * v).sum::<f32>() / h.len() as f32;
        let range = match self.peaks.back() {
            Some(&(_, p)) => {
                let p = p.round() as i64;
                (p - 8).max(1) as usize..(p + 9).min(TAPS as i64 - 1) as usize
            }
            None => 1..TAPS - 1,
        };
        let Some(j) = range.max_by(|a, b| h[*a].abs().total_cmp(&h[*b].abs())) else {
            return;
        };
        if h[j] * h[j] < 30.0 * energy {
            self.peaks.clear();
            return;
        }
        let (l, c, r) = (h[j - 1].abs() as f64, h[j].abs() as f64, h[j + 1].abs() as f64);
        let curve = l - 2.0 * c + r;
        let frac = if curve < 0.0 { (0.5 * (l - r) / curve).clamp(-0.5, 0.5) } else { 0.0 };
        let t = self.proc_pos as f64;
        self.peaks.push_back((t, j as f64 + frac));
        self.correct_drift(t, Series::Taps);
    }

    /// Fold a clear slope in one series into the track's drift rate.
    fn correct_drift(&mut self, t: f64, series: Series) {
        let Some(track) = self.track else { return };
        let points = match series {
            Series::Taps => &mut self.peaks,
            Series::Estimates => &mut self.offsets,
        };
        while points.front().is_some_and(|p| t - p.0 > TRACK_SPAN) {
            points.pop_front();
        }
        if let Some((slope, span)) = slope(points) {
            // Correct only motion that clearly exceeds measurement noise.
            if slope.abs() * RATE as f64 > MIN_DRIFT && (slope * span).abs() > MIN_MOVE {
                let rate = (track.rate + slope).clamp(-MAX_DRIFT, MAX_DRIFT);
                self.track = Some(Track { at: t, lag: track.lag_at(t), rate });
                self.peaks.clear();
                self.offsets.clear();
            }
        }
    }

    fn trim(&mut self) {
        let keep = self.in_pos.saturating_sub(delay::WINDOW as u64 + RATE);
        self.mic.keep_from(keep);
        let lead = (delay::MAX_LEAD + TAPS as i64 + 2 * N as i64) as u64;
        self.reference
            .keep_from(keep.min(self.out_pos.saturating_sub(lead)));
    }
}

enum Series {
    Taps,
    Estimates,
}

/// Robust (Theil-Sen) slope of (time, position) once the points span long
/// enough to tell drift from measurement noise; a reflection picked now and
/// then instead of the direct path doesn't drag it.
fn slope(points: &VecDeque<(f64, f64)>) -> Option<(f64, f64)> {
    let span = points.back()?.0 - points.front()?.0;
    if span < FIT_SPAN || points.len() < 5 {
        return None;
    }
    let mut slopes = Vec::new();
    for (i, a) in points.iter().enumerate() {
        for b in points.iter().skip(i + 1) {
            if b.0 - a.0 >= RATE as f64 {
                slopes.push((b.1 - a.1) / (b.0 - a.0));
            }
        }
    }
    if slopes.is_empty() {
        return None;
    }
    slopes.sort_by(|a, b| a.total_cmp(b));
    Some((slopes[slopes.len() / 2], span))
}

/// Reference frame for the mic block at `pos`: the N samples before and the
/// N aligned with it, MARGIN early.
fn frame_start(track: &Track, pos: f64) -> f64 {
    pos - (track.lag_at(pos) - MARGIN as f64) - N as f64
}

/// Output held from lock until the filter converges.
struct Settle {
    start: u64,
    raw: Vec<f32>,
    out: Vec<f32>,
}

fn scale(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v *= SCALE);
}

/// The meeting's canceller as the dictation and command-key taps see it.
#[derive(Default)]
pub struct Shared(Mutex<TapState>);

#[derive(Default)]
struct TapState {
    reference: Ring,
    model: Option<Model>,
    /// Where the running system stream's sample 0 sits on the meeting clock.
    sys_offset: Option<u64>,
}

impl Shared {
    fn clear(&self) {
        *self.0.lock().unwrap() = TapState::default();
    }
}

/// The capture thread's handle: the canceller plus what it shares with taps.
pub struct MicCleaner {
    aec: Canceller,
    shared: Arc<Shared>,
}

impl MicCleaner {
    pub fn new(shared: Arc<Shared>) -> Self {
        shared.clear();
        Self {
            aec: Canceller::new(),
            shared,
        }
    }

    /// The system stream (re)started with its sample 0 at `offset`.
    pub fn system_started(&mut self, offset: u64) {
        self.shared.0.lock().unwrap().sys_offset = Some(offset);
    }

    /// System audio drained from the stream, starting at `at`.
    pub fn reference(&mut self, at: u64, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        self.aec.push_reference(at, samples);
        let mut s = self.shared.0.lock().unwrap();
        s.reference.put(at, samples);
        let keep = s.reference.end().saturating_sub(TAP_HISTORY);
        s.reference.keep_from(keep);
    }

    /// The system stream stopped or restarted.
    pub fn system_lost(&mut self) {
        self.aec.reference_restarted();
        let mut s = self.shared.0.lock().unwrap();
        s.model = None;
        s.sys_offset = None;
    }

    pub fn process(&mut self, mic: &[f32], last: bool) -> Vec<f32> {
        let out = if last {
            self.aec.finish(mic)
        } else {
            self.aec.process(mic)
        };
        let model = if last { None } else { self.aec.model() };
        self.shared.0.lock().unwrap().model = model;
        out
    }

    pub fn held(&self) -> u64 {
        self.aec.held()
    }
}

impl Drop for MicCleaner {
    fn drop(&mut self) {
        self.shared.clear();
    }
}

/// Clean a dictation or command-key take that tapped the meeting's mic at
/// `start_ms`, with the meeting's converged canceller. Unchanged when the
/// canceller isn't locked (no leak known, or no system audio).
pub fn clean_tap(
    shared: &Shared,
    peek: impl FnOnce() -> Option<(usize, Vec<f32>)>,
    start_ms: i64,
    samples: Vec<f32>,
) -> Vec<f32> {
    let start = start_ms.max(0) as u64 * RATE / 1000;
    let (model, reference) = {
        let s = shared.0.lock().unwrap();
        let Some(model) = s.model.clone() else {
            return samples;
        };
        let lag = model.track.lag_at(start as f64).ceil() as i64;
        let from = (start as i64 - lag - TAPS as i64 - 2 * N as i64 - 16).max(0) as u64;
        let mut held = vec![0.0; s.reference.end().saturating_sub(from) as usize];
        s.reference.read(from as i64, &mut held);
        let mut ring = Ring {
            data: VecDeque::new(),
            base: from,
        };
        ring.put(from, &held);
        // What the capture thread hasn't drained yet (the last second or so).
        if let (Some(offset), Some((at, tail))) = (s.sys_offset, peek()) {
            ring.put(offset + at as u64, &tail);
        }
        (model, ring)
    };
    let mut aec = Canceller::from_model(&model, start);
    aec.reference = reference;
    aec.finish(&samples)
}

/// `clean_tap` for a take on the running meeting's mic.
pub(crate) fn clean_meeting_tap(app: &tauri::AppHandle, start_ms: i64, samples: Vec<f32>) -> Vec<f32> {
    use tauri::Manager;
    let st = app.state::<crate::AppState>();
    #[cfg(target_os = "macos")]
    let peek = || crate::syscapture::peek(&st.sys);
    #[cfg(not(target_os = "macos"))]
    let peek = || None;
    clean_tap(&st.meeting.echo, peek, start_ms, samples)
}
