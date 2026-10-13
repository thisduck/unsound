use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::Serialize;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};

/// Whisper expects 16 kHz mono f32 samples.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordingResult {
    pub duration_secs: f32,
    pub sample_count: usize,
}

struct ActiveRecording {
    stop_tx: mpsc::Sender<()>,
    done_rx: mpsc::Receiver<()>,
    buffer: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
    channels: u16,
    /// Absolute raw-sample index of `buffer[0]`. Meeting recordings drop audio
    /// once it's been drained (it's on disk by then), so indices into the
    /// buffer are offset by what's been trimmed — memory stays flat for hours.
    base: usize,
    /// Absolute raw index already handed out by `drain_16k`.
    drained: usize,
    /// Streaming resampler for `drain_16k`: one continuous filter across
    /// drains, so chunk boundaries add no padding (a fresh resampler per 1 s
    /// chunk drifted the mic timeline ~4.5 s/hour against system audio).
    stream: Option<StreamResampler>,
    owner: Owner,
    /// Open taps: id → absolute raw start index (frame-aligned). A tap is a
    /// slice of the running recording claimed by someone else — dictation or
    /// a spoken command during a meeting — so both can share the one mic.
    taps: Vec<(u64, usize)>,
    next_tap: u64,
}

/// Who started the recording (and so who may stop it).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Owner {
    Dictation,
    Meeting,
}

#[derive(Default)]
pub struct AudioState {
    active: Mutex<Option<ActiveRecording>>,
    /// The last finished recording, as 16 kHz mono samples ready for whisper.
    pub last_recording: Mutex<Vec<f32>>,
    /// A dictation take running as a tap on a meeting's recording.
    pub dictation_tap: Mutex<Option<u64>>,
    pub capture_error: Arc<Mutex<Option<String>>>,
}

/// A continuous downmix + resample to 16 kHz mono. Input that doesn't fill a
/// whole resampler chunk waits for the next call instead of being padded.
struct StreamResampler {
    resampler: Option<rubato::SincFixedIn<f32>>,
    pending: Vec<f32>,
}

const STREAM_CHUNK: usize = 1024;

impl StreamResampler {
    fn new(from: u32) -> Result<Self, String> {
        let resampler = if from == WHISPER_SAMPLE_RATE {
            None
        } else {
            Some(new_resampler(from, WHISPER_SAMPLE_RATE, STREAM_CHUNK)?)
        };
        Ok(Self {
            resampler,
            pending: Vec::new(),
        })
    }

    fn push(&mut self, mono: &[f32]) -> Vec<f32> {
        use rubato::Resampler;
        let Some(r) = self.resampler.as_mut() else {
            return mono.to_vec();
        };
        self.pending.extend_from_slice(mono);
        let mut out = Vec::new();
        let mut pos = 0;
        while pos + STREAM_CHUNK <= self.pending.len() {
            match r.process(&[&self.pending[pos..pos + STREAM_CHUNK]], None) {
                Ok(frames) => out.extend_from_slice(&frames[0]),
                Err(e) => eprintln!("[audio] resample failed: {e}"),
            }
            pos += STREAM_CHUNK;
        }
        self.pending.drain(..pos);
        out
    }
}

/// Names of available input devices, for the microphone picker.
pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    host.input_devices()
        .map(|devices| devices.filter_map(|d| d.description().ok().map(|desc| desc.name().to_string())).collect())
        .unwrap_or_default()
}

/// Resolve a device by name, falling back to the system default when the
/// name is empty or no longer present (e.g. an unplugged USB mic).
fn input_device(host: &cpal::Host, name: &str) -> Option<cpal::Device> {
    if !name.is_empty() {
        if let Ok(mut devices) = host.input_devices() {
            if let Some(d) = devices.find(|d| d.description().map(|desc| desc.name() == name).unwrap_or(false)) {
                return Some(d);
            }
        }
    }
    host.default_input_device()
}

pub fn start_recording(
    app: AppHandle,
    state: &AudioState,
    device_name: String,
    owner: Owner,
) -> Result<(), String> {
    let mut active = state.active.lock().unwrap();
    if active.is_some() {
        return Err("already recording".into());
    }

    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    // The cpal stream is !Send, so it lives on its own thread; this channel
    // reports the negotiated stream config (or an error) back to us.
    let (config_tx, config_rx) = mpsc::channel::<Result<(u32, u16), String>>();

    let thread_buffer = buffer.clone();
    let failure = state.capture_error.clone();
    *failure.lock().unwrap() = None;
    std::thread::spawn(move || {
        let build = || -> Result<(cpal::Stream, u32, u16), String> {
            let host = cpal::default_host();
            let device = input_device(&host, &device_name)
                .ok_or("no microphone found (check input devices and permissions)")?;
            let config = device.default_input_config().map_err(|e| e.to_string())?;
            let sample_rate = config.sample_rate();
            let channels = config.channels();
            let failure = failure.clone();
            let err_fn = move |e| { *failure.lock().unwrap() = Some(format!("{e}")); };

            // Emit a level event roughly every 50 ms so the UI can show a meter.
            let emit_every = (sample_rate as usize * channels as usize) / 20;
            let mut since_emit = 0usize;
            let app = app.clone();
            let buf = thread_buffer.clone();

            let stream = match config.sample_format() {
                cpal::SampleFormat::F32 => device
                    .build_input_stream(
                        config.clone().into(),
                        move |data: &[f32], _: &_| {
                            let mut buf = buf.lock().unwrap();
                            buf.extend_from_slice(data);
                            since_emit += data.len();
                            if since_emit >= emit_every {
                                since_emit = 0;
                                let rms = (data.iter().map(|s| s * s).sum::<f32>()
                                    / data.len().max(1) as f32)
                                    .sqrt();
                                let _ = app.emit("audio-level", rms);
                            }
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| e.to_string())?,
                cpal::SampleFormat::I16 => device
                    .build_input_stream(
                        config.clone().into(),
                        move |data: &[i16], _: &_| {
                            let mut buf = buf.lock().unwrap();
                            buf.extend(data.iter().map(|&s| s as f32 / i16::MAX as f32));
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| e.to_string())?,
                cpal::SampleFormat::U16 => device
                    .build_input_stream(
                        config.clone().into(),
                        move |data: &[u16], _: &_| {
                            let mut buf = buf.lock().unwrap();
                            buf.extend(
                                data.iter()
                                    .map(|&s| (s as f32 - 32768.0) / 32768.0),
                            );
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| e.to_string())?,
                other => return Err(format!("unsupported sample format: {other:?}")),
            };
            stream.play().map_err(|e| e.to_string())?;
            Ok((stream, sample_rate, channels))
        };

        match build() {
            Ok((stream, sample_rate, channels)) => {
                let _ = config_tx.send(Ok((sample_rate, channels)));
                // Keep the stream alive until stop is requested.
                let _ = stop_rx.recv();
                drop(stream);
                let _ = done_tx.send(());
            }
            Err(e) => {
                let _ = config_tx.send(Err(e));
            }
        }
    });

    let (sample_rate, channels) = config_rx
        .recv()
        .map_err(|_| "audio thread died before reporting a config".to_string())??;

    *active = Some(ActiveRecording {
        stop_tx,
        done_rx,
        buffer,
        sample_rate,
        channels,
        base: 0,
        drained: 0,
        stream: None,
        owner,
        taps: Vec::new(),
        next_tap: 1,
    });
    Ok(())
}

/// Who owns the running recording, if any.
pub fn owner(state: &AudioState) -> Option<Owner> {
    state.active.lock().unwrap().as_ref().map(|r| r.owner)
}

/// Pull the audio captured since the last call, as 16 kHz mono ready for
/// Whisper — the mic side of live meeting transcription. Returns empty when
/// not recording or nothing new has arrived. Resampling is continuous across
/// calls, so the returned stream has exactly the recording's timeline.
/// Meeting recordings then forget the drained audio (unless a tap still
/// needs it), keeping memory flat however long the meeting runs.
pub fn drain_16k(state: &AudioState) -> Vec<f32> {
    let mut active = state.active.lock().unwrap();
    let Some(rec) = active.as_mut() else {
        return Vec::new();
    };
    drain_recording(rec)
}

fn drain_recording(rec: &mut ActiveRecording) -> Vec<f32> {
    let mut raw = rec.buffer.lock().unwrap();
    let end = rec.base + raw.len();
    if end <= rec.drained {
        return Vec::new();
    }
    // cpal callback buffers are frame-aligned, so `raw.len()` is always a whole
    // number of frames — no mid-frame split to worry about.
    let slice = raw[rec.drained - rec.base..].to_vec();
    rec.drained = end;
    if rec.owner == Owner::Meeting {
        let keep_from = rec
            .taps
            .iter()
            .map(|(_, start)| *start)
            .min()
            .unwrap_or(rec.drained)
            .min(rec.drained);
        if keep_from > rec.base {
            raw.drain(..keep_from - rec.base);
            rec.base = keep_from;
        }
    }
    drop(raw);
    let channels = rec.channels;
    if rec.stream.is_none() {
        match StreamResampler::new(rec.sample_rate) {
            Ok(s) => rec.stream = Some(s),
            Err(e) => {
                eprintln!("[audio] {e}");
                return Vec::new();
            }
        }
    }
    let mono = downmix(&slice, channels);
    rec.stream.as_mut().map(|s| s.push(&mono)).unwrap_or_default()
}

/// Claim the audio from now on as a separate take while the recording keeps
/// running (dictation or a spoken command during a meeting). Returns the tap
/// id and where it starts on the 16 kHz timeline, in ms.
pub fn tap_begin(state: &AudioState) -> Result<(u64, i64), String> {
    let mut active = state.active.lock().unwrap();
    let rec = active.as_mut().ok_or("not recording")?;
    let len = rec.buffer.lock().unwrap().len();
    let start = rec.base + len;
    let id = rec.next_tap;
    rec.next_tap += 1;
    rec.taps.push((id, start));
    let ms = frames_to_ms(start / rec.channels.max(1) as usize, rec.sample_rate);
    Ok((id, ms))
}

/// The tap's audio so far (16 kHz mono) without ending it — live captions.
pub fn tap_snapshot(state: &AudioState, id: u64) -> Vec<f32> {
    let active = state.active.lock().unwrap();
    let Some(rec) = active.as_ref() else {
        return Vec::new();
    };
    let Some((_, start)) = rec.taps.iter().find(|(t, _)| *t == id).copied() else {
        return Vec::new();
    };
    let raw = rec.buffer.lock().unwrap();
    let slice = raw[start.saturating_sub(rec.base).min(raw.len())..].to_vec();
    let (rate, channels) = (rec.sample_rate, rec.channels);
    drop(raw);
    drop(active);
    resample(&downmix(&slice, channels), rate, WHISPER_SAMPLE_RATE).unwrap_or_default()
}

/// End a tap: its audio as 16 kHz mono, plus where it ended (ms on the
/// recording's timeline).
pub fn tap_end(state: &AudioState, id: u64) -> Result<(Vec<f32>, i64), String> {
    let mut active = state.active.lock().unwrap();
    let rec = active.as_mut().ok_or("not recording")?;
    let idx = rec
        .taps
        .iter()
        .position(|(t, _)| *t == id)
        .ok_or("that take already ended")?;
    let (_, start) = rec.taps.remove(idx);
    let raw = rec.buffer.lock().unwrap();
    let end = rec.base + raw.len();
    let slice = raw[start.saturating_sub(rec.base).min(raw.len())..].to_vec();
    let (rate, channels) = (rec.sample_rate, rec.channels);
    drop(raw);
    let end_ms = frames_to_ms(end / channels.max(1) as usize, rate);
    drop(active);
    let samples = resample(&downmix(&slice, channels), rate, WHISPER_SAMPLE_RATE)?;
    Ok((samples, end_ms))
}

/// Where the recording is now on its 16 kHz timeline, in ms.
pub fn position_ms(state: &AudioState) -> Option<i64> {
    let active = state.active.lock().unwrap();
    let rec = active.as_ref()?;
    let len = rec.buffer.lock().unwrap().len();
    Some(frames_to_ms((rec.base + len) / rec.channels.max(1) as usize, rec.sample_rate))
}

fn frames_to_ms(frames: usize, rate: u32) -> i64 {
    (frames as i64 * 1000) / rate.max(1) as i64
}

/// Hand a meeting's recording over to the dictation take tapping it: the
/// meeting is ending, but the take isn't, so the stream keeps running and the
/// take's later `stop_recording` ends it normally, from where the tap began.
/// Returns false when there's no dictation tap to hand over to.
pub fn hand_over_to_dictation(state: &AudioState) -> bool {
    let Some(tap) = state.dictation_tap.lock().unwrap().take() else {
        return false;
    };
    let mut active = state.active.lock().unwrap();
    let Some(rec) = active.as_mut() else {
        return false;
    };
    let Some(idx) = rec.taps.iter().position(|(t, _)| *t == tap) else {
        return false;
    };
    let (_, start) = rec.taps.remove(idx);
    let mut raw = rec.buffer.lock().unwrap();
    let cut = start.saturating_sub(rec.base).min(raw.len());
    raw.drain(..cut);
    drop(raw);
    rec.base = 0;
    rec.drained = 0;
    rec.stream = None;
    rec.owner = Owner::Dictation;
    rec.taps.clear();
    true
}

/// A copy of the *entire* audio captured so far, as 16 kHz mono — for live
/// dictation captions, which re-transcribe the whole take as it grows. Unlike
/// `drain_16k` this doesn't advance any cursor, so it never disturbs the final
/// transcription taken at stop. Empty when not recording.
pub fn snapshot_16k(state: &AudioState) -> Vec<f32> {
    // Dictating during a meeting: the take is just the tapped slice.
    if let Some(tap) = *state.dictation_tap.lock().unwrap() {
        return tap_snapshot(state, tap);
    }
    let active = state.active.lock().unwrap();
    let Some(rec) = active.as_ref() else {
        return Vec::new();
    };
    let raw = rec.buffer.lock().unwrap().clone();
    let (sample_rate, channels) = (rec.sample_rate, rec.channels);
    drop(active);
    let mono = downmix(&raw, channels);
    resample(&mono, sample_rate, WHISPER_SAMPLE_RATE).unwrap_or_default()
}

pub fn stop_recording(state: &AudioState) -> Result<RecordingResult, String> {
    let rec = {
        let mut active = state.active.lock().unwrap();
        match active.as_ref().map(|r| r.owner) {
            None => return Err("not recording".into()),
            // Never let a dictation stop end a meeting's capture.
            Some(Owner::Meeting) => return Err("the microphone is recording a meeting".into()),
            Some(Owner::Dictation) => active.take().expect("checked above"),
        }
    };

    let _ = rec.stop_tx.send(());
    let _ = rec.done_rx.recv_timeout(std::time::Duration::from_secs(5));

    let raw = std::mem::take(&mut *rec.buffer.lock().unwrap());
    let mono = downmix(&raw, rec.channels);
    let samples = resample(&mono, rec.sample_rate, WHISPER_SAMPLE_RATE)?;

    let result = RecordingResult {
        duration_secs: samples.len() as f32 / WHISPER_SAMPLE_RATE as f32,
        sample_count: samples.len(),
    };
    *state.last_recording.lock().unwrap() = samples;
    Ok(result)
}

/// End a meeting's mic capture, returning the last undrained audio (16 kHz,
/// including what was waiting inside the resampler) for the final VAD flush.
/// If a dictation take is tapping the mic, the stream is handed to it instead
/// of stopped, so the take can finish.
pub fn stop_meeting_recording(state: &AudioState) -> Vec<f32> {
    if owner(state) != Some(Owner::Meeting) {
        return Vec::new();
    }
    let mut tail = Vec::new();
    if state.dictation_tap.lock().unwrap().is_some() {
        tail = drain_16k(state);
        if let Some(rest) = flush_stream(state) { tail.extend(rest); }
        if hand_over_to_dictation(state) { return tail; }
    }
    let mut active = state.active.lock().unwrap();
    let Some(rec) = active.as_mut().filter(|r|r.owner == Owner::Meeting) else { return tail };
    let _ = rec.stop_tx.send(());
    let _ = rec.done_rx.recv_timeout(std::time::Duration::from_secs(5));
    // Drain AFTER the producer stops: its final callback may arrive while
    // shutdown is in progress. Keep the recorder slot reserved through this.
    tail.extend(drain_recording(rec));
    if let Some(rest) = flush_recording(rec) { tail.extend(rest); }
    active.take();
    tail
}

#[cfg(test)]
mod reliability_tests {
    use super::*;

    // Inject the same buffer/channels used by CPAL, with its producer gone.
    // These tests do not pretend to emulate a physical USB device reconnect.
    fn disconnected(state: &AudioState) -> Arc<Mutex<Vec<f32>>> {
        let (stop_tx, _) = mpsc::channel();
        let (_, done_rx) = mpsc::channel();
        let buffer = Arc::new(Mutex::new(Vec::new()));
        *state.active.lock().unwrap() = Some(ActiveRecording {
            stop_tx, done_rx, buffer: buffer.clone(), sample_rate: 16_000,
            channels: 1, base: 0, drained: 0, stream: None,
            owner: Owner::Meeting, taps: Vec::new(), next_tap: 0,
        });
        buffer
    }

    #[test]
    fn shutdown_keeps_samples_from_the_final_device_callback() {
        let state = AudioState::default();
        let buffer = disconnected(&state);
        let (stop_tx, stop_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        {
            let mut active = state.active.lock().unwrap();
            let rec = active.as_mut().unwrap();
            rec.stop_tx = stop_tx;
            rec.done_rx = done_rx;
        }
        let producer = std::thread::spawn(move || {
            stop_rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
            buffer.lock().unwrap().extend([0.75; 160]);
            done_tx.send(()).unwrap();
        });
        assert_eq!(stop_meeting_recording(&state), vec![0.75;160]);
        producer.join().unwrap();
        assert!(stop_meeting_recording(&state).is_empty());
    }

    #[test]
    fn disconnected_producer_drains_tail_once_and_releases_recording() {
        let state = AudioState::default();
        let buffer = disconnected(&state);
        buffer.lock().unwrap().extend([0.25; 1600]);
        assert_eq!(drain_16k(&state), vec![0.25; 1600]);
        buffer.lock().unwrap().extend([0.5; 800]);
        assert_eq!(stop_meeting_recording(&state), vec![0.5; 800]);
        assert!(stop_meeting_recording(&state).is_empty());
        assert_eq!(owner(&state), None);
        let next = disconnected(&state);
        next.lock().unwrap().extend([0.75; 400]);
        assert_eq!(stop_meeting_recording(&state), vec![0.75; 400]);
    }

    #[test]
    fn capture_buffer_is_trimmed_and_dictation_cannot_stop_a_meeting() {
        let state = AudioState::default();
        let buffer = disconnected(&state);
        for _ in 0..120 {
            buffer.lock().unwrap().extend([0.25; 16_000]);
            assert_eq!(drain_16k(&state).len(), 16_000);
            assert!(buffer.lock().unwrap().is_empty());
        }
        assert_eq!(position_ms(&state), Some(120_000));
        assert!(stop_recording(&state).is_err());
        assert_eq!(owner(&state), Some(Owner::Meeting));
        stop_meeting_recording(&state);
    }

    #[test]
    fn meeting_stop_preserves_an_active_dictation_tap_after_disconnect() {
        let state = AudioState::default();
        let buffer = disconnected(&state);
        buffer.lock().unwrap().extend([0.25; 1600]);
        drain_16k(&state);
        let (tap, _) = tap_begin(&state).unwrap();
        *state.dictation_tap.lock().unwrap() = Some(tap);
        buffer.lock().unwrap().extend([0.5; 800]);
        assert_eq!(stop_meeting_recording(&state), vec![0.5; 800]);
        assert_eq!(owner(&state), Some(Owner::Dictation));
        assert_eq!(stop_recording(&state).unwrap().sample_count, 800);
        assert_eq!(*state.last_recording.lock().unwrap(), vec![0.5; 800]);
    }
}

/// Push the resampler's held-back input through (zero-padded) at the end.
fn flush_stream(state: &AudioState) -> Option<Vec<f32>> {
    let mut active = state.active.lock().unwrap();
    let rec = active.as_mut()?;
    flush_recording(rec)
}

fn flush_recording(rec: &mut ActiveRecording) -> Option<Vec<f32>> {
    use rubato::Resampler;
    let rate = rec.sample_rate;
    let stream = rec.stream.as_mut()?;
    let pending = std::mem::take(&mut stream.pending);
    let r = stream.resampler.as_mut()?;
    if pending.is_empty() {
        return None;
    }
    let out = r.process_partial(Some(&[&pending[..]]), None).ok()?;
    // Padding produced output past the real input; keep only its share.
    let keep = pending.len() * WHISPER_SAMPLE_RATE as usize / rate.max(1) as usize;
    Some(out[0][..keep.min(out[0].len())].to_vec())
}

fn downmix(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    samples
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

fn resample(input: &[f32], from: u32, to: u32) -> Result<Vec<f32>, String> {
    if from == to || input.is_empty() {
        return Ok(input.to_vec());
    }
    use rubato::Resampler;
    let chunk = 1024;
    let mut resampler = new_resampler(from, to, chunk)?;

    let mut out = Vec::with_capacity(input.len() * to as usize / from as usize + chunk);
    let mut pos = 0;
    while pos + chunk <= input.len() {
        let frames = resampler
            .process(&[&input[pos..pos + chunk]], None)
            .map_err(|e| e.to_string())?;
        out.extend_from_slice(&frames[0]);
        pos += chunk;
    }
    if pos < input.len() {
        let frames = resampler
            .process_partial(Some(&[&input[pos..]]), None)
            .map_err(|e| e.to_string())?;
        out.extend_from_slice(&frames[0]);
    }
    Ok(out)
}

fn new_resampler(from: u32, to: u32, chunk: usize) -> Result<rubato::SincFixedIn<f32>, String> {
    use rubato::{SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };
    SincFixedIn::<f32>::new(to as f64 / from as f64, 2.0, params, chunk, 1).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Draining in uneven pieces must produce the same timeline as one pass:
    /// no padding at chunk boundaries (the old per-drain resampler added
    /// ~0.12%, i.e. seconds per hour).
    #[test]
    fn stream_resampler_keeps_the_timeline() {
        let mut s = StreamResampler::new(48_000).unwrap();
        let mut out = 0usize;
        let mut fed = 0usize;
        for i in 0..600 {
            // Irregular drain sizes, like 1 s polls of 512-frame callbacks.
            let n = if i % 2 == 0 { 48_512 } else { 47_488 };
            out += s.push(&vec![0.1; n]).len();
            fed += n;
        }
        let expected = fed / 3;
        let diff = expected as i64 - out as i64;
        // Only what's still waiting inside the filter may be missing.
        assert!((0..=STREAM_CHUNK as i64).contains(&diff), "diff {diff}");
    }
}
