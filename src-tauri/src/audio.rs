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
    /// How many raw samples have already been handed out by `drain_16k`, so
    /// meeting streaming can pull only the newly-captured audio each cycle.
    drained: usize,
}

#[derive(Default)]
pub struct AudioState {
    active: Mutex<Option<ActiveRecording>>,
    /// The last finished recording, as 16 kHz mono samples ready for whisper.
    pub last_recording: Mutex<Vec<f32>>,
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

pub fn start_recording(app: AppHandle, state: &AudioState, device_name: String) -> Result<(), String> {
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
    std::thread::spawn(move || {
        let build = || -> Result<(cpal::Stream, u32, u16), String> {
            let host = cpal::default_host();
            let device = input_device(&host, &device_name)
                .ok_or("no microphone found (check input devices and permissions)")?;
            let config = device.default_input_config().map_err(|e| e.to_string())?;
            let sample_rate = config.sample_rate();
            let channels = config.channels();
            let err_fn = |e| eprintln!("audio stream error: {e}");

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
        drained: 0,
    });
    Ok(())
}

/// Pull the audio captured since the last call, as 16 kHz mono ready for
/// Whisper — the mic side of live meeting transcription. Returns empty when
/// not recording or nothing new has arrived. Each chunk is resampled
/// independently; the tiny boundary discontinuity is inaudible for speech.
pub fn drain_16k(state: &AudioState) -> Vec<f32> {
    let mut active = state.active.lock().unwrap();
    let Some(rec) = active.as_mut() else {
        return Vec::new();
    };
    let raw = rec.buffer.lock().unwrap();
    if raw.len() <= rec.drained {
        return Vec::new();
    }
    // cpal callback buffers are frame-aligned, so `raw.len()` is always a whole
    // number of frames — no mid-frame split to worry about.
    let slice = raw[rec.drained..].to_vec();
    rec.drained = raw.len();
    let (sample_rate, channels) = (rec.sample_rate, rec.channels);
    drop(raw);
    drop(active);

    let mono = downmix(&slice, channels);
    resample(&mono, sample_rate, WHISPER_SAMPLE_RATE).unwrap_or_default()
}

pub fn stop_recording(state: &AudioState) -> Result<RecordingResult, String> {
    let rec = state
        .active
        .lock()
        .unwrap()
        .take()
        .ok_or("not recording")?;

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
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
        WindowFunction,
    };
    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };
    let chunk = 1024;
    let mut resampler = SincFixedIn::<f32>::new(to as f64 / from as f64, 2.0, params, chunk, 1)
        .map_err(|e| e.to_string())?;

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
