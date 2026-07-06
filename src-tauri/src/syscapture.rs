//! System-audio capture via ScreenCaptureKit (macOS 13+).
//!
//! This is the "them" half of a meeting: whatever is coming out of the
//! speakers — the remote participants in Meet/Zoom/etc. The microphone ("me")
//! stays on the existing `cpal` path in `audio.rs`. Keeping the two channels
//! separate is what lets us label who-said-what without a diarization model,
//! and leaves room to cluster the system channel into multiple speakers later.
//!
//! ScreenCaptureKit's audio capture (`SCStreamConfiguration.capturesAudio`)
//! needs macOS 13, so everything here is runtime-gated by `is_supported()` and
//! the feature is simply hidden on older systems. It also requires the Screen
//! Recording permission — the OS prompts on first use.
//!
//! We ask SCK for 16 kHz mono float PCM directly (a rate it supports), which is
//! exactly what Whisper wants, so there is no resampling on this path.

#![cfg(target_os = "macos")]

use std::ffi::c_void;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use block2::RcBlock;
use dispatch2::{DispatchQueue, DispatchQueueAttr, DispatchRetained};
use objc2::rc::Retained;
use objc2::runtime::{NSObject, NSObjectProtocol, ProtocolObject};
use objc2::{available, define_class, msg_send, AnyThread, DefinedClass};
use objc2_core_audio_types::{AudioBuffer, AudioBufferList};
use objc2_core_media::{CMBlockBuffer, CMSampleBuffer};
use objc2_foundation::{NSArray, NSError};
use objc2_screen_capture_kit::{
    SCContentFilter, SCShareableContent, SCStream, SCStreamConfiguration, SCStreamOutput,
    SCStreamOutputType, SCWindow,
};

/// SCK delivers this rate/layout to us, which is what Whisper consumes.
pub const CAPTURE_SAMPLE_RATE: u32 = 16_000;

extern "C" {
    /// Release the +1 CMBlockBuffer handed back by the audio-buffer-list call.
    fn CFRelease(cf: *const c_void);
}

/// Log the first audio buffer's shape once, to confirm what SCK delivers.
static LOGGED_FORMAT: AtomicBool = AtomicBool::new(false);

/// Whether system-audio capture is possible here. SCStream audio is macOS 13+.
pub fn is_supported() -> bool {
    available!(macos = 13.0)
}

fn err_msg(err: *mut NSError) -> String {
    if err.is_null() {
        return "unknown error".into();
    }
    // Safe: non-null NSError from a completion handler.
    let err = unsafe { &*err };
    err.localizedDescription().to_string()
}

// ---- the SCStreamOutput delegate -------------------------------------------

/// Instance state for the delegate: the shared sink captured samples land in.
struct Ivars {
    sink: Arc<Mutex<Vec<f32>>>,
}

define_class!(
    // SAFETY: NSObject has no subclassing requirements and we don't impl Drop.
    #[unsafe(super(NSObject))]
    #[name = "UnsoundSysAudioOutput"]
    #[ivars = Ivars]
    struct AudioOutput;

    unsafe impl NSObjectProtocol for AudioOutput {}

    unsafe impl SCStreamOutput for AudioOutput {
        #[unsafe(method(stream:didOutputSampleBuffer:ofType:))]
        fn did_output(
            &self,
            _stream: &SCStream,
            sample_buffer: &CMSampleBuffer,
            kind: SCStreamOutputType,
        ) {
            // We only added an audio output, but guard anyway.
            if kind.0 != SCStreamOutputType::Audio.0 {
                return;
            }
            let samples = extract_mono(sample_buffer);
            if !samples.is_empty() {
                self.ivars().sink.lock().unwrap().extend_from_slice(&samples);
            }
        }
    }
);

impl AudioOutput {
    fn new(sink: Arc<Mutex<Vec<f32>>>) -> Retained<Self> {
        let this = Self::alloc().set_ivars(Ivars { sink });
        unsafe { msg_send![super(this), init] }
    }
}

/// Copy the sample buffer's PCM out as mono f32. We requested one 16 kHz mono
/// channel, so the common case is a single planar float buffer; we also fold
/// interleaved stereo down defensively in case a host ignores the request.
fn extract_mono(sbuf: &CMSampleBuffer) -> Vec<f32> {
    let mut abl = AudioBufferList {
        mNumberBuffers: 0,
        mBuffers: [AudioBuffer {
            mNumberChannels: 0,
            mDataByteSize: 0,
            mData: std::ptr::null_mut(),
        }; 1],
    };
    let mut block_buffer: *mut CMBlockBuffer = std::ptr::null_mut();
    let mut size_needed: usize = 0;

    // SAFETY: pointers are valid for the duration of the call; the returned
    // block buffer owns the memory `abl.mBuffers[..].mData` points at, so we
    // copy before releasing it below.
    let status = unsafe {
        sbuf.audio_buffer_list_with_retained_block_buffer(
            &mut size_needed,
            &mut abl,
            std::mem::size_of::<AudioBufferList>(),
            None,
            None,
            0,
            &mut block_buffer,
        )
    };
    if status != 0 || abl.mNumberBuffers == 0 {
        if !block_buffer.is_null() {
            unsafe { CFRelease(block_buffer as *const c_void) };
        }
        return Vec::new();
    }

    // Frame count, so we can infer the sample format from the byte size rather
    // than assume it. SCK delivers either 32-bit float or 16-bit int PCM, and
    // reading one as the other collapses speech to near-silence.
    let num_samples = unsafe { sbuf.num_samples() }.max(0) as usize;
    let buf = abl.mBuffers[0];
    let channels = buf.mNumberChannels.max(1) as usize;
    let byte_size = buf.mDataByteSize as usize;
    let bytes_per_sample = if num_samples > 0 {
        (byte_size / (num_samples * channels).max(1)).max(1)
    } else {
        4
    };

    let mut out: Vec<f32> = Vec::new();
    if !buf.mData.is_null() && byte_size > 0 {
        if bytes_per_sample == 2 {
            // Int16 PCM → normalize to [-1, 1].
            let n = byte_size / 2;
            // SAFETY: mData points at n contiguous i16 samples.
            let slice = unsafe { std::slice::from_raw_parts(buf.mData as *const i16, n) };
            if channels <= 1 {
                out.extend(slice.iter().map(|&s| s as f32 / 32768.0));
            } else {
                out.reserve(n / channels);
                for frame in slice.chunks_exact(channels) {
                    out.push(
                        frame.iter().map(|&s| s as f32 / 32768.0).sum::<f32>() / channels as f32,
                    );
                }
            }
        } else {
            // Float32 PCM.
            let n = byte_size / 4;
            // SAFETY: mData points at n contiguous f32 samples.
            let slice = unsafe { std::slice::from_raw_parts(buf.mData as *const f32, n) };
            if channels <= 1 {
                out.extend_from_slice(slice);
            } else {
                out.reserve(n / channels);
                for frame in slice.chunks_exact(channels) {
                    out.push(frame.iter().copied().sum::<f32>() / channels as f32);
                }
            }
        }
    }

    // One-shot diagnostics: confirms what SCK actually hands us.
    if !LOGGED_FORMAT.swap(true, Ordering::Relaxed) {
        let peak = out.iter().fold(0f32, |m, &s| m.max(s.abs()));
        eprintln!(
            "[syscapture] first audio buffer: buffers={}, channels={}, bytes={}, numSamples={}, inferred bytes/sample={}, mono peak={:.4}",
            abl.mNumberBuffers, channels, byte_size, num_samples, bytes_per_sample, peak
        );
    }

    if !block_buffer.is_null() {
        unsafe { CFRelease(block_buffer as *const c_void) };
    }
    out
}

// ---- capture lifecycle -----------------------------------------------------

struct Active {
    stop_tx: mpsc::Sender<()>,
    done_rx: mpsc::Receiver<()>,
    buffer: Arc<Mutex<Vec<f32>>>,
    /// Samples already handed out by `drain`, for live meeting streaming.
    drained: usize,
}

/// Mirrors `AudioState`: one capture at a time, plus the finished samples.
#[derive(Default)]
pub struct SysCaptureState {
    active: Mutex<Option<Active>>,
    /// The last finished capture, 16 kHz mono, ready for Whisper.
    pub last: Mutex<Vec<f32>>,
}

/// Result of a finished system-audio capture, parallel to `RecordingResult`.
pub struct CaptureResult {
    pub duration_secs: f32,
    pub sample_count: usize,
}

pub fn start(state: &SysCaptureState) -> Result<(), String> {
    if !is_supported() {
        return Err("system-audio capture needs macOS 13 or newer".into());
    }
    let mut active = state.active.lock().unwrap();
    if active.is_some() {
        return Err("already capturing system audio".into());
    }

    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    // The SCStream and its Objective-C friends are !Send, so like the cpal
    // stream they live entirely on their own thread; this channel reports the
    // start result (or an error) back to us.
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();

    let thread_buffer = buffer.clone();
    std::thread::spawn(move || {
        // Build the whole pipeline. Anything that fails reports Err and bails.
        let build = || -> Result<(Retained<SCStream>, Retained<AudioOutput>, DispatchRetained<DispatchQueue>), String> {
            // 1. Fetch the shareable content (displays). This is async; block
            // the worker thread until the completion handler answers.
            let (content_tx, content_rx) =
                mpsc::channel::<Result<Retained<SCShareableContent>, String>>();
            let content_handler = RcBlock::new(
                move |content: *mut SCShareableContent, err: *mut NSError| {
                    if content.is_null() {
                        let _ = content_tx.send(Err(err_msg(err)));
                    } else {
                        // Retain: the pointer is borrowed for the callback only.
                        match unsafe { Retained::retain(content) } {
                            Some(c) => {
                                let _ = content_tx.send(Ok(c));
                            }
                            None => {
                                let _ = content_tx.send(Err("no shareable content".into()));
                            }
                        }
                    }
                },
            );
            unsafe { SCShareableContent::getShareableContentWithCompletionHandler(&content_handler) };
            let content = content_rx
                .recv()
                .map_err(|_| "screen-recording permission was denied, or capture is unavailable".to_string())??;

            // 2. A content filter over the first display (audio ignores the
            // visual content, but SCK requires a display-backed filter).
            let displays = unsafe { content.displays() };
            let display = displays
                .firstObject()
                .ok_or("no display is available to capture from")?;
            let no_windows = NSArray::<SCWindow>::new();
            let filter = unsafe {
                SCContentFilter::initWithDisplay_excludingWindows(
                    SCContentFilter::alloc(),
                    &display,
                    &no_windows,
                )
            };

            // 3. Configuration: 16 kHz mono audio, our own app excluded, and a
            // tiny video frame we never read (SCK still captures video).
            let config = unsafe { SCStreamConfiguration::new() };
            unsafe {
                config.setCapturesAudio(true);
                config.setSampleRate(CAPTURE_SAMPLE_RATE as isize);
                config.setChannelCount(1);
                config.setExcludesCurrentProcessAudio(true);
                config.setWidth(2);
                config.setHeight(2);
            }

            // 4. The stream, our delegate, and a serial delivery queue.
            let output = AudioOutput::new(thread_buffer.clone());
            let stream = unsafe {
                SCStream::initWithFilter_configuration_delegate(
                    SCStream::alloc(),
                    &filter,
                    &config,
                    None,
                )
            };
            let queue = DispatchQueue::new("com.unsound.sysaudio", DispatchQueueAttr::SERIAL);
            let proto = ProtocolObject::from_ref(&*output);
            unsafe {
                stream.addStreamOutput_type_sampleHandlerQueue_error(
                    proto,
                    SCStreamOutputType::Audio,
                    Some(&queue),
                )
            }
            .map_err(|e| err_msg(Retained::as_ptr(&e) as *mut NSError))?;

            // 5. Start, waiting for the completion handler's verdict.
            let (start_tx, start_rx) = mpsc::channel::<Result<(), String>>();
            let start_handler = RcBlock::new(move |err: *mut NSError| {
                if err.is_null() {
                    let _ = start_tx.send(Ok(()));
                } else {
                    let _ = start_tx.send(Err(err_msg(err)));
                }
            });
            unsafe { stream.startCaptureWithCompletionHandler(Some(&start_handler)) };
            start_rx
                .recv()
                .map_err(|_| "system-audio capture failed to start".to_string())??;

            Ok((stream, output, queue))
        };

        match build() {
            Ok((stream, _output, _queue)) => {
                let _ = ready_tx.send(Ok(()));
                // Hold the stream (and its delegate/queue) alive until stop.
                let _ = stop_rx.recv();
                let (fin_tx, fin_rx) = mpsc::channel::<()>();
                let stop_handler = RcBlock::new(move |_err: *mut NSError| {
                    let _ = fin_tx.send(());
                });
                unsafe { stream.stopCaptureWithCompletionHandler(Some(&stop_handler)) };
                let _ = fin_rx.recv_timeout(Duration::from_secs(3));
                drop(stream);
                let _ = done_tx.send(());
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        }
    });

    ready_rx
        .recv()
        .map_err(|_| "system-audio thread died before reporting readiness".to_string())??;

    *active = Some(Active {
        stop_tx,
        done_rx,
        buffer,
        drained: 0,
    });
    Ok(())
}

/// Pull the system audio captured since the last call (already 16 kHz mono) —
/// the "them" side of live meeting transcription. Empty when not capturing or
/// nothing new has arrived.
pub fn drain(state: &SysCaptureState) -> Vec<f32> {
    let mut active = state.active.lock().unwrap();
    let Some(a) = active.as_mut() else {
        return Vec::new();
    };
    let buf = a.buffer.lock().unwrap();
    if buf.len() <= a.drained {
        return Vec::new();
    }
    let out = buf[a.drained..].to_vec();
    a.drained = buf.len();
    out
}

pub fn stop(state: &SysCaptureState) -> Result<CaptureResult, String> {
    let cap = state
        .active
        .lock()
        .unwrap()
        .take()
        .ok_or("not capturing system audio")?;

    let _ = cap.stop_tx.send(());
    let _ = cap.done_rx.recv_timeout(Duration::from_secs(5));

    let samples = std::mem::take(&mut *cap.buffer.lock().unwrap());
    let result = CaptureResult {
        duration_secs: samples.len() as f32 / CAPTURE_SAMPLE_RATE as f32,
        sample_count: samples.len(),
    };
    *state.last.lock().unwrap() = samples;
    Ok(result)
}

/// Write mono f32 samples to a 16-bit PCM WAV — a listenable artifact for
/// verifying capture independently of transcription.
pub fn write_wav_16(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    let data_size = (samples.len() * 2) as u32;
    let byte_rate = sample_rate * 2; // mono, 16-bit
    let mut buf = Vec::with_capacity(44 + data_size as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_size).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // format = PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // channels = mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, buf).map_err(|e| e.to_string())
}
