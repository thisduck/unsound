//! Decode an uploaded audio file to the 16 kHz mono f32 samples whisper wants.
//!
//! Symphonia handles wav/mp3/m4a/aac/flac/ogg-vorbis/caf/mkv. Opus (WhatsApp
//! voice notes) has no Symphonia decoder, so Ogg-Opus is demuxed with the
//! `ogg` crate and decoded with libopus. The real format is sniffed from the
//! file's bytes, never its extension — WhatsApp downloads sometimes carry a
//! bogus `.jpg`/`.dat` name while the bytes are still Opus.

use std::fs;
use std::path::Path;

use crate::audio::WHISPER_SAMPLE_RATE;

/// Read a file, decode whatever audio it contains, and return 16 kHz mono.
pub fn decode_to_samples(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("could not read file: {e}"))?;
    if bytes.len() < 16 {
        return Err("file is empty or too small to be audio".into());
    }
    if let Some(kind) = image_kind(&bytes) {
        return Err(format!(
            "this looks like {kind}, not audio — pick an audio file or voice note"
        ));
    }

    let (samples, rate, channels) = if is_ogg_opus(&bytes) {
        decode_ogg_opus(&bytes)?
    } else if is_mp4(&bytes) {
        // Symphonia reads plain m4a but not fragmented MP4 (WhatsApp exports
        // heavily fragmented AAC). Try Symphonia first, then remux the AAC
        // ourselves so both work.
        match decode_symphonia(&bytes, path) {
            Ok(r) => r,
            Err(_) => decode_mp4_aac(&bytes)?,
        }
    } else {
        decode_symphonia(&bytes, path)?
    };

    if samples.is_empty() {
        return Err("no audio could be decoded from this file".into());
    }
    let mono = downmix(&samples, channels);
    resample(&mono, rate, WHISPER_SAMPLE_RATE)
}

/// Detect genuine image files so they get a helpful message rather than a
/// cryptic decode failure. The WhatsApp case is NOT caught here — those bytes
/// are Opus, so `is_ogg_opus` handles them regardless of a `.jpg` name.
fn image_kind(b: &[u8]) -> Option<&'static str> {
    if b.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("a JPEG image")
    } else if b.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("a PNG image")
    } else if b.starts_with(b"GIF87a") || b.starts_with(b"GIF89a") {
        Some("a GIF image")
    } else if b.len() >= 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        Some("a WebP image")
    } else if b.starts_with(b"\x00\x00\x01\x00") {
        Some("an icon")
    } else {
        None
    }
}

/// Ogg-Opus starts with an "OggS" page whose first packet is "OpusHead".
fn is_ogg_opus(b: &[u8]) -> bool {
    b.starts_with(b"OggS") && b.windows(8).take(128).any(|w| w == b"OpusHead")
}

/// MP4/ISO-BMFF files carry an "ftyp" box near the start.
fn is_mp4(b: &[u8]) -> bool {
    b.len() >= 12 && &b[4..8] == b"ftyp"
}

/// Decode AAC out of any MP4 (including fragmented) by re-framing each raw
/// AAC sample as ADTS, then handing the self-framing ADTS stream to
/// Symphonia's AAC decoder.
fn decode_mp4_aac(bytes: &[u8]) -> Result<(Vec<f32>, u32, u16), String> {
    use mp4::{Mp4Reader, TrackType};

    let cursor = std::io::Cursor::new(bytes.to_vec());
    let size = bytes.len() as u64;
    let mut mp4 = Mp4Reader::read_header(cursor, size)
        .map_err(|e| format!("could not read MP4: {e}"))?;

    // First audio track.
    let (track_id, freq_idx, obj_type, chan) = mp4
        .tracks()
        .iter()
        .find(|(_, t)| t.track_type().ok() == Some(TrackType::Audio))
        .map(|(id, t)| {
            (
                *id,
                t.sample_freq_index().ok(),
                t.audio_profile().ok(),
                t.channel_config().ok(),
            )
        })
        .ok_or("this MP4 has no audio track to transcribe")?;

    let freq_idx = freq_idx.ok_or("unsupported AAC sample rate in this file")?;
    let sample_rate = freq_idx.freq();
    let adts_freq = adts_freq_index(sample_rate);
    // Object type: default to AAC-LC (2) if unreadable; ADTS profile is type-1.
    let profile_bits = obj_type.map(|o| o as u8).unwrap_or(2).saturating_sub(1) & 0x3;
    let chan_val = chan.map(|c| c as u8).unwrap_or(1);
    let channels = chan_val.max(1) as u16;

    let count = mp4.sample_count(track_id).map_err(|e| e.to_string())?;
    let mut adts: Vec<u8> = Vec::with_capacity(bytes.len());
    for i in 1..=count {
        if let Ok(Some(sample)) = mp4.read_sample(track_id, i) {
            let frame_len = sample.bytes.len() + 7;
            if frame_len >= (1 << 13) {
                continue;
            }
            // 7-byte ADTS header (no CRC).
            adts.push(0xFF);
            adts.push(0xF1);
            adts.push((profile_bits << 6) | ((adts_freq & 0xF) << 2) | ((chan_val >> 2) & 0x1));
            adts.push(((chan_val & 0x3) << 6) | ((frame_len >> 11) as u8 & 0x3));
            adts.push((frame_len >> 3) as u8);
            adts.push(((frame_len as u8 & 0x7) << 5) | 0x1F);
            adts.push(0xFC);
            adts.extend_from_slice(&sample.bytes);
        }
    }
    if adts.is_empty() {
        return Err("no AAC audio frames found in this file".into());
    }

    let (samples, rate, ch) = decode_adts(&adts)?;
    Ok((samples, if rate > 0 { rate } else { sample_rate }, if ch > 0 { ch } else { channels }))
}

/// The 4-bit ADTS sampling-frequency index for a rate in Hz.
fn adts_freq_index(hz: u32) -> u8 {
    match hz {
        96000 => 0, 88200 => 1, 64000 => 2, 48000 => 3, 44100 => 4, 32000 => 5,
        24000 => 6, 22050 => 7, 16000 => 8, 12000 => 9, 11025 => 10, 8000 => 11,
        7350 => 12, _ => 4,
    }
}

/// Decode a raw ADTS AAC byte stream to interleaved f32 via Symphonia.
fn decode_adts(adts: &[u8]) -> Result<(Vec<f32>, u32, u16), String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let mss = MediaSourceStream::new(Box::new(std::io::Cursor::new(adts.to_vec())), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("aac");
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("AAC re-frame failed: {e}"))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("no AAC track after re-framing")?;
    let track_id = track.id;
    let mut rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut channels = track.codec_params.channels.map(|c| c.count() as u16).unwrap_or(0);
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("no AAC decoder: {e}"))?;

    let mut samples: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                rate = spec.rate;
                channels = spec.channels.count() as u16;
                let mut sb = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                sb.copy_interleaved_ref(decoded);
                samples.extend_from_slice(sb.samples());
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(_) => break,
        }
    }
    Ok((samples, rate, channels))
}

fn decode_symphonia(bytes: &[u8], path: &Path) -> Result<(Vec<f32>, u32, u16), String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let source = std::io::Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(source), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|_| {
            "unsupported audio format — try wav, mp3, m4a, flac, ogg, or an opus voice note"
                .to_string()
        })?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("file has no decodable audio track")?;
    let track_id = track.id;
    let mut rate = track.codec_params.sample_rate.unwrap_or(WHISPER_SAMPLE_RATE);
    let mut channels = track.codec_params.channels.map(|c| c.count() as u16).unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("no decoder for this audio codec: {e}"))?;

    let mut samples: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break, // end of stream
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                rate = spec.rate;
                channels = spec.channels.count() as u16;
                let mut sb = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                sb.copy_interleaved_ref(decoded);
                samples.extend_from_slice(sb.samples());
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(format!("decode error: {e}")),
        }
    }
    Ok((samples, rate, channels))
}

fn decode_ogg_opus(bytes: &[u8]) -> Result<(Vec<f32>, u32, u16), String> {
    use opus::{Channels, Decoder};

    let mut reader = ogg::PacketReader::new(std::io::Cursor::new(bytes.to_vec()));

    // First packet: OpusHead — channel count at byte 9.
    let head = loop {
        match reader.read_packet().map_err(|e| e.to_string())? {
            Some(p) if p.data.starts_with(b"OpusHead") => break p,
            Some(_) => continue,
            None => return Err("not a valid Opus stream".into()),
        }
    };
    let src_channels = *head.data.get(9).unwrap_or(&1);
    let channels = if src_channels >= 2 { Channels::Stereo } else { Channels::Mono };
    let ch_count = if src_channels >= 2 { 2u16 } else { 1u16 };

    // libopus always decodes to 48 kHz; we resample afterward.
    const OPUS_RATE: u32 = 48_000;
    let mut decoder = Decoder::new(OPUS_RATE, channels)
        .map_err(|e| format!("failed to init Opus decoder: {e}"))?;

    let mut samples: Vec<f32> = Vec::new();
    let mut buf = vec![0f32; (OPUS_RATE as usize / 1000 * 120) * ch_count as usize];
    while let Some(packet) = reader.read_packet().map_err(|e| e.to_string())? {
        // Skip the OpusTags comment header.
        if packet.data.starts_with(b"OpusTags") {
            continue;
        }
        match decoder.decode_float(&packet.data, &mut buf, false) {
            Ok(frames) => samples.extend_from_slice(&buf[..frames * ch_count as usize]),
            Err(_) => continue,
        }
    }
    Ok((samples, OPUS_RATE, ch_count))
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
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
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
