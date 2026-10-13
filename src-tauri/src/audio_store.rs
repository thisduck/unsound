//! Meeting audio on disk. Both channels are written as they're captured —
//! `<id>.wav` (system audio, "them"; the name predates the mic file) and
//! `<id>.mic.wav` ("me") — as 16 kHz mono 16-bit WAV, so a meeting never has
//! to fit in memory, a crash loses at most a few seconds, and any moment can
//! be played back from the transcript.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};

pub const RATE: u32 = 16_000;
const HEADER: u64 = 44;
/// Rewrite the header's sizes about this often, so a file cut off by a crash
/// still says how much audio it holds.
const HEADER_EVERY_SAMPLES: u64 = RATE as u64 * 5;

pub fn dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("meeting-audio");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// The system-audio ("them") file for a meeting.
pub fn system_path(app: &AppHandle, id: &str) -> Result<PathBuf, String> {
    Ok(dir(app)?.join(format!("{id}.wav")))
}

/// The microphone ("me") file for a meeting.
pub fn mic_path(app: &AppHandle, id: &str) -> Result<PathBuf, String> {
    Ok(dir(app)?.join(format!("{id}.mic.wav")))
}

pub fn has_audio(app: &AppHandle, id: &str) -> bool {
    [system_path(app, id), mic_path(app, id)]
        .into_iter()
        .flatten()
        .any(|p| p.metadata().map(|m| m.len() > HEADER).unwrap_or(false))
}

pub fn delete(app: &AppHandle, id: &str) {
    for p in [system_path(app, id), mic_path(app, id)].into_iter().flatten() {
        let _ = std::fs::remove_file(p);
    }
}

fn header(data_bytes: u32) -> [u8; 44] {
    let mut h = [0u8; 44];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&(36 + data_bytes).to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&1u16.to_le_bytes()); // PCM
    h[22..24].copy_from_slice(&1u16.to_le_bytes()); // mono
    h[24..28].copy_from_slice(&RATE.to_le_bytes());
    h[28..32].copy_from_slice(&(RATE * 2).to_le_bytes());
    h[32..34].copy_from_slice(&2u16.to_le_bytes());
    h[34..36].copy_from_slice(&16u16.to_le_bytes());
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&data_bytes.to_le_bytes());
    h
}

/// Appends 16 kHz mono audio to a WAV file as it arrives.
pub struct WavWriter {
    file: File,
    samples: u64,
    since_header: u64,
}

impl WavWriter {
    pub fn create(path: &Path) -> Result<Self, String> {
        let mut file = File::create(path).map_err(|e| format!("can't create {path:?}: {e}"))?;
        file.write_all(&header(0)).map_err(|e| e.to_string())?;
        Ok(Self {
            file,
            samples: 0,
            since_header: 0,
        })
    }

    pub fn append(&mut self, samples: &[f32]) -> Result<(), String> {
        if samples.is_empty() {
            return Ok(());
        }
        let mut buf = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            buf.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
        }
        self.file.write_all(&buf).map_err(|e| e.to_string())?;
        self.samples += samples.len() as u64;
        self.since_header += samples.len() as u64;
        if self.since_header >= HEADER_EVERY_SAMPLES {
            self.write_header()?;
        }
        Ok(())
    }

    /// Pad with silence up to `total` samples — keeps a channel that went
    /// quiet (or stopped delivering) aligned with the meeting clock.
    pub fn pad_to(&mut self, total: u64) -> Result<(), String> {
        if total > self.samples {
            let n = (total - self.samples) as usize;
            self.append(&vec![0.0; n])?;
        }
        Ok(())
    }

    fn write_header(&mut self) -> Result<(), String> {
        let bytes = (self.samples * 2).min(u32::MAX as u64 - 36) as u32;
        self.file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
        self.file.write_all(&header(bytes)).map_err(|e| e.to_string())?;
        self.file.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
        self.since_header = 0;
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64, String> {
        self.write_header()?;
        self.file.sync_all().map_err(|e| e.to_string())?;
        Ok(self.samples)
    }
}

/// Make a WAV cut off mid-write (crash, force quit) consistent: its header
/// sizes are set from the file length.
pub fn repair(path: &Path) -> Result<(), String> {
    let Ok(mut file) = OpenOptions::new().read(true).write(true).open(path) else {
        return Ok(());
    };
    let len = file.metadata().map_err(|e| e.to_string())?.len();
    if len < HEADER {
        return Ok(());
    }
    let data = ((len - HEADER) / 2 * 2).min(u32::MAX as u64 - 36) as u32;
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    file.write_all(&header(data)).map_err(|e| e.to_string())
}

/// Read `[start_ms, end_ms)` of a WAV we wrote as f32 samples, seeking
/// straight to the range (hour-long files are never loaded whole).
pub fn read_range(path: &Path, start_ms: i64, end_ms: i64) -> Result<Vec<f32>, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let total = file.metadata().map_err(|e| e.to_string())?.len().saturating_sub(HEADER) / 2;
    let from = ((start_ms.max(0) as u64) * RATE as u64 / 1000).min(total);
    let to = ((end_ms.max(0) as u64) * RATE as u64 / 1000).min(total);
    if to <= from {
        return Ok(Vec::new());
    }
    file.seek(SeekFrom::Start(HEADER + from * 2))
        .map_err(|e| e.to_string())?;
    let mut bytes = vec![0u8; ((to - from) * 2) as usize];
    file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    Ok(bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect())
}

/// A playable WAV (both channels mixed) of a stretch of a meeting.
pub fn clip(app: &AppHandle, id: &str, start_ms: i64, end_ms: i64) -> Result<Vec<u8>, String> {
    let mut mix: Vec<f32> = Vec::new();
    for path in [system_path(app, id)?, mic_path(app, id)?] {
        if !path.exists() {
            continue;
        }
        let part = read_range(&path, start_ms, end_ms)?;
        if mix.len() < part.len() {
            mix.resize(part.len(), 0.0);
        }
        for (m, s) in mix.iter_mut().zip(part) {
            *m += s;
        }
    }
    if mix.is_empty() {
        return Err("no audio was saved for this part of the meeting".into());
    }
    let mut out = Vec::with_capacity(44 + mix.len() * 2);
    out.extend_from_slice(&header((mix.len() * 2) as u32));
    for s in mix {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_reads_and_repairs() {
        let dir = std::env::temp_dir().join(format!("unsound-wav-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.wav");
        let mut w = WavWriter::create(&path).unwrap();
        let tone: Vec<f32> = (0..32_000).map(|i| ((i % 100) as f32 / 100.0) - 0.5).collect();
        w.append(&tone).unwrap();
        w.pad_to(48_000).unwrap();
        assert_eq!(w.finish().unwrap(), 48_000);
        assert_eq!(path.metadata().unwrap().len(), HEADER + 96_000);
        let mid = read_range(&path, 1000, 1500).unwrap();
        assert_eq!(mid.len(), 8000);
        assert!((mid[0] - tone[16_000]).abs() < 0.001);
        assert!(read_range(&path, 2500, 2600).unwrap().iter().all(|s| *s == 0.0));
        // A stale header (crash) is fixed from the file length.
        let mut f = OpenOptions::new().write(true).open(&path).unwrap();
        f.write_all(&header(0)).unwrap();
        repair(&path).unwrap();
        let decoded = crate::audiofile::decode_to_samples(&path).unwrap();
        assert_eq!(decoded.len(), 48_000);
        let _ = std::fs::remove_dir_all(dir);
    }
}
