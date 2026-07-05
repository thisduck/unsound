use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tauri::{AppHandle, Manager};

pub const DEFAULT_SHORTCUT: &str = "cmd+shift+space";

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    /// Press once to start, again to stop. "cmd+shift+space" form.
    pub hands_free: Vec<String>,
    /// Hold to record, release to finish.
    pub push_to_talk: Vec<String>,
    /// Input device name; empty string means the system default.
    pub mic_device: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            hands_free: vec![DEFAULT_SHORTCUT.into()],
            push_to_talk: vec![],
            mic_device: String::new(),
        }
    }
}

/// On-disk shape, including the pre-multi-shortcut `shortcut` field.
#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct RawSettings {
    shortcut: Option<String>,
    hands_free: Option<Vec<String>>,
    push_to_talk: Option<Vec<String>>,
    mic_device: Option<String>,
}

fn settings_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir.join("settings.json"))
}

pub fn load(app: &AppHandle) -> Settings {
    let raw: Option<RawSettings> = settings_path(app)
        .ok()
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|data| serde_json::from_str(&data).ok());
    let Some(raw) = raw else {
        return Settings::default();
    };
    Settings {
        hands_free: raw.hands_free.unwrap_or_else(|| match raw.shortcut {
            // Migrate the old single shortcut; it was hands-free behavior.
            Some(s) if !s.trim().is_empty() => vec![s],
            Some(_) => vec![], // explicitly disabled
            None => vec![DEFAULT_SHORTCUT.into()],
        }),
        push_to_talk: raw.push_to_talk.unwrap_or_default(),
        mic_device: raw.mic_device.unwrap_or_default(),
    }
}

pub fn save(app: &AppHandle, settings: &Settings) -> Result<(), String> {
    let data = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    fs::write(settings_path(app)?, data).map_err(|e| e.to_string())
}
