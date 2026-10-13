use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tauri::{AppHandle, Manager};

pub const DEFAULT_SHORTCUT: &str = "cmd+shift+space";

/// A learned correction: `from` is what whisper tends to hear, `to` is what
/// the speaker actually means. The `to` side also biases recognition.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DictEntry {
    pub from: String,
    pub to: String,
}

/// A writing style the cleanup model imitates, defined by pasted samples
/// of the user's own writing.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Style {
    pub id: String,
    pub name: String,
    /// Explicit style rules (e.g. "all lowercase, even 'i'"); these carry
    /// more weight with small models than inference from samples.
    #[serde(default)]
    pub notes: String,
    /// Deterministically lowercase the output in code — casing is mechanical
    /// and small models are unreliable at it.
    #[serde(default)]
    pub lowercase: bool,
    #[serde(default)]
    pub samples: Vec<String>,
}

/// A tool the user describes in their own words instead of a built-in
/// integration: what it's for, where it lives, and their rules for it
/// ("if no project is mentioned, use BLAH"). The assistant picks guides by
/// `when`, follows `guidance` while drafting, and — in browser mode — does
/// the work on `site` in the user's own, signed-in browser.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ToolGuide {
    pub id: String,
    pub name: String,
    /// When to use it, e.g. "creating or updating Jira tickets".
    pub when: String,
    /// Where to start in the browser, e.g. "https://acme.atlassian.net".
    pub site: String,
    /// The user's rules: defaults, conventions, what never to do.
    pub guidance: String,
    /// "browser" (do it on the site in my browser), "auto" (a built-in
    /// integration when there is one, else the browser), or "draft" (just
    /// draft it for me to paste).
    pub mode: String,
    /// Fill everything in but leave the final click (Create/Send/Save) to me.
    pub leave_final_step: bool,
    pub enabled: bool,
}

impl Default for ToolGuide {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            when: String::new(),
            site: String::new(),
            guidance: String::new(),
            mode: "browser".into(),
            leave_final_step: false,
            enabled: true,
        }
    }
}

/// The meeting assistant: what it listens for, which model thinks, and the
/// tools it may use.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AssistantSettings {
    pub enabled: bool,
    /// Say this name, then a request, during a meeting ("unsound, draft…").
    pub wake_word: String,
    pub listen_for_wake_word: bool,
    /// Which model thinks for the assistant:
    ///  • "local" — a downloaded GGUF, in-process;
    ///  • "openai" / "anthropic" — their APIs, with the user's key;
    ///  • "compatible" — any OpenAI-compatible server (Fireworks, Groq,
    ///    Together, OpenRouter, Ollama, LM Studio, llama-server…);
    ///  • "claude" / "codex" — the user's own Claude or ChatGPT subscription,
    ///    through the Claude Code or Codex CLI they're signed into.
    /// ("endpoint" is the old name for "compatible".)
    pub backend: String,
    /// Model id for the local backend; empty = the meeting summary model.
    pub local_model: String,
    /// Model for "openai" / "anthropic".
    pub cloud_model: String,
    /// Model for "claude" / "codex" ("" = the CLI's default).
    pub cli_model: String,
    pub endpoint_url: String,
    pub endpoint_model: String,
    /// Actions that run without asking first (drafting never needs asking).
    pub auto_run: Vec<String>,
    pub jira: crate::tools::jira::JiraConfig,
    pub mcp_servers: Vec<crate::tools::mcp::McpServer>,
    /// Allow handing open-ended requests to Claude Code (`claude -p`).
    pub claude_code: bool,
    pub claude_allowed_tools: Vec<String>,
    /// Claude Code model for hand-offs and browser runs ("" = its default;
    /// "sonnet" is quicker and cheaper for clicking through a site).
    pub claude_model: String,
    /// The user's own tool descriptions and rules.
    pub guides: Vec<ToolGuide>,
    /// Always-included context for drafts: name, role, team, sign-off.
    pub about_me: String,
    /// How website work runs: "chrome" (Claude in Chrome through Claude Code,
    /// in the user's signed-in Chrome), "local" (the assistant's own model
    /// driving the browser through an MCP browser server), or "off".
    pub browser: String,
    /// The MCP browser server for "local" (Playwright MCP in extension mode
    /// attaches to the user's running, signed-in Chrome).
    pub browser_server: crate::tools::mcp::McpServer,
}

/// Playwright MCP's extension mode: attaches to the user's running Chrome
/// (signed-in tabs) through the "Playwright MCP Bridge" extension.
pub fn default_browser_server() -> crate::tools::mcp::McpServer {
    crate::tools::mcp::McpServer {
        name: "browser".into(),
        command: "npx".into(),
        args: vec!["-y".into(), "@playwright/mcp@latest".into(), "--extension".into()],
        enabled: true,
        ..Default::default()
    }
}

impl Default for AssistantSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            wake_word: "unsound".into(),
            listen_for_wake_word: true,
            backend: "local".into(),
            local_model: String::new(),
            cloud_model: String::new(),
            cli_model: String::new(),
            endpoint_url: String::new(),
            endpoint_model: String::new(),
            auto_run: vec!["note".into()],
            jira: Default::default(),
            mcp_servers: Vec::new(),
            claude_code: false,
            claude_allowed_tools: Vec::new(),
            claude_model: String::new(),
            guides: Vec::new(),
            about_me: String::new(),
            browser: "off".into(),
            browser_server: default_browser_server(),
        }
    }
}

/// Keychain keys for the cloud backends' API keys.
pub const ENDPOINT_KEY: &str = "assistant-endpoint-key";
pub const OPENAI_KEY: &str = "openai-api-key";
pub const ANTHROPIC_KEY: &str = "anthropic-api-key";

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MeetingSettings {
    /// Offer to record when a call app (Zoom, Meet, Teams…) takes the mic.
    pub detect_meetings: bool,
    /// Title meetings and name attendees from the current calendar event.
    pub use_calendar: bool,
    /// Speaker-embedding model for live speaker labels; empty = default.
    pub embedding_model: String,
    /// Summary model for background notes during a meeting; empty = the one
    /// the Meetings tab passes at start.
    pub summary_model: String,
}

impl Default for MeetingSettings {
    fn default() -> Self {
        Self {
            detect_meetings: true,
            use_calendar: false,
            embedding_model: String::new(),
            summary_model: String::new(),
        }
    }
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    /// Press once to start, again to stop. "cmd+shift+space" form.
    pub hands_free: Vec<String>,
    /// Hold to record, release to finish.
    pub push_to_talk: Vec<String>,
    /// Input device name; empty string means the system default.
    pub mic_device: String,
    pub styles: Vec<Style>,
    /// Style id applied by default; empty string means neutral (no style).
    pub default_style: String,
    /// Personal dictionary built from the user's corrections.
    pub dictionary: Vec<DictEntry>,
    /// Hold to speak a request to the assistant (during a meeting, or any time).
    pub command_keys: Vec<String>,
    pub assistant: AssistantSettings,
    pub meetings: MeetingSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            hands_free: vec![DEFAULT_SHORTCUT.into()],
            push_to_talk: vec![],
            mic_device: String::new(),
            styles: vec![],
            default_style: String::new(),
            dictionary: vec![],
            command_keys: vec![],
            assistant: AssistantSettings::default(),
            meetings: MeetingSettings::default(),
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
    styles: Option<Vec<Style>>,
    default_style: Option<String>,
    dictionary: Option<Vec<DictEntry>>,
    command_keys: Option<Vec<String>>,
    assistant: Option<AssistantSettings>,
    meetings: Option<MeetingSettings>,
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
        styles: raw.styles.unwrap_or_default(),
        default_style: raw.default_style.unwrap_or_default(),
        dictionary: raw.dictionary.unwrap_or_default(),
        command_keys: raw.command_keys.unwrap_or_default(),
        assistant: raw.assistant.unwrap_or_default(),
        meetings: raw.meetings.unwrap_or_default(),
    }
}

pub fn save(app: &AppHandle, settings: &Settings) -> Result<(), String> {
    let data = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    fs::write(settings_path(app)?, data).map_err(|e| e.to_string())
}
