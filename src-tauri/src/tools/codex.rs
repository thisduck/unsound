//! Locate the Codex CLI (`codex`), so the assistant can think with the user's
//! own ChatGPT subscription — the same way claude_code.rs uses Claude Code.

use super::shell_env;
use std::path::PathBuf;

/// The real `codex` binary: on the login PATH first, then the usual spots.
pub fn find_codex() -> Option<PathBuf> {
    if let Some(p) = shell_env::which("codex") {
        return Some(p);
    }
    let home = std::env::var("HOME").unwrap_or_default();
    [
        format!("{home}/.local/bin/codex"),
        "/opt/homebrew/bin/codex".to_string(),
        "/usr/local/bin/codex".to_string(),
        format!("{home}/.npm-global/bin/codex"),
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|p| shell_env::is_executable(p))
}
