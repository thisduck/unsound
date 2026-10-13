//! Hand open-ended tasks ("look up the ticket we discussed and add a comment")
//! to Claude Code running headless. It brings the user's own MCP servers,
//! browser tools and auth, so unsound doesn't have to reimplement them. Only
//! the tools the user allowlists run; anything else is denied, never prompted.

use super::shell_env;
use serde_json::Value;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClaudeResult {
    pub text: String,
    pub cost_usd: Option<f64>,
    pub is_error: bool,
}

/// The real `claude` binary. In an interactive shell `claude` may be a
/// function or alias wrapping `command claude`, so look for the executable
/// file itself: on the login PATH first, then the usual install spots.
pub fn find_claude() -> Option<PathBuf> {
    if let Some(p) = shell_env::which("claude") {
        return Some(p);
    }
    let home = std::env::var("HOME").unwrap_or_default();
    [
        format!("{home}/.claude/local/claude"),
        format!("{home}/.local/bin/claude"),
        "/opt/homebrew/bin/claude".to_string(),
        "/usr/local/bin/claude".to_string(),
        format!("{home}/.npm-global/bin/claude"),
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|p| shell_env::is_executable(p))
}

/// How to run Claude Code beyond the prompt.
pub struct RunOpts<'a> {
    /// Becomes `--allowedTools` (Claude Code's own names, e.g.
    /// `mcp__atlassian__createJiraIssue`); empty means read-only defaults.
    pub allowed_tools: &'a [String],
    /// Drive the user's own Chrome through Claude in Chrome (`--chrome`).
    /// That connection rides on the claude.ai sign-in, and an API key in the
    /// environment silently takes precedence and disables it — so these runs
    /// drop inherited API-key auth.
    pub chrome: bool,
    /// `--model` ("" = the user's default).
    pub model: &'a str,
    /// Killed if it outlives this.
    pub timeout: Duration,
}

/// Run `claude -p` with `prompt` and return its final answer. `allowed_tools`
/// becomes `--allowedTools`; empty means read-only defaults. The process is
/// killed if it outlives `timeout`.
pub async fn run(
    prompt: &str,
    allowed_tools: &[String],
    timeout: Duration,
) -> Result<ClaudeResult, String> {
    run_with(
        prompt,
        &RunOpts {
            allowed_tools,
            chrome: false,
            model: "",
            timeout,
        },
    )
    .await
}

/// `run`, with the full set of options.
pub async fn run_with(prompt: &str, opts: &RunOpts<'_>) -> Result<ClaudeResult, String> {
    let (allowed_tools, timeout) = (opts.allowed_tools, opts.timeout);
    let bin = find_claude().ok_or(
        "Claude Code isn't installed (or isn't on your PATH) — see claude.com/claude-code",
    )?;
    let cwd = std::env::temp_dir().join("unsound-claude");
    std::fs::create_dir_all(&cwd).map_err(|e| e.to_string())?;

    let mut cmd = tokio::process::Command::new(&bin);
    cmd.arg("-p")
        .arg(prompt)
        .args(["--output-format", "json"]);
    if opts.chrome {
        cmd.arg("--chrome");
        for key in ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"] {
            cmd.env_remove(key);
        }
    }
    if !opts.model.trim().is_empty() {
        cmd.args(["--model", opts.model.trim()]);
    }
    let tools: Vec<&str> = allowed_tools
        .iter()
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .collect();
    if !tools.is_empty() {
        // Variadic flag, so it goes last where it can't swallow anything.
        cmd.arg("--allowedTools").arg(tools.join(","));
    }
    cmd.current_dir(&cwd)
        .env("PATH", shell_env::login_path())
        // Mirror the user's shell wrapper, which blanks these for claude.
        .env("NODE_OPTIONS", "")
        .env("NODE_EXTRA_CA_CERTS", "")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let child = cmd
        .spawn()
        .map_err(|e| format!("could not start Claude Code: {e}"))?;
    // Dropping the future on timeout drops the child, which kills it.
    let out = tokio::time::timeout(timeout, child.wait_with_output())
        .await
        .map_err(|_| format!("Claude Code didn't finish within {}s", timeout.as_secs()))?
        .map_err(|e| e.to_string())?;

    let stdout = String::from_utf8_lossy(&out.stdout);
    match parse_result(&stdout) {
        Some(r) => Ok(r),
        None => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let detail = if stderr.trim().is_empty() { stdout.trim() } else { stderr.trim() };
            let tail: String = detail
                .chars()
                .rev()
                .take(600)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            Err(format!(
                "Claude Code exited ({}) without a result: {tail}",
                out.status
            ))
        }
    }
}

/// The `--output-format json` result object. Scans from the end so any stray
/// log lines before it don't matter.
fn parse_result(stdout: &str) -> Option<ClaudeResult> {
    let candidates = std::iter::once(stdout.trim()).chain(stdout.lines().rev().map(str::trim));
    for chunk in candidates {
        if !chunk.starts_with('{') {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(chunk) else {
            continue;
        };
        if v.get("result").is_none() && v["type"] != "result" {
            continue;
        }
        let is_error = v["is_error"].as_bool().unwrap_or(false)
            || v["subtype"].as_str().is_some_and(|s| s.starts_with("error"));
        let text = v["result"].as_str().unwrap_or_default().to_string();
        let text = if text.is_empty() && is_error {
            v["subtype"].as_str().unwrap_or("error").to_string()
        } else {
            text
        };
        return Some(ClaudeResult {
            text,
            cost_usd: v["total_cost_usd"].as_f64().or_else(|| v["cost_usd"].as_f64()),
            is_error,
        });
    }
    None
}

#[cfg(test)]
mod tests {

    /// Drives the user's real Chrome through Claude in Chrome (read-only).
    /// `cargo test --lib -- --ignored chrome_smoke --nocapture`
    #[tokio::test]
    #[ignore]
    async fn chrome_smoke() {
        let r = run_with(
            "Open https://en.wikipedia.org/wiki/Main_Page in a new tab and reply with the title of today's featured article, and nothing else. Don't click anything else.",
            &RunOpts {
                allowed_tools: &["mcp__claude-in-chrome".to_string()],
                chrome: true,
                model: "sonnet",
                timeout: Duration::from_secs(300),
            },
        )
        .await
        .unwrap();
        println!("{r:?}");
        assert!(!r.is_error);
    }

    use super::*;

    #[test]
    fn parses_result_object() {
        let out = "some log line\n{\"type\":\"result\",\"subtype\":\"success\",\"is_error\":false,\"result\":\"OK\",\"total_cost_usd\":0.0123}\n";
        let r = parse_result(out).unwrap();
        assert_eq!(r.text, "OK");
        assert_eq!(r.cost_usd, Some(0.0123));
        assert!(!r.is_error);
        let err = parse_result(r#"{"type":"result","subtype":"error_max_turns","is_error":true}"#).unwrap();
        assert!(err.is_error);
        assert_eq!(err.text, "error_max_turns");
        assert!(parse_result("not json").is_none());
    }

    /// Spends a few cents: `cargo test --lib tools::claude_code -- --ignored --nocapture`
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn smoke_reply_ok() {
        eprintln!("claude at {:?}", find_claude());
        let r = run("Reply with exactly: OK", &[], Duration::from_secs(90))
            .await
            .expect("claude run");
        eprintln!("result: {r:?}");
        assert!(!r.is_error);
        assert!(r.text.contains("OK"));
    }
}
