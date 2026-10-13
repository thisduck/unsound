//! The user's real `PATH`. Apps launched from Finder/Dock inherit launchd's
//! bare `/usr/bin:/bin:/usr/sbin:/sbin`, so `npx`, `uvx`, `claude` and anything
//! else installed via Homebrew/nvm/npm is invisible to them. We ask the login
//! shell once and cache the answer; tools that spawn children use it.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

const START: &str = "__UNSOUND_PATH_START__";
const END: &str = "__UNSOUND_PATH_END__";

static PATH: OnceLock<String> = OnceLock::new();

/// The login shell's `PATH`, plus the usual install locations as a fallback
/// (deduplicated, shell entries first). Cached for the life of the process.
pub fn login_path() -> String {
    PATH.get_or_init(resolve).clone()
}

/// Find an executable on [`login_path`]. Absolute/relative paths are returned
/// as-is when they point at a file.
pub fn which(bin: &str) -> Option<PathBuf> {
    if bin.contains('/') {
        let p = PathBuf::from(bin);
        return is_executable(&p).then_some(p);
    }
    login_path()
        .split(':')
        .filter(|d| !d.is_empty())
        .map(|d| Path::new(d).join(bin))
        .find(|p| is_executable(p))
}

pub(crate) fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(p)
        .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

fn resolve() -> String {
    let shell = std::env::var("SHELL")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "/bin/zsh".into());
    // Interactive + login so both .zprofile and .zshrc run (nvm and friends
    // usually live in .zshrc). Markers fence off anything the rc files print.
    let script = format!("printf '{START}%s{END}' \"$PATH\"");
    let from_shell = ["-ilc", "-lc"]
        .iter()
        .find_map(|flags| run_with_timeout(&shell, flags, &script, Duration::from_secs(5)));
    let mut dirs: Vec<String> = Vec::new();
    let mut push = |d: String| {
        if !d.is_empty() && !dirs.contains(&d) {
            dirs.push(d);
        }
    };
    for d in from_shell.unwrap_or_default().split(':') {
        push(d.to_string());
    }
    let home = std::env::var("HOME").unwrap_or_default();
    for d in [
        "/opt/homebrew/bin".to_string(),
        "/opt/homebrew/sbin".to_string(),
        "/usr/local/bin".to_string(),
        format!("{home}/.local/bin"),
        format!("{home}/.npm-global/bin"),
        format!("{home}/.claude/local"),
        format!("{home}/.bun/bin"),
        format!("{home}/.cargo/bin"),
        "/usr/bin".to_string(),
        "/bin".to_string(),
        "/usr/sbin".to_string(),
        "/sbin".to_string(),
    ] {
        push(d);
    }
    if let Ok(current) = std::env::var("PATH") {
        for d in current.split(':') {
            push(d.to_string());
        }
    }
    dirs.join(":")
}

/// Run `shell <flags> <script>` and pull the fenced PATH out of its stdout.
/// A hung rc file (e.g. one waiting on the network) must not hang the app.
fn run_with_timeout(shell: &str, flags: &str, script: &str, limit: Duration) -> Option<String> {
    let mut child = Command::new(shell)
        .arg(flags)
        .arg(script)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if started.elapsed() < limit => std::thread::sleep(Duration::from_millis(25)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
    let out = child.wait_with_output().ok()?;
    extract(&String::from_utf8_lossy(&out.stdout))
}

fn extract(stdout: &str) -> Option<String> {
    let start = stdout.rfind(START)? + START.len();
    let end = start + stdout[start..].find(END)?;
    let path = stdout[start..end].trim();
    (!path.is_empty()).then(|| path.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_path_between_markers_despite_rc_noise() {
        let out = format!("Welcome!\n{START}/a:/b{END}\n");
        assert_eq!(extract(&out).as_deref(), Some("/a:/b"));
        assert_eq!(extract("no markers"), None);
    }

    #[test]
    fn login_path_includes_fallbacks_and_finds_sh() {
        let p = login_path();
        assert!(p.split(':').any(|d| d == "/usr/bin"));
        assert!(which("sh").is_some());
        assert!(which("definitely-not-a-real-binary-xyz").is_none());
    }
}
