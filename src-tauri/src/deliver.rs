use enigo::{Enigo, Keyboard, Settings as EnigoSettings};
use tauri::AppHandle;

/// Type `text` into the frontmost app as synthesized keystrokes. The
/// clipboard is deliberately never touched — only the explicit copy buttons
/// in the UI use it. Requires the Accessibility permission on macOS.
pub fn deliver_text(app: &AppHandle, text: &str) -> Result<(), String> {
    // enigo goes through macOS input APIs that must run on the main thread
    // (they assert otherwise since macOS 26).
    let (tx, rx) = std::sync::mpsc::channel::<Result<(), String>>();
    // Trail a space so back-to-back dictations don't run their words together
    // ("hello" + "world" → "hello world"). Skip if empty or already spaced.
    let text = if text.is_empty() || text.ends_with(char::is_whitespace) {
        text.to_string()
    } else {
        format!("{text} ")
    };
    app.run_on_main_thread(move || {
        let _ = tx.send(type_text(&text));
    })
    .map_err(|e| e.to_string())?;
    rx.recv_timeout(std::time::Duration::from_secs(10))
        .map_err(|_| "typing the text timed out".to_string())?
}

fn type_text(text: &str) -> Result<(), String> {
    let mut enigo = Enigo::new(&EnigoSettings::default())
        .map_err(|e| format!("keystroke synthesis unavailable: {e}"))?;
    enigo
        .text(text)
        .map_err(|e| format!("typing failed (is Accessibility allowed?): {e}"))
}
