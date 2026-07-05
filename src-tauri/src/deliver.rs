use enigo::{Direction, Enigo, Key, Keyboard, Settings as EnigoSettings};
use tauri::AppHandle;

/// Put `text` on the clipboard and synthesize a paste keystroke so it lands
/// in the frontmost app. Requires the Accessibility permission on macOS; if
/// the keystroke is blocked the text is still on the clipboard.
pub fn deliver_text(app: &AppHandle, text: &str) -> Result<(), String> {
    let mut clipboard = arboard::Clipboard::new().map_err(|e| e.to_string())?;
    clipboard
        .set_text(text.to_string())
        .map_err(|e| e.to_string())?;

    // Give the pasteboard a moment to settle before the keystroke.
    std::thread::sleep(std::time::Duration::from_millis(120));

    // enigo's key lookup uses macOS text-input-source APIs that must run on
    // the main thread (they assert otherwise since macOS 26).
    let (tx, rx) = std::sync::mpsc::channel::<Result<(), String>>();
    app.run_on_main_thread(move || {
        let _ = tx.send(paste_keystroke());
    })
    .map_err(|e| e.to_string())?;
    rx.recv_timeout(std::time::Duration::from_secs(3))
        .map_err(|_| "paste keystroke timed out (text is on the clipboard)".to_string())?
}

fn paste_keystroke() -> Result<(), String> {
    let mut enigo = Enigo::new(&EnigoSettings::default())
        .map_err(|e| format!("keystroke synthesis unavailable: {e}"))?;

    #[cfg(target_os = "macos")]
    let modifier = Key::Meta;
    #[cfg(not(target_os = "macos"))]
    let modifier = Key::Control;

    enigo
        .key(modifier, Direction::Press)
        .and_then(|_| enigo.key(Key::Unicode('v'), Direction::Click))
        .and_then(|_| enigo.key(modifier, Direction::Release))
        .map_err(|e| format!("paste keystroke failed (text is on the clipboard): {e}"))?;
    Ok(())
}
