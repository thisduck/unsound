use enigo::{Direction, Enigo, Key, Keyboard, Settings as EnigoSettings};

/// Put `text` on the clipboard and synthesize a paste keystroke so it lands
/// in the frontmost app. Requires the Accessibility permission on macOS; if
/// the keystroke is blocked the text is still on the clipboard.
pub fn deliver_text(text: &str) -> Result<(), String> {
    let mut clipboard = arboard::Clipboard::new().map_err(|e| e.to_string())?;
    clipboard
        .set_text(text.to_string())
        .map_err(|e| e.to_string())?;

    // Give the pasteboard a moment to settle before the keystroke.
    std::thread::sleep(std::time::Duration::from_millis(120));

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
