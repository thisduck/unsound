/// Name of the app currently in the foreground — the app a shortcut
/// dictation is about to type into. No permission needed on macOS.
#[cfg(target_os = "macos")]
pub fn frontmost_app() -> Option<String> {
    use objc2_app_kit::NSWorkspace;
    let workspace = NSWorkspace::sharedWorkspace();
    let app = workspace.frontmostApplication()?;
    let name = app.localizedName()?;
    Some(name.to_string())
}

#[cfg(not(target_os = "macos"))]
pub fn frontmost_app() -> Option<String> {
    None
}
