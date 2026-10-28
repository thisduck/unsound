//! A persistent, interactive reminder that microphone capture is still running.
use tauri::{AppHandle, Emitter, Manager};

const LABEL: &str = "meeting-indicator";
const WIDTH: f64 = 336.0;
const HEIGHT: f64 = 76.0;

pub fn create(app: &AppHandle) -> tauri::Result<()> {
    let window = tauri::WebviewWindowBuilder::new(
        app,
        LABEL,
        tauri::WebviewUrl::App("index.html#meeting-indicator".into()),
    )
    .title("unsound — meeting recording")
    .inner_size(WIDTH, HEIGHT)
    .decorations(false)
    .transparent(true)
    .always_on_top(true)
    .visible_on_all_workspaces(true)
    .visible(false)
    .focused(false)
    .resizable(false)
    .closable(false)
    .minimizable(false)
    .maximizable(false)
    .skip_taskbar(true)
    .shadow(false)
    .accept_first_mouse(true)
    .build()?;
    #[cfg(target_os = "macos")]
    {
        use objc2_app_kit::{NSWindow, NSWindowCollectionBehavior};
        // Setup runs on the main thread, and Tauri owns this NSWindow.
        let native = unsafe { &*(window.ns_window()? as *const NSWindow) };
        let behavior = native.collectionBehavior()
            & !(NSWindowCollectionBehavior::FullScreenPrimary
                | NSWindowCollectionBehavior::FullScreenNone);
        native.setCollectionBehavior(behavior | NSWindowCollectionBehavior::FullScreenAuxiliary);
    }
    #[cfg(not(target_os = "macos"))]
    let _ = window;
    Ok(())
}

/// Read the latest capture state on the UI thread: an older stop must never
/// hide a newer recording. Leave a dragged window in place until capture ends.
pub fn refresh(app: &AppHandle) {
    let handle = app.clone();
    let result = app.run_on_main_thread(move || {
        let Some(window) = handle.get_webview_window(LABEL) else {
            return;
        };
        let result = if crate::meeting::is_recording(&handle) {
            if !window.is_visible().unwrap_or(false) {
                position(&handle, &window);
            }
            show_without_focus(&window)
        } else {
            window.hide()
        };
        if let Err(error) = result {
            eprintln!("[meeting-indicator] could not update visibility: {error}");
        }
        let _ = handle.emit_to(LABEL, "meeting-recording-changed", ());
    });
    if let Err(error) = result {
        eprintln!("[meeting-indicator] could not schedule update: {error}");
    }
}

fn show_without_focus(window: &tauri::WebviewWindow) -> tauri::Result<()> {
    #[cfg(target_os = "macos")]
    {
        // Called on the UI thread. Tauri's show() makes the window key, which
        // would take keyboard focus away from the call when recording starts.
        let native = unsafe { &*(window.ns_window()? as *const objc2_app_kit::NSWindow) };
        native.orderFrontRegardless();
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    window.show()
}

fn position(app: &AppHandle, window: &tauri::WebviewWindow) {
    let monitor = app
        .cursor_position()
        .ok()
        .and_then(|pos| app.monitor_from_point(pos.x, pos.y).ok().flatten())
        .or_else(|| app.primary_monitor().ok().flatten());
    let Some(monitor) = monitor else { return };
    let scale = monitor.scale_factor();
    let origin = monitor.position();
    let x = origin.x as f64 + monitor.size().width as f64 - (WIDTH + 16.0) * scale;
    let y = origin.y as f64 + 36.0 * scale;
    let _ = window.set_position(tauri::PhysicalPosition::new(x, y));
}
