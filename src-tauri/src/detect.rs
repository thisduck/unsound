//! Meeting detection: notice when a call app (Zoom, Teams, Meet in a
//! browser, …) starts using the microphone and offer to record it, via a small
//! clickable prompt window in the top-right corner.
//!
//! On macOS 14.2+ CoreAudio lists its client processes, each with its bundle id
//! and whether it's pulling input right now — so we know exactly which app took
//! the mic. On older systems we fall back to "is the input device busy?" plus
//! the list of running call apps (browsers can't be told apart there, so they
//! don't count). A background thread polls every couple of seconds; nothing is
//! recorded until the user says so.

use serde::Serialize;
use tauri::AppHandle;

/// Payload of `meeting-detected` / `meeting-app-released`.
#[derive(Clone, Serialize, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DetectedApp {
    pub app: String,
    pub bundle_id: String,
}

/// A process currently pulling microphone input.
#[derive(Clone, Debug)]
pub struct MicUser {
    pub pid: i32,
    pub bundle_id: String,
}

/// Call apps we prompt for, by bundle-id prefix (helpers such as
/// `com.google.Chrome.helper` hold the mic on the app's behalf).
const KNOWN_APPS: &[(&str, &str, bool)] = &[
    // (bundle id prefix, friendly name, is a browser)
    ("us.zoom.xos", "Zoom", false),
    ("com.microsoft.teams2", "Teams", false),
    ("com.microsoft.teams", "Teams", false),
    ("com.tinyspeck.slackmacgap", "Slack", false),
    ("com.cisco.webexmeetingsapp", "Webex", false),
    ("Cisco-Systems.Spark", "Webex", false),
    ("com.apple.FaceTime", "FaceTime", false),
    ("com.hnc.Discord", "Discord", false),
    ("com.google.Chrome", "Chrome (Meet?)", true),
    ("com.apple.Safari", "Safari (Meet?)", true),
    ("company.thebrowser.Browser", "Arc (Meet?)", true),
    ("org.mozilla.firefox", "Firefox (Meet?)", true),
    ("com.microsoft.edgemac", "Edge (Meet?)", true),
    ("com.brave.Browser", "Brave (Meet?)", true),
];

/// Match a bundle id against the known call apps: `(canonical id, name, browser)`.
fn known_app(bundle_id: &str) -> Option<(&'static str, &'static str, bool)> {
    KNOWN_APPS
        .iter()
        .find(|(prefix, _, _)| {
            bundle_id == *prefix
                || bundle_id
                    .strip_prefix(prefix)
                    .is_some_and(|rest| rest.starts_with('.'))
        })
        .copied()
}

#[cfg(target_os = "macos")]
pub use imp::*;

#[cfg(not(target_os = "macos"))]
mod stub {
    use super::*;
    pub fn start(_app: &AppHandle) {}
    pub fn set_enabled(_enabled: bool) {}
    pub fn create_prompt_window(_app: &AppHandle) -> tauri::Result<()> {
        Ok(())
    }
    pub fn show_prompt(_app: &AppHandle, _app_name: &str, _bundle_id: &str) {}
    pub fn hide_prompt(_app: &AppHandle) {}
    pub fn mic_users() -> Option<Vec<MicUser>> {
        None
    }
}
#[cfg(not(target_os = "macos"))]
pub use stub::*;

#[cfg(target_os = "macos")]
mod imp {
    use super::*;
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};
    use tauri::{Emitter, Manager};

    const POLL: Duration = Duration::from_secs(2);
    /// A call app must be off the mic this long before we call it "released"
    /// (and before it may prompt again) — rides out brief device hand-offs.
    const RELEASE_AFTER: Duration = Duration::from_secs(5);

    const PROMPT_LABEL: &str = "prompt";
    const PROMPT_W: f64 = 360.0;
    const PROMPT_H: f64 = 120.0;
    const PROMPT_MARGIN: f64 = 16.0;
    /// Clear of the menu bar.
    const PROMPT_TOP: f64 = 36.0;

    static STARTED: AtomicBool = AtomicBool::new(false);
    static ENABLED: AtomicBool = AtomicBool::new(true);

    // ---- CoreAudio (plain C API; the framework is already linked by cpal) ----

    #[repr(C)]
    struct PropertyAddress {
        selector: u32,
        scope: u32,
        element: u32,
    }

    #[link(name = "CoreAudio", kind = "framework")]
    extern "C" {
        fn AudioObjectGetPropertyDataSize(
            object: u32,
            address: *const PropertyAddress,
            qualifier_size: u32,
            qualifier: *const c_void,
            out_size: *mut u32,
        ) -> i32;
        fn AudioObjectGetPropertyData(
            object: u32,
            address: *const PropertyAddress,
            qualifier_size: u32,
            qualifier: *const c_void,
            io_size: *mut u32,
            out: *mut c_void,
        ) -> i32;
    }

    const fn fourcc(s: &[u8; 4]) -> u32 {
        u32::from_be_bytes(*s)
    }
    const SYSTEM_OBJECT: u32 = 1;
    const SCOPE_GLOBAL: u32 = fourcc(b"glob");
    const ELEMENT_MAIN: u32 = 0;
    const PROCESS_OBJECT_LIST: u32 = fourcc(b"prs#"); // macOS 14.2+
    const PROCESS_PID: u32 = fourcc(b"ppid");
    const PROCESS_BUNDLE_ID: u32 = fourcc(b"pbid");
    const PROCESS_IS_RUNNING_INPUT: u32 = fourcc(b"piri");
    const DEFAULT_INPUT_DEVICE: u32 = fourcc(b"dIn ");
    const DEVICE_IS_RUNNING_SOMEWHERE: u32 = fourcc(b"gone");

    fn address(selector: u32) -> PropertyAddress {
        PropertyAddress {
            selector,
            scope: SCOPE_GLOBAL,
            element: ELEMENT_MAIN,
        }
    }

    /// Read a fixed-size property (u32/i32/pointer-sized).
    fn get<T: Copy + Default>(object: u32, selector: u32) -> Option<T> {
        let addr = address(selector);
        let mut value = T::default();
        let mut size = std::mem::size_of::<T>() as u32;
        // SAFETY: `value` is a valid, correctly sized out-buffer.
        let status = unsafe {
            AudioObjectGetPropertyData(
                object,
                &addr,
                0,
                std::ptr::null(),
                &mut size,
                &mut value as *mut T as *mut c_void,
            )
        };
        (status == 0).then_some(value)
    }

    /// Read a CFString property (returned +1; released here).
    fn get_string(object: u32, selector: u32) -> Option<String> {
        use core_foundation::base::TCFType;
        use core_foundation::string::{CFString, CFStringRef};
        let ptr: usize = get(object, selector)?;
        if ptr == 0 {
            return None;
        }
        // SAFETY: CoreAudio hands back a retained CFStringRef we now own.
        let s = unsafe { CFString::wrap_under_create_rule(ptr as CFStringRef) };
        Some(s.to_string())
    }

    /// Every process CoreAudio knows about that is pulling input right now.
    /// `None` when the per-process API isn't available (before macOS 14.2).
    pub fn mic_users() -> Option<Vec<MicUser>> {
        let addr = address(PROCESS_OBJECT_LIST);
        let mut size = 0u32;
        // SAFETY: plain size query on the system object.
        let status = unsafe {
            AudioObjectGetPropertyDataSize(SYSTEM_OBJECT, &addr, 0, std::ptr::null(), &mut size)
        };
        if status != 0 {
            return None;
        }
        let count = size as usize / std::mem::size_of::<u32>();
        let mut ids = vec![0u32; count];
        // SAFETY: `ids` holds exactly `size` bytes.
        let status = unsafe {
            AudioObjectGetPropertyData(
                SYSTEM_OBJECT,
                &addr,
                0,
                std::ptr::null(),
                &mut size,
                ids.as_mut_ptr() as *mut c_void,
            )
        };
        if status != 0 {
            return None;
        }
        ids.truncate(size as usize / std::mem::size_of::<u32>());
        Some(
            ids.into_iter()
                .filter(|&id| get::<u32>(id, PROCESS_IS_RUNNING_INPUT).unwrap_or(0) != 0)
                .map(|id| MicUser {
                    pid: get::<i32>(id, PROCESS_PID).unwrap_or(-1),
                    bundle_id: get_string(id, PROCESS_BUNDLE_ID).unwrap_or_default(),
                })
                .collect(),
        )
    }

    /// Pre-14.2 fallback: is anyone using the default input device?
    fn input_busy() -> bool {
        let Some(device) = get::<u32>(SYSTEM_OBJECT, DEFAULT_INPUT_DEVICE).filter(|&d| d != 0)
        else {
            return false;
        };
        get::<u32>(device, DEVICE_IS_RUNNING_SOMEWHERE).unwrap_or(0) != 0
    }

    /// Running (non-browser) call apps, for the fallback path.
    fn running_call_apps() -> Vec<String> {
        use objc2_app_kit::NSWorkspace;
        let workspace = NSWorkspace::sharedWorkspace();
        workspace
            .runningApplications()
            .iter()
            .filter_map(|a| a.bundleIdentifier().map(|b| b.to_string()))
            .filter(|b| known_app(b).is_some_and(|(_, _, browser)| !browser))
            .collect()
    }

    /// One poll: the call apps holding the mic (canonical id → name), and
    /// whether unsound itself is using it. `own_mic` is only knowable on the
    /// per-process path.
    fn snapshot() -> (HashMap<&'static str, &'static str>, bool) {
        let own_pid = std::process::id() as i32;
        let mut holders = HashMap::new();
        match mic_users() {
            Some(users) => {
                let mut own = false;
                for u in users {
                    if u.pid == own_pid {
                        own = true;
                        continue;
                    }
                    if let Some((id, name, _)) = known_app(&u.bundle_id) {
                        holders.insert(id, name);
                    }
                }
                (holders, own)
            }
            None => {
                if input_busy() {
                    for b in running_call_apps() {
                        if let Some((id, name, _)) = known_app(&b) {
                            holders.insert(id, name);
                        }
                    }
                }
                (holders, false)
            }
        }
    }

    /// Start the detector thread (once; later calls are no-ops).
    pub fn start(app: &AppHandle) {
        if STARTED.swap(true, Ordering::SeqCst) {
            return;
        }
        let app = app.clone();
        std::thread::spawn(move || run(app));
    }

    /// Turn prompting on/off (the thread keeps running, idle, when off).
    pub fn set_enabled(enabled: bool) {
        ENABLED.store(enabled, Ordering::SeqCst);
    }

    struct Tracked {
        name: &'static str,
        /// When it was last seen holding the mic.
        last_seen: Instant,
    }

    fn run(app: AppHandle) {
        // Call apps that have held the mic and haven't yet been gone for
        // `RELEASE_AFTER` — each prompts at most once per stint on the mic.
        let mut tracked: HashMap<&'static str, Tracked> = HashMap::new();
        // The app the prompt was last shown for.
        let mut prompted_for: Option<&'static str> = None;
        loop {
            std::thread::sleep(POLL);
            if !ENABLED.load(Ordering::SeqCst) {
                tracked.clear();
                continue;
            }
            let (holders, own_mic) = snapshot();
            let now = Instant::now();

            for (&id, &name) in &holders {
                match tracked.get_mut(id) {
                    Some(t) => t.last_seen = now,
                    None => {
                        tracked.insert(id, Tracked { name, last_seen: now });
                        // Only a fresh grab of the mic prompts, and only when
                        // we aren't already recording or dictating.
                        if !crate::meeting::is_recording(&app) && !own_mic {
                            let payload = DetectedApp {
                                app: name.to_string(),
                                bundle_id: id.to_string(),
                            };
                            eprintln!("[detect] {name} ({id}) took the mic");
                            let _ = app.emit("meeting-detected", &payload);
                            place_and_show(&app);
                            prompted_for = Some(id);
                        }
                    }
                }
            }

            let released: Vec<&'static str> = tracked
                .iter()
                .filter(|(id, t)| {
                    !holders.contains_key(*id) && now.duration_since(t.last_seen) >= RELEASE_AFTER
                })
                .map(|(id, _)| *id)
                .collect();
            for id in released {
                if let Some(t) = tracked.remove(id) {
                    eprintln!("[detect] {} ({id}) released the mic", t.name);
                    let _ = app.emit(
                        "meeting-app-released",
                        DetectedApp {
                            app: t.name.to_string(),
                            bundle_id: id.to_string(),
                        },
                    );
                    // The call ended before anyone answered the prompt.
                    if prompted_for == Some(id) {
                        prompted_for = None;
                        hide_prompt(&app);
                    }
                }
            }
        }
    }

    // ---- the prompt window -------------------------------------------------

    /// A small always-on-top card with Record / Not now. Hidden until needed;
    /// unlike the dictation overlay it takes clicks.
    pub fn create_prompt_window(app: &AppHandle) -> tauri::Result<()> {
        tauri::WebviewWindowBuilder::new(
            app,
            PROMPT_LABEL,
            tauri::WebviewUrl::App("index.html#prompt".into()),
        )
        .title("unsound")
        .inner_size(PROMPT_W, PROMPT_H)
        .decorations(false)
        .transparent(true)
        .always_on_top(true)
        .visible(false)
        .focused(false)
        .resizable(false)
        .skip_taskbar(true)
        .shadow(false)
        // Buttons work on the first click, without activating unsound first.
        .accept_first_mouse(true)
        .visible_on_all_workspaces(true)
        .build()?;
        Ok(())
    }

    /// Top-right of the monitor the cursor is on.
    fn position(app: &AppHandle, window: &tauri::WebviewWindow) {
        let monitor = app
            .cursor_position()
            .ok()
            .and_then(|pos| app.monitor_from_point(pos.x, pos.y).ok().flatten())
            .or_else(|| app.primary_monitor().ok().flatten());
        let Some(monitor) = monitor else { return };
        let scale = monitor.scale_factor();
        let mpos = monitor.position();
        let msize = monitor.size();
        let x = mpos.x as f64 + msize.width as f64 - (PROMPT_W + PROMPT_MARGIN) * scale;
        let y = mpos.y as f64 + PROMPT_TOP * scale;
        let _ = window.set_position(tauri::PhysicalPosition::new(x, y));
    }

    fn place_and_show(app: &AppHandle) {
        if let Some(w) = app.get_webview_window(PROMPT_LABEL) {
            position(app, &w);
            let _ = w.show();
        }
    }

    /// Show the prompt for `app_name` (also sends it `meeting-detected`, so the
    /// card has its text even when shown by hand rather than by detection).
    pub fn show_prompt(app: &AppHandle, app_name: &str, bundle_id: &str) {
        let payload = DetectedApp {
            app: app_name.to_string(),
            bundle_id: bundle_id.to_string(),
        };
        let _ = app.emit_to(PROMPT_LABEL, "meeting-detected", payload);
        place_and_show(app);
    }

    pub fn hide_prompt(app: &AppHandle) {
        if let Some(w) = app.get_webview_window(PROMPT_LABEL) {
            let _ = w.hide();
        }
    }
}

/// The prompt's own Record / Not now / timeout buttons hide it through this.
#[tauri::command]
pub fn hide_meeting_prompt(app: AppHandle) {
    hide_prompt(&app);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_known_apps_and_their_helpers() {
        assert_eq!(known_app("us.zoom.xos").map(|a| a.1), Some("Zoom"));
        assert_eq!(
            known_app("com.google.Chrome.helper").map(|a| a.0),
            Some("com.google.Chrome")
        );
        // teams2 must not be swallowed by the shorter "com.microsoft.teams".
        assert_eq!(
            known_app("com.microsoft.teams2").map(|a| a.0),
            Some("com.microsoft.teams2")
        );
        // A prefix match must end on a component boundary.
        assert!(known_app("us.zoom.xosfake").is_none());
        assert!(known_app("com.apple.Music").is_none());
    }
}
