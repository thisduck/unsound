//! macOS privacy permissions: status checks that never prompt, requests, and
//! deep links into System Settings. The frontend's onboarding and Settings ›
//! Permissions page are built on `permissions_all` / `permissions_request`.

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionStatus {
    /// Whether synthesized keystrokes (auto-paste) are allowed.
    pub accessibility: bool,
}

pub fn status() -> PermissionStatus {
    PermissionStatus {
        accessibility: accessibility_trusted(),
    }
}

/// What macOS reports. Accessibility and Screen Recording only expose a
/// yes/no, so "not granted" shows up as `NotDetermined` for them.
#[derive(Clone, Copy, Serialize, PartialEq, Eq, Debug)]
#[serde(rename_all = "camelCase")]
pub enum Access {
    Granted,
    Denied,
    NotDetermined,
    /// Blocked by a device-management profile or parental controls.
    Restricted,
    /// macOS doesn't say (Notifications; Automation while Mail and Reminders
    /// are closed).
    Unknown,
    /// Not available on this OS version.
    Unsupported,
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Debug)]
#[serde(rename_all = "camelCase")]
pub enum Kind {
    Microphone,
    Accessibility,
    ScreenRecording,
    Calendar,
    Automation,
    Notifications,
}

pub const KINDS: [Kind; 6] = [
    Kind::Microphone,
    Kind::Accessibility,
    Kind::ScreenRecording,
    Kind::Calendar,
    Kind::Automation,
    Kind::Notifications,
];

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Entry {
    pub kind: Kind,
    pub access: Access,
    /// Calendar only: whether "Use my calendar" is on in meeting settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

/// Current status without showing any system prompt.
pub fn check(kind: Kind) -> Access {
    match kind {
        Kind::Microphone => mac::microphone(),
        Kind::Accessibility => {
            if accessibility_trusted() {
                Access::Granted
            } else {
                Access::NotDetermined
            }
        }
        Kind::ScreenRecording => mac::screen_recording(),
        Kind::Calendar => from_calendar(crate::calendar::access()),
        Kind::Automation => mac::automation(),
        Kind::Notifications => Access::Unknown,
    }
}

fn from_calendar(a: crate::calendar::CalAccess) -> Access {
    use crate::calendar::CalAccess;
    match a {
        CalAccess::Granted => Access::Granted,
        CalAccess::Denied => Access::Denied,
        CalAccess::NotDetermined => Access::NotDetermined,
    }
}

/// AVAuthorizationStatus → Access.
fn from_av(raw: isize) -> Access {
    match raw {
        0 => Access::NotDetermined,
        1 => Access::Restricted,
        2 => Access::Denied,
        3 => Access::Granted,
        _ => Access::Unknown,
    }
}

/// AEDeterminePermissionToAutomateTarget result → Access.
fn from_apple_events(status: i32) -> Access {
    match status {
        0 => Access::Granted,
        -1743 => Access::Denied,        // errAEEventNotPermitted
        -1744 => Access::NotDetermined, // errAEEventWouldRequireUserConsent
        _ => Access::Unknown,           // -600 procNotFound: target not running
    }
}

/// One status for several Automation targets: any refusal wins, then any
/// undecided target, and only all-granted counts as granted.
fn combine(items: &[Access]) -> Access {
    if items.contains(&Access::Denied) {
        Access::Denied
    } else if items.contains(&Access::NotDetermined) {
        Access::NotDetermined
    } else if !items.is_empty() && items.iter().all(|a| *a == Access::Granted) {
        Access::Granted
    } else if items.contains(&Access::Granted) {
        // Granted for the running app; the other hasn't been asked yet.
        Access::NotDetermined
    } else {
        Access::Unknown
    }
}

/// The System Settings pane for a permission.
pub fn settings_url(kind: Kind, bundle_id: &str, modern: bool) -> String {
    let privacy = |pane: &str| format!("x-apple.systempreferences:com.apple.preference.security?{pane}");
    match kind {
        Kind::Microphone => privacy("Privacy_Microphone"),
        Kind::Accessibility => privacy("Privacy_Accessibility"),
        Kind::ScreenRecording => privacy("Privacy_ScreenCapture"),
        Kind::Calendar => privacy("Privacy_Calendars"),
        Kind::Automation => privacy("Privacy_Automation"),
        Kind::Notifications if modern => {
            format!("x-apple.systempreferences:com.apple.Notifications-Settings.extension?id={bundle_id}")
        }
        Kind::Notifications => "x-apple.systempreferences:com.apple.preference.notifications".into(),
    }
}

pub fn open_settings(app: &AppHandle, kind: Kind) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let modern = objc2::available!(macos = 13.0);
        let url = settings_url(kind, &app.config().identifier, modern);
        std::process::Command::new("/usr/bin/open")
            .arg(url)
            .spawn()
            .map(|_| ())
            .map_err(|e| format!("could not open System Settings: {e}"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (app, kind);
        Err("open your system privacy settings".into())
    }
}

/// Show the system prompt where macOS has one; where it won't prompt again
/// (already refused, or no prompt exists) open the matching Settings pane.
/// Blocks while a prompt is up.
pub fn request(app: &AppHandle, kind: Kind) -> Result<Access, String> {
    let before = check(kind);
    if matches!(before, Access::Granted | Access::Unsupported) {
        return Ok(before);
    }
    if matches!(before, Access::Denied | Access::Restricted) || kind == Kind::Notifications {
        open_settings(app, kind)?;
        return Ok(before);
    }
    match kind {
        Kind::Microphone => mac::request_microphone_access(),
        Kind::Accessibility => {
            request_accessibility();
        }
        Kind::ScreenRecording => mac::request_screen_recording(),
        Kind::Calendar => {
            crate::calendar::request_access()?;
        }
        Kind::Automation => mac::request_automation(),
        Kind::Notifications => {}
    }
    Ok(check(kind))
}

#[cfg(target_os = "macos")]
fn accessibility_trusted() -> bool {
    macos_accessibility_client::accessibility::application_is_trusted()
}

#[cfg(not(target_os = "macos"))]
fn accessibility_trusted() -> bool {
    true
}

/// Ask macOS to show the Accessibility permission dialog (no-op elsewhere).
pub fn request_accessibility() -> bool {
    #[cfg(target_os = "macos")]
    {
        macos_accessibility_client::accessibility::application_is_trusted_with_prompt()
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

/// Touch the default input device so the OS shows its microphone permission
/// prompt now instead of mid-first-recording.
pub fn request_microphone() -> Result<(), String> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let (tx, rx) = std::sync::mpsc::channel::<Result<(), String>>();
    std::thread::spawn(move || {
        let attempt = || -> Result<(), String> {
            let host = cpal::default_host();
            let device = host
                .default_input_device()
                .ok_or("no microphone found (check input devices and permissions)")?;
            let config = device.default_input_config().map_err(|e| e.to_string())?;
            let err_fn = |e| eprintln!("mic test stream error: {e}");
            let stream = match config.sample_format() {
                cpal::SampleFormat::I16 => device.build_input_stream(
                    config.clone().into(),
                    |_: &[i16], _: &_| {},
                    err_fn,
                    None,
                ),
                cpal::SampleFormat::U16 => device.build_input_stream(
                    config.clone().into(),
                    |_: &[u16], _: &_| {},
                    err_fn,
                    None,
                ),
                _ => device.build_input_stream(
                    config.clone().into(),
                    |_: &[f32], _: &_| {},
                    err_fn,
                    None,
                ),
            }
            .map_err(|e| e.to_string())?;
            stream.play().map_err(|e| e.to_string())?;
            std::thread::sleep(std::time::Duration::from_millis(400));
            drop(stream);
            Ok(())
        };
        let _ = tx.send(attempt());
    });
    rx.recv()
        .map_err(|_| "microphone check did not respond".to_string())?
}

#[cfg(target_os = "macos")]
mod mac {
    use super::{combine, from_apple_events, from_av, Access};
    use block2::RcBlock;
    use objc2::msg_send;
    use objc2::runtime::{AnyClass, Bool};
    use objc2_foundation::NSString;
    use std::ffi::c_void;
    use std::time::Duration;

    /// Apps the assistant scripts through osascript (tools/apple.rs).
    const AUTOMATION_TARGETS: [&str; 2] = ["com.apple.mail", "com.apple.reminders"];

    #[link(name = "AVFoundation", kind = "framework")]
    extern "C" {
        static AVMediaTypeAudio: &'static NSString;
    }

    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGPreflightScreenCaptureAccess() -> bool;
        fn CGRequestScreenCaptureAccess() -> bool;
    }

    #[repr(C)]
    struct AEDesc {
        descriptor_type: u32,
        data_handle: *mut c_void,
    }

    #[link(name = "CoreServices", kind = "framework")]
    extern "C" {
        fn AECreateDesc(type_code: u32, data: *const c_void, size: isize, result: *mut AEDesc) -> i16;
        fn AEDisposeDesc(desc: *mut AEDesc) -> i16;
        fn AEDeterminePermissionToAutomateTarget(
            target: *const AEDesc,
            event_class: u32,
            event_id: u32,
            ask_user_if_needed: u8,
        ) -> i32;
    }

    const TYPE_APPLICATION_BUNDLE_ID: u32 = u32::from_be_bytes(*b"bund");
    const TYPE_WILDCARD: u32 = u32::from_be_bytes(*b"****");

    fn capture_device() -> Option<&'static AnyClass> {
        AnyClass::get(c"AVCaptureDevice")
    }

    pub fn microphone() -> Access {
        let Some(cls) = capture_device() else {
            return Access::Unknown;
        };
        // SAFETY: class method taking an AVMediaType (NSString) and returning
        // AVAuthorizationStatus (NSInteger); never prompts.
        let raw: isize = unsafe { msg_send![cls, authorizationStatusForMediaType: AVMediaTypeAudio] };
        from_av(raw)
    }

    /// Show the microphone prompt and wait for the answer.
    pub fn request_microphone_access() {
        let Some(cls) = capture_device() else { return };
        let (tx, rx) = std::sync::mpsc::channel::<bool>();
        let handler = RcBlock::new(move |granted: Bool| {
            let _ = tx.send(granted.as_bool());
        });
        // SAFETY: AVFoundation copies the completion block.
        let _: () = unsafe {
            msg_send![cls, requestAccessForMediaType: AVMediaTypeAudio, completionHandler: &*handler]
        };
        let _ = rx.recv_timeout(Duration::from_secs(300));
    }

    pub fn screen_recording() -> Access {
        // SAFETY: plain CoreGraphics query (macOS 10.15+, below our minimum).
        if unsafe { CGPreflightScreenCaptureAccess() } {
            Access::Granted
        } else if crate::syscapture::is_supported() {
            Access::NotDetermined
        } else {
            // System audio capture needs macOS 13.
            Access::Unsupported
        }
    }

    /// Shows the prompt once per install; later calls return at once.
    pub fn request_screen_recording() {
        // SAFETY: plain CoreGraphics call.
        unsafe { CGRequestScreenCaptureAccess() };
    }

    fn apple_events(bundle_id: &str, ask: bool) -> Access {
        let mut desc = AEDesc {
            descriptor_type: 0,
            data_handle: std::ptr::null_mut(),
        };
        // SAFETY: AECreateDesc copies the bytes; the descriptor is disposed below.
        unsafe {
            if AECreateDesc(
                TYPE_APPLICATION_BUNDLE_ID,
                bundle_id.as_ptr().cast(),
                bundle_id.len() as isize,
                &mut desc,
            ) != 0
            {
                return Access::Unknown;
            }
            let status = AEDeterminePermissionToAutomateTarget(&desc, TYPE_WILDCARD, TYPE_WILDCARD, ask as u8);
            AEDisposeDesc(&mut desc);
            from_apple_events(status)
        }
    }

    pub fn automation() -> Access {
        let items: Vec<Access> = AUTOMATION_TARGETS
            .iter()
            .map(|id| apple_events(id, false))
            .collect();
        combine(&items)
    }

    /// Prompts for each running target app; closed apps are asked the first
    /// time the assistant uses them.
    pub fn request_automation() {
        for id in AUTOMATION_TARGETS {
            if apple_events(id, false) == Access::NotDetermined {
                apple_events(id, true);
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod mac {
    use super::Access;
    pub fn microphone() -> Access {
        Access::Granted
    }
    pub fn request_microphone_access() {}
    pub fn screen_recording() -> Access {
        Access::Unsupported
    }
    pub fn request_screen_recording() {}
    pub fn automation() -> Access {
        Access::Unsupported
    }
    pub fn request_automation() {}
}

fn entries(app: &AppHandle) -> Vec<Entry> {
    let use_calendar = crate::settings::load(app).meetings.use_calendar;
    KINDS
        .iter()
        .map(|&kind| Entry {
            kind,
            access: check(kind),
            enabled: (kind == Kind::Calendar).then_some(use_calendar),
        })
        .collect()
}

/// Granting calendar access from the permissions UI means the user wants
/// meetings titled from it, so turn "Use my calendar" on too.
fn enable_calendar(app: &AppHandle) -> Result<(), String> {
    let mut s = crate::settings::load(app);
    if !s.meetings.use_calendar {
        s.meetings.use_calendar = true;
        crate::settings::save(app, &s)?;
        let _ = app.emit("settings-changed", ());
    }
    Ok(())
}

/// Every permission's status at once. Never prompts.
#[tauri::command]
pub async fn permissions_all(app: AppHandle) -> Result<Vec<Entry>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let list = entries(&app);
        // fn-key shortcuts fail to start without Accessibility; start them
        // once it's granted, without needing a relaunch.
        if list
            .iter()
            .any(|e| e.kind == Kind::Accessibility && e.access == Access::Granted)
            && crate::hotkeys::listener_failed()
        {
            crate::hotkeys::ensure_listener(&app);
        }
        list
    })
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn permissions_request(app: AppHandle, kind: Kind) -> Result<Access, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let access = request(&app, kind)?;
        if kind == Kind::Calendar && access == Access::Granted {
            enable_calendar(&app)?;
        }
        Ok(access)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub fn permissions_open_settings(app: AppHandle, kind: Kind) -> Result<(), String> {
    open_settings(&app, kind)
}

/// Restart so permissions macOS applies only at launch (Screen Recording)
/// take effect. Refused while recording or processing.
#[tauri::command]
pub fn relaunch_app(app: AppHandle) -> Result<(), String> {
    if crate::updates::busy_reason(&app)?.is_some() {
        return Err("unsound is still recording or processing. Relaunch when it's done.".into());
    }
    app.restart();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn av_status_maps_every_case() {
        assert_eq!(from_av(0), Access::NotDetermined);
        assert_eq!(from_av(1), Access::Restricted);
        assert_eq!(from_av(2), Access::Denied);
        assert_eq!(from_av(3), Access::Granted);
        assert_eq!(from_av(9), Access::Unknown);
    }

    #[test]
    fn apple_event_status_maps_tcc_errors() {
        assert_eq!(from_apple_events(0), Access::Granted);
        assert_eq!(from_apple_events(-1743), Access::Denied);
        assert_eq!(from_apple_events(-1744), Access::NotDetermined);
        assert_eq!(from_apple_events(-600), Access::Unknown);
    }

    #[test]
    fn automation_combines_targets() {
        use Access::*;
        assert_eq!(combine(&[Granted, Granted]), Granted);
        assert_eq!(combine(&[Granted, Denied]), Denied);
        assert_eq!(combine(&[NotDetermined, Unknown]), NotDetermined);
        assert_eq!(combine(&[Granted, Unknown]), NotDetermined);
        assert_eq!(combine(&[Unknown, Unknown]), Unknown);
        assert_eq!(combine(&[]), Unknown);
    }

    #[test]
    fn calendar_reuses_eventkit_mapping() {
        use crate::calendar::CalAccess;
        assert_eq!(from_calendar(CalAccess::Granted), Access::Granted);
        assert_eq!(from_calendar(CalAccess::Denied), Access::Denied);
        assert_eq!(from_calendar(CalAccess::NotDetermined), Access::NotDetermined);
    }

    #[test]
    fn settings_urls_open_the_matching_pane() {
        let url = |k| settings_url(k, "com.unsound.app", true);
        assert!(url(Kind::Microphone).ends_with("?Privacy_Microphone"));
        assert!(url(Kind::Accessibility).ends_with("?Privacy_Accessibility"));
        assert!(url(Kind::ScreenRecording).ends_with("?Privacy_ScreenCapture"));
        assert!(url(Kind::Calendar).ends_with("?Privacy_Calendars"));
        assert!(url(Kind::Automation).ends_with("?Privacy_Automation"));
        assert!(url(Kind::Notifications).ends_with("Notifications-Settings.extension?id=com.unsound.app"));
        assert_eq!(
            settings_url(Kind::Notifications, "x", false),
            "x-apple.systempreferences:com.apple.preference.notifications"
        );
    }

    /// Calls the real OS checks (none of them prompt). `--ignored` to run.
    #[test]
    #[ignore]
    fn live_checks_do_not_prompt() {
        for kind in KINDS {
            eprintln!("{kind:?}: {:?}", check(kind));
        }
    }

    #[test]
    fn wire_names_match_the_frontend() {
        assert_eq!(serde_json::to_string(&Kind::ScreenRecording).unwrap(), "\"screenRecording\"");
        assert_eq!(serde_json::to_string(&Access::NotDetermined).unwrap(), "\"notDetermined\"");
        let k: Kind = serde_json::from_str("\"calendar\"").unwrap();
        assert_eq!(k, Kind::Calendar);
        assert_eq!(KINDS.len(), 6);
    }
}
