//! Low-level global key listener (our own CGEventTap) for shortcuts the
//! regular macOS hotkey API can't express — anything involving the fn key —
//! and for live key capture in the settings UI. Needs the Accessibility
//! permission; plain combos stay on tauri-plugin-global-shortcut, which
//! works without it.
//!
//! Deliberately reads only keycodes from events: keycode → key-name mapping
//! is a local table, never the TSM/input-source APIs (which assert when
//! called off the main thread and would crash the app).

use serde::Serialize;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use tauri::{AppHandle, Emitter};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mode {
    HandsFree,
    PushToTalk,
}

/// A key event as the matcher sees it.
enum Tap {
    Mod(&'static str),
    Key(&'static str),
    Escape,
}

#[derive(Clone, PartialEq)]
struct Binding {
    mods: BTreeSet<&'static str>,
    key: Option<String>,
    combo: String,
    mode: Mode,
}

#[derive(Default)]
struct Matcher {
    bindings: Vec<Binding>,
    mods_down: BTreeSet<&'static str>,
    peak_mods: BTreeSet<&'static str>,
    capturing: bool,
    committed: bool,
    /// Index of the push-to-talk binding currently held down.
    active_ptt: Option<usize>,
}

static MATCHER: Mutex<Matcher> = Mutex::new(Matcher {
    bindings: Vec::new(),
    mods_down: BTreeSet::new(),
    peak_mods: BTreeSet::new(),
    capturing: false,
    committed: false,
    active_ptt: None,
});

// 0 = not started, 1 = starting, 2 = running, 3 = failed (no permission)
static LISTENER_STATE: AtomicU8 = AtomicU8::new(0);
static APP: OnceLock<AppHandle> = OnceLock::new();

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureUpdate {
    pub combo: String,
}

/// Parse "fn+space" into a Binding. Only called for combos we canonically
/// produced ourselves, so the vocabulary is closed.
fn parse(combo: &str, mode: Mode) -> Result<Binding, String> {
    let mut mods = BTreeSet::new();
    let mut key = None;
    for part in combo.split('+') {
        match part {
            "cmd" | "super" | "command" => {
                mods.insert("cmd");
            }
            "ctrl" | "control" => {
                mods.insert("ctrl");
            }
            "alt" | "option" => {
                mods.insert("alt");
            }
            "shift" => {
                mods.insert("shift");
            }
            "fn" => {
                mods.insert("fn");
            }
            k if !k.is_empty() => {
                if key.replace(k.to_string()).is_some() {
                    return Err(format!("'{combo}' has more than one non-modifier key"));
                }
            }
            _ => {}
        }
    }
    if mods.is_empty() && key.is_none() {
        return Err(format!("'{combo}' is empty"));
    }
    Ok(Binding {
        mods,
        key,
        combo: combo.to_string(),
        mode,
    })
}

pub fn set_bindings(bindings: &[(String, Mode)]) -> Result<(), String> {
    let parsed = bindings
        .iter()
        .map(|(c, m)| parse(c, *m))
        .collect::<Result<Vec<_>, _>>()?;
    let mut matcher = MATCHER.lock().unwrap();
    matcher.bindings = parsed;
    matcher.active_ptt = None;
    Ok(())
}

/// Start (or confirm) the event-tap listener. Returns true if it is running.
pub fn ensure_listener(app: &AppHandle) -> bool {
    let _ = APP.set(app.clone());
    match LISTENER_STATE.load(Ordering::SeqCst) {
        2 => return true,
        3 => {
            // A previous attempt failed (permission missing). Allow a retry —
            // the user may have granted Accessibility since.
            LISTENER_STATE.store(0, Ordering::SeqCst);
        }
        _ => {}
    }
    if LISTENER_STATE
        .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        // Another thread is starting it; give it a moment.
        std::thread::sleep(std::time::Duration::from_millis(250));
        return LISTENER_STATE.load(Ordering::SeqCst) == 2;
    }
    std::thread::spawn(|| match listener::run() {
        Ok(()) => {}
        Err(e) => {
            eprintln!("fn-key listener unavailable (grant Accessibility): {e}");
            LISTENER_STATE.store(3, Ordering::SeqCst);
        }
    });
    std::thread::sleep(std::time::Duration::from_millis(250));
    LISTENER_STATE.load(Ordering::SeqCst) == 2
}

pub fn start_capture(app: &AppHandle) -> bool {
    let native = ensure_listener(app);
    if native {
        let mut m = MATCHER.lock().unwrap();
        m.capturing = true;
        m.committed = false;
        m.peak_mods = m.mods_down.clone();
    }
    native
}

pub fn cancel_capture() {
    let mut m = MATCHER.lock().unwrap();
    m.capturing = false;
    m.committed = false;
    m.peak_mods.clear();
}

fn combo_string(mods: &BTreeSet<&'static str>, key: Option<&str>) -> String {
    // Fixed display order, independent of BTreeSet's alphabetical order.
    let order = ["cmd", "ctrl", "alt", "shift", "fn"];
    let mut parts: Vec<String> = order
        .iter()
        .filter(|m| mods.contains(*m))
        .map(|m| m.to_string())
        .collect();
    if let Some(k) = key {
        parts.push(k.to_string());
    }
    parts.join("+")
}

fn emit(event: &str, payload: Option<CaptureUpdate>) {
    if let Some(app) = APP.get() {
        match payload {
            Some(p) => {
                let _ = app.emit(event, p);
            }
            None => {
                let _ = app.emit(event, ());
            }
        }
    }
}

fn on_key(tap: Tap, down: bool) {
    if down {
        on_press(tap);
    } else {
        on_release(tap);
    }
}

/// While unsound uses fn (a binding or an in-progress capture), bare-fn key
/// events are swallowed so macOS doesn't open the emoji picker — the same
/// trick Wispr Flow uses. Everything returns to normal when the app quits.
fn should_swallow_fn() -> bool {
    let m = MATCHER.lock().unwrap();
    m.capturing || m.bindings.iter().any(|b| b.mods.contains("fn"))
}

fn on_press(tap: Tap) {
    let mut m = MATCHER.lock().unwrap();

    if m.capturing {
        match tap {
            Tap::Escape => {
                m.capturing = false;
                m.peak_mods.clear();
                drop(m);
                emit("capture-cancel", None);
            }
            Tap::Mod(md) => {
                m.mods_down.insert(md);
                m.peak_mods = m.mods_down.clone();
                let combo = combo_string(&m.mods_down, None);
                drop(m);
                emit("capture-update", Some(CaptureUpdate { combo }));
            }
            Tap::Key(k) => {
                let combo = combo_string(&m.mods_down, Some(k));
                m.capturing = false;
                m.committed = true;
                m.peak_mods.clear();
                drop(m);
                emit("capture-commit", Some(CaptureUpdate { combo }));
            }
        }
        return;
    }

    match tap {
        Tap::Escape => {}
        Tap::Mod(md) => {
            m.mods_down.insert(md);
            // Modifier-only bindings (e.g. bare fn) fire when the held set
            // matches exactly.
            let hit = m
                .bindings
                .iter()
                .enumerate()
                .find(|(_, b)| b.key.is_none() && b.mods == m.mods_down)
                .map(|(i, b)| (i, b.mode));
            if let Some((i, mode)) = hit {
                match mode {
                    Mode::HandsFree => {
                        drop(m);
                        emit("hotkey-toggle", None);
                    }
                    Mode::PushToTalk => {
                        m.active_ptt = Some(i);
                        drop(m);
                        emit("ptt-down", None);
                    }
                }
            }
        }
        Tap::Key(k) => {
            let hit = m
                .bindings
                .iter()
                .enumerate()
                .find(|(_, b)| b.key.as_deref() == Some(k) && b.mods == m.mods_down)
                .map(|(i, b)| (i, b.mode));
            if let Some((i, mode)) = hit {
                // e.g. bare fn started push-to-talk, then Space arrived making
                // it fn+space: the key combo wins, the held PTT is abandoned.
                let cancel = m.active_ptt.take().is_some();
                match mode {
                    Mode::HandsFree => {
                        drop(m);
                        if cancel {
                            emit("ptt-cancel", None);
                        }
                        emit("hotkey-toggle", None);
                    }
                    Mode::PushToTalk => {
                        m.active_ptt = Some(i);
                        drop(m);
                        if cancel {
                            emit("ptt-cancel", None);
                        }
                        emit("ptt-down", None);
                    }
                }
            }
        }
    }
}

fn on_release(tap: Tap) {
    let mut m = MATCHER.lock().unwrap();

    if m.capturing || (m.committed && m.mods_down.is_empty()) {
        if let Tap::Mod(md) = tap {
            m.mods_down.remove(md);
            if m.capturing {
                if m.mods_down.is_empty() && !m.peak_mods.is_empty() {
                    // Modifier-only combo: commit what was held (bare fn).
                    let combo = combo_string(&m.peak_mods.clone(), None);
                    m.capturing = false;
                    m.peak_mods.clear();
                    drop(m);
                    emit("capture-commit", Some(CaptureUpdate { combo }));
                    return;
                }
                let combo = combo_string(&m.mods_down, None);
                drop(m);
                emit("capture-update", Some(CaptureUpdate { combo }));
            }
        }
        return;
    }
    m.committed = false;

    match tap {
        Tap::Escape => {}
        Tap::Mod(md) => {
            // Releasing a modifier that an active push-to-talk depends on
            // ends it.
            let ends_ptt = m
                .active_ptt
                .and_then(|i| m.bindings.get(i))
                .map(|b| b.mods.contains(md))
                .unwrap_or(false);
            m.mods_down.remove(md);
            if ends_ptt {
                m.active_ptt = None;
                drop(m);
                emit("ptt-up", None);
            }
        }
        Tap::Key(k) => {
            let ends_ptt = m
                .active_ptt
                .and_then(|i| m.bindings.get(i))
                .map(|b| b.key.as_deref() == Some(k))
                .unwrap_or(false);
            if ends_ptt {
                m.active_ptt = None;
                drop(m);
                emit("ptt-up", None);
            }
        }
    }
}

#[cfg(target_os = "macos")]
mod listener {
    use super::{on_key, should_swallow_fn, Tap, LISTENER_STATE};
    use core_foundation::base::TCFType;
    use core_foundation::runloop::{kCFRunLoopCommonModes, CFRunLoop};
    use core_graphics::event::{
        CGEventFlags, CGEventTap, CGEventTapLocation, CGEventTapOptions, CGEventTapPlacement,
        CGEventType, CallbackResult, EventField,
    };
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicPtr, Ordering};

    static TAP_PORT: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

    extern "C" {
        fn CGEventTapEnable(tap: *mut c_void, enable: bool);
    }

    const ESCAPE: i64 = 53;

    fn mod_from_keycode(code: i64) -> Option<(&'static str, CGEventFlags)> {
        match code {
            54 | 55 => Some(("cmd", CGEventFlags::CGEventFlagCommand)),
            56 | 60 => Some(("shift", CGEventFlags::CGEventFlagShift)),
            58 | 61 => Some(("alt", CGEventFlags::CGEventFlagAlternate)),
            59 | 62 => Some(("ctrl", CGEventFlags::CGEventFlagControl)),
            63 => Some(("fn", CGEventFlags::CGEventFlagSecondaryFn)),
            _ => None,
        }
    }

    fn name_from_keycode(code: i64) -> Option<&'static str> {
        Some(match code {
            0 => "a",
            11 => "b",
            8 => "c",
            2 => "d",
            14 => "e",
            3 => "f",
            5 => "g",
            4 => "h",
            34 => "i",
            38 => "j",
            40 => "k",
            37 => "l",
            46 => "m",
            45 => "n",
            31 => "o",
            35 => "p",
            12 => "q",
            15 => "r",
            1 => "s",
            17 => "t",
            32 => "u",
            9 => "v",
            13 => "w",
            7 => "x",
            16 => "y",
            6 => "z",
            29 => "0",
            18 => "1",
            19 => "2",
            20 => "3",
            21 => "4",
            23 => "5",
            22 => "6",
            26 => "7",
            28 => "8",
            25 => "9",
            49 => "space",
            51 => "backspace",
            117 => "delete",
            36 => "enter",
            48 => "tab",
            115 => "home",
            119 => "end",
            116 => "pageup",
            121 => "pagedown",
            123 => "left",
            124 => "right",
            125 => "down",
            126 => "up",
            122 => "f1",
            120 => "f2",
            99 => "f3",
            118 => "f4",
            96 => "f5",
            97 => "f6",
            98 => "f7",
            100 => "f8",
            101 => "f9",
            109 => "f10",
            103 => "f11",
            111 => "f12",
            _ => return None,
        })
    }

    pub fn run() -> Result<(), String> {
        // An active (Default) tap so bare-fn events can be dropped before the
        // system acts on them; events we return Keep for pass through intact.
        let tap = CGEventTap::new(
            CGEventTapLocation::HID,
            CGEventTapPlacement::HeadInsertEventTap,
            CGEventTapOptions::Default,
            vec![
                CGEventType::KeyDown,
                CGEventType::KeyUp,
                CGEventType::FlagsChanged,
            ],
            |_proxy, etype, event| {
                match etype {
                    CGEventType::TapDisabledByTimeout | CGEventType::TapDisabledByUserInput => {
                        // macOS disables slow taps; turn ours back on.
                        let port = TAP_PORT.load(Ordering::SeqCst);
                        if !port.is_null() {
                            unsafe { CGEventTapEnable(port, true) };
                        }
                        return CallbackResult::Keep;
                    }
                    _ => {}
                }
                let code = event.get_integer_value_field(EventField::KEYBOARD_EVENT_KEYCODE);
                match etype {
                    CGEventType::KeyDown | CGEventType::KeyUp => {
                        let down = matches!(etype, CGEventType::KeyDown);
                        if code == ESCAPE {
                            if down {
                                on_key(Tap::Escape, true);
                            }
                        } else if let Some(name) = name_from_keycode(code) {
                            on_key(Tap::Key(name), down);
                        }
                    }
                    CGEventType::FlagsChanged => {
                        if let Some((name, flag)) = mod_from_keycode(code) {
                            let down = event.get_flags().contains(flag);
                            on_key(Tap::Mod(name), down);
                            if name == "fn" && should_swallow_fn() {
                                return CallbackResult::Drop;
                            }
                        }
                    }
                    _ => {}
                }
                CallbackResult::Keep
            },
        )
        .map_err(|_| "event tap creation failed (Accessibility permission needed)".to_string())?;

        let source = tap
            .mach_port()
            .create_runloop_source(0)
            .map_err(|_| "run loop source creation failed".to_string())?;
        let run_loop = CFRunLoop::get_current();
        run_loop.add_source(&source, unsafe { kCFRunLoopCommonModes });
        tap.enable();
        TAP_PORT.store(
            tap.mach_port().as_concrete_TypeRef() as *mut c_void,
            Ordering::SeqCst,
        );
        LISTENER_STATE.store(2, Ordering::SeqCst);
        CFRunLoop::run_current(); // never returns while the tap lives
        Ok(())
    }
}

#[cfg(not(target_os = "macos"))]
mod listener {
    pub fn run() -> Result<(), String> {
        Err("fn-key shortcuts are only supported on macOS".to_string())
    }
}
