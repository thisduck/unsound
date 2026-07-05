//! Low-level global key listener (CGEventTap via rdev) for shortcuts the
//! regular macOS hotkey API can't express — anything involving the fn key —
//! and for live key capture in the settings UI. Needs the Accessibility
//! permission; plain combos stay on tauri-plugin-global-shortcut, which
//! works without it.

use serde::Serialize;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use tauri::{AppHandle, Emitter};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mode {
    HandsFree,
    PushToTalk,
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
static SUPPRESS_EVENTS: AtomicBool = AtomicBool::new(false);

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
    std::thread::spawn(|| {
        LISTENER_STATE.store(2, Ordering::SeqCst);
        if let Err(e) = rdev::listen(on_event) {
            eprintln!("fn-key listener unavailable (grant Accessibility): {e:?}");
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

/// While the plugin-registered (non-fn) shortcuts are being re-applied we
/// don't want double events; unused for now but kept for symmetry.
#[allow(dead_code)]
pub fn suppress(on: bool) {
    SUPPRESS_EVENTS.store(on, Ordering::SeqCst);
}

fn modifier_of(key: &rdev::Key) -> Option<&'static str> {
    use rdev::Key::*;
    match key {
        MetaLeft | MetaRight => Some("cmd"),
        ShiftLeft | ShiftRight => Some("shift"),
        Alt | AltGr => Some("alt"),
        ControlLeft | ControlRight => Some("ctrl"),
        Function => Some("fn"),
        _ => None,
    }
}

fn key_name(key: &rdev::Key) -> Option<String> {
    use rdev::Key::*;
    let name = match key {
        Space => "space".into(),
        Backspace => "backspace".into(),
        Delete => "delete".into(),
        Home => "home".into(),
        End => "end".into(),
        PageUp => "pageup".into(),
        PageDown => "pagedown".into(),
        Tab => "tab".into(),
        Return => "enter".into(),
        UpArrow => "up".into(),
        DownArrow => "down".into(),
        LeftArrow => "left".into(),
        RightArrow => "right".into(),
        F1 => "f1".into(),
        F2 => "f2".into(),
        F3 => "f3".into(),
        F4 => "f4".into(),
        F5 => "f5".into(),
        F6 => "f6".into(),
        F7 => "f7".into(),
        F8 => "f8".into(),
        F9 => "f9".into(),
        F10 => "f10".into(),
        F11 => "f11".into(),
        F12 => "f12".into(),
        KeyA => "a".into(),
        KeyB => "b".into(),
        KeyC => "c".into(),
        KeyD => "d".into(),
        KeyE => "e".into(),
        KeyF => "f".into(),
        KeyG => "g".into(),
        KeyH => "h".into(),
        KeyI => "i".into(),
        KeyJ => "j".into(),
        KeyK => "k".into(),
        KeyL => "l".into(),
        KeyM => "m".into(),
        KeyN => "n".into(),
        KeyO => "o".into(),
        KeyP => "p".into(),
        KeyQ => "q".into(),
        KeyR => "r".into(),
        KeyS => "s".into(),
        KeyT => "t".into(),
        KeyU => "u".into(),
        KeyV => "v".into(),
        KeyW => "w".into(),
        KeyX => "x".into(),
        KeyY => "y".into(),
        KeyZ => "z".into(),
        Num0 => "0".into(),
        Num1 => "1".into(),
        Num2 => "2".into(),
        Num3 => "3".into(),
        Num4 => "4".into(),
        Num5 => "5".into(),
        Num6 => "6".into(),
        Num7 => "7".into(),
        Num8 => "8".into(),
        Num9 => "9".into(),
        _ => return None,
    };
    Some(name)
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

fn on_event(event: rdev::Event) {
    match event.event_type {
        rdev::EventType::KeyPress(key) => on_press(key),
        rdev::EventType::KeyRelease(key) => on_release(key),
        _ => {}
    }
}

fn on_press(key: rdev::Key) {
    let mut m = MATCHER.lock().unwrap();

    if m.capturing {
        if key == rdev::Key::Escape {
            m.capturing = false;
            m.peak_mods.clear();
            drop(m);
            emit("capture-cancel", None);
            return;
        }
        if let Some(md) = modifier_of(&key) {
            m.mods_down.insert(md);
            m.peak_mods = m.mods_down.clone();
            let combo = combo_string(&m.mods_down, None);
            drop(m);
            emit("capture-update", Some(CaptureUpdate { combo }));
        } else if let Some(k) = key_name(&key) {
            let combo = combo_string(&m.mods_down, Some(&k));
            m.capturing = false;
            m.committed = true;
            m.peak_mods.clear();
            drop(m);
            emit("capture-commit", Some(CaptureUpdate { combo }));
        }
        return;
    }

    if SUPPRESS_EVENTS.load(Ordering::SeqCst) {
        if let Some(md) = modifier_of(&key) {
            m.mods_down.insert(md);
        }
        return;
    }

    if let Some(md) = modifier_of(&key) {
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
    } else if let Some(k) = key_name(&key) {
        let hit = m
            .bindings
            .iter()
            .enumerate()
            .find(|(_, b)| b.key.as_deref() == Some(k.as_str()) && b.mods == m.mods_down)
            .map(|(i, b)| (i, b.mode));
        if let Some((i, mode)) = hit {
            // e.g. bare fn started push-to-talk, then Space arrived making it
            // fn+space: the key combo wins, the held PTT is abandoned.
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

fn on_release(key: rdev::Key) {
    let mut m = MATCHER.lock().unwrap();

    if m.capturing || (m.committed && m.mods_down.is_empty()) {
        if let Some(md) = modifier_of(&key) {
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

    if let Some(md) = modifier_of(&key) {
        // Releasing a modifier that an active push-to-talk depends on ends it.
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
    } else if let Some(k) = key_name(&key) {
        let ends_ptt = m
            .active_ptt
            .and_then(|i| m.bindings.get(i))
            .map(|b| b.key.as_deref() == Some(k.as_str()))
            .unwrap_or(false);
        if ends_ptt {
            m.active_ptt = None;
            drop(m);
            emit("ptt-up", None);
        }
    }
}
