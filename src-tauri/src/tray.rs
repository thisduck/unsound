use crate::{audio, deliver, settings};
use tauri::menu::{CheckMenuItem, IsMenuItem, Menu, MenuItem, PredefinedMenuItem, Submenu};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Manager, Wry};

pub const TRAY_ID: &str = "unsound-tray";

const MIC_DEFAULT_ID: &str = "mic:";

pub fn build_menu(app: &AppHandle) -> Result<Menu<Wry>, tauri::Error> {
    let open = MenuItem::with_id(app, "open", "Open unsound", true, None::<&str>)?;
    let paste = MenuItem::with_id(app, "paste-last", "Paste last take", true, None::<&str>)?;

    let selected = settings::load(app).mic_device;
    let mut mic_items: Vec<CheckMenuItem<Wry>> = vec![CheckMenuItem::with_id(
        app,
        MIC_DEFAULT_ID,
        "System default",
        true,
        selected.is_empty(),
        None::<&str>,
    )?];
    for name in audio::list_input_devices() {
        mic_items.push(CheckMenuItem::with_id(
            app,
            format!("mic:{name}"),
            &name,
            true,
            selected == name,
            None::<&str>,
        )?);
    }
    let mic_refs: Vec<&dyn IsMenuItem<Wry>> =
        mic_items.iter().map(|i| i as &dyn IsMenuItem<Wry>).collect();
    let microphone = Submenu::with_items(app, "Microphone", true, &mic_refs)?;

    let quit = MenuItem::with_id(app, "quit", "Quit unsound", true, None::<&str>)?;

    Menu::with_items(
        app,
        &[
            &open,
            &paste,
            &PredefinedMenuItem::separator(app)?,
            &microphone,
            &PredefinedMenuItem::separator(app)?,
            &quit,
        ],
    )
}

pub fn init(app: &AppHandle) -> Result<(), tauri::Error> {
    let menu = build_menu(app)?;
    TrayIconBuilder::with_id(TRAY_ID)
        .icon(app.default_window_icon().cloned().expect("app icon missing"))
        .icon_as_template(false)
        .tooltip("unsound")
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(|app, event| on_menu_event(app, event.id().as_ref()))
        .build(app)?;
    Ok(())
}

fn on_menu_event(app: &AppHandle, id: &str) {
    match id {
        "open" => {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }
        "paste-last" => {
            let text = app
                .state::<crate::AppState>()
                .last_output
                .lock()
                .unwrap()
                .clone();
            if !text.is_empty() {
                std::thread::spawn(move || {
                    let _ = deliver::deliver_text(&text);
                });
            }
        }
        "quit" => app.exit(0),
        mic if mic.starts_with("mic:") => {
            let device = mic.trim_start_matches("mic:").to_string();
            let mut s = settings::load(app);
            s.mic_device = device;
            let _ = settings::save(app, &s);
            // Rebuild so the checkmark moves; notify the UI so its picker follows.
            refresh_menu(app);
            let _ = app.emit("settings-changed", ());
        }
        _ => {}
    }
}

/// Rebuild the tray menu (called after mic changes or device list refreshes).
pub fn refresh_menu(app: &AppHandle) {
    if let Some(tray) = app.tray_by_id(TRAY_ID) {
        if let Ok(menu) = build_menu(app) {
            let _ = tray.set_menu(Some(menu));
        }
    }
}
