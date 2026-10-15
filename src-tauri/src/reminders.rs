//! Persistent reminders; the backend continues while the main window is hidden.
use crate::{store, util};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_notification::NotificationExt;

pub fn start(app: AppHandle) {
    std::thread::spawn(move || loop {
        let db = app.state::<store::Db>();
        match store::due_reminders(&db, util::now_ms(), true) {
            Ok(items) => {
                for item in items {
                    match store::claim_reminder(&db, &item, util::now_ms()) {
                        Ok(true) => {}
                        Ok(false) => continue,
                        Err(e) => {
                            eprintln!("[reminders] could not claim reminder: {e}");
                            continue;
                        }
                    }
                    let result = (|| -> Result<(), String> {
                        if app
                            .notification()
                            .permission_state()
                            .map_err(|e| e.to_string())?
                            != tauri_plugin_notification::PermissionState::Granted
                        {
                            return Err("Notifications are disabled. This reminder is still available in unsound.".into());
                        }
                        app.notification()
                            .builder()
                            .title("Meeting follow-up")
                            .body(&item.text)
                            .show()
                            .map_err(|e| e.to_string())
                    })();
                    if let Err(e) = store::reminder_result(
                        &db,
                        &item,
                        util::now_ms(),
                        result.as_ref().err().map(String::as_str),
                    ) {
                        eprintln!("[reminders] could not save delivery status: {e}");
                    }
                    let _ = app.emit("action-items-changed", ());
                }
            }
            Err(e) => eprintln!("[reminders] could not read reminders: {e}"),
        }
        std::thread::sleep(std::time::Duration::from_secs(15));
    });
}
