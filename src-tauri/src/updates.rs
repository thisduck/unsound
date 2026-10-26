//! Signed updates are downloaded separately from installation. Only the native
//! installer may reserve a restart, sharing a gate with every capture entry.
use crate::{audio, meeting, store, AppState};
use serde::Serialize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_updater::{Update, UpdaterExt};

static INSTALLING: Mutex<bool> = Mutex::new(false);
static ACTIVE_WORK: AtomicUsize = AtomicUsize::new(0);

pub struct WorkPermit;
impl Drop for WorkPermit {
    fn drop(&mut self) {
        ACTIVE_WORK.fetch_sub(1, Ordering::SeqCst);
    }
}
pub fn work_permit() -> Result<WorkPermit, String> {
    let _gate = capture_permit()?;
    ACTIVE_WORK.fetch_add(1, Ordering::SeqCst);
    Ok(WorkPermit)
}

pub fn capture_permit() -> Result<MutexGuard<'static, bool>, String> {
    let gate = INSTALLING.lock().unwrap();
    if *gate {
        return Err(
            "An update is being installed. Recording will be available after restart.".into(),
        );
    }
    Ok(gate)
}

#[derive(Default)]
pub struct Updates {
    prepared: tokio::sync::Mutex<Option<Prepared>>,
}
struct Prepared {
    update: Update,
    bytes: Option<Vec<u8>>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateInfo {
    version: String,
    current_version: String,
    notes: String,
}

#[derive(Clone, Serialize)]
pub struct Progress {
    downloaded: u64,
    total: Option<u64>,
}

#[tauri::command]
pub async fn check_app_update(
    app: AppHandle,
    state: State<'_, Updates>,
) -> Result<Option<UpdateInfo>, String> {
    let mut prepared = state
        .prepared
        .try_lock()
        .map_err(|_| "An update operation is already in progress.")?;
    let mut update = app
        .updater_builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| e.to_string())?
        .check()
        .await
        .map_err(|e| e.to_string())?;
    if let Some(update) = update.as_mut() {
        update.timeout = Some(Duration::from_secs(300));
        // Update metadata can be served only from the configured endpoint; the
        // archive is also restricted to this repository and verified by Tauri.
        if !valid_download_url(update.download_url.as_str()) {
            return Err("The update download is not from the unsound release repository.".into());
        }
    }
    let info = update.as_ref().map(|u| UpdateInfo {
        version: u.version.clone(),
        current_version: u.current_version.clone(),
        notes: u.body.clone().unwrap_or_default(),
    });
    *prepared = update.map(|update| Prepared {
        update,
        bytes: None,
    });
    Ok(info)
}

fn valid_download_url(url: &str) -> bool {
    url.starts_with("https://github.com/thisduck/unsound/releases/download/")
}

#[tauri::command]
pub async fn download_app_update(app: AppHandle, state: State<'_, Updates>) -> Result<(), String> {
    let mut prepared = state
        .prepared
        .try_lock()
        .map_err(|_| "An update operation is already in progress.")?;
    let prepared = prepared.as_mut().ok_or("Check for an update first.")?;
    prepared.bytes = None;
    let mut downloaded = 0;
    // download() verifies the signature before returning any installable bytes.
    let bytes = prepared
        .update
        .download(
            |chunk, total| {
                downloaded += chunk as u64;
                let _ = app.emit("app-update-progress", Progress { downloaded, total });
            },
            || {},
        )
        .await
        .map_err(|e| e.to_string())?;
    prepared.bytes = Some(bytes);
    Ok(())
}

fn busy_reason(app: &AppHandle) -> Result<Option<String>, String> {
    if ACTIVE_WORK.load(Ordering::SeqCst) > 0 {
        return Ok(Some(
            "Wait for current processing to finish before updating.".into(),
        ));
    }
    #[cfg(target_os = "macos")]
    if crate::syscapture::is_recording(&app.state::<AppState>().sys) {
        return Ok(Some(
            "Finish your recording before restarting to update.".into(),
        ));
    }
    if audio::owner(&app.state::<AppState>().audio).is_some() || meeting::is_recording(app) {
        return Ok(Some(
            "Finish your recording before restarting to update.".into(),
        ));
    }
    if meeting::statuses(app)?
        .iter()
        .any(|s| matches!(s.state.as_str(), "recording" | "finishing"))
    {
        return Ok(Some(
            "Wait for meeting transcription and notes to finish before updating.".into(),
        ));
    }
    if !store::tasks_with_status(
        &app.state::<store::Db>(),
        &["heard", "planning", "drafting", "running"],
    )?
    .is_empty()
    {
        return Ok(Some(
            "Wait for assistant tasks to finish before updating.".into(),
        ));
    }
    Ok(None)
}

#[tauri::command]
pub fn app_update_blocker(app: AppHandle) -> Result<Option<String>, String> {
    busy_reason(&app)
}

#[tauri::command]
pub async fn install_app_update(app: AppHandle) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        #[cfg(target_os = "macos")]
        {
            let exe = std::env::current_exe().map_err(|e| e.to_string())?;
            let bundle = exe
                .ancestors()
                .nth(3)
                .ok_or("Install a release in Applications before updating.")?;
            if bundle.extension().and_then(|v| v.to_str()) != Some("app") {
                return Err("Install a release in Applications before updating.".into());
            }
        }
        let state = app.state::<Updates>();
        let prepared = state
            .prepared
            .try_lock()
            .map_err(|_| "An update operation is already in progress.")?;
        let prepared = prepared.as_ref().ok_or("Check for an update first.")?;
        let bytes = prepared
            .bytes
            .as_ref()
            .ok_or("Download and verify the update first.")?;
        {
            let mut gate = capture_permit()?;
            if let Some(reason) = busy_reason(&app)? {
                return Err(reason);
            }
            *gate = true;
        }
        // Restore capture on any installation error or unwinding. On success,
        // keep it reserved until the process is replaced.
        struct Reservation;
        impl Drop for Reservation {
            fn drop(&mut self) {
                *INSTALLING.lock().unwrap() = false;
            }
        }
        let _reservation = Reservation;
        prepared.update.install(bytes).map_err(|e| e.to_string())?;
        app.restart();
    })
    .await
    .map_err(|e| e.to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn updates_reject_untrusted_download_locations() {
        assert!(valid_download_url(
            "https://github.com/thisduck/unsound/releases/download/v1.0.0/unsound.app.tar.gz"
        ));
        for url in [
            "http://github.com/thisduck/unsound/releases/download/a",
            "https://github.com.evil.test/thisduck/unsound/releases/download/a",
            "https://github.com/other/app/releases/download/a",
        ] {
            assert!(!valid_download_url(url));
        }
    }
    #[test]
    fn updates_install_reservation_blocks_capture_and_recovers() {
        let work = work_permit().unwrap();
        assert_eq!(ACTIVE_WORK.load(Ordering::SeqCst), 1);
        drop(work);
        assert_eq!(ACTIVE_WORK.load(Ordering::SeqCst), 0);
        *INSTALLING.lock().unwrap() = true;
        assert!(capture_permit().is_err());
        assert!(work_permit().is_err());
        *INSTALLING.lock().unwrap() = false;
        assert!(capture_permit().is_ok());
    }
}
