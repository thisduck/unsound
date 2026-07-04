use serde::Serialize;

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
