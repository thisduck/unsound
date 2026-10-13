//! Credentials the assistant needs (Jira API token, endpoint keys) live in the
//! macOS Keychain, never in settings.json — that file is plain text and easy
//! to paste into a bug report.

#[cfg(target_os = "macos")]
const SERVICE: &str = "com.unsound.app";

/// The stored secret, or `None` when absent (or the Keychain is unavailable).
#[cfg(target_os = "macos")]
pub fn get(key: &str) -> Option<String> {
    let entry = keyring::Entry::new(SERVICE, key).ok()?;
    match entry.get_password() {
        Ok(v) => Some(v),
        Err(keyring::Error::NoEntry) => None,
        Err(e) => {
            eprintln!("[secrets] reading '{key}' failed: {e}");
            None
        }
    }
}

/// Store (or replace) a secret. An empty value deletes it instead.
#[cfg(target_os = "macos")]
pub fn set(key: &str, value: &str) -> Result<(), String> {
    if value.is_empty() {
        return delete(key);
    }
    keyring::Entry::new(SERVICE, key)
        .and_then(|e| e.set_password(value))
        .map_err(|e| format!("could not save to the Keychain: {e}"))
}

/// Remove a secret; removing one that isn't there is not an error.
#[cfg(target_os = "macos")]
pub fn delete(key: &str) -> Result<(), String> {
    let entry = keyring::Entry::new(SERVICE, key).map_err(|e| e.to_string())?;
    match entry.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(format!("could not remove from the Keychain: {e}")),
    }
}

#[cfg(not(target_os = "macos"))]
pub fn get(_key: &str) -> Option<String> {
    None
}

#[cfg(not(target_os = "macos"))]
pub fn set(_key: &str, _value: &str) -> Result<(), String> {
    Err("secure credential storage is only available on macOS".into())
}

#[cfg(not(target_os = "macos"))]
pub fn delete(_key: &str) -> Result<(), String> {
    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_throwaway_key() {
        let key = "unsound-test-key";
        set(key, "s3cret").unwrap();
        assert_eq!(get(key).as_deref(), Some("s3cret"));
        set(key, "").unwrap(); // empty deletes
        assert_eq!(get(key), None);
        delete(key).unwrap(); // deleting a missing key is fine
    }
}
