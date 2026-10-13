//! Read-only calendar lookup via EventKit: the event you're in right now, so a
//! meeting can start with a real title and the names of the people on it
//! (which also gives speaker detection a sensible head count).
//!
//! Needs the Calendars permission (full access on macOS 14+, where write-only
//! access can't read events). Nothing here leaves the machine.

use serde::Serialize;

/// A calendar event overlapping "now".
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CalEvent {
    pub title: String,
    /// ISO-8601 UTC, the same shape as the frontend's `toISOString()`.
    pub start: String,
    pub end: String,
    /// Display names, else email addresses; the current user left out.
    pub attendees: Vec<String>,
    pub organizer: Option<String>,
    pub location: String,
}

#[derive(Clone, Copy, Serialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
pub enum CalAccess {
    NotDetermined,
    Denied,
    Granted,
}

/// How far either side of now an event may sit and still count as "the one
/// you're in" — covers joining a few minutes early or late.
#[cfg(target_os = "macos")]
const WINDOW_SECS: f64 = 15.0 * 60.0;

#[cfg(target_os = "macos")]
pub fn access() -> CalAccess {
    use objc2_event_kit::{EKAuthorizationStatus, EKEntityType, EKEventStore};
    let status = unsafe { EKEventStore::authorizationStatusForEntityType(EKEntityType::Event) };
    match status {
        // FullAccess shares its value with the pre-14 `Authorized`.
        EKAuthorizationStatus::FullAccess => CalAccess::Granted,
        EKAuthorizationStatus::NotDetermined => CalAccess::NotDetermined,
        // Restricted, Denied, and WriteOnly (which can't read events).
        _ => CalAccess::Denied,
    }
}

/// Ask for calendar access, blocking until the user answers the system
/// prompt. Returns whether access was granted; already-decided access returns
/// at once without prompting.
#[cfg(target_os = "macos")]
pub fn request_access() -> Result<bool, String> {
    use block2::RcBlock;
    use objc2::runtime::Bool;
    use objc2_event_kit::{EKEntityType, EKEventStore};
    use objc2_foundation::NSError;
    use std::sync::mpsc;
    use std::time::Duration;

    match access() {
        CalAccess::Granted => return Ok(true),
        CalAccess::Denied => return Ok(false),
        CalAccess::NotDetermined => {}
    }
    let store = unsafe { EKEventStore::new() };
    let (tx, rx) = mpsc::channel::<Result<bool, String>>();
    let handler = RcBlock::new(move |granted: Bool, err: *mut NSError| {
        let result = if granted.as_bool() {
            Ok(true)
        } else if err.is_null() {
            Ok(false)
        } else {
            // Safe: non-null NSError from the completion handler.
            Err(unsafe { &*err }.localizedDescription().to_string())
        };
        let _ = tx.send(result);
    });
    // The block pointer is only borrowed for the call; EventKit copies it.
    let block = RcBlock::as_ptr(&handler);
    if objc2::available!(macos = 14.0) {
        unsafe { store.requestFullAccessToEventsWithCompletion(block) };
    } else {
        #[allow(deprecated)]
        unsafe {
            store.requestAccessToEntityType_completion(EKEntityType::Event, block)
        };
    }
    // Generous: the user may take a while to read the prompt.
    let result = rx
        .recv_timeout(Duration::from_secs(300))
        .map_err(|_| "no answer to the calendar permission prompt".to_string())?;
    drop(store);
    result
}

/// The (non-all-day) event overlapping the next/last 15 minutes whose start is
/// closest to now, or `None` without access or when nothing is scheduled.
#[cfg(target_os = "macos")]
pub fn current_event() -> Option<CalEvent> {
    use objc2_event_kit::EKEventStore;
    use objc2_foundation::NSDate;

    if access() != CalAccess::Granted {
        return None;
    }
    let store = unsafe { EKEventStore::new() };
    let from = NSDate::dateWithTimeIntervalSinceNow(-WINDOW_SECS);
    let to = NSDate::dateWithTimeIntervalSinceNow(WINDOW_SECS);
    let now = NSDate::now().timeIntervalSince1970();
    let predicate =
        unsafe { store.predicateForEventsWithStartDate_endDate_calendars(&from, &to, None) };
    let events = unsafe { store.eventsMatchingPredicate(&predicate) };

    let best = events
        .iter()
        .filter(|e| !unsafe { e.isAllDay() })
        .min_by(|a, b| {
            let da = (unsafe { a.startDate() }.timeIntervalSince1970() - now).abs();
            let db = (unsafe { b.startDate() }.timeIntervalSince1970() - now).abs();
            da.total_cmp(&db)
        })?;

    let attendees = unsafe { best.attendees() }
        .map(|list| {
            list.iter()
                .filter(|p| !unsafe { p.isCurrentUser() })
                .filter_map(|p| participant_label(&p))
                .collect()
        })
        .unwrap_or_default();
    Some(CalEvent {
        title: unsafe { best.title() }.to_string(),
        start: iso_utc(unsafe { best.startDate() }.timeIntervalSince1970()),
        end: iso_utc(unsafe { best.endDate() }.timeIntervalSince1970()),
        attendees,
        organizer: unsafe { best.organizer() }.and_then(|p| participant_label(&p)),
        location: unsafe { best.location() }
            .map(|l| l.to_string())
            .unwrap_or_default(),
    })
}

/// A participant's display name, else the address from their `mailto:` URL.
#[cfg(target_os = "macos")]
fn participant_label(p: &objc2_event_kit::EKParticipant) -> Option<String> {
    if let Some(name) = unsafe { p.name() } {
        let name = name.to_string();
        if !name.trim().is_empty() {
            return Some(name.trim().to_string());
        }
    }
    let url = unsafe { p.URL() }.absoluteString()?.to_string();
    let addr = url.strip_prefix("mailto:").unwrap_or(&url).trim().to_string();
    (!addr.is_empty()).then_some(addr)
}

#[cfg(not(target_os = "macos"))]
pub fn access() -> CalAccess {
    CalAccess::NotDetermined
}

#[cfg(not(target_os = "macos"))]
pub fn request_access() -> Result<bool, String> {
    Ok(false)
}

#[cfg(not(target_os = "macos"))]
pub fn current_event() -> Option<CalEvent> {
    None
}

/// Seconds since the Unix epoch → "YYYY-MM-DDTHH:MM:SS.mmmZ" (no date crate).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn iso_utc(secs: f64) -> String {
    let total_ms = (secs * 1000.0).round() as i64;
    let ms = total_ms.rem_euclid(1000);
    let t = total_ms.div_euclid(1000);
    let (days, sod) = (t.div_euclid(86_400), t.rem_euclid(86_400));
    // Howard Hinnant's civil-from-days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{ms:03}Z",
        sod / 3600,
        sod % 3600 / 60,
        sod % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_utc_like_to_iso_string() {
        assert_eq!(iso_utc(0.0), "1970-01-01T00:00:00.000Z");
        // 2026-07-29T16:00:12.345Z
        assert_eq!(iso_utc(1_785_340_812.345), "2026-07-29T16:00:12.345Z");
        // Leap day.
        assert_eq!(iso_utc(1_709_164_800.0), "2024-02-29T00:00:00.000Z");
    }
}
