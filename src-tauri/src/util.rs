//! Small shared helpers: timestamps without a date crate, and the transcript
//! time format used everywhere a meeting position is shown to a person or a
//! model ("12:34", "1:02:03").

use std::time::{SystemTime, UNIX_EPOCH};

/// Now as ISO-8601 UTC with milliseconds — the same shape the frontend's
/// `toISOString()` writes, so stored timestamps sort lexicographically.
pub fn now_iso() -> String {
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    iso_from_unix_ms(d.as_millis() as i64)
}

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Unix milliseconds → "YYYY-MM-DDTHH:MM:SS.mmmZ".
pub fn iso_from_unix_ms(ms: i64) -> String {
    let secs = ms.div_euclid(1000);
    let milli = ms.rem_euclid(1000);
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{milli:03}Z",
        tod / 3600,
        (tod / 60) % 60,
        tod % 60
    )
}

/// Parse the ISO-8601 UTC shapes we write ("…Z", optional fraction) back to
/// Unix milliseconds. Offsets other than Z aren't produced by us; `None`.
#[cfg(test)]
pub fn unix_ms_from_iso(s: &str) -> Option<i64> {
    let s = s.trim();
    let s = s.strip_suffix('Z').unwrap_or(s);
    let (date, time) = s.split_once('T')?;
    let mut dp = date.split('-');
    let y: i64 = dp.next()?.parse().ok()?;
    let mo: i64 = dp.next()?.parse().ok()?;
    let d: i64 = dp.next()?.parse().ok()?;
    let (hms, frac) = time.split_once('.').unwrap_or((time, "0"));
    let mut tp = hms.split(':');
    let h: i64 = tp.next()?.parse().ok()?;
    let mi: i64 = tp.next()?.parse().ok()?;
    let sec: i64 = tp.next().unwrap_or("0").parse().ok()?;
    let frac3: String = frac.chars().chain("000".chars()).take(3).collect();
    let milli: i64 = frac3.parse().ok()?;
    let days = days_from_civil(y, mo, d);
    Some(((days * 86_400 + h * 3600 + mi * 60 + sec) * 1000) + milli)
}

// Howard Hinnant's date algorithms (public domain).
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// A meeting position as people read it: "4:05", or "1:02:03" past an hour.
pub fn clock(ms: i64) -> String {
    let s = ms.max(0) / 1000;
    let (h, m, sec) = (s / 3600, (s / 60) % 60, s % 60);
    if h > 0 {
        format!("{h}:{m:02}:{sec:02}")
    } else {
        format!("{m}:{sec:02}")
    }
}

/// Inverse of `clock` ("4:05" / "1:02:03" → ms), for timestamps a model cites.
pub fn parse_clock(s: &str) -> Option<i64> {
    let parts: Vec<i64> = s
        .trim()
        .split(':')
        .map(|p| p.trim().parse::<i64>().ok())
        .collect::<Option<_>>()?;
    let secs = match parts.as_slice() {
        [m, s] if *s < 60 => m * 60 + s,
        [h, m, s] if *m < 60 && *s < 60 => h * 3600 + m * 60 + s,
        _ => return None,
    };
    Some(secs * 1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_round_trips() {
        let ms = 1_785_340_812_345; // 2026-07-29T16:00:12.345Z
        let iso = iso_from_unix_ms(ms);
        assert_eq!(iso, "2026-07-29T16:00:12.345Z");
        assert_eq!(unix_ms_from_iso(&iso), Some(ms));
        assert_eq!(unix_ms_from_iso("2026-07-29T16:00:12Z"), Some(ms - 345));
    }

    #[test]
    fn clock_formats_and_parses() {
        assert_eq!(clock(65_000), "1:05");
        assert_eq!(clock(3_723_000), "1:02:03");
        assert_eq!(parse_clock("1:05"), Some(65_000));
        assert_eq!(parse_clock("1:02:03"), Some(3_723_000));
        assert_eq!(parse_clock("1:75"), None);
    }
}
