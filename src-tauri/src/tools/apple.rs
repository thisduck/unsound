//! Local, account-free actions on the Mac: clipboard, a Mail draft, a
//! Reminders item, a calendar event. Values reach AppleScript as `argv`, never
//! spliced into script source, so a quote or backslash in a meeting transcript
//! can't break (or inject into) the script.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

/// Put `text` on the general pasteboard.
pub fn copy_to_clipboard(text: &str) -> Result<(), String> {
    let mut child = Command::new("/usr/bin/pbcopy")
        // pbcopy picks its encoding from the locale; GUI apps may have none.
        .env("LANG", "en_US.UTF-8")
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not run pbcopy: {e}"))?;
    child
        .stdin
        .take()
        .ok_or("pbcopy has no stdin")?
        .write_all(text.as_bytes())
        .map_err(|e| e.to_string())?;
    let status = child.wait().map_err(|e| e.to_string())?;
    status
        .success()
        .then_some(())
        .ok_or_else(|| "pbcopy failed".to_string())
}

/// RFC 6068 `mailto:` URL. Everything outside the unreserved set is
/// percent-encoded (spaces as `%20`, newlines as `%0D%0A`) so it survives any
/// mail client's parser.
pub fn mailto_url(to: &[String], cc: &[String], subject: &str, body: &str) -> String {
    let addrs = |list: &[String]| {
        list.iter()
            .map(|a| a.trim())
            .filter(|a| !a.is_empty())
            .map(|a| encode(a, b"@"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let mut query = Vec::new();
    let cc = addrs(cc);
    if !cc.is_empty() {
        query.push(format!("cc={cc}"));
    }
    if !subject.is_empty() {
        query.push(format!("subject={}", encode(subject, b"")));
    }
    if !body.is_empty() {
        let crlf = body.replace("\r\n", "\n").replace('\n', "\r\n");
        query.push(format!("body={}", encode(&crlf, b"")));
    }
    let mut url = format!("mailto:{}", addrs(to));
    if !query.is_empty() {
        url.push('?');
        url.push_str(&query.join("&"));
    }
    url
}

fn encode(s: &str, keep: &[u8]) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) || keep.contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

const MAIL_DRAFT_SCRIPT: &str = r#"on run argv
    set theSubject to item 1 of argv
    set theBody to item 2 of argv
    set toList to paragraphs of (item 3 of argv)
    set ccList to paragraphs of (item 4 of argv)
    tell application "Mail"
        set msg to make new outgoing message with properties {subject:theSubject, content:theBody, visible:true}
        tell msg
            repeat with addr in toList
                if (length of (contents of addr)) > 0 then make new to recipient at end of to recipients with properties {address:(contents of addr)}
            end repeat
            repeat with addr in ccList
                if (length of (contents of addr)) > 0 then make new cc recipient at end of cc recipients with properties {address:(contents of addr)}
            end repeat
        end tell
        activate
    end tell
end run
"#;

/// Open a visible Mail.app draft for the user to review and send. If Mail
/// isn't usable (not set up, or Automation permission denied) fall back to a
/// `mailto:` link, which opens whatever mail client is the default.
pub async fn open_mail_draft(
    to: &[String],
    cc: &[String],
    subject: &str,
    body: &str,
) -> Result<(), String> {
    let args = vec![
        subject.to_string(),
        body.to_string(),
        to.join("\n"),
        cc.join("\n"),
    ];
    match run_osascript(MAIL_DRAFT_SCRIPT, &args).await {
        Ok(_) => Ok(()),
        Err(e) => {
            eprintln!("[apple] Mail draft failed ({e}); falling back to mailto:");
            open_url(&mailto_url(to, cc, subject, body)).await
        }
    }
}

const REMINDER_SCRIPT: &str = r#"on run argv
    set theTitle to item 1 of argv
    set theNotes to item 2 of argv
    set hasDue to (item 3 of argv) is "1"
    set hasTime to (item 4 of argv) is "1"
    if hasDue then
        set d to current date
        -- Day first, so e.g. the 31st never overflows into a 30-day month.
        set day of d to 1
        set year of d to (item 5 of argv) as integer
        set month of d to (item 6 of argv) as integer
        set day of d to (item 7 of argv) as integer
        set hours of d to (item 8 of argv) as integer
        set minutes of d to (item 9 of argv) as integer
        set seconds of d to 0
    end if
    tell application "Reminders"
        set r to make new reminder with properties {name:theTitle}
        if theNotes is not "" then set body of r to theNotes
        if hasDue then
            if hasTime then
                set due date of r to d
            else
                set allday due date of r to d
            end if
        end if
    end tell
end run
"#;

/// Add a reminder to the default Reminders list. `due` is local time:
/// `YYYY-MM-DD` (all-day) or `YYYY-MM-DDTHH:MM[:SS]` (any zone suffix ignored).
pub async fn add_reminder(title: &str, notes: &str, due: Option<&str>) -> Result<(), String> {
    let parsed = match due.map(str::trim).filter(|d| !d.is_empty()) {
        Some(d) => Some(parse_local(d).ok_or_else(|| format!("unrecognized due date '{d}'"))?),
        None => None,
    };
    let (has_due, has_time, y, mo, da, h, mi) = match parsed {
        Some(p) => ("1", if p.time { "1" } else { "0" }, p.y, p.mo, p.d, p.h, p.mi),
        None => ("0", "0", 2000, 1, 1, 0, 0),
    };
    let args = vec![
        title.to_string(),
        notes.to_string(),
        has_due.to_string(),
        has_time.to_string(),
        y.to_string(),
        mo.to_string(),
        da.to_string(),
        h.to_string(),
        mi.to_string(),
    ];
    run_osascript(REMINDER_SCRIPT, &args)
        .await
        .map(|_| ())
        .map_err(|e| format!("could not add the reminder: {e}"))
}

/// A single-event iCalendar file (RFC 5545): CRLF line endings, lines folded at
/// 75 octets, text escaped. Date-only `start` makes an all-day event; a
/// missing/invalid `end` defaults to 30 minutes (or one day). Times with `Z` or
/// an offset are written in UTC; bare times float (Calendar reads them as
/// local). Names without an email go in the description — ATTENDEE needs a URI.
pub fn calendar_ics(
    title: &str,
    start_iso: &str,
    end_iso: &str,
    notes: &str,
    location: &str,
    attendees: &[String],
) -> String {
    let now = now_utc();
    let uid = format!("{}-{}@unsound", now.stamp(), unique_suffix());
    let mut lines = vec![
        "BEGIN:VCALENDAR".to_string(),
        "VERSION:2.0".to_string(),
        "PRODID:-//unsound//meeting assistant//EN".to_string(),
        "CALSCALE:GREGORIAN".to_string(),
        "METHOD:PUBLISH".to_string(),
        "BEGIN:VEVENT".to_string(),
        format!("UID:{uid}"),
        format!("DTSTAMP:{}Z", now.stamp()),
    ];
    match parse_iso(start_iso) {
        Some(start) if !start.time => {
            let end = parse_iso(end_iso)
                .filter(|e| !e.time && e.naive_minutes() > start.naive_minutes())
                .unwrap_or_else(|| start.plus_minutes(24 * 60));
            lines.push(format!("DTSTART;VALUE=DATE:{}", start.date_stamp()));
            lines.push(format!("DTEND;VALUE=DATE:{}", end.date_stamp()));
        }
        Some(start) => {
            let end = parse_iso(end_iso)
                .filter(|e| e.time && e.utc_minutes() > start.utc_minutes())
                .unwrap_or_else(|| start.plus_minutes(30));
            lines.push(format!("DTSTART:{}", start.ics_datetime()));
            lines.push(format!("DTEND:{}", end.ics_datetime()));
        }
        None => {
            // No usable start: default to the next whole hour (UTC-written).
            let start = now.plus_minutes(60 - now.mi as i64).with_zero_seconds();
            lines.push(format!("DTSTART:{}", start.ics_datetime()));
            lines.push(format!("DTEND:{}", start.plus_minutes(30).ics_datetime()));
        }
    }
    lines.push(format!("SUMMARY:{}", ics_text(title.trim())));
    let (emails, names): (Vec<&String>, Vec<&String>) =
        attendees.iter().filter(|a| !a.trim().is_empty()).partition(|a| a.contains('@'));
    let mut description = notes.trim().to_string();
    if !names.is_empty() {
        let list = names.iter().map(|n| n.trim()).collect::<Vec<_>>().join(", ");
        if !description.is_empty() {
            description.push_str("\n\n");
        }
        description.push_str(&format!("Attendees: {list}"));
    }
    if !description.is_empty() {
        lines.push(format!("DESCRIPTION:{}", ics_text(&description)));
    }
    if !location.trim().is_empty() {
        lines.push(format!("LOCATION:{}", ics_text(location.trim())));
    }
    for e in emails {
        let e = e.trim();
        // "Priya <priya@x.com>" → CN + address
        let (cn, addr) = match (e.find('<'), e.rfind('>')) {
            (Some(a), Some(b)) if b > a => (e[..a].trim(), e[a + 1..b].trim()),
            _ => ("", e),
        };
        if cn.is_empty() {
            lines.push(format!("ATTENDEE;ROLE=REQ-PARTICIPANT:mailto:{addr}"));
        } else {
            lines.push(format!(
                "ATTENDEE;CN=\"{}\";ROLE=REQ-PARTICIPANT:mailto:{addr}",
                cn.replace('"', "")
            ));
        }
    }
    lines.push("END:VEVENT".to_string());
    lines.push("END:VCALENDAR".to_string());
    let mut out = String::new();
    for l in lines {
        out.push_str(&fold(&l));
        out.push_str("\r\n");
    }
    out
}

/// Write the event to a temp `.ics` and open it: Calendar shows its import
/// sheet, so the user confirms the calendar and nothing needs a permission.
pub async fn add_calendar_event(
    title: &str,
    start_iso: &str,
    end_iso: &str,
    notes: &str,
    location: &str,
    attendees: &[String],
) -> Result<PathBuf, String> {
    let dir = std::env::temp_dir().join("unsound-events");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("{}-{}.ics", now_utc().stamp(), unique_suffix()));
    let ics = calendar_ics(title, start_iso, end_iso, notes, location, attendees);
    std::fs::write(&path, ics).map_err(|e| e.to_string())?;
    let status = tokio::process::Command::new("/usr/bin/open")
        .arg(&path)
        .status()
        .await
        .map_err(|e| format!("could not open the event: {e}"))?;
    status
        .success()
        .then_some(path)
        .ok_or_else(|| "Calendar could not open the event file".to_string())
}

async fn open_url(url: &str) -> Result<(), String> {
    let status = tokio::process::Command::new("/usr/bin/open")
        .arg(url)
        .status()
        .await
        .map_err(|e| e.to_string())?;
    status
        .success()
        .then_some(())
        .ok_or_else(|| format!("could not open {url}"))
}

/// Run an AppleScript from stdin with `args` as its `argv`; stdout on success.
async fn run_osascript(script: &str, args: &[String]) -> Result<String, String> {
    use tokio::io::AsyncWriteExt;
    let mut child = tokio::process::Command::new("/usr/bin/osascript")
        .arg("-")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("could not run osascript: {e}"))?;
    let mut stdin = child.stdin.take().ok_or("osascript has no stdin")?;
    stdin.write_all(script.as_bytes()).await.map_err(|e| e.to_string())?;
    drop(stdin);
    // The first run can sit on the Automation permission prompt; give the user
    // time to answer it.
    let out = tokio::time::timeout(std::time::Duration::from_secs(120), child.wait_with_output())
        .await
        .map_err(|_| "timed out waiting for the app (is a permission prompt open?)".to_string())?
        .map_err(|e| e.to_string())?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        Err(if err.contains("-1743") {
            "unsound isn't allowed to control that app — enable it in System Settings → Privacy & Security → Automation".into()
        } else {
            err
        })
    }
}

// ---- dates (no date crate: civil-calendar arithmetic) ----------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct DateTime {
    y: i64,
    mo: u32,
    d: u32,
    h: u32,
    mi: u32,
    s: u32,
    /// false → date only (all-day)
    time: bool,
    /// Offset from UTC in minutes when the string carried a zone (Z = 0).
    offset_min: Option<i64>,
}

impl DateTime {
    fn naive_minutes(&self) -> i64 {
        days_from_civil(self.y, self.mo, self.d) * 1440 + (self.h * 60 + self.mi) as i64
    }
    fn utc_minutes(&self) -> i64 {
        self.naive_minutes() - self.offset_min.unwrap_or(0)
    }
    fn from_minutes(total: i64, s: u32, time: bool, offset_min: Option<i64>) -> Self {
        let days = total.div_euclid(1440);
        let rem = total.rem_euclid(1440);
        let (y, mo, d) = civil_from_days(days);
        DateTime { y, mo, d, h: (rem / 60) as u32, mi: (rem % 60) as u32, s, time, offset_min }
    }
    fn plus_minutes(&self, m: i64) -> Self {
        Self::from_minutes(self.naive_minutes() + m, self.s, self.time, self.offset_min)
    }
    fn with_zero_seconds(mut self) -> Self {
        self.s = 0;
        self
    }
    fn date_stamp(&self) -> String {
        format!("{:04}{:02}{:02}", self.y, self.mo, self.d)
    }
    fn stamp(&self) -> String {
        format!("{}T{:02}{:02}{:02}", self.date_stamp(), self.h, self.mi, self.s)
    }
    /// UTC with `Z` when the zone is known, else floating local time.
    fn ics_datetime(&self) -> String {
        match self.offset_min {
            Some(_) => {
                let utc = Self::from_minutes(self.utc_minutes(), self.s, true, Some(0));
                format!("{}Z", utc.stamp())
            }
            None => self.stamp(),
        }
    }
}

/// `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM[:SS[.fff]]`, optional `Z` / `±HH:MM`; a
/// space works in place of the `T`.
fn parse_iso(s: &str) -> Option<DateTime> {
    let s = s.trim();
    if s.len() < 10 {
        return None;
    }
    let y: i64 = s.get(0..4)?.parse().ok()?;
    if s.get(4..5)? != "-" || s.get(7..8)? != "-" {
        return None;
    }
    let mo: u32 = s.get(5..7)?.parse().ok()?;
    let d: u32 = s.get(8..10)?.parse().ok()?;
    if !(1..=12).contains(&mo) || d < 1 || d > days_in_month(y, mo) {
        return None;
    }
    let rest = &s[10..];
    if rest.is_empty() {
        return Some(DateTime { y, mo, d, h: 0, mi: 0, s: 0, time: false, offset_min: None });
    }
    let rest = rest.strip_prefix('T').or_else(|| rest.strip_prefix(' '))?;
    let h: u32 = rest.get(0..2)?.parse().ok()?;
    if rest.get(2..3)? != ":" {
        return None;
    }
    let mi: u32 = rest.get(3..5)?.parse().ok()?;
    let mut tail = &rest[5..];
    let mut sec = 0;
    if let Some(t) = tail.strip_prefix(':') {
        sec = t.get(0..2)?.parse().ok()?;
        tail = &t[2..];
        if let Some(t) = tail.strip_prefix('.') {
            tail = t.trim_start_matches(|c: char| c.is_ascii_digit());
        }
    }
    if h > 23 || mi > 59 || sec > 60 {
        return None;
    }
    let offset_min = match tail {
        "" => None,
        "Z" | "z" => Some(0),
        t if (t.starts_with('+') || t.starts_with('-')) && t.len() >= 3 => {
            let sign = if t.starts_with('-') { -1 } else { 1 };
            let body = t[1..].replace(':', "");
            let oh: i64 = body.get(0..2)?.parse().ok()?;
            let om: i64 = body.get(2..4).unwrap_or("0").parse().ok()?;
            Some(sign * (oh * 60 + om))
        }
        _ => return None,
    };
    Some(DateTime { y, mo, d, h, mi, s: sec.min(59), time: true, offset_min })
}

/// A due date as the user's local wall-clock (zones dropped on purpose: the
/// assistant resolves "Friday at 3" in the user's own time).
struct Local {
    y: i64,
    mo: u32,
    d: u32,
    h: u32,
    mi: u32,
    time: bool,
}

fn parse_local(s: &str) -> Option<Local> {
    let p = parse_iso(s)?;
    Some(Local { y: p.y, mo: p.mo, d: p.d, h: p.h, mi: p.mi, time: p.time })
}

fn now_utc() -> DateTime {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let mut dt = DateTime::from_minutes(secs.div_euclid(60), (secs.rem_euclid(60)) as u32, true, Some(0));
    dt.s = secs.rem_euclid(60) as u32;
    dt
}

fn unique_suffix() -> String {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{:x}{:x}{:x}", std::process::id(), nanos, N.fetch_add(1, Ordering::Relaxed))
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(y) => 29,
        _ => 28,
    }
}

/// Days since 1970-01-01 (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = m as i64;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

// ---- iCalendar text ---------------------------------------------------------

fn ics_text(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace(';', "\\;")
        .replace(',', "\\,")
        .replace("\r\n", "\\n")
        .replace('\n', "\\n")
}

/// Fold at 75 octets without splitting a UTF-8 character; continuation lines
/// start with a single space (which counts toward their 75).
fn fold(line: &str) -> String {
    let mut out = String::new();
    let mut used = 0;
    for ch in line.chars() {
        let len = ch.len_utf8();
        if used + len > 75 {
            out.push_str("\r\n ");
            used = 1;
        }
        out.push(ch);
        used += len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mailto_encodes_everything_but_addresses() {
        let url = mailto_url(
            &["priya@acme.com".into(), " ".into()],
            &["sam@acme.com".into()],
            "Q3 plan & next steps",
            "Hi,\nLine two — ok?",
        );
        assert_eq!(
            url,
            "mailto:priya@acme.com?cc=sam@acme.com&subject=Q3%20plan%20%26%20next%20steps&body=Hi%2C%0D%0ALine%20two%20%E2%80%94%20ok%3F"
        );
        assert_eq!(mailto_url(&[], &[], "", ""), "mailto:");
    }

    #[test]
    fn iso_parsing_and_zone_conversion() {
        let d = parse_iso("2026-10-02").unwrap();
        assert!(!d.time);
        let t = parse_iso("2026-10-02T15:30").unwrap();
        assert_eq!((t.h, t.mi, t.offset_min), (15, 30, None));
        assert_eq!(t.ics_datetime(), "20261002T153000");
        let z = parse_iso("2026-12-31T23:30:00-05:00").unwrap();
        assert_eq!(z.ics_datetime(), "20270101T043000Z");
        let z2 = parse_iso("2026-03-01T00:15:00.123+01:00").unwrap();
        assert_eq!(z2.ics_datetime(), "20260228T231500Z");
        assert!(parse_iso("2026-02-30").is_none());
        assert!(parse_iso("next friday").is_none());
    }

    #[test]
    fn civil_round_trip() {
        for days in [-1, 0, 59, 10_000, 20_454, 25_000] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days);
        }
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(civil_from_days(0), (1970, 1, 1));
    }

    #[test]
    fn ics_is_valid_shape() {
        let ics = calendar_ics(
            "Follow-up; pricing, v2",
            "2026-10-02T15:00:00Z",
            "",
            "Discuss the numbers\nfrom Tuesday",
            "Room 4",
            &["Priya <priya@acme.com>".into(), "Sam".into()],
        );
        assert!(ics.starts_with("BEGIN:VCALENDAR\r\n"));
        assert!(ics.ends_with("END:VCALENDAR\r\n"));
        assert!(!ics.replace("\r\n", "").contains('\n'));
        assert!(ics.contains("DTSTART:20261002T150000Z\r\n"));
        assert!(ics.contains("DTEND:20261002T153000Z\r\n"));
        assert!(ics.contains("SUMMARY:Follow-up\\; pricing\\, v2\r\n"));
        assert!(ics.contains("ATTENDEE;CN=\"Priya\";ROLE=REQ-PARTICIPANT:mailto:priya@acme.com"));
        assert!(ics.contains("Attendees: Sam"));
        for line in ics.split("\r\n") {
            assert!(line.len() <= 75, "line too long: {line}");
        }
        let all_day = calendar_ics("Offsite", "2026-10-05", "", "", "", &[]);
        assert!(all_day.contains("DTSTART;VALUE=DATE:20261005\r\n"));
        assert!(all_day.contains("DTEND;VALUE=DATE:20261006\r\n"));
    }

    #[test]
    fn folding_respects_utf8() {
        let long = format!("DESCRIPTION:{}", "é".repeat(80));
        let folded = fold(&long);
        for line in folded.split("\r\n") {
            assert!(line.len() <= 75);
        }
        assert_eq!(folded.replace("\r\n ", ""), long);
    }

    /// The scripts must compile against the apps' dictionaries (osacompile
    /// doesn't run them, so nothing is created).
    #[test]
    fn applescripts_compile() {
        for (name, src) in [("mail", MAIL_DRAFT_SCRIPT), ("reminder", REMINDER_SCRIPT)] {
            let dir = std::env::temp_dir().join("unsound-osacompile-test");
            std::fs::create_dir_all(&dir).unwrap();
            let src_path = dir.join(format!("{name}.applescript"));
            let out_path = dir.join(format!("{name}.scpt"));
            std::fs::write(&src_path, src).unwrap();
            let out = Command::new("/usr/bin/osacompile")
                .arg("-o")
                .arg(&out_path)
                .arg(&src_path)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{name} failed to compile: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }

    /// argv passes arbitrary text through untouched — no escaping needed.
    #[test]
    fn osascript_argv_round_trips_hostile_text() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let hostile = "He said \"hi\" \\ then\n\"; do shell script \"echo pwned\" --";
        let out = rt
            .block_on(run_osascript("on run argv\nreturn item 1 of argv\nend run", &[hostile.into()]))
            .unwrap();
        assert_eq!(out, hostile.trim());
    }
}
