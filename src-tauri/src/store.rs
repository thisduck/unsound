//! Local SQLite store for dictation history and meetings.
//!
//! One database file lives beside the models in the app data directory
//! (`unsound.db`). It holds three things: `takes` (the dictation history that
//! used to live in the frontend's localStorage), `meetings`, and the per-
//! utterance `segments` that make up a meeting transcript. Nothing here ever
//! leaves the machine — same privacy contract as the rest of the app.

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tauri::{AppHandle, Manager};

/// Managed Tauri state wrapping a single serialized connection. The app's
/// concurrency is low (one recording at a time), so a mutex is plenty and
/// avoids a pool dependency.
pub struct Db(pub Mutex<Connection>);

/// A dictation take — one pass of record → transcribe → refine. Mirrors the
/// frontend `Take` interface exactly so history round-trips without churn.
/// `at` is an ISO-8601 UTC string (sorts lexicographically, no date crate).
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Take {
    pub id: String,
    pub at: String,
    #[serde(default)]
    pub raw: String,
    #[serde(default)]
    pub refined: String,
    #[serde(default)]
    pub stt_model: String,
    #[serde(default)]
    pub llm_model: String,
    #[serde(default)]
    pub app: Option<String>,
    #[serde(default)]
    pub lang: Option<String>,
}

/// One utterance in a meeting. `speaker` is a free-form label so today's
/// "me"/"them" split extends to named or numbered participants later without a
/// schema change. `source` records which channel it came from (mic vs. system
/// audio), which is what a future diarizer would re-cluster on.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    #[serde(default)]
    pub id: i64,
    #[serde(default = "default_speaker")]
    pub speaker: String,
    #[serde(default = "default_source")]
    pub source: String,
    #[serde(default)]
    pub start_ms: i64,
    #[serde(default)]
    pub end_ms: i64,
    #[serde(default)]
    pub text: String,
}

fn default_speaker() -> String {
    "them".into()
}
fn default_source() -> String {
    "system".into()
}

/// A meeting: its metadata plus, when fetched in full, its segments.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Meeting {
    pub id: String,
    #[serde(default)]
    pub title: String,
    pub started_at: String,
    #[serde(default)]
    pub ended_at: Option<String>,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub stt_model: String,
    #[serde(default)]
    pub llm_model: String,
    #[serde(default)]
    pub lang: Option<String>,
    /// Populated by `get_meeting`; empty in list views.
    #[serde(default)]
    pub segments: Vec<Segment>,
    /// Convenience count for list views (segments stays empty there).
    #[serde(default)]
    pub segment_count: i64,
}

/// Open (creating if needed) the database and run migrations.
pub fn open(app: &AppHandle) -> Result<Db, String> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let conn = Connection::open(dir.join("unsound.db")).map_err(|e| e.to_string())?;
    conn.pragma_update(None, "journal_mode", "WAL")
        .map_err(|e| e.to_string())?;
    conn.pragma_update(None, "foreign_keys", "ON")
        .map_err(|e| e.to_string())?;
    migrate(&conn).map_err(|e| e.to_string())?;
    Ok(Db(Mutex::new(conn)))
}

/// Schema versioning via `PRAGMA user_version`, so future changes are additive
/// and ordered. v0 → v1 creates the initial tables.
fn migrate(conn: &Connection) -> rusqlite::Result<()> {
    let version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if version < 1 {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS takes (
                id         TEXT PRIMARY KEY,
                at         TEXT NOT NULL,
                raw        TEXT NOT NULL DEFAULT '',
                refined    TEXT NOT NULL DEFAULT '',
                stt_model  TEXT NOT NULL DEFAULT '',
                llm_model  TEXT NOT NULL DEFAULT '',
                app        TEXT,
                lang       TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_takes_at ON takes(at DESC);

            CREATE TABLE IF NOT EXISTS meetings (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL DEFAULT '',
                started_at TEXT NOT NULL,
                ended_at   TEXT,
                summary    TEXT NOT NULL DEFAULT '',
                notes      TEXT NOT NULL DEFAULT '',
                stt_model  TEXT NOT NULL DEFAULT '',
                llm_model  TEXT NOT NULL DEFAULT '',
                lang       TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_meetings_started ON meetings(started_at DESC);

            CREATE TABLE IF NOT EXISTS segments (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                speaker    TEXT NOT NULL DEFAULT 'them',
                source     TEXT NOT NULL DEFAULT 'system',
                start_ms   INTEGER NOT NULL DEFAULT 0,
                end_ms     INTEGER NOT NULL DEFAULT 0,
                text       TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_segments_meeting ON segments(meeting_id, start_ms);
            "#,
        )?;
        conn.pragma_update(None, "user_version", 1)?;
    }
    Ok(())
}

// ---- takes (dictation history) ---------------------------------------------

pub fn list_takes(db: &Db, limit: i64) -> Result<Vec<Take>, String> {
    let conn = db.0.lock().unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT id, at, raw, refined, stt_model, llm_model, app, lang
             FROM takes ORDER BY at DESC LIMIT ?1",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([limit], row_to_take)
        .map_err(|e| e.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| e.to_string())
}

fn row_to_take(r: &rusqlite::Row) -> rusqlite::Result<Take> {
    Ok(Take {
        id: r.get(0)?,
        at: r.get(1)?,
        raw: r.get(2)?,
        refined: r.get(3)?,
        stt_model: r.get(4)?,
        llm_model: r.get(5)?,
        app: r.get(6)?,
        lang: r.get(7)?,
    })
}

/// Insert or update a take by id (the whole record is upserted at once).
pub fn save_take(db: &Db, t: &Take) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    conn.execute(
        "INSERT INTO takes (id, at, raw, refined, stt_model, llm_model, app, lang)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
         ON CONFLICT(id) DO UPDATE SET
            at=excluded.at, raw=excluded.raw, refined=excluded.refined,
            stt_model=excluded.stt_model, llm_model=excluded.llm_model,
            app=excluded.app, lang=excluded.lang",
        params![t.id, t.at, t.raw, t.refined, t.stt_model, t.llm_model, t.app, t.lang],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn delete_take(db: &Db, id: &str) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    conn.execute("DELETE FROM takes WHERE id = ?1", [id])
        .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn clear_takes(db: &Db) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    conn.execute("DELETE FROM takes", [])
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// One-time bulk import of the frontend's old localStorage history. Only
/// inserts rows whose id isn't already present, so it's safe to call twice.
pub fn import_takes(db: &Db, takes: &[Take]) -> Result<usize, String> {
    let mut conn = db.0.lock().unwrap();
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    let mut n = 0;
    for t in takes {
        let changed = tx
            .execute(
                "INSERT OR IGNORE INTO takes
                 (id, at, raw, refined, stt_model, llm_model, app, lang)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![t.id, t.at, t.raw, t.refined, t.stt_model, t.llm_model, t.app, t.lang],
            )
            .map_err(|e| e.to_string())?;
        n += changed;
    }
    tx.commit().map_err(|e| e.to_string())?;
    Ok(n)
}

// ---- meetings --------------------------------------------------------------

pub fn create_meeting(db: &Db, m: &Meeting) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    conn.execute(
        "INSERT INTO meetings (id, title, started_at, summary, notes, stt_model, llm_model, lang)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![m.id, m.title, m.started_at, m.summary, m.notes, m.stt_model, m.llm_model, m.lang],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Append transcribed utterances to an in-progress meeting.
pub fn add_segments(db: &Db, meeting_id: &str, segs: &[Segment]) -> Result<(), String> {
    let mut conn = db.0.lock().unwrap();
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    for s in segs {
        tx.execute(
            "INSERT INTO segments (meeting_id, speaker, source, start_ms, end_ms, text)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![meeting_id, s.speaker, s.source, s.start_ms, s.end_ms, s.text],
        )
        .map_err(|e| e.to_string())?;
    }
    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
}

pub fn end_meeting(
    db: &Db,
    id: &str,
    ended_at: &str,
    summary: &str,
    title: Option<&str>,
) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    // Only overwrite the title when one is supplied (auto-title on end).
    conn.execute(
        "UPDATE meetings
         SET ended_at = ?2, summary = ?3, title = COALESCE(?4, title)
         WHERE id = ?1",
        params![id, ended_at, summary, title],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn update_meeting_notes(db: &Db, id: &str, notes: &str) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    conn.execute(
        "UPDATE meetings SET notes = ?2 WHERE id = ?1",
        params![id, notes],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn set_meeting_summary(db: &Db, id: &str, summary: &str) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    conn.execute(
        "UPDATE meetings SET summary = ?2 WHERE id = ?1",
        params![id, summary],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn rename_meeting(db: &Db, id: &str, title: &str) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    conn.execute(
        "UPDATE meetings SET title = ?2 WHERE id = ?1",
        params![id, title],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn delete_meeting(db: &Db, id: &str) -> Result<(), String> {
    // segments cascade via the foreign key.
    let conn = db.0.lock().unwrap();
    conn.execute("DELETE FROM meetings WHERE id = ?1", [id])
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Meetings for the list view: metadata + a segment count, newest first.
pub fn list_meetings(db: &Db) -> Result<Vec<Meeting>, String> {
    let conn = db.0.lock().unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT m.id, m.title, m.started_at, m.ended_at, m.summary, m.notes,
                    m.stt_model, m.llm_model, m.lang,
                    (SELECT COUNT(*) FROM segments s WHERE s.meeting_id = m.id)
             FROM meetings m ORDER BY m.started_at DESC",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| {
            Ok(Meeting {
                id: r.get(0)?,
                title: r.get(1)?,
                started_at: r.get(2)?,
                ended_at: r.get(3)?,
                summary: r.get(4)?,
                notes: r.get(5)?,
                stt_model: r.get(6)?,
                llm_model: r.get(7)?,
                lang: r.get(8)?,
                segments: Vec::new(),
                segment_count: r.get(9)?,
            })
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| e.to_string())
}

/// A single meeting with all of its segments in timeline order.
pub fn get_meeting(db: &Db, id: &str) -> Result<Option<Meeting>, String> {
    let conn = db.0.lock().unwrap();
    let mut meeting = conn
        .query_row(
            "SELECT id, title, started_at, ended_at, summary, notes,
                    stt_model, llm_model, lang
             FROM meetings WHERE id = ?1",
            [id],
            |r| {
                Ok(Meeting {
                    id: r.get(0)?,
                    title: r.get(1)?,
                    started_at: r.get(2)?,
                    ended_at: r.get(3)?,
                    summary: r.get(4)?,
                    notes: r.get(5)?,
                    stt_model: r.get(6)?,
                    llm_model: r.get(7)?,
                    lang: r.get(8)?,
                    segments: Vec::new(),
                    segment_count: 0,
                })
            },
        )
        .optional()
        .map_err(|e| e.to_string())?;

    if let Some(m) = meeting.as_mut() {
        let mut stmt = conn
            .prepare(
                "SELECT id, speaker, source, start_ms, end_ms, text
                 FROM segments WHERE meeting_id = ?1 ORDER BY start_ms, id",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([id], |r| {
                Ok(Segment {
                    id: r.get(0)?,
                    speaker: r.get(1)?,
                    source: r.get(2)?,
                    start_ms: r.get(3)?,
                    end_ms: r.get(4)?,
                    text: r.get(5)?,
                })
            })
            .map_err(|e| e.to_string())?;
        m.segments = rows
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| e.to_string())?;
        m.segment_count = m.segments.len() as i64;
    }
    Ok(meeting)
}
