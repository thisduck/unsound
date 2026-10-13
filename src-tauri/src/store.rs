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
    /// Per-meeting speaker label overrides (speaker key → display name), e.g.
    /// "them:0" → "Priya". Populated by `get_meeting`.
    #[serde(default)]
    pub speaker_names: std::collections::HashMap<String, String>,
    /// Calendar attendees (display names) captured at start, when available.
    #[serde(default)]
    pub attendees: Vec<String>,
    /// Moments the user flagged while recording. Populated by `get_meeting`.
    #[serde(default)]
    pub marks: Vec<Mark>,
    /// Structured follow-ups extracted by the summary (plus any added by hand).
    #[serde(default)]
    pub action_items: Vec<ActionItem>,
    /// Whether any recorded audio is on disk for playback. Filled in by the
    /// command layer (it knows where audio lives), not the store.
    #[serde(default)]
    pub has_audio: bool,
}

/// A moment flagged during a meeting ("that was important").
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Mark {
    #[serde(default)]
    pub id: i64,
    pub at_ms: i64,
    #[serde(default)]
    pub label: String,
}

/// One follow-up: who does what by when, and where in the meeting it came up.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActionItem {
    #[serde(default)]
    pub id: i64,
    #[serde(default)]
    pub meeting_id: Option<String>,
    pub text: String,
    #[serde(default)]
    pub owner: String,
    #[serde(default)]
    pub due: String,
    #[serde(default)]
    pub at_ms: Option<i64>,
    #[serde(default)]
    pub done: bool,
    /// Added or edited by the user — survives summary regeneration.
    #[serde(default)]
    pub manual: bool,
    #[serde(default)]
    pub created_at: String,
    /// Meeting title, filled for the cross-meeting "open items" list.
    #[serde(default)]
    pub meeting_title: String,
}

/// A cached extraction over one stretch of a meeting's transcript. `signature`
/// is a hash of the exact text it was extracted from, so edits, renames and
/// re-detected speakers invalidate just the chunks they touch.
#[derive(Clone)]
pub struct Chunk {
    pub idx: i64,
    pub start_ms: i64,
    pub end_ms: i64,
    pub signature: String,
    pub facts: String,
}

/// Something the user asked unsound to do ("unsound, draft an email…").
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Task {
    pub id: String,
    #[serde(default)]
    pub meeting_id: Option<String>,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    /// Position in the meeting timeline where it was asked.
    #[serde(default)]
    pub at_ms: Option<i64>,
    pub utterance: String,
    /// How it was given: "wake" (spoken name), "hotkey", or "typed".
    #[serde(default)]
    pub trigger: String,
    /// email | jira_issue | reminder | calendar_event | slack_message | note | delegate | other
    #[serde(default)]
    pub action: String,
    /// "now" or "after_meeting".
    #[serde(default)]
    pub run_when: String,
    /// heard → planning → drafting → ready → running → done; or waiting (for
    /// the meeting to end), failed, dismissed.
    #[serde(default)]
    pub status: String,
    /// The request restated as a clear instruction.
    #[serde(default)]
    pub instruction: String,
    #[serde(default)]
    pub span_start_ms: Option<i64>,
    #[serde(default)]
    pub span_end_ms: Option<i64>,
    /// The drafted output (JSON, shape depends on `action`).
    #[serde(default)]
    pub draft: serde_json::Value,
    /// What running it produced: {message, url?}.
    #[serde(default)]
    pub result: serde_json::Value,
    #[serde(default)]
    pub error: String,
    /// Agent steps for delegated work (JSON array), for transparency.
    #[serde(default)]
    pub log: serde_json::Value,
}

/// A remembered voice: one person's speaker-embedding centroid per model.
#[derive(Clone)]
pub struct VoiceprintRow {
    pub name: String,
    pub model: String,
    pub embedding: Vec<u8>,
    pub samples: i64,
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
    if version < 2 {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS speaker_names (
                meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                speaker    TEXT NOT NULL,
                name       TEXT NOT NULL,
                PRIMARY KEY (meeting_id, speaker)
            );
            "#,
        )?;
        conn.pragma_update(None, "user_version", 2)?;
    }
    if version < 3 {
        conn.execute_batch(
            r#"
            ALTER TABLE meetings ADD COLUMN attendees TEXT NOT NULL DEFAULT '[]';

            CREATE TABLE IF NOT EXISTS meeting_chunks (
                meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                idx        INTEGER NOT NULL,
                start_ms   INTEGER NOT NULL,
                end_ms     INTEGER NOT NULL,
                signature  TEXT NOT NULL,
                facts      TEXT NOT NULL,
                PRIMARY KEY (meeting_id, idx)
            );

            CREATE TABLE IF NOT EXISTS action_items (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id TEXT REFERENCES meetings(id) ON DELETE CASCADE,
                text       TEXT NOT NULL,
                owner      TEXT NOT NULL DEFAULT '',
                due        TEXT NOT NULL DEFAULT '',
                at_ms      INTEGER,
                done       INTEGER NOT NULL DEFAULT 0,
                manual     INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_action_items_meeting ON action_items(meeting_id);
            CREATE INDEX IF NOT EXISTS idx_action_items_open ON action_items(done, created_at DESC);

            CREATE TABLE IF NOT EXISTS marks (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                at_ms      INTEGER NOT NULL,
                label      TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_marks_meeting ON marks(meeting_id, at_ms);

            CREATE TABLE IF NOT EXISTS tasks (
                id            TEXT PRIMARY KEY,
                meeting_id    TEXT REFERENCES meetings(id) ON DELETE CASCADE,
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL,
                at_ms         INTEGER,
                utterance     TEXT NOT NULL,
                trigger       TEXT NOT NULL DEFAULT 'wake',
                action        TEXT NOT NULL DEFAULT '',
                run_when      TEXT NOT NULL DEFAULT 'now',
                status        TEXT NOT NULL DEFAULT 'heard',
                instruction   TEXT NOT NULL DEFAULT '',
                span_start_ms INTEGER,
                span_end_ms   INTEGER,
                draft         TEXT NOT NULL DEFAULT 'null',
                result        TEXT NOT NULL DEFAULT 'null',
                error         TEXT NOT NULL DEFAULT '',
                log           TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_meeting ON tasks(meeting_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC);

            CREATE TABLE IF NOT EXISTS voiceprints (
                name       TEXT NOT NULL COLLATE NOCASE,
                model      TEXT NOT NULL,
                embedding  BLOB NOT NULL,
                samples    INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (name, model)
            );

            CREATE TABLE IF NOT EXISTS meeting_speakers (
                meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                speaker    TEXT NOT NULL,
                model      TEXT NOT NULL,
                embedding  BLOB NOT NULL,
                segments   INTEGER NOT NULL DEFAULT 0,
                speech_ms  INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (meeting_id, speaker)
            );

            -- Full-text search over transcripts and meeting text. External
            -- content tables kept in sync by triggers; porter stemming so
            -- "deciding" finds "decided".
            CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
                text, content='segments', content_rowid='id', tokenize='porter unicode61'
            );
            CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
                INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
                INSERT INTO segments_fts(segments_fts, rowid, text) VALUES ('delete', old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE OF text ON segments BEGIN
                INSERT INTO segments_fts(segments_fts, rowid, text) VALUES ('delete', old.id, old.text);
                INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
            END;
            INSERT INTO segments_fts(segments_fts) VALUES ('rebuild');

            CREATE VIRTUAL TABLE IF NOT EXISTS meetings_fts USING fts5(
                title, summary, notes, content='meetings', content_rowid='rowid',
                tokenize='porter unicode61'
            );
            CREATE TRIGGER IF NOT EXISTS meetings_ai AFTER INSERT ON meetings BEGIN
                INSERT INTO meetings_fts(rowid, title, summary, notes)
                VALUES (new.rowid, new.title, new.summary, new.notes);
            END;
            CREATE TRIGGER IF NOT EXISTS meetings_ad AFTER DELETE ON meetings BEGIN
                INSERT INTO meetings_fts(meetings_fts, rowid, title, summary, notes)
                VALUES ('delete', old.rowid, old.title, old.summary, old.notes);
            END;
            CREATE TRIGGER IF NOT EXISTS meetings_au AFTER UPDATE OF title, summary, notes ON meetings BEGIN
                INSERT INTO meetings_fts(meetings_fts, rowid, title, summary, notes)
                VALUES ('delete', old.rowid, old.title, old.summary, old.notes);
                INSERT INTO meetings_fts(rowid, title, summary, notes)
                VALUES (new.rowid, new.title, new.summary, new.notes);
            END;
            INSERT INTO meetings_fts(meetings_fts) VALUES ('rebuild');
            "#,
        )?;
        conn.pragma_update(None, "user_version", 3)?;
    }
    Ok(())
}

/// Give every meeting without an `ended_at` one: its start plus the end of its
/// last transcribed line (or just its start when it has none). Returns ids.
fn finish_open_meetings(conn: &Connection) -> rusqlite::Result<Vec<String>> {
    let ids: Vec<String> = {
        let mut stmt = conn.prepare("SELECT id FROM meetings WHERE ended_at IS NULL")?;
        let rows = stmt.query_map([], |r| r.get(0))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    conn.execute(
        "UPDATE meetings SET ended_at = strftime('%Y-%m-%dT%H:%M:%fZ', started_at,
            '+' || (COALESCE((SELECT MAX(end_ms) FROM segments s WHERE s.meeting_id = meetings.id), 0) / 1000.0)
            || ' seconds')
         WHERE ended_at IS NULL",
        [],
    )?;
    Ok(ids)
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
        "INSERT INTO meetings (id, title, started_at, summary, notes, stt_model, llm_model, lang, attendees)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            m.id,
            m.title,
            m.started_at,
            m.summary,
            m.notes,
            m.stt_model,
            m.llm_model,
            m.lang,
            serde_json::to_string(&m.attendees).unwrap_or_else(|_| "[]".into())
        ],
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

/// Record when a meeting ended (set by the backend when capture stops).
pub fn set_meeting_ended(db: &Db, id: &str, ended_at: &str) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    conn.execute(
        "UPDATE meetings SET ended_at = ?2 WHERE id = ?1",
        params![id, ended_at],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Close out meetings left open — every meeting before v3 (the frontend never
/// called end_meeting), and any cut off by a crash or quit mid-recording. Call
/// at launch, when nothing can be recording. Returns the ids it finished.
pub fn recover_open_meetings(db: &Db) -> Result<Vec<String>, String> {
    let conn = db.0.lock().unwrap();
    finish_open_meetings(&conn).map_err(|e| e.to_string())
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
                    (SELECT COUNT(*) FROM segments s WHERE s.meeting_id = m.id),
                    m.attendees
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
                speaker_names: std::collections::HashMap::new(),
                attendees: parse_names(&r.get::<_, String>(10)?),
                marks: Vec::new(),
                action_items: Vec::new(),
                has_audio: false,
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
                    stt_model, llm_model, lang, attendees
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
                    speaker_names: std::collections::HashMap::new(),
                    attendees: parse_names(&r.get::<_, String>(9)?),
                    marks: Vec::new(),
                    action_items: Vec::new(),
                    has_audio: false,
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

        let mut stmt = conn
            .prepare("SELECT speaker, name FROM speaker_names WHERE meeting_id = ?1")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(|e| e.to_string())?;
        for row in rows {
            let (speaker, name) = row.map_err(|e| e.to_string())?;
            m.speaker_names.insert(speaker, name);
        }
        m.marks = marks_on(&conn, id)?;
        m.action_items = action_items_on(&conn, Some(id), false)?;
    }
    Ok(meeting)
}

fn parse_names(json: &str) -> Vec<String> {
    serde_json::from_str(json).unwrap_or_default()
}

/// Set (or clear, when `name` is blank) a per-meeting speaker display name.
pub fn set_speaker_name(db: &Db, meeting_id: &str, speaker: &str, name: &str) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    let name = name.trim();
    if name.is_empty() {
        conn.execute(
            "DELETE FROM speaker_names WHERE meeting_id = ?1 AND speaker = ?2",
            params![meeting_id, speaker],
        )
        .map_err(|e| e.to_string())?;
    } else {
        conn.execute(
            "INSERT INTO speaker_names (meeting_id, speaker, name) VALUES (?1, ?2, ?3)
             ON CONFLICT(meeting_id, speaker) DO UPDATE SET name = excluded.name",
            params![meeting_id, speaker, name],
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Edit one segment's transcript text.
pub fn update_segment_text(db: &Db, segment_id: i64, text: &str) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    conn.execute(
        "UPDATE segments SET text = ?2 WHERE id = ?1",
        params![segment_id, text],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Relabel segments' speakers by id (used after diarization assigns Speaker N).
pub fn update_segment_speakers(db: &Db, updates: &[(i64, String)]) -> Result<(), String> {
    let mut conn = db.0.lock().unwrap();
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    for (id, speaker) in updates {
        tx.execute(
            "UPDATE segments SET speaker = ?2 WHERE id = ?1",
            params![id, speaker],
        )
        .map_err(|e| e.to_string())?;
    }
    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
}

/// A search result: one matching transcript line (with its position), or a
/// match in a meeting's title/summary/notes. `snippet` wraps matched terms in
/// « » for highlighting.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchHit {
    pub meeting_id: String,
    pub title: String,
    pub started_at: String,
    pub snippet: String,
    /// "transcript" | "meeting"
    pub kind: String,
    pub segment_id: Option<i64>,
    pub start_ms: Option<i64>,
    pub speaker: Option<String>,
    /// The name given to that speaker in its meeting, if any.
    pub speaker_name: Option<String>,
}

/// Turn free text into an FTS5 query: every word must appear (prefix match on
/// each, so "deci" finds "decision"), or — `any` — at least one does.
/// Quoting each term keeps punctuation and FTS operators inert.
pub fn fts_query(text: &str, any: bool) -> Option<String> {
    let terms: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .map(|t| t.trim_matches('\'').to_lowercase())
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{}\"*", t.replace('"', "")))
        .collect();
    if terms.is_empty() {
        return None;
    }
    Some(terms.join(if any { " OR " } else { " " }))
}

/// Full-text search across every meeting: transcript lines first (best BM25
/// match first, a few per meeting), then meetings whose title, summary or
/// notes match. Falls back to matching any word when all-words finds nothing.
pub fn search_meetings(db: &Db, query: &str) -> Result<Vec<SearchHit>, String> {
    let Some(all) = fts_query(query, false) else {
        return Ok(Vec::new());
    };
    let mut hits = run_search(db, &all)?;
    if hits.is_empty() {
        if let Some(any) = fts_query(query, true).filter(|q| *q != all) {
            hits = run_search(db, &any)?;
        }
    }
    Ok(hits)
}

fn run_search(db: &Db, q: &str) -> Result<Vec<SearchHit>, String> {
    let conn = db.0.lock().unwrap();
    let mut out: Vec<SearchHit> = Vec::new();
    let mut per_meeting: std::collections::HashMap<String, usize> = Default::default();
    {
        let mut stmt = conn
            .prepare(
                "SELECT s.meeting_id, m.title, m.started_at,
                        snippet(segments_fts, 0, '«', '»', '…', 24),
                        s.id, s.start_ms, s.speaker,
                        (SELECT name FROM speaker_names sn
                         WHERE sn.meeting_id = s.meeting_id AND sn.speaker = s.speaker)
                 FROM segments_fts
                 JOIN segments s ON s.id = segments_fts.rowid
                 JOIN meetings m ON m.id = s.meeting_id
                 WHERE segments_fts MATCH ?1
                 ORDER BY bm25(segments_fts)
                 LIMIT 300",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([q], |r| {
                Ok(SearchHit {
                    meeting_id: r.get(0)?,
                    title: r.get(1)?,
                    started_at: r.get(2)?,
                    snippet: r.get(3)?,
                    kind: "transcript".into(),
                    segment_id: Some(r.get(4)?),
                    start_ms: Some(r.get(5)?),
                    speaker: Some(r.get(6)?),
                    speaker_name: r.get(7)?,
                })
            })
            .map_err(|e| e.to_string())?;
        for hit in rows {
            let hit = hit.map_err(|e| e.to_string())?;
            let n = per_meeting.entry(hit.meeting_id.clone()).or_default();
            if *n < 5 {
                *n += 1;
                out.push(hit);
            }
        }
    }
    let mut stmt = conn
        .prepare(
            "SELECT m.id, m.title, m.started_at,
                    snippet(meetings_fts, -1, '«', '»', '…', 24)
             FROM meetings_fts
             JOIN meetings m ON m.rowid = meetings_fts.rowid
             WHERE meetings_fts MATCH ?1
             ORDER BY bm25(meetings_fts)
             LIMIT 50",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([q], |r| {
            Ok(SearchHit {
                meeting_id: r.get(0)?,
                title: r.get(1)?,
                started_at: r.get(2)?,
                snippet: r.get(3)?,
                kind: "meeting".into(),
                segment_id: None,
                start_ms: None,
                speaker: None,
                speaker_name: None,
            })
        })
        .map_err(|e| e.to_string())?;
    for hit in rows {
        out.push(hit.map_err(|e| e.to_string())?);
    }
    Ok(out)
}

/// Segments matching an FTS query, for grounding answers: (meeting id,
/// segment id, start_ms). Ranked by BM25.
pub fn search_segment_ids(
    db: &Db,
    q: &str,
    meeting_id: Option<&str>,
    limit: i64,
) -> Result<Vec<(String, i64, i64)>, String> {
    let conn = db.0.lock().unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT s.meeting_id, s.id, s.start_ms
             FROM segments_fts JOIN segments s ON s.id = segments_fts.rowid
             WHERE segments_fts MATCH ?1 AND (?2 IS NULL OR s.meeting_id = ?2)
             ORDER BY bm25(segments_fts) LIMIT ?3",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(params![q, meeting_id, limit], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| e.to_string())
}

// ---- marks ------------------------------------------------------------------

fn marks_on(conn: &Connection, meeting_id: &str) -> Result<Vec<Mark>, String> {
    let mut stmt = conn
        .prepare("SELECT id, at_ms, label FROM marks WHERE meeting_id = ?1 ORDER BY at_ms")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([meeting_id], |r| {
            Ok(Mark {
                id: r.get(0)?,
                at_ms: r.get(1)?,
                label: r.get(2)?,
            })
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| e.to_string())
}

pub fn add_mark(db: &Db, meeting_id: &str, at_ms: i64, label: &str) -> Result<Mark, String> {
    let conn = db.0.lock().unwrap();
    conn.execute(
        "INSERT INTO marks (meeting_id, at_ms, label) VALUES (?1, ?2, ?3)",
        params![meeting_id, at_ms, label],
    )
    .map_err(|e| e.to_string())?;
    Ok(Mark {
        id: conn.last_insert_rowid(),
        at_ms,
        label: label.to_string(),
    })
}

pub fn delete_mark(db: &Db, id: i64) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    conn.execute("DELETE FROM marks WHERE id = ?1", [id])
        .map_err(|e| e.to_string())?;
    Ok(())
}

// ---- action items -----------------------------------------------------------

fn row_to_item(r: &rusqlite::Row) -> rusqlite::Result<ActionItem> {
    Ok(ActionItem {
        id: r.get(0)?,
        meeting_id: r.get(1)?,
        text: r.get(2)?,
        owner: r.get(3)?,
        due: r.get(4)?,
        at_ms: r.get(5)?,
        done: r.get::<_, i64>(6)? != 0,
        manual: r.get::<_, i64>(7)? != 0,
        created_at: r.get(8)?,
        meeting_title: r.get::<_, Option<String>>(9)?.unwrap_or_default(),
    })
}

const ITEM_COLUMNS: &str = "a.id, a.meeting_id, a.text, a.owner, a.due, a.at_ms, a.done, a.manual,
     a.created_at, m.title";

fn action_items_on(
    conn: &Connection,
    meeting_id: Option<&str>,
    open_only: bool,
) -> Result<Vec<ActionItem>, String> {
    let sql = format!(
        "SELECT {ITEM_COLUMNS} FROM action_items a LEFT JOIN meetings m ON m.id = a.meeting_id
         WHERE (?1 IS NULL OR a.meeting_id = ?1) AND (?2 = 0 OR a.done = 0)
         ORDER BY {}",
        if meeting_id.is_some() { "COALESCE(a.at_ms, 0), a.id" } else { "a.created_at DESC, a.id" }
    );
    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(params![meeting_id, open_only as i64], row_to_item)
        .map_err(|e| e.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| e.to_string())
}

/// Every open follow-up across all meetings, newest first.
pub fn open_action_items(db: &Db) -> Result<Vec<ActionItem>, String> {
    let conn = db.0.lock().unwrap();
    action_items_on(&conn, None, true)
}

/// Replace a meeting's summary-extracted items with a fresh set, keeping any
/// the user touched (checked off or edited) and skipping new ones that repeat
/// a kept item.
pub fn replace_auto_action_items(
    db: &Db,
    meeting_id: &str,
    items: &[ActionItem],
) -> Result<(), String> {
    let mut conn = db.0.lock().unwrap();
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    tx.execute(
        "DELETE FROM action_items WHERE meeting_id = ?1 AND manual = 0 AND done = 0",
        [meeting_id],
    )
    .map_err(|e| e.to_string())?;
    let mut kept: Vec<String> = {
        let mut stmt = tx
            .prepare("SELECT text FROM action_items WHERE meeting_id = ?1")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([meeting_id], |r| r.get::<_, String>(0))
            .map_err(|e| e.to_string())?;
        rows.filter_map(Result::ok).map(|t| normalize(&t)).collect()
    };
    let now = crate::util::now_iso();
    for it in items {
        // Skip blanks, items the user kept, and repeats within this batch.
        let key = normalize(&it.text);
        if key.is_empty() || kept.contains(&key) {
            continue;
        }
        kept.push(key);
        tx.execute(
            "INSERT INTO action_items (meeting_id, text, owner, due, at_ms, done, manual, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, 0, 0, ?6)",
            params![meeting_id, it.text.trim(), it.owner.trim(), it.due.trim(), it.at_ms, now],
        )
        .map_err(|e| e.to_string())?;
    }
    tx.commit().map_err(|e| e.to_string())
}

fn normalize(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Insert (id 0) or update an item; any user edit marks it manual.
pub fn save_action_item(db: &Db, item: &ActionItem) -> Result<ActionItem, String> {
    let conn = db.0.lock().unwrap();
    if item.id == 0 {
        conn.execute(
            "INSERT INTO action_items (meeting_id, text, owner, due, at_ms, done, manual, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1, ?7)",
            params![
                item.meeting_id,
                item.text,
                item.owner,
                item.due,
                item.at_ms,
                item.done as i64,
                crate::util::now_iso()
            ],
        )
        .map_err(|e| e.to_string())?;
    } else {
        conn.execute(
            "UPDATE action_items SET text = ?2, owner = ?3, due = ?4, done = ?5, manual = 1 WHERE id = ?1",
            params![item.id, item.text, item.owner, item.due, item.done as i64],
        )
        .map_err(|e| e.to_string())?;
    }
    let id = if item.id == 0 { conn.last_insert_rowid() } else { item.id };
    conn.query_row(
        &format!(
            "SELECT {ITEM_COLUMNS} FROM action_items a LEFT JOIN meetings m ON m.id = a.meeting_id
             WHERE a.id = ?1"
        ),
        [id],
        row_to_item,
    )
    .map_err(|e| e.to_string())
}

pub fn delete_action_item(db: &Db, id: i64) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    conn.execute("DELETE FROM action_items WHERE id = ?1", [id])
        .map_err(|e| e.to_string())?;
    Ok(())
}

// ---- summary chunks ---------------------------------------------------------

pub fn chunks(db: &Db, meeting_id: &str) -> Result<Vec<Chunk>, String> {
    let conn = db.0.lock().unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT idx, start_ms, end_ms, signature, facts FROM meeting_chunks
             WHERE meeting_id = ?1 ORDER BY idx",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([meeting_id], |r| {
            Ok(Chunk {
                idx: r.get(0)?,
                start_ms: r.get(1)?,
                end_ms: r.get(2)?,
                signature: r.get(3)?,
                facts: r.get(4)?,
            })
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| e.to_string())
}

pub fn save_chunk(db: &Db, meeting_id: &str, c: &Chunk) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    conn.execute(
        "INSERT INTO meeting_chunks (meeting_id, idx, start_ms, end_ms, signature, facts)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(meeting_id, idx) DO UPDATE SET start_ms = excluded.start_ms,
            end_ms = excluded.end_ms, signature = excluded.signature, facts = excluded.facts",
        params![meeting_id, c.idx, c.start_ms, c.end_ms, c.signature, c.facts],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Drop cached chunks at or past `from_idx` (the chunk layout changed).
pub fn delete_chunks_from(db: &Db, meeting_id: &str, from_idx: i64) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    conn.execute(
        "DELETE FROM meeting_chunks WHERE meeting_id = ?1 AND idx >= ?2",
        params![meeting_id, from_idx],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

// ---- tasks ------------------------------------------------------------------

const TASK_COLUMNS: &str = "id, meeting_id, created_at, updated_at, at_ms, utterance, trigger, action,
     run_when, status, instruction, span_start_ms, span_end_ms, draft, result, error, log";

fn row_to_task(r: &rusqlite::Row) -> rusqlite::Result<Task> {
    let json = |i: usize| -> rusqlite::Result<serde_json::Value> {
        Ok(serde_json::from_str(&r.get::<_, String>(i)?).unwrap_or(serde_json::Value::Null))
    };
    Ok(Task {
        id: r.get(0)?,
        meeting_id: r.get(1)?,
        created_at: r.get(2)?,
        updated_at: r.get(3)?,
        at_ms: r.get(4)?,
        utterance: r.get(5)?,
        trigger: r.get(6)?,
        action: r.get(7)?,
        run_when: r.get(8)?,
        status: r.get(9)?,
        instruction: r.get(10)?,
        span_start_ms: r.get(11)?,
        span_end_ms: r.get(12)?,
        draft: json(13)?,
        result: json(14)?,
        error: r.get(15)?,
        log: json(16)?,
    })
}

/// Insert or fully replace a task (stamps `updated_at`).
pub fn save_task(db: &Db, t: &Task) -> Result<Task, String> {
    let conn = db.0.lock().unwrap();
    let now = crate::util::now_iso();
    let created = if t.created_at.is_empty() { now.clone() } else { t.created_at.clone() };
    conn.execute(
        "INSERT INTO tasks (id, meeting_id, created_at, updated_at, at_ms, utterance, trigger, action,
            run_when, status, instruction, span_start_ms, span_end_ms, draft, result, error, log)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)
         ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at, at_ms = excluded.at_ms,
            utterance = excluded.utterance, trigger = excluded.trigger, action = excluded.action,
            run_when = excluded.run_when, status = excluded.status, instruction = excluded.instruction,
            span_start_ms = excluded.span_start_ms, span_end_ms = excluded.span_end_ms,
            draft = excluded.draft, result = excluded.result, error = excluded.error, log = excluded.log",
        params![
            t.id,
            t.meeting_id,
            created,
            now,
            t.at_ms,
            t.utterance,
            t.trigger,
            t.action,
            t.run_when,
            t.status,
            t.instruction,
            t.span_start_ms,
            t.span_end_ms,
            t.draft.to_string(),
            t.result.to_string(),
            t.error,
            t.log.to_string()
        ],
    )
    .map_err(|e| e.to_string())?;
    let mut saved = t.clone();
    saved.created_at = created;
    saved.updated_at = now;
    Ok(saved)
}

pub fn get_task(db: &Db, id: &str) -> Result<Option<Task>, String> {
    let conn = db.0.lock().unwrap();
    conn.query_row(
        &format!("SELECT {TASK_COLUMNS} FROM tasks WHERE id = ?1"),
        [id],
        row_to_task,
    )
    .optional()
    .map_err(|e| e.to_string())
}

/// Tasks for one meeting (oldest first), or — `None` — every task, newest first.
pub fn list_tasks(db: &Db, meeting_id: Option<&str>) -> Result<Vec<Task>, String> {
    let conn = db.0.lock().unwrap();
    let sql = match meeting_id {
        Some(_) => format!("SELECT {TASK_COLUMNS} FROM tasks WHERE meeting_id = ?1 ORDER BY created_at"),
        None => format!("SELECT {TASK_COLUMNS} FROM tasks WHERE ?1 IS NULL ORDER BY created_at DESC LIMIT 500"),
    };
    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(params![meeting_id], row_to_task)
        .map_err(|e| e.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| e.to_string())
}

/// Tasks whose status is one of `statuses` (e.g. unfinished work at launch).
pub fn tasks_with_status(db: &Db, statuses: &[&str]) -> Result<Vec<Task>, String> {
    let conn = db.0.lock().unwrap();
    let placeholders = vec!["?"; statuses.len()].join(",");
    let sql = format!("SELECT {TASK_COLUMNS} FROM tasks WHERE status IN ({placeholders}) ORDER BY created_at");
    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(rusqlite::params_from_iter(statuses.iter()), row_to_task)
        .map_err(|e| e.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| e.to_string())
}

pub fn delete_task(db: &Db, id: &str) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    conn.execute("DELETE FROM tasks WHERE id = ?1", [id])
        .map_err(|e| e.to_string())?;
    Ok(())
}

// ---- voices -----------------------------------------------------------------

pub fn voiceprints(db: &Db, model: &str) -> Result<Vec<VoiceprintRow>, String> {
    let conn = db.0.lock().unwrap();
    let mut stmt = conn
        .prepare("SELECT name, model, embedding, samples FROM voiceprints WHERE model = ?1 ORDER BY name")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([model], |r| {
            Ok(VoiceprintRow {
                name: r.get(0)?,
                model: r.get(1)?,
                embedding: r.get(2)?,
                samples: r.get(3)?,
            })
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| e.to_string())
}

/// Names with a remembered voice (any model), for the settings list.
pub fn voiceprint_names(db: &Db) -> Result<Vec<(String, i64)>, String> {
    let conn = db.0.lock().unwrap();
    let mut stmt = conn
        .prepare("SELECT name, MAX(samples) FROM voiceprints GROUP BY name COLLATE NOCASE ORDER BY name")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .map_err(|e| e.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| e.to_string())
}

pub fn save_voiceprint(db: &Db, v: &VoiceprintRow) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    conn.execute(
        "INSERT INTO voiceprints (name, model, embedding, samples, updated_at) VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(name, model) DO UPDATE SET embedding = excluded.embedding,
            samples = excluded.samples, updated_at = excluded.updated_at",
        params![v.name, v.model, v.embedding, v.samples, crate::util::now_iso()],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn delete_voiceprint(db: &Db, name: &str) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    conn.execute("DELETE FROM voiceprints WHERE name = ?1", [name])
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// A meeting speaker's voice centroid: (speaker, model, embedding, segments, speech_ms).
pub type SpeakerCentroid = (String, String, Vec<u8>, i64, i64);

pub fn set_meeting_speakers(db: &Db, meeting_id: &str, rows: &[SpeakerCentroid]) -> Result<(), String> {
    let mut conn = db.0.lock().unwrap();
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    tx.execute("DELETE FROM meeting_speakers WHERE meeting_id = ?1", [meeting_id])
        .map_err(|e| e.to_string())?;
    for (speaker, model, emb, segs, ms) in rows {
        tx.execute(
            "INSERT INTO meeting_speakers (meeting_id, speaker, model, embedding, segments, speech_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![meeting_id, speaker, model, emb, segs, ms],
        )
        .map_err(|e| e.to_string())?;
    }
    tx.commit().map_err(|e| e.to_string())
}

pub fn meeting_speaker(db: &Db, meeting_id: &str, speaker: &str) -> Result<Option<SpeakerCentroid>, String> {
    let conn = db.0.lock().unwrap();
    conn.query_row(
        "SELECT speaker, model, embedding, segments, speech_ms FROM meeting_speakers
         WHERE meeting_id = ?1 AND speaker = ?2",
        params![meeting_id, speaker],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
    )
    .optional()
    .map_err(|e| e.to_string())
}

/// Relabel speakers across a meeting (e.g. `them:3` → `them:1` after
/// clusters merge), carrying display names along.
pub fn remap_speakers(db: &Db, meeting_id: &str, remap: &[(String, String)]) -> Result<(), String> {
    if remap.is_empty() {
        return Ok(());
    }
    let mut conn = db.0.lock().unwrap();
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    for (from, to) in remap {
        tx.execute(
            "UPDATE segments SET speaker = ?3 WHERE meeting_id = ?1 AND speaker = ?2",
            params![meeting_id, from, to],
        )
        .map_err(|e| e.to_string())?;
        // Keep a name given to the merged-away speaker unless the target has one.
        tx.execute(
            "INSERT OR IGNORE INTO speaker_names (meeting_id, speaker, name)
             SELECT meeting_id, ?3, name FROM speaker_names WHERE meeting_id = ?1 AND speaker = ?2",
            params![meeting_id, from, to],
        )
        .map_err(|e| e.to_string())?;
        tx.execute(
            "DELETE FROM speaker_names WHERE meeting_id = ?1 AND speaker = ?2",
            params![meeting_id, from],
        )
        .map_err(|e| e.to_string())?;
    }
    tx.commit().map_err(|e| e.to_string())
}

/// Append a line to a meeting's notes (voice "note that…" commands).
pub fn append_meeting_note(db: &Db, meeting_id: &str, line: &str) -> Result<String, String> {
    let conn = db.0.lock().unwrap();
    conn.execute(
        "UPDATE meetings SET notes = CASE WHEN trim(notes) = '' THEN ?2 ELSE rtrim(notes) || char(10) || ?2 END
         WHERE id = ?1",
        params![meeting_id, line],
    )
    .map_err(|e| e.to_string())?;
    conn.query_row("SELECT notes FROM meetings WHERE id = ?1", [meeting_id], |r| r.get(0))
        .map_err(|e| e.to_string())
}

pub fn set_attendees(db: &Db, meeting_id: &str, names: &[String]) -> Result<(), String> {
    let conn = db.0.lock().unwrap();
    conn.execute(
        "UPDATE meetings SET attendees = ?2 WHERE id = ?1",
        params![meeting_id, serde_json::to_string(names).unwrap_or_else(|_| "[]".into())],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Apply a literal replacement across one meeting's transcript ("fix
/// everywhere"), whole words only, case-insensitive. Returns lines changed.
pub fn replace_in_meeting(db: &Db, meeting_id: &str, from: &str, to: &str) -> Result<usize, String> {
    let segs = {
        let conn = db.0.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT id, text FROM segments WHERE meeting_id = ?1")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([meeting_id], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))
            .map_err(|e| e.to_string())?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| e.to_string())?
    };
    let mut updates = Vec::new();
    for (id, text) in segs {
        let next = replace_words(&text, from, to);
        if next != text {
            updates.push((id, next));
        }
    }
    let mut conn = db.0.lock().unwrap();
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    for (id, text) in &updates {
        tx.execute("UPDATE segments SET text = ?2 WHERE id = ?1", params![id, text])
            .map_err(|e| e.to_string())?;
    }
    tx.commit().map_err(|e| e.to_string())?;
    Ok(updates.len())
}

/// Case-insensitive whole-word(s) replacement of `from` in `text`.
pub fn replace_words(text: &str, from: &str, to: &str) -> String {
    let from = from.trim();
    if from.is_empty() {
        return text.to_string();
    }
    let lower = text.to_lowercase();
    let needle = from.to_lowercase();
    // Lowercasing can change byte lengths for some scripts; only do the
    // index-mapped replacement when it doesn't.
    if lower.len() != text.len() {
        return text.replace(from, to);
    }
    let is_word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric());
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while let Some(pos) = lower[i..].find(&needle) {
        let start = i + pos;
        let end = start + needle.len();
        let before = text[..start].chars().next_back();
        let after = text[end..].chars().next();
        out.push_str(&text[i..start]);
        if is_word(before) || is_word(after) {
            out.push_str(&text[start..end]);
        } else {
            out.push_str(to);
        }
        i = end;
    }
    out.push_str(&text[i..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts_query_quotes_terms() {
        assert_eq!(fts_query("pricing deck", false).unwrap(), "\"pricing\"* \"deck\"*");
        assert_eq!(fts_query("a OR b", true).unwrap(), "\"a\"* OR \"or\"* OR \"b\"*");
        assert!(fts_query("  ?!  ", false).is_none());
    }

    #[test]
    fn replaces_whole_words_only() {
        assert_eq!(replace_words("Ask Pria and pria's team", "pria", "Priya"), "Ask Priya and Priya's team");
        assert_eq!(replace_words("capricorn", "pri", "X"), "capricorn");
    }

    #[test]
    fn migrates_and_searches_in_memory() {
        let conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        migrate(&conn).unwrap();
        let db = Db(Mutex::new(conn));
        let m = Meeting {
            id: "m1".into(),
            title: "Pricing sync".into(),
            started_at: "2026-07-29T16:00:00.000Z".into(),
            ended_at: None,
            summary: String::new(),
            notes: String::new(),
            stt_model: String::new(),
            llm_model: String::new(),
            lang: None,
            segments: vec![],
            segment_count: 0,
            speaker_names: Default::default(),
            attendees: vec!["Priya".into()],
            marks: vec![],
            action_items: vec![],
            has_audio: false,
        };
        create_meeting(&db, &m).unwrap();
        add_segments(
            &db,
            "m1",
            &[Segment {
                id: 0,
                speaker: "them:0".into(),
                source: "system".into(),
                start_ms: 61_000,
                end_ms: 64_000,
                text: "We decided the enterprise tier ships in March".into(),
            }],
        )
        .unwrap();
        let hits = search_meetings(&db, "decid enterprise").unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].start_ms, Some(61_000));
        assert!(hits[0].snippet.contains('«'));
        let hits = search_meetings(&db, "pricing").unwrap();
        assert_eq!(hits[0].kind, "meeting");
        // The open meeting is closed at start + its last line.
        assert_eq!(recover_open_meetings(&db).unwrap(), vec!["m1".to_string()]);
        let got = get_meeting(&db, "m1").unwrap().unwrap();
        assert_eq!(got.ended_at.as_deref(), Some("2026-07-29T16:01:04.000Z"));
        assert_eq!(got.attendees, vec!["Priya".to_string()]);
    }
}
