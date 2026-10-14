//! Durable transcription work. Audio ranges stay on disk; one worker reads one
//! bounded range at a time. A result and its completion marker commit together.
use crate::store::{Db, Segment};
use rusqlite::{params, OptionalExtension};
use serde::Serialize;

pub const MAX_SAMPLES: i64 = 25 * 16_000;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    pub meeting_id: String,
    pub state: String,
    pub error: String,
    pub warning: String,
    pub mic_samples: i64,
    pub system_samples: i64,
    pub pending: i64,
    pub pending_audio_ms: i64,
    pub recording: bool,
}

pub struct Job {
    pub id: i64,
    pub source: String,
    pub speaker: String,
    pub start: i64,
    pub end: i64,
    pub queued_at_ms: i64,
}

pub fn begin(
    db: &Db,
    id: &str,
    model: &str,
    language: Option<&str>,
    embedding: &str,
) -> Result<(), String> {
    let mut conn = db.0.lock().unwrap();
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    tx.execute(
        "INSERT INTO meeting_work(meeting_id,state,embedding_model) VALUES (?1,'recording',?2)",
        params![id, embedding],
    )
    .map_err(|e| e.to_string())?;
    tx.execute(
        "UPDATE meetings SET stt_model=?2,lang=?3 WHERE id=?1",
        params![id, model, language],
    )
    .map_err(|e| e.to_string())?;
    tx.commit().map_err(|e| e.to_string())
}

pub fn embedding_model(db: &Db, id: &str) -> Result<String, String> {
    db.0.lock()
        .unwrap()
        .query_row(
            "SELECT embedding_model FROM meeting_work WHERE meeting_id=?1",
            [id],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())
}

pub fn state(db: &Db, id: &str, state: &str, error: &str) -> Result<(), String> {
    db.0.lock()
        .unwrap()
        .execute(
            "UPDATE meeting_work SET state=?2,error=?3 WHERE meeting_id=?1",
            params![id, state, error],
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
}

pub fn warning(db: &Db, id: &str, message: &str) -> Result<(), String> {
    db.0.lock()
        .unwrap()
        .execute(
            "UPDATE meeting_work SET warning=?2 WHERE meeting_id=?1",
            params![id, message],
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
}

pub fn statuses(db: &Db) -> Result<Vec<Status>, String> {
    let conn = db.0.lock().unwrap();
    let mut q = conn
        .prepare(
            "SELECT w.meeting_id,w.state,w.error,w.warning,w.mic_samples,w.system_samples,
        (SELECT COUNT(*) FROM transcription_jobs j WHERE j.meeting_id=w.meeting_id AND done=0),
        (SELECT COALESCE(SUM(end_sample-start_sample),0)/16 FROM transcription_jobs j WHERE j.meeting_id=w.meeting_id AND done=0)
        FROM meeting_work w ORDER BY w.rowid DESC",
        )
        .map_err(|e| e.to_string())?;
    let rows = q
        .query_map([], |r| {
            Ok(Status {
                meeting_id: r.get(0)?,
                state: r.get(1)?,
                error: r.get(2)?,
                warning: r.get(3)?,
                mic_samples: r.get(4)?,
                system_samples: r.get(5)?,
                pending: r.get(6)?,
                pending_audio_ms: r.get(7)?,
                recording: false,
            })
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<Result<_, _>>().map_err(|e| e.to_string())
}

pub fn saved(db: &Db, id: &str, source: &str, samples: i64) -> Result<(), String> {
    let column = if source == "mic" {
        "mic_samples"
    } else {
        "system_samples"
    };
    db.0.lock()
        .unwrap()
        .execute(
            &format!("UPDATE meeting_work SET {column}=?2 WHERE meeting_id=?1"),
            params![id, samples],
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
}

pub fn enqueue(
    db: &Db,
    id: &str,
    source: &str,
    speaker: &str,
    start: i64,
    end: i64,
    skipped: bool,
) -> Result<(), String> {
    if start < 0 || end < start {
        return Err("invalid audio range".into());
    }
    let mut conn = db.0.lock().unwrap();
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    let mut pos = start;
    while pos < end {
        let next = (pos + MAX_SAMPLES).min(end);
        tx.execute("INSERT OR IGNORE INTO transcription_jobs(meeting_id,source,speaker,start_sample,end_sample,done,queued_at_ms)
            VALUES (?1,?2,?3,?4,?5,?6,?7)", params![id,source,speaker,pos,next,skipped,crate::meeting_metrics::unix_ms()])
            .map_err(|e|e.to_string())?;
        pos = next;
    }
    tx.commit().map_err(|e| e.to_string())
}

pub fn next(db: &Db, id: &str) -> Result<Option<Job>, String> {
    db.0.lock()
        .unwrap()
        .query_row(
            "SELECT j.id,j.source,j.speaker,j.start_sample,j.end_sample,j.queued_at_ms FROM transcription_jobs j
         JOIN meeting_work w ON w.meeting_id=j.meeting_id WHERE j.meeting_id=?1 AND j.done=0
         AND w.state IN ('recording','finishing') ORDER BY j.start_sample,j.id LIMIT 1",
            [id],
            |r| {
                Ok(Job {
                    id: r.get(0)?,
                    source: r.get(1)?,
                    speaker: r.get(2)?,
                    start: r.get(3)?,
                    end: r.get(4)?,
                    queued_at_ms: r.get(5)?,
                })
            },
        )
        .optional()
        .map_err(|e| e.to_string())
}

/// Fetch only bounded metadata; preserve Whisper batching without loading a
/// backlog into memory. Other engines request a single job.
pub fn next_batch(db: &Db, id: &str, batch: bool, gap: i64) -> Result<Vec<Job>, String> {
    let conn = db.0.lock().unwrap();
    let mut q = conn
        .prepare(
            "SELECT j.id,j.source,j.speaker,j.start_sample,j.end_sample,j.queued_at_ms FROM transcription_jobs j
        JOIN meeting_work w ON w.meeting_id=j.meeting_id WHERE j.meeting_id=?1 AND done=0
        AND w.state IN ('recording','finishing') ORDER BY j.start_sample,j.id LIMIT 32",
        )
        .map_err(|e| e.to_string())?;
    let rows = q
        .query_map([id], |r| {
            Ok(Job {
                id: r.get(0)?,
                source: r.get(1)?,
                speaker: r.get(2)?,
                start: r.get(3)?,
                end: r.get(4)?,
                    queued_at_ms: r.get(5)?,
            })
        })
        .map_err(|e| e.to_string())?;
    let mut jobs: Vec<Job> = Vec::new();
    let mut total = 0;
    for row in rows {
        let job = row.map_err(|e| e.to_string())?;
        let size = job.end - job.start;
        if let Some(first) = jobs.first() {
            if !batch || job.source != first.source || total + size + gap > MAX_SAMPLES {
                break;
            }
            total += gap;
        }
        total += size;
        jobs.push(job);
    }
    Ok(jobs)
}

#[cfg(test)]
pub fn complete(db: &Db, id: &str, job: i64, segments: &[Segment]) -> Result<bool, String> {
    complete_batch(db, id, &[job], segments)
}

/// Return false for a canceled or already committed result. Never emit it as saved.
pub fn complete_batch(
    db: &Db,
    id: &str,
    jobs: &[i64],
    segments: &[Segment],
) -> Result<bool, String> {
    if jobs.is_empty() {
        return Ok(false);
    }
    let mut conn = db.0.lock().unwrap();
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    for job in jobs {
        let changed = tx.execute("UPDATE transcription_jobs SET done=1 WHERE id=?1 AND meeting_id=?2 AND done=0
        AND EXISTS (SELECT 1 FROM meeting_work WHERE meeting_id=?2 AND state IN ('recording','finishing'))",
        params![job,id]).map_err(|e|e.to_string())?;
        if changed == 0 {
            return Ok(false);
        }
    }
    for s in segments {
        tx.execute("INSERT INTO segments(meeting_id,speaker,source,start_ms,end_ms,text) VALUES (?1,?2,?3,?4,?5,?6)",
            params![id,s.speaker,s.source,s.start_ms,s.end_ms,s.text]).map_err(|e|e.to_string())?;
    }
    tx.commit().map_err(|e| e.to_string())?;
    Ok(true)
}

pub fn exclusion(db: &Db, id: &str, start_ms: i64, end_ms: Option<i64>) -> Result<(), String> {
    db.0.lock()
        .unwrap()
        .execute(
            "INSERT INTO transcription_exclusions(meeting_id,start_ms,end_ms) VALUES (?1,?2,?3)
        ON CONFLICT(meeting_id,start_ms) DO UPDATE SET end_ms=excluded.end_ms",
            params![id, start_ms, end_ms],
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Apply persisted tap exclusions even to a job queued before the tap ended.
/// Keep the sample count unchanged so transcript timestamps stay on the clock.
pub fn mask_private_audio(
    db: &Db,
    id: &str,
    source: &str,
    start: i64,
    samples: &mut [f32],
) -> Result<(), String> {
    if source != "mic" {
        return Ok(());
    }
    let end = start + samples.len() as i64;
    let conn = db.0.lock().unwrap();
    let mut q = conn.prepare("SELECT start_ms*16,COALESCE(end_ms*16,?2) FROM transcription_exclusions WHERE meeting_id=?1")
        .map_err(|e|e.to_string())?;
    let ranges = q
        .query_map(params![id, end], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
        })
        .map_err(|e| e.to_string())?;
    for range in ranges {
        let (a, b) = range.map_err(|e| e.to_string())?;
        let from = (a.max(start) - start).clamp(0, samples.len() as i64) as usize;
        let to = (b.min(end) - start).clamp(0, samples.len() as i64) as usize;
        if from < to {
            samples[from..to].fill(0.0);
        }
    }
    Ok(())
}

/// Fill gaps, including speech still inside VAD when the process died. Completed
/// and deliberately excluded ranges are coverage too, so replay cannot duplicate
/// transcript or expose a dictation/command tap. Called only after capture stops.
pub fn recover_channel(db: &Db, id: &str, source: &str, total: i64) -> Result<(), String> {
    let mut covered: Vec<(i64, i64)> = {
        let conn = db.0.lock().unwrap();
        let mut q = conn.prepare("SELECT start_sample,end_sample FROM transcription_jobs WHERE meeting_id=?1 AND source=?2")
            .map_err(|e|e.to_string())?;
        let rows = q
            .query_map(params![id, source], |r| Ok((r.get(0)?, r.get(1)?)))
            .map_err(|e| e.to_string())?;
        let mut ranges = rows
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        if source == "mic" {
            let mut q = conn.prepare("SELECT start_ms*16,COALESCE(end_ms*16,?2) FROM transcription_exclusions WHERE meeting_id=?1")
                .map_err(|e|e.to_string())?;
            let rows = q
                .query_map(params![id, total], |r| Ok((r.get(0)?, r.get(1)?)))
                .map_err(|e| e.to_string())?;
            ranges.extend(
                rows.collect::<Result<Vec<_>, _>>()
                    .map_err(|e| e.to_string())?,
            );
        }
        ranges
    };
    covered.sort_unstable();
    let speaker = if source == "mic" { "me" } else { "them" };
    let mut cursor = 0;
    for (start, end) in covered {
        if start > cursor {
            enqueue(db, id, source, speaker, cursor, start.min(total), false)?;
        }
        cursor = cursor.max(end).min(total);
        if cursor == total {
            break;
        }
    }
    if cursor < total {
        enqueue(db, id, source, speaker, cursor, total, false)?;
    }
    saved(db, id, source, total)
}

#[cfg(test)]
mod reliability_tests {
    use super::*;
    use crate::store;
    use serde_json::json;

    fn fixture() -> (tempfile::TempDir, Db) {
        let dir = tempfile::tempdir().unwrap();
        let db = store::open_path(&dir.path().join("db")).unwrap();
        store::create_meeting(
            &db,
            &serde_json::from_value(json!({"id":"m","startedAt":"2026-01-01T00:00:00Z"})).unwrap(),
        )
        .unwrap();
        begin(&db, "m", "test-model", None, "").unwrap();
        (dir, db)
    }
    fn segment(text: &str) -> Segment {
        serde_json::from_value(json!({"text":text,"endMs":1000})).unwrap()
    }

    #[test]
    fn storage_interrupt_rolls_back_result_and_completion_together() {
        let (_dir, db) = fixture();
        enqueue(&db, "m", "mic", "me", 0, 16_000, false).unwrap();
        let job = next(&db, "m").unwrap().unwrap();
        db.0.lock()
            .unwrap()
            .execute_batch(
                "CREATE TEMP TRIGGER fail_result BEFORE INSERT ON segments
            WHEN NEW.text='second' BEGIN SELECT RAISE(ABORT,'storage interrupted'); END;",
            )
            .unwrap();
        let result = [segment("first"), segment("second")];
        assert!(complete(&db, "m", job.id, &result).is_err());
        assert_eq!(next(&db, "m").unwrap().unwrap().id, job.id);
        assert!(store::get_meeting(&db, "m")
            .unwrap()
            .unwrap()
            .segments
            .is_empty());
        db.0.lock()
            .unwrap()
            .execute_batch("DROP TRIGGER fail_result")
            .unwrap();
        assert!(complete(&db, "m", job.id, &result).unwrap());
        assert!(!complete(&db, "m", job.id, &result).unwrap());
        assert_eq!(
            store::get_meeting(&db, "m")
                .unwrap()
                .unwrap()
                .segments
                .len(),
            2
        );
        assert!(next(&db, "m").unwrap().is_none());
    }

    #[test]
    fn canceled_inference_cannot_commit_late_and_can_be_retried() {
        let (_dir, db) = fixture();
        enqueue(&db, "m", "mic", "me", 0, 16_000, false).unwrap();
        let job = next(&db, "m").unwrap().unwrap();
        state(&db, "m", "paused", "").unwrap();
        assert!(!complete(&db, "m", job.id, &[segment("late")]).unwrap());
        assert_eq!(statuses(&db).unwrap()[0].pending, 1);
        assert!(next(&db, "m").unwrap().is_none());
        state(&db, "m", "finishing", "").unwrap();
        assert!(complete(&db, "m", job.id, &[segment("retried")]).unwrap());
        assert_eq!(
            store::get_meeting(&db, "m").unwrap().unwrap().segments[0].text,
            "retried"
        );
    }

    #[test]
    fn restart_fills_tail_and_gaps_without_repeating_completed_or_private_audio() {
        let (dir, db) = fixture();
        enqueue(&db, "m", "mic", "me", 16_000, 32_000, false).unwrap();
        let first = next(&db, "m").unwrap().unwrap();
        complete(&db, "m", first.id, &[segment("already saved")]).unwrap();
        exclusion(&db, "m", 2000, Some(3000)).unwrap();
        exclusion(&db, "m", 4000, None).unwrap(); // command still held at crash
        drop(db);
        let db = store::open_path(&dir.path().join("db")).unwrap();
        recover_channel(&db, "m", "mic", 80_000).unwrap();
        recover_channel(&db, "m", "mic", 80_000).unwrap();
        assert_eq!(statuses(&db).unwrap()[0].pending, 2);
        let j = next(&db, "m").unwrap().unwrap();
        assert_eq!((j.start, j.end), (0, 16_000));
        complete(&db, "m", j.id, &[]).unwrap(); // recognized silence is a completed job
        let j = next(&db, "m").unwrap().unwrap();
        assert_eq!((j.start, j.end), (48_000, 64_000));
        complete(&db, "m", j.id, &[segment("tail")]).unwrap();
        assert!(next(&db, "m").unwrap().is_none());
        assert_eq!(
            store::get_meeting(&db, "m")
                .unwrap()
                .unwrap()
                .segments
                .len(),
            2
        );
    }

    #[test]
    fn arbitrarily_long_backlog_is_split_into_bounded_disk_ranges() {
        let (_dir, db) = fixture();
        recover_channel(&db, "m", "system", MAX_SAMPLES * 1000 + 53).unwrap();
        assert_eq!(statuses(&db).unwrap()[0].pending, 1001);
        let mut cursor = 0;
        while let Some(job) = next(&db, "m").unwrap() {
            assert_eq!(job.start, cursor);
            assert!(job.end - job.start <= MAX_SAMPLES);
            cursor = job.end;
            complete(&db, "m", job.id, &[]).unwrap();
        }
        assert_eq!(cursor, MAX_SAMPLES * 1000 + 53);
    }

    #[test]
    fn whisper_batches_short_turns_with_a_bound_and_commits_all_markers_atomically() {
        let (_dir, db) = fixture();
        for n in 0..20 {
            enqueue(&db, "m", "mic", "me", n * 32_000, (n + 1) * 32_000, false).unwrap();
        }
        let jobs = next_batch(&db, "m", true, 4800).unwrap();
        assert!(jobs.len() > 1);
        assert!(
            jobs.iter().map(|j| j.end - j.start).sum::<i64>() + (jobs.len() as i64 - 1) * 4800
                <= MAX_SAMPLES
        );
        assert_eq!(next_batch(&db, "m", false, 4800).unwrap().len(), 1);
        let ids: Vec<_> = jobs.iter().map(|j| j.id).collect();
        db.0.lock()
            .unwrap()
            .execute_batch(
                "CREATE TEMP TRIGGER fail_batch BEFORE INSERT ON segments
            BEGIN SELECT RAISE(ABORT,'storage interrupted'); END;",
            )
            .unwrap();
        assert!(complete_batch(&db, "m", &ids, &[segment("batch")]).is_err());
        assert_eq!(statuses(&db).unwrap()[0].pending, 20);
        db.0.lock()
            .unwrap()
            .execute_batch("DROP TRIGGER fail_batch")
            .unwrap();
        assert!(complete_batch(&db, "m", &ids, &[segment("batch")]).unwrap());
        assert!(!complete_batch(&db, "m", &ids, &[segment("duplicate")]).unwrap());
        assert_eq!(statuses(&db).unwrap()[0].pending, 20 - ids.len() as i64);
        assert_eq!(
            store::get_meeting(&db, "m")
                .unwrap()
                .unwrap()
                .segments
                .len(),
            1
        );
    }

    #[test]
    fn pending_audio_cannot_replay_private_taps() {
        let (_dir, db) = fixture();
        enqueue(&db, "m", "mic", "me", 0, 16_000, false).unwrap();
        exclusion(&db, "m", 500, Some(750)).unwrap();
        exclusion(&db, "m", 900, None).unwrap();
        let mut samples = vec![0.5; 16_000];
        mask_private_audio(&db, "m", "mic", 0, &mut samples).unwrap();
        assert!(samples[..8000].iter().all(|s| *s == 0.5));
        assert!(samples[8000..12000].iter().all(|s| *s == 0.0));
        assert!(samples[12000..14400].iter().all(|s| *s == 0.5));
        assert!(samples[14400..].iter().all(|s| *s == 0.0));
        let mut remote = vec![0.5; 16_000];
        mask_private_audio(&db, "m", "system", 0, &mut remote).unwrap();
        assert!(remote.iter().all(|s| *s == 0.5));
    }

    #[test]
    fn pending_work_and_failures_survive_reopen_and_delete_cascades() {
        let (dir, db) = fixture();
        enqueue(&db, "m", "mic", "me", 0, 16_000, false).unwrap();
        saved(&db, "m", "mic", 16_000).unwrap();
        state(&db, "m", "error", "disk full after 1 second").unwrap();
        drop(db);
        let db = store::open_path(&dir.path().join("db")).unwrap();
        let status = statuses(&db).unwrap().remove(0);
        assert_eq!((status.pending, status.mic_samples), (1, 16_000));
        assert_eq!(status.error, "disk full after 1 second");
        store::delete_meeting(&db, "m").unwrap();
        assert!(statuses(&db).unwrap().is_empty());
        assert!(next(&db, "m").unwrap().is_none());
    }
}
