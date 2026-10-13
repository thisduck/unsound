//! Fault tests use production SQLite/WAV code and disposable directories.
//! No microphone, model downloads, network providers, or real profile required.
use crate::{audio_store, meeting_work, store};
use serde_json::json;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn meeting(id: &str) -> store::Meeting {
    serde_json::from_value(json!({"id": id, "startedAt": "2026-01-01T00:00:00Z"})).unwrap()
}

fn segment(text: &str) -> store::Segment {
    serde_json::from_value(json!({"text": text, "startMs": 0, "endMs": 1000})).unwrap()
}

#[test]
fn storage_failure_mid_batch_rolls_back_and_retry_is_exact() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unsound.db");
    let db = store::open_path(&path).unwrap();
    store::create_meeting(&db, &meeting("m")).unwrap();
    store::add_segments(&db, "m", &[segment("before interruption")]).unwrap();
    // Abort on the SECOND insert, after the transaction has already written.
    db.0.lock()
        .unwrap()
        .execute_batch(
            "CREATE TEMP TRIGGER fail_storage BEFORE INSERT ON segments
         WHEN NEW.text = 'second' BEGIN SELECT RAISE(ABORT, 'injected storage failure'); END;",
        )
        .unwrap();
    let batch = [segment("first"), segment("second")];
    assert!(store::add_segments(&db, "m", &batch)
        .unwrap_err()
        .contains("injected storage"));
    assert_eq!(
        store::get_meeting(&db, "m")
            .unwrap()
            .unwrap()
            .segments
            .len(),
        1
    );
    db.0.lock()
        .unwrap()
        .execute_batch("DROP TRIGGER fail_storage")
        .unwrap();
    store::add_segments(&db, "m", &batch).unwrap();
    drop(db);
    let db = store::open_path(&path).unwrap();
    let m = store::get_meeting(&db, "m").unwrap().unwrap();
    assert_eq!(
        m.segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>(),
        ["before interruption", "first", "second"]
    );
}

#[test]
fn storage_becomes_read_only_preserves_last_saved_notes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unsound.db");
    let db = store::open_path(&path).unwrap();
    store::create_meeting(&db, &meeting("m")).unwrap();
    store::update_meeting_notes(&db, "m", "saved notes").unwrap();
    db.0.lock()
        .unwrap()
        .pragma_update(None, "query_only", true)
        .unwrap();
    assert!(store::update_meeting_notes(&db, "m", "unsaved notes").is_err());
    assert!(store::add_segments(&db, "m", &[segment("unsaved")]).is_err());
    drop(db);
    let db = store::open_path(&path).unwrap();
    let m = store::get_meeting(&db, "m").unwrap().unwrap();
    assert_eq!(m.notes, "saved notes");
    assert!(m.segments.is_empty());
    store::update_meeting_notes(&db, "m", "retried notes").unwrap();
    assert_eq!(
        store::get_meeting(&db, "m").unwrap().unwrap().notes,
        "retried notes"
    );
}

// This subprocess test only performs work when invoked by the parent below.
// The parent kills it while SQLite and the unfinished WAV are still open.
#[test]
fn crash_writer_child() {
    let Some(root) = std::env::var_os("UNSOUND_CRASH_TEST_DIR") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    let db = store::open_path(&root.join("unsound.db")).unwrap();
    store::create_meeting(&db, &meeting("interrupted")).unwrap();
    store::update_meeting_notes(&db, "interrupted", "committed notes").unwrap();
    meeting_work::begin(&db, "interrupted", "test-model", None, "").unwrap();
    meeting_work::enqueue(&db, "interrupted", "system", "them", 0, 16_000, false).unwrap();
    meeting_work::enqueue(&db, "interrupted", "system", "them", 16_000, 32_000, false).unwrap();
    let first = meeting_work::next(&db, "interrupted").unwrap().unwrap();
    meeting_work::complete(
        &db,
        "interrupted",
        first.id,
        &[segment("committed transcript")],
    )
    .unwrap();
    let mut wav = audio_store::WavWriter::create(&root.join("interrupted.wav")).unwrap();
    wav.append(&vec![0.25; 48_000]).unwrap(); // header still says zero samples
                                              // Leave a real uncommitted transaction behind as well.
    db.0.lock().unwrap().execute_batch(
        "BEGIN IMMEDIATE; UPDATE meetings SET notes = 'uncommitted notes' WHERE id = 'interrupted';"
    ).unwrap();
    std::fs::write(root.join("ready"), b"ready").unwrap();
    loop {
        std::thread::park();
    }
}

struct KillOnDrop(Child);
impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn killed_process_recovers_commits_and_requeues_unfinished_and_unqueued_audio() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = KillOnDrop(
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "reliability_tests::crash_writer_child",
                "--nocapture",
            ])
            .env("UNSOUND_CRASH_TEST_DIR", dir.path())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(15);
    while !dir.path().join("ready").exists() {
        assert!(
            child.0.try_wait().unwrap().is_none(),
            "crash writer exited before ready"
        );
        assert!(
            Instant::now() < deadline,
            "crash writer did not become ready"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    child.0.kill().unwrap();
    assert!(!child.0.wait().unwrap().success());
    let db = store::open_path(&dir.path().join("unsound.db")).unwrap();
    assert_eq!(store::recover_open_meetings(&db).unwrap(), ["interrupted"]);
    let m = store::get_meeting(&db, "interrupted").unwrap().unwrap();
    assert_eq!(m.notes, "committed notes");
    assert_eq!(m.segments.len(), 1);
    assert_eq!(m.segments[0].text, "committed transcript");
    assert!(m.ended_at.is_some());
    let wav = dir.path().join("interrupted.wav");
    let before = std::fs::read(&wav).unwrap();
    assert_eq!(&before[40..44], &[0, 0, 0, 0]);
    audio_store::repair(&wav).unwrap();
    let samples = crate::audiofile::decode_to_samples(&wav).unwrap();
    assert_eq!(samples.len(), 48_000);
    assert!(samples.iter().all(|s| (*s - 0.25).abs() < 0.001));
    // A queued second and an unqueued VAD tail both survive. The committed
    // first second is never transcribed again, including on a second restart.
    meeting_work::recover_channel(&db, "interrupted", "system", 48_000).unwrap();
    meeting_work::state(&db, "interrupted", "finishing", "").unwrap();
    for (start, end, text) in [
        (16_000, 32_000, "queued"),
        (32_000, 48_000, "unqueued tail"),
    ] {
        let job = meeting_work::next(&db, "interrupted").unwrap().unwrap();
        assert_eq!((job.start, job.end), (start, end));
        let input = audio_store::read_samples(&wav, start, end).unwrap();
        assert_eq!(input.len(), 16_000);
        assert!(input.iter().all(|s| (*s - 0.25).abs() < 0.001));
        meeting_work::complete(&db, "interrupted", job.id, &[segment(text)]).unwrap();
    }
    drop(db);
    let db = store::open_path(&dir.path().join("unsound.db")).unwrap();
    meeting_work::recover_channel(&db, "interrupted", "system", 48_000).unwrap();
    assert!(meeting_work::next(&db, "interrupted").unwrap().is_none());
    assert!(store::recover_open_meetings(&db).unwrap().is_empty());
    assert_eq!(
        store::get_meeting(&db, "interrupted")
            .unwrap()
            .unwrap()
            .segments
            .len(),
        3
    );
    audio_store::repair(&wav).unwrap();
    assert_eq!(std::fs::read(&wav).unwrap()[44..], before[44..]);
}
