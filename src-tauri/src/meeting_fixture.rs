//! Deterministic hardware/model substitutes, compiled only for tests or automation.
//! Persistence, worker scheduling, cancellation, and startup recovery stay real.
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Default)]
pub struct Fixture {
    audio: Mutex<VecDeque<(Vec<f32>, Vec<f32>)>>,
    #[cfg(feature = "automation")]
    pub system_audio: AtomicBool,
    #[cfg(feature = "automation")]
    pub disconnected: AtomicBool,
    pub fail_audio: AtomicBool,
    stall: AtomicUsize, // 1 honors cancellation; 2 models a stuck native call
    pub calls: AtomicUsize,
    pub in_flight: AtomicUsize,
    timeout_ms: AtomicU64,
}

impl Fixture {
    pub fn feed(&self) -> Result<(), String> {
        let mut queue = self.audio.lock().unwrap();
        if queue.len() == 4 {
            return Err("fixture audio buffer is full".into());
        }
        queue.push_back((vec![0.25; 16_000], Vec::new()));
        Ok(())
    }

    #[cfg(test)]
    pub fn drain(&self) -> Vec<f32> {
        self.audio.lock().unwrap().drain(..).flat_map(|(mic,_)|mic).collect()
    }

    #[cfg(feature = "automation")]
    pub fn disconnect_with_tail(&self) -> Result<(), String> {
        let mut queue = self.audio.lock().unwrap();
        if queue.len() == 4 {
            return Err("fixture audio buffer is full".into());
        }
        queue.push_back((vec![0.25; 16_000], Vec::new()));
        self.disconnected.store(true, Ordering::SeqCst);
        Ok(())
    }

    #[cfg(feature = "automation")]
    pub fn capture_input(&self) -> (bool, Vec<f32>, Vec<f32>) {
        let mut queue = self.audio.lock().unwrap();
        let disconnected = self.disconnected.swap(false, Ordering::SeqCst);
        let (mut mic,mut system)=(Vec::new(),Vec::new());
        for (m,s) in queue.drain(..) {mic.extend(m);system.extend(s);}
        (disconnected, mic, system)
    }

    #[cfg(feature = "automation")]
    pub fn feed_pcm(&self,mic:Vec<f32>,system:Vec<f32>)->Result<(),String> {
        if mic.len()>crate::meeting_work::MAX_SAMPLES as usize || system.len()>crate::meeting_work::MAX_SAMPLES as usize {return Err("benchmark frame exceeds 25 seconds".into());}
        let mut queue=self.audio.lock().unwrap();
        if queue.len()>=4 {return Err("fixture audio buffer is full".into());}
        queue.push_back((mic,system));Ok(())
    }

    pub fn check_save(&self) -> Result<(), String> {
        if self.fail_audio.load(Ordering::SeqCst) {
            Err("injected audio storage failure".into())
        } else {
            Ok(())
        }
    }

    pub fn timeout(&self) -> Duration {
        match self.timeout_ms.load(Ordering::SeqCst) {
            0 => Duration::from_secs(90),
            ms => Duration::from_millis(ms),
        }
    }

    pub fn transcribe(&self, samples: &[f32], cancel: &AtomicBool) -> Result<String, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        loop {
            let mode = self.stall.load(Ordering::SeqCst);
            if mode == 0 || (mode == 1 && cancel.load(Ordering::SeqCst)) {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        if cancel.load(Ordering::SeqCst) {
            return Err("fixture transcription canceled".into());
        }
        // Inspect actual PCM read from the saved WAV, not a canned queue result.
        if !samples.iter().any(|s| *s > 0.2) {
            return Err("fixture audio is missing or corrupt".into());
        }
        Ok("Fixture speech survived.".into())
    }
}

#[cfg(feature = "automation")]
mod native {
    use super::*;
    use crate::{audio_store, meeting, store::Db};
    use std::collections::HashMap;
    use tauri::{AppHandle, Manager};

    pub const MODEL_PREFIX: &str = "e2e-fixture:";
    pub fn valid_id(id: &str) -> bool {
        id.starts_with("e2e-")
            && id.len() <= 80
            && id.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-')
    }
    #[derive(Default)]
    pub struct Fixtures(Mutex<HashMap<String, Arc<Fixture>>>);

    pub fn get(app: &AppHandle, id: &str) -> Arc<Fixture> {
        app.state::<Fixtures>()
            .0
            .lock()
            .unwrap()
            .entry(id.into())
            .or_default()
            .clone()
    }

    pub fn for_model(app: &AppHandle, model: &str) -> Option<Arc<Fixture>> {
        model
            .strip_prefix(MODEL_PREFIX)
            .filter(|id| valid_id(id))
            .map(|id| get(app, id))
    }

    pub fn timeout(app: &AppHandle, id: &str) -> Duration {
        app.state::<Fixtures>()
            .0
            .lock()
            .unwrap()
            .get(id)
            .map(|f| f.timeout())
            .unwrap_or(Duration::from_secs(90))
    }

    #[derive(serde::Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum Action {
        Feed,
        Stall,
        StallUncancellable,
        Release,
        Watchdog,
        FailAudio,
        Disconnect,
        FailTranscript,
        Clear,
        Snapshot,
    }

    #[derive(serde::Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Snapshot {
        calls: usize,
        in_flight: usize,
        audio_samples: u64,
        process_id: u32,
    }

    #[tauri::command]
    pub fn meeting_fixture(app: AppHandle, id: String, action: Action) -> Result<Snapshot, String> {
        if app.config().identifier != "com.unsound.e2e" {
            return Err("fixture commands require the isolated automation profile".into());
        }
        // IDs also become filenames and SQL trigger names: admit no path/SQL syntax.
        if !valid_id(&id) {
            return Err("invalid fixture id".into());
        }
        let db = app.state::<Db>();
        if crate::store::get_meeting(&db, &id)?.is_none() {
            return Err("create the fixture meeting first".into());
        }
        let f = get(&app, &id);
        let trigger = format!("fixture_{}", id.replace('-', "_"));
        match action {
            Action::Feed => {
                if meeting::active_meeting(&app).as_deref() != Some(&id) {
                    return Err("fixture is not recording".into());
                }
                f.feed()?;
            }
            Action::Stall => f.stall.store(1, Ordering::SeqCst),
            Action::StallUncancellable => f.stall.store(2, Ordering::SeqCst),
            Action::Release => f.stall.store(0, Ordering::SeqCst),
            Action::Watchdog => f.timeout_ms.store(1000, Ordering::SeqCst),
            Action::FailAudio => f.fail_audio.store(true, Ordering::SeqCst),
            Action::Disconnect => f.disconnect_with_tail()?,
            Action::FailTranscript => {
                db.0.lock().unwrap().execute_batch(&format!(
                    "CREATE TEMP TRIGGER {trigger} BEFORE INSERT ON segments
                     WHEN NEW.meeting_id='{id}' BEGIN SELECT RAISE(ABORT,'injected transcript storage failure'); END;"
                )).map_err(|e| e.to_string())?;
            }
            Action::Clear => {
                f.stall.store(0, Ordering::SeqCst);
                f.fail_audio.store(false, Ordering::SeqCst);
                f.disconnected.store(false, Ordering::SeqCst);
                f.timeout_ms.store(0, Ordering::SeqCst);
                db.0.lock()
                    .unwrap()
                    .execute_batch(&format!("DROP TRIGGER IF EXISTS {trigger}"))
                    .map_err(|e| e.to_string())?;
            }
            Action::Snapshot => {}
        }
        let path = audio_store::mic_path(&app, &id)?;
        Ok(Snapshot {
            calls: f.calls.load(Ordering::SeqCst),
            in_flight: f.in_flight.load(Ordering::SeqCst),
            audio_samples: if path.exists() {
                audio_store::sample_count(&path)? as u64
            } else {
                0
            },
            process_id: std::process::id(),
        })
    }
}
#[cfg(feature = "automation")]
pub use native::*;

#[cfg(test)]
mod reliability_tests {
    use super::*;
    use std::time::Instant;

    fn wait_started(f: &Fixture) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while f.in_flight.load(Ordering::SeqCst) == 0 {
            assert!(Instant::now() < deadline, "fixture did not enter inference");
            std::thread::yield_now();
        }
    }

    #[test]
    fn stalled_model_honors_cancellation_without_consuming_more_audio() {
        let f = Arc::new(Fixture::default());
        f.stall.store(1, Ordering::SeqCst);
        let cancel = Arc::new(AtomicBool::new(false));
        let (worker, flag) = (f.clone(), cancel.clone());
        let join = std::thread::spawn(move || worker.transcribe(&[0.25], &flag));
        wait_started(&f);
        for _ in 0..4 {
            f.feed().unwrap();
        }
        assert!(f.feed().is_err());
        assert_eq!(f.drain().len(), 64_000);
        assert!(f.drain().is_empty());
        cancel.store(true, Ordering::SeqCst);
        assert!(join.join().unwrap().unwrap_err().contains("canceled"));
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.in_flight.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn noncooperative_model_returns_late_but_never_reports_canceled_audio_as_success() {
        let f = Arc::new(Fixture::default());
        f.stall.store(2, Ordering::SeqCst);
        let cancel = Arc::new(AtomicBool::new(false));
        let (worker, flag) = (f.clone(), cancel.clone());
        let join = std::thread::spawn(move || worker.transcribe(&[0.25], &flag));
        wait_started(&f);
        cancel.store(true, Ordering::SeqCst);
        assert!(!join.is_finished());
        f.stall.store(0, Ordering::SeqCst);
        assert!(join.join().unwrap().is_err());
    }

    #[test]
    fn audio_failure_preserves_the_last_checkpoint_and_saved_pcm_is_transcribable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.wav");
        let mut wav = crate::audio_store::WavWriter::create(&path).unwrap();
        let f = Fixture::default();
        f.feed().unwrap();
        f.check_save().and_then(|_| wav.append(&f.drain())).unwrap();
        wav.checkpoint().unwrap();
        f.fail_audio.store(true, Ordering::SeqCst);
        f.feed().unwrap();
        assert!(f.check_save().and_then(|_| wav.append(&f.drain())).is_err());
        wav.finish().unwrap();
        assert_eq!(crate::audio_store::sample_count(&path).unwrap(), 16_000);
        let pcm = crate::audio_store::read_samples(&path, 0, 16_000).unwrap();
        assert_eq!(
            f.transcribe(&pcm, &AtomicBool::new(false)).unwrap(),
            "Fixture speech survived."
        );
        assert!(f.transcribe(&[0.0; 16], &AtomicBool::new(false)).is_err());
        assert_eq!(f.timeout(), Duration::from_secs(90));
    }
}
