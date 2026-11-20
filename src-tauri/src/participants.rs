//! People observed in a meeting (from the call app's roster, see `roster`)
//! and what they are used for: conservative speaker naming, and the names in
//! the Whisper prompt.
//!
//! Naming never guesses. It names a remote voice only when
//! - one other person was alone with you on the roster for at least
//!   `SOLO_MS`, and the remote channel during that time is one voice that
//!   spoke mostly then; or
//! - every person but one on the roster is already matched to a voice (by
//!   voiceprint or by you), the number of voices equals the number of people,
//!   and exactly one voice is left unnamed.
//! Names you set and voiceprint matches are never overwritten. Everything
//! else is left to one-click suggestions in the rename UI.

use crate::store::{self, Db};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A roster showing one other person, unchanged this long, is a 1:1. Shorter
/// is someone joining or leaving.
pub const SOLO_MS: i64 = 15_000;
/// A voice is named from a 1:1 when at least this share of its speech fell
/// inside that person's 1:1 stretches...
const NAME_SHARE: f64 = 0.6;
/// ...and this share of the remote speech in those stretches is that voice.
const SOLE_VOICE_SHARE: f64 = 0.9;
/// Minimum speech behind a 1:1 name: more while recording (a wrong name on
/// screen is worse than none), less once clusters are final.
const LIVE_MIN_MS: i64 = 6_000;
const FINAL_MIN_MS: i64 = 2_000;
/// A remote voice counts as a real speaker with this much speech.
const SPEAKER_MIN_SEGMENTS: usize = 3;
const SPEAKER_MIN_MS: i64 = 4_000;
const SPEAKER_MIN_SHARE: f64 = 0.05;
/// Live naming re-checks at most this often.
const LIVE_EVERY: Duration = Duration::from_secs(10);

/// How a speaker got its name (`speaker_names.source`). "user" is the column
/// default, written by the store for every rename you make.
#[allow(dead_code)]
pub const SOURCE_USER: &str = "user";
pub const SOURCE_VOICE: &str = "voice";
pub const SOURCE_ROSTER: &str = "roster";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Person {
    pub name: String,
    /// The local user (marked by the app, or a name seen as you before).
    #[serde(default)]
    pub me: bool,
    /// The app whose roster showed them.
    #[serde(default)]
    pub app: String,
    /// When they were first and last seen, on the meeting clock.
    #[serde(default)]
    pub first_ms: i64,
    #[serde(default)]
    pub last_ms: i64,
}

/// A stretch in which the roster showed exactly one other person.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoloRun {
    pub name: String,
    pub start_ms: i64,
    pub end_ms: i64,
}

/// Everyone seen on the roster over the meeting (`meetings.participants`).
/// Calendar invitees live separately in `meetings.attendees`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Participants {
    #[serde(default)]
    pub people: Vec<Person>,
    #[serde(default)]
    pub solo: Vec<SoloRun>,
}

impl Participants {
    /// People other than you, in order of appearance.
    pub fn others(&self) -> impl Iterator<Item = &Person> {
        self.people.iter().filter(|p| !p.me)
    }
}

fn key(name: &str) -> String {
    name.trim().to_lowercase()
}

/// Folds roster reads into the meeting's `Participants`.
pub struct Observer {
    pub record: Participants,
    /// Index of the solo run the previous read extended, if it still runs.
    open: Option<usize>,
    self_names: Vec<String>,
}

impl Observer {
    pub fn new(record: Participants, self_names: Vec<String>) -> Self {
        Self { record, open: None, self_names: self_names.iter().map(|n| key(n)).collect() }
    }

    /// Record one read at `at_ms`: `(app, rows)`, or `None` when there was no
    /// roster (panel closed, no permission…). Returns whether to save.
    pub fn observe(&mut self, roster: Option<(&str, &[crate::roster::Entry])>, at_ms: i64) -> bool {
        let Some((app, rows)) = roster.filter(|(_, rows)| !rows.is_empty()) else {
            // No evidence about who is there: a 1:1 can't continue through it.
            self.open = None;
            return false;
        };
        let mut others = Vec::new();
        for row in rows {
            let k = key(&row.name);
            let me = row.me.unwrap_or(false) || (row.me.is_none() && self.self_names.contains(&k));
            match self.record.people.iter_mut().find(|p| key(&p.name) == k) {
                Some(p) => {
                    p.last_ms = p.last_ms.max(at_ms);
                    p.first_ms = p.first_ms.min(at_ms);
                    p.me |= me;
                }
                None => self.record.people.push(Person {
                    name: row.name.clone(),
                    me,
                    app: app.to_string(),
                    first_ms: at_ms,
                    last_ms: at_ms,
                }),
            }
            let is_me = self.record.people.iter().any(|p| key(&p.name) == k && p.me);
            if !is_me && !others.contains(&k) {
                others.push(k);
            }
        }
        if let [other] = &others[..] {
            match self.open.and_then(|i| self.record.solo.get_mut(i)) {
                Some(run) if key(&run.name) == *other => run.end_ms = at_ms.max(run.end_ms),
                _ => {
                    let name = rows.iter().find(|r| key(&r.name) == *other).map(|r| r.name.clone()).unwrap_or_default();
                    self.record.solo.push(SoloRun { name, start_ms: at_ms, end_ms: at_ms });
                    self.open = Some(self.record.solo.len() - 1);
                }
            }
        } else {
            self.open = None;
        }
        true
    }
}

// ---- naming ------------------------------------------------------------------

/// Whether a set name and a roster name are the same person: equal ignoring
/// case, or a bare first name that only one roster person has.
pub fn same_person(given: &str, roster_name: &str, roster: &[&str]) -> bool {
    let (g, r) = (key(given), key(roster_name));
    if g.is_empty() || r.is_empty() {
        return false;
    }
    if g == r {
        return true;
    }
    let first = |s: &str| s.split_whitespace().next().unwrap_or("").to_string();
    if g.contains(' ') || first(&r) != g {
        return false;
    }
    roster.iter().filter(|n| first(&key(n)) == g).count() == 1
}

/// A change to one speaker's name.
#[derive(Clone, Debug, PartialEq)]
pub enum Change {
    Name { speaker: String, name: String },
    /// Remove a roster-sourced name the final evidence no longer supports.
    Clear { speaker: String },
}

/// Decide roster-based names. Pure: `spoken` is the remote channel's segments
/// as `(speaker, start_ms, end_ms)`, `names` maps speaker → (name, source).
/// While recording (`final_pass` false) existing names are kept and only
/// added to; at the end roster names are re-derived from scratch.
pub fn decide(
    p: &Participants,
    spoken: &[(String, i64, i64)],
    names: &HashMap<String, (String, String)>,
    final_pass: bool,
) -> Vec<Change> {
    // Remote voices and their speech.
    let mut voices: HashMap<&str, (usize, i64)> = HashMap::new();
    for (sp, s, e) in spoken {
        if sp.starts_with("them:") {
            let v = voices.entry(sp.as_str()).or_default();
            v.0 += 1;
            v.1 += (e - s).max(0);
        }
    }
    let remote_ms: i64 = voices.values().map(|v| v.1).sum();
    let mut speakers: Vec<&str> = voices
        .iter()
        .filter(|(_, (n, ms))| {
            *n >= SPEAKER_MIN_SEGMENTS
                && *ms >= SPEAKER_MIN_MS
                && *ms as f64 >= SPEAKER_MIN_SHARE * remote_ms as f64
        })
        .map(|(k, _)| *k)
        .collect();
    speakers.sort();

    let others: Vec<&str> = {
        let mut seen = Vec::new();
        for o in p.others() {
            if !seen.iter().any(|s: &&str| key(s) == key(&o.name)) {
                seen.push(o.name.as_str());
            }
        }
        seen
    };
    // Names that stand: yours and voiceprints always; roster names only live.
    let mut assigned: HashMap<String, String> = names
        .iter()
        .filter(|(sp, (n, src))| *sp != "me" && !n.trim().is_empty() && (!final_pass || src != SOURCE_ROSTER))
        .map(|(sp, (n, _))| (sp.clone(), n.trim().to_string()))
        .collect();
    let taken = |assigned: &HashMap<String, String>, person: &str| {
        assigned.values().any(|n| same_person(n, person, &others))
    };
    let mut derived: Vec<(String, String)> = Vec::new();

    // 1:1 stretches.
    let min_ms = if final_pass { FINAL_MIN_MS } else { LIVE_MIN_MS };
    let solo: Vec<&SoloRun> = p.solo.iter().filter(|r| r.end_ms - r.start_ms >= SOLO_MS).collect();
    let inside = |sp: &str, person: &str| -> i64 {
        spoken
            .iter()
            .filter(|(s, _, _)| s == sp)
            .map(|(_, s, e)| {
                solo.iter()
                    .filter(|r| key(&r.name) == key(person))
                    .map(|r| (e.min(&r.end_ms) - s.max(&r.start_ms)).max(0))
                    .sum::<i64>()
            })
            .sum()
    };
    let mut solo_people: Vec<&str> = Vec::new();
    for r in &solo {
        if !solo_people.iter().any(|n| key(n) == key(&r.name)) {
            solo_people.push(&r.name);
        }
    }
    for person in solo_people {
        if !others.iter().any(|o| key(o) == key(person)) || taken(&assigned, person) {
            continue;
        }
        let in_runs: Vec<(&str, i64)> = voices.keys().map(|sp| (*sp, inside(sp, person))).collect();
        let total: i64 = in_runs.iter().map(|(_, ms)| ms).sum();
        let Some(&(sp, ms)) = in_runs.iter().max_by_key(|(sp, ms)| (*ms, std::cmp::Reverse(*sp))) else {
            continue;
        };
        let voice_ms = voices[sp].1;
        let named = assigned.contains_key(sp);
        if !named
            && speakers.contains(&sp)
            && ms >= min_ms
            && ms as f64 >= SOLE_VOICE_SHARE * total as f64
            && ms as f64 >= NAME_SHARE * voice_ms as f64
        {
            assigned.insert(sp.to_string(), person.to_string());
            derived.push((sp.to_string(), person.to_string()));
        }
    }

    // Elimination: needs at least two people, so it never reduces to an
    // unchecked "one voice, one name" guess.
    if others.len() >= 2 && speakers.len() == others.len() {
        let open_speakers: Vec<&str> = speakers.iter().copied().filter(|s| !assigned.contains_key(*s)).collect();
        let open_people: Vec<&str> = others.iter().copied().filter(|o| !taken(&assigned, o)).collect();
        if let ([sp], [person]) = (&open_speakers[..], &open_people[..]) {
            assigned.insert(sp.to_string(), person.to_string());
            derived.push((sp.to_string(), person.to_string()));
        }
    }

    let mut changes: Vec<Change> = Vec::new();
    for (speaker, name) in &derived {
        let current = names.get(speaker).map(|(n, _)| n.trim());
        if current != Some(name.as_str()) {
            changes.push(Change::Name { speaker: speaker.clone(), name: name.clone() });
        }
    }
    if final_pass {
        let mut stale: Vec<&String> = names
            .iter()
            .filter(|(sp, (_, src))| src == SOURCE_ROSTER && !derived.iter().any(|(d, _)| d == *sp))
            .map(|(sp, _)| sp)
            .collect();
        stale.sort();
        changes.extend(stale.into_iter().map(|sp| Change::Clear { speaker: sp.clone() }));
    }
    changes
}

/// Decide and store roster-based names for a meeting. Returns what changed as
/// `(speaker, name)`, an empty name meaning cleared.
pub fn apply(db: &Db, meeting_id: &str, final_pass: bool) -> Result<Vec<(String, String)>, String> {
    let (_, participants) = store::meeting_people(db, meeting_id)?;
    if participants.people.is_empty() {
        return Ok(Vec::new());
    }
    let spoken = store::speaker_spans(db, meeting_id)?;
    let names = store::speaker_names_with_source(db, meeting_id)?;
    let mut out = Vec::new();
    for change in decide(&participants, &spoken, &names, final_pass) {
        match change {
            Change::Name { speaker, name } => {
                if store::set_speaker_name_auto(db, meeting_id, &speaker, &name, SOURCE_ROSTER, &[SOURCE_ROSTER])? {
                    out.push((speaker, name));
                }
            }
            Change::Clear { speaker } => {
                if store::clear_speaker_name_from(db, meeting_id, &speaker, SOURCE_ROSTER)? {
                    out.push((speaker, String::new()));
                }
            }
        }
    }
    Ok(out)
}

/// `apply`, announcing each change to the UI. Failures are logged, not raised.
pub fn name_speakers(app: &tauri::AppHandle, meeting_id: &str, final_pass: bool) {
    use tauri::{Emitter, Manager};
    match apply(&app.state::<Db>(), meeting_id, final_pass) {
        Ok(changes) => {
            for (speaker, name) in changes {
                let _ = app.emit(
                    "meeting-speaker-named",
                    serde_json::json!({ "meetingId": meeting_id, "speaker": speaker, "name": name }),
                );
            }
        }
        Err(e) => eprintln!("[participants] naming skipped: {e}"),
    }
}

// ---- the Whisper prompt ------------------------------------------------------

/// Names to prime for a meeting: people seen on the call (others first, then
/// you), then calendar invitees.
pub fn prompt_names(attendees: &[String], p: &Participants) -> Vec<String> {
    let people = p
        .others()
        .chain(p.people.iter().filter(|x| x.me))
        .map(|x| x.name.as_str())
        .chain(attendees.iter().map(String::as_str));
    crate::glossary::name_terms(people)
}

/// The transcription thread's view of participants: the current prompt, kept
/// up to date as people appear, and throttled live naming.
pub struct Live {
    dictionary: Vec<String>,
    names: Vec<String>,
    prompt: Option<String>,
    next_naming: Instant,
}

impl Live {
    pub fn new(settings: &crate::settings::Settings) -> Self {
        let dictionary = crate::glossary::dictionary_terms(settings);
        Self {
            prompt: crate::glossary::prompt(&[], &dictionary),
            dictionary,
            names: Vec::new(),
            next_naming: Instant::now(),
        }
    }

    /// The prompt for the next batch, including anyone seen since the last.
    pub fn prompt(&mut self, db: &Db, meeting_id: &str) -> Option<&str> {
        if let Ok((attendees, p)) = store::meeting_people(db, meeting_id) {
            let names = prompt_names(&attendees, &p);
            if names != self.names {
                self.prompt = crate::glossary::prompt(&names, &self.dictionary);
                self.names = names;
            }
        }
        self.prompt.as_deref()
    }

    /// After a committed batch: name voices the roster makes certain.
    pub fn after_batch(&mut self, app: &tauri::AppHandle, meeting_id: &str) {
        if Instant::now() < self.next_naming {
            return;
        }
        self.next_naming = Instant::now() + LIVE_EVERY;
        name_speakers(app, meeting_id, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roster::Entry;

    fn e(name: &str, me: Option<bool>) -> Entry {
        Entry { name: name.into(), me }
    }

    fn person(name: &str, me: bool) -> Person {
        Person { name: name.into(), me, app: "Zoom".into(), first_ms: 0, last_ms: 60_000 }
    }

    fn run(name: &str, start_ms: i64, end_ms: i64) -> SoloRun {
        SoloRun { name: name.into(), start_ms, end_ms }
    }

    /// `n` segments of 3 s each for `speaker`, starting at `from_ms`, 5 s apart.
    fn talk(speaker: &str, from_ms: i64, n: usize) -> Vec<(String, i64, i64)> {
        (0..n).map(|i| (speaker.to_string(), from_ms + i as i64 * 5_000, from_ms + i as i64 * 5_000 + 3_000)).collect()
    }

    fn names(v: &[(&str, &str, &str)]) -> HashMap<String, (String, String)> {
        v.iter().map(|(sp, n, src)| (sp.to_string(), (n.to_string(), src.to_string()))).collect()
    }

    fn named(sp: &str, n: &str) -> Change {
        Change::Name { speaker: sp.into(), name: n.into() }
    }

    #[test]
    fn observer_keeps_union_self_and_solo_runs() {
        let mut o = Observer::new(Participants::default(), vec![]);
        // Names arrive cleaned by the roster reader.
        let zoom = [e("Adnan", Some(true)), e("Priya", Some(false))];
        assert!(o.observe(Some(("Zoom", &zoom)), 5_000));
        assert!(o.observe(Some(("Zoom", &zoom)), 10_000));
        assert!(o.observe(Some(("Zoom", &zoom)), 25_000));
        assert_eq!(o.record.solo, vec![run("Priya", 5_000, 25_000)]);
        // Panel closed: the run stops; reopening starts a new one.
        assert!(!o.observe(None, 30_000));
        assert!(o.observe(Some(("Zoom", &zoom)), 35_000));
        assert_eq!(o.record.solo.len(), 2);
        // A third person joins: no solo, union grows.
        let three = [zoom[0].clone(), zoom[1].clone(), e("Marcus", Some(false))];
        assert!(o.observe(Some(("Zoom", &three)), 40_000));
        assert_eq!(o.record.solo.len(), 2);
        let people: Vec<(&str, bool, i64, i64)> =
            o.record.people.iter().map(|p| (p.name.as_str(), p.me, p.first_ms, p.last_ms)).collect();
        assert_eq!(people, vec![("Adnan", true, 5_000, 40_000), ("Priya", false, 5_000, 40_000), ("Marcus", false, 40_000, 40_000)]);
    }

    #[test]
    fn observer_uses_known_self_names_when_the_app_does_not_mark_you() {
        let mut o = Observer::new(Participants::default(), vec!["Adnan Ali".into()]);
        let teams = [e("adnan ali", None), e("Priya Raman", None)];
        o.observe(Some(("Teams", &teams)), 1_000);
        o.observe(Some(("Teams", &teams)), 20_000);
        assert!(o.record.people[0].me);
        assert_eq!(o.record.solo, vec![run("Priya Raman", 1_000, 20_000)]);
        // An explicit "not me" from the app wins over the remembered name.
        let mut o = Observer::new(Participants::default(), vec!["Priya".into()]);
        o.observe(Some(("Zoom", &[e("Priya", Some(false))])), 0);
        assert!(!o.record.people[0].me);
    }

    #[test]
    fn same_person_matches_case_and_unique_first_names() {
        let roster = ["Priya Raman", "Marcus Lee", "Marcus Chen"];
        assert!(same_person("priya raman", "Priya Raman", &roster));
        assert!(same_person("Priya", "Priya Raman", &roster));
        assert!(!same_person("Marcus", "Marcus Lee", &roster));
        assert!(!same_person("Priya R", "Priya Raman", &roster));
        assert!(!same_person("", "Priya Raman", &roster));
    }

    fn one_on_one() -> Participants {
        Participants { people: vec![person("Adnan", true), person("Priya", false)], solo: vec![run("Priya", 0, 60_000)] }
    }

    #[test]
    fn stable_one_on_one_names_the_single_remote_voice() {
        let spoken = talk("them:0", 1_000, 8);
        assert_eq!(decide(&one_on_one(), &spoken, &HashMap::new(), false), vec![named("them:0", "Priya")]);
        assert_eq!(decide(&one_on_one(), &spoken, &HashMap::new(), true), vec![named("them:0", "Priya")]);
    }

    #[test]
    fn one_on_one_needs_stability_and_enough_speech() {
        let mut p = one_on_one();
        p.solo = vec![run("Priya", 0, 10_000)];
        assert!(decide(&p, &talk("them:0", 1_000, 8), &HashMap::new(), false).is_empty());
        // Live needs 6 s inside the 1:1; the final pass accepts less.
        let few: Vec<(String, i64, i64)> =
            [(1_000, 2_500), (6_000, 7_500), (11_000, 12_500)].iter().map(|&(s, e)| ("them:0".into(), s, e)).collect();
        assert!(decide(&one_on_one(), &few, &HashMap::new(), false).is_empty());
        assert_eq!(decide(&one_on_one(), &few, &HashMap::new(), true), vec![named("them:0", "Priya")]);
    }

    #[test]
    fn two_voices_during_a_one_on_one_name_nobody() {
        // Two people share one account, or someone the roster never showed.
        let mut spoken = talk("them:0", 1_000, 6);
        spoken.extend(talk("them:1", 2_000, 6));
        assert!(decide(&one_on_one(), &spoken, &HashMap::new(), true).is_empty());
    }

    #[test]
    fn a_voice_mostly_heard_outside_the_one_on_one_is_not_named() {
        // Alice spoke before the roster was ever seen; Priya joined later.
        let mut p = one_on_one();
        p.solo = vec![run("Priya", 100_000, 130_000)];
        let mut spoken = talk("them:0", 0, 20);
        spoken.extend(talk("them:0", 100_000, 2));
        assert!(decide(&p, &spoken, &HashMap::new(), true).is_empty());
    }

    #[test]
    fn never_overwrites_user_or_voiceprint_names() {
        let spoken = talk("them:0", 1_000, 8);
        for src in [SOURCE_USER, SOURCE_VOICE] {
            let n = names(&[("them:0", "Priyanka", src)]);
            assert!(decide(&one_on_one(), &spoken, &n, false).is_empty(), "{src}");
            assert!(decide(&one_on_one(), &spoken, &n, true).is_empty(), "{src}");
        }
        // Someone else already called Priya by you: the name isn't reused.
        let mut spoken2 = spoken.clone();
        spoken2.extend(talk("manual:1", 2_000, 3));
        let n = names(&[("manual:1", "Priya", SOURCE_USER)]);
        assert!(decide(&one_on_one(), &spoken2, &n, true).is_empty());
    }

    fn group() -> Participants {
        Participants {
            people: vec![person("Adnan", true), person("Priya Raman", false), person("Marcus", false), person("Lee", false)],
            solo: vec![],
        }
    }

    fn three_voices() -> Vec<(String, i64, i64)> {
        let mut s = talk("them:0", 0, 5);
        s.extend(talk("them:1", 30_000, 5));
        s.extend(talk("them:2", 60_000, 5));
        s
    }

    #[test]
    fn elimination_names_the_last_voice() {
        let n = names(&[("them:0", "Priya", SOURCE_VOICE), ("them:2", "Lee", SOURCE_USER)]);
        assert_eq!(decide(&group(), &three_voices(), &n, true), vec![named("them:1", "Marcus")]);
    }

    #[test]
    fn elimination_refuses_when_counts_differ_or_two_are_open() {
        // Two open voices.
        let n = names(&[("them:0", "Priya", SOURCE_VOICE)]);
        assert!(decide(&group(), &three_voices(), &n, true).is_empty());
        // A fourth voice the roster doesn't explain.
        let mut four = three_voices();
        four.extend(talk("them:3", 90_000, 5));
        let n = names(&[("them:0", "Priya", SOURCE_VOICE), ("them:2", "Lee", SOURCE_USER), ("them:3", "Dana", SOURCE_USER)]);
        assert!(decide(&group(), &four, &n, true).is_empty());
        // Someone on the roster never spoke (two voices, three people).
        let two: Vec<_> = three_voices().into_iter().filter(|s| s.0 != "them:2").collect();
        let n = names(&[("them:0", "Priya", SOURCE_VOICE)]);
        assert!(decide(&group(), &two, &n, true).is_empty());
    }

    #[test]
    fn elimination_needs_two_people_and_ignores_phantom_voices() {
        // One person, one voice, no stable 1:1: that's a guess, not elimination.
        let p = Participants { people: vec![person("Adnan", true), person("Priya", false)], solo: vec![] };
        assert!(decide(&p, &talk("them:0", 0, 8), &HashMap::new(), true).is_empty());
        // A two-line phantom voice doesn't count as a speaker.
        let mut s = three_voices();
        s.extend(talk("them:9", 5_000, 2));
        let n = names(&[("them:0", "Priya", SOURCE_VOICE), ("them:2", "Lee", SOURCE_USER)]);
        assert_eq!(decide(&group(), &s, &n, true), vec![named("them:1", "Marcus")]);
    }

    #[test]
    fn final_pass_rederives_roster_names() {
        let spoken = talk("them:0", 1_000, 8);
        // Still supported: no change.
        let n = names(&[("them:0", "Priya", SOURCE_ROSTER)]);
        assert!(decide(&one_on_one(), &spoken, &n, true).is_empty());
        // Clusters merged differently: the old roster name is withdrawn.
        let n = names(&[("them:4", "Priya", SOURCE_ROSTER)]);
        assert_eq!(
            decide(&one_on_one(), &spoken, &n, true),
            vec![named("them:0", "Priya"), Change::Clear { speaker: "them:4".into() }]
        );
        // Live passes only add.
        assert_eq!(decide(&one_on_one(), &spoken, &n, false), vec![]);
    }

    #[test]
    fn prompt_names_put_the_call_before_invitees() {
        let p = Participants { people: vec![person("Adnan Ali", true), person("Priya Raman", false)], solo: vec![] };
        assert_eq!(
            prompt_names(&["Marcus".into(), "dana@example.com".into()], &p),
            vec!["Priya Raman", "Priya", "Adnan Ali", "Adnan", "Marcus"]
        );
    }

    #[test]
    fn stored_naming_respects_sources_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let db = store::open_path(&dir.path().join("t.db")).unwrap();
        let m: store::Meeting = serde_json::from_value(serde_json::json!({"id":"m","startedAt":"2026-11-20T10:00:00Z"})).unwrap();
        store::create_meeting(&db, &m).unwrap();
        let segs: Vec<store::Segment> = talk("them:0", 1_000, 8)
            .into_iter()
            .map(|(speaker, start_ms, end_ms)| store::Segment { id: 0, speaker, source: "system".into(), start_ms, end_ms, text: "hi".into() })
            .collect();
        store::add_segments(&db, "m", &segs).unwrap();
        // No roster yet: nothing happens.
        assert!(apply(&db, "m", false).unwrap().is_empty());
        store::set_participants(&db, "m", &one_on_one()).unwrap();
        assert_eq!(store::meeting_people(&db, "m").unwrap().1, one_on_one());
        assert_eq!(apply(&db, "m", false).unwrap(), vec![("them:0".into(), "Priya".into())]);
        assert_eq!(
            store::speaker_names_with_source(&db, "m").unwrap()["them:0"],
            ("Priya".to_string(), SOURCE_ROSTER.to_string())
        );
        // A rename by you becomes a user name, which naming never touches.
        store::set_speaker_name(&db, "m", "them:0", "Priya R.").unwrap();
        assert!(apply(&db, "m", true).unwrap().is_empty());
        assert_eq!(store::speaker_names_with_source(&db, "m").unwrap()["them:0"].1, SOURCE_USER);
        // A voiceprint name can't be overwritten by the roster either.
        assert!(!store::set_speaker_name_auto(&db, "m", "them:0", "X", SOURCE_ROSTER, &[SOURCE_ROSTER]).unwrap());
        assert!(!store::clear_speaker_name_from(&db, "m", "them:0", SOURCE_ROSTER).unwrap());
        // The roster's own names give way to a voiceprint.
        store::set_speaker_name(&db, "m", "them:0", "").unwrap();
        assert!(store::set_speaker_name_auto(&db, "m", "them:0", "Priya", SOURCE_ROSTER, &[]).unwrap());
        assert!(store::set_speaker_name_auto(&db, "m", "them:0", "Priyanka", SOURCE_VOICE, &[SOURCE_ROSTER]).unwrap());
        assert_eq!(store::get_meeting(&db, "m").unwrap().unwrap().speaker_names["them:0"], "Priyanka");
        // Merging speakers carries the source along.
        store::remap_speakers(&db, "m", &[("them:0".into(), "them:1".into())]).unwrap();
        assert_eq!(store::speaker_names_with_source(&db, "m").unwrap()["them:1"].1, SOURCE_VOICE);
    }
}
