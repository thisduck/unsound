//! Meeting notes, built incrementally.
//!
//! The transcript is cut into fixed five-minute stretches ("chunks"). Each is
//! run through the extraction prompt once and cached with a signature of the
//! exact text it saw — during the meeting, as soon as a stretch is complete,
//! so by the time the meeting ends almost everything is already extracted and
//! only the final compose remains (seconds, not minutes). Working in stretches
//! also keeps every part of a long meeting in view of a small model and never
//! overflows its context. Edits, re-detected speakers or a changed transcript
//! invalidate only the chunks whose text changed.
//!
//! Also here: grounded Q&A over one meeting (reusing a cached context so
//! follow-ups are instant) and across all meetings (full-text retrieval).

use crate::llm::{self, Gen, Slot};
use crate::store::{self, ActionItem, Chunk, Db, Meeting, Segment};
use crate::util::{clock, parse_clock};
use crate::AppState;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager};

/// Length of one extraction stretch. Fixed boundaries (not "since the last
/// chunk") keep cached chunks valid as the transcript grows.
pub const CHUNK_MS: i64 = 5 * 60 * 1000;
/// A stretch counts as complete this long after its end (late segments from
/// the transcription queue still land in it).
const SETTLE_MS: i64 = 30_000;

/// Serializes background LLM work (live extraction, summaries, assistant
/// drafts) so the jobs take turns on the GPU instead of thrashing it — and so
/// a summary never races the extraction of the chunk it needs.
pub static LLM_TURN: Mutex<()> = Mutex::new(());

/// How a speaker key reads to a person or a model: a name when one's set,
/// "Me" for the mic, "Speaker N" for a detected participant, else "Them".
pub fn speaker_label(speaker: &str, names: &HashMap<String, String>) -> String {
    if let Some(n) = names.get(speaker).filter(|n| !n.trim().is_empty()) {
        return n.trim().to_string();
    }
    default_label(speaker)
}

fn default_label(speaker: &str) -> String {
    if speaker == "me" {
        return "Me".into();
    }
    if let Some(n) = speaker.strip_prefix("them:").and_then(|n| n.parse::<i32>().ok()) {
        return format!("Speaker {}", n + 1);
    }
    "Them".into()
}

/// "[m:ss] Speaker: text" — the transcript line format every prompt uses.
pub fn render_line(s: &Segment, names: &HashMap<String, String>) -> String {
    format!("[{}] {}: {}", clock(s.start_ms), speaker_label(&s.speaker, names), s.text.trim())
}

/// Replace default speaker labels in model output with the names set since
/// ("Speaker 2" → "Priya"), without touching "Speaker 12".
pub fn apply_names(text: &str, names: &HashMap<String, String>) -> String {
    let mut out = text.to_string();
    // Longest labels first so "Speaker 12" is handled before "Speaker 1".
    let mut pairs: Vec<(String, String)> = names
        .iter()
        .filter(|(_, n)| !n.trim().is_empty())
        .map(|(k, n)| (default_label(k), n.trim().to_string()))
        .filter(|(label, _)| label != "Them")
        .collect();
    pairs.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    for (label, name) in pairs {
        let mut result = String::with_capacity(out.len());
        let mut rest = out.as_str();
        while let Some(i) = rest.find(&label) {
            let after = rest[i + label.len()..].chars().next();
            result.push_str(&rest[..i]);
            if after.is_some_and(|c| c.is_ascii_digit()) {
                result.push_str(&label);
            } else {
                result.push_str(&name);
            }
            rest = &rest[i + label.len()..];
        }
        result.push_str(rest);
        out = result;
    }
    out
}

/// Planned stretches: index → its segments (by start time).
fn plan(segments: &[Segment]) -> BTreeMap<i64, Vec<&Segment>> {
    let mut chunks: BTreeMap<i64, Vec<&Segment>> = BTreeMap::new();
    for s in segments.iter().filter(|s| !s.text.trim().is_empty()) {
        chunks.entry(s.start_ms / CHUNK_MS).or_default().push(s);
    }
    chunks
}

/// The text extraction sees for a stretch (default labels — names are applied
/// afterwards, so renaming someone doesn't force re-extraction).
fn chunk_text(segs: &[&Segment]) -> String {
    let none = HashMap::new();
    segs.iter().map(|s| render_line(s, &none)).collect::<Vec<_>>().join("\n")
}

fn signature(model_path: &Path, text: &str) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    model_path.to_string_lossy().hash(&mut h);
    llm::MEETING_EXTRACT_PROMPT.hash(&mut h);
    text.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Make sure every stretch (only those complete by `upto_ms`, when given) has
/// an up-to-date extraction; returns them in timeline order. Each extraction
/// takes the LLM turn separately, so a live meeting's other jobs interleave.
pub fn ensure_chunks(
    app: &AppHandle,
    meeting_id: &str,
    model_path: &Path,
    upto_ms: Option<i64>,
) -> Result<Vec<Chunk>, String> {
    let db = app.state::<Db>();
    let meeting = store::get_meeting(&db, meeting_id)?.ok_or("meeting not found")?;
    let cached: HashMap<i64, Chunk> = store::chunks(&db, meeting_id)?
        .into_iter()
        .map(|c| (c.idx, c))
        .collect();
    let planned = plan(&meeting.segments);
    let mut out = Vec::new();
    for (idx, segs) in &planned {
        let (start, end) = (idx * CHUNK_MS, (idx + 1) * CHUNK_MS);
        if upto_ms.is_some_and(|upto| end + SETTLE_MS > upto) {
            break;
        }
        let text = chunk_text(segs);
        let sig = signature(model_path, &text);
        if let Some(c) = cached.get(idx).filter(|c| c.signature == sig) {
            out.push(c.clone());
            continue;
        }
        let facts = {
            let _turn = LLM_TURN.lock().unwrap_or_else(|p| p.into_inner());
            // Someone else (the live notes thread) may have just done it.
            let fresh = store::chunks(&db, meeting_id)?
                .into_iter()
                .find(|c| c.idx == *idx && c.signature == sig);
            if let Some(c) = fresh {
                out.push(c);
                continue;
            }
            extract(app, model_path, start, end, &text)?
        };
        let chunk = Chunk {
            idx: *idx,
            start_ms: start,
            end_ms: end,
            signature: sig,
            facts,
        };
        store::save_chunk(&db, meeting_id, &chunk)?;
        let _ = app.emit(
            "meeting-notes-progress",
            serde_json::json!({ "meetingId": meeting_id, "throughMs": end }),
        );
        out.push(chunk);
    }
    // Stretches that no longer exist (e.g. transcript edited away).
    if upto_ms.is_none() {
        let last = planned.keys().next_back().copied().unwrap_or(-1);
        store::delete_chunks_from(&db, meeting_id, last + 1)?;
    }
    Ok(out)
}

fn extract(app: &AppHandle, model_path: &Path, start: i64, end: i64, text: &str) -> Result<String, String> {
    extract_with(&app.state::<AppState>().llm, model_path, start, end, text)
}

fn extract_with(llm_state: &llm::LlmState, model_path: &Path, start: i64, end: i64, text: &str) -> Result<String, String> {
    let messages = [
        ("system", llm::MEETING_EXTRACT_PROMPT.to_string()),
        (
            "user",
            format!("Excerpt ({}–{}):\n\n{text}", clock(start), clock(end)),
        ),
    ];
    let gen = Gen {
        max_tokens: 1200,
        ..Default::default()
    };
    let facts = llm::chat(llm_state, model_path, &messages, &gen, None, &mut |_| {})?;
    Ok(facts.trim().to_string())
}

fn is_empty_facts(facts: &str) -> bool {
    let t = facts.trim();
    t.is_empty() || t.contains("(nothing substantive)") && t.lines().count() <= 1
}

/// Everything compose and the action-item pass need, as prompt text.
fn material(meeting: &Meeting, chunks: &[Chunk]) -> String {
    let names = &meeting.speaker_names;
    let mut m = String::new();
    if !meeting.title.trim().is_empty() {
        m.push_str(&format!("Meeting: {}\n", meeting.title.trim()));
    }
    if !meeting.attendees.is_empty() {
        m.push_str(&format!("Invited: {}\n", meeting.attendees.join(", ")));
    }
    let named: Vec<String> = names
        .iter()
        .filter(|(_, n)| !n.trim().is_empty())
        .map(|(k, n)| format!("{} is {}", default_label(k), n.trim()))
        .collect();
    if !named.is_empty() {
        m.push_str(&format!("Speakers: {}\n", named.join("; ")));
    }
    if !meeting.notes.trim().is_empty() {
        m.push_str(&format!("\nMy own notes:\n{}\n", meeting.notes.trim()));
    }
    if !meeting.marks.is_empty() {
        m.push_str("\nMoments I flagged as important:\n");
        for mark in &meeting.marks {
            let label = if mark.label.trim().is_empty() { "(flagged)" } else { mark.label.trim() };
            m.push_str(&format!("- ({}) {label}\n", clock(mark.at_ms)));
        }
    }
    m.push_str("\nExtracted points:\n");
    for c in chunks.iter().filter(|c| !is_empty_facts(&c.facts)) {
        m.push_str(&format!(
            "\n### {}–{}\n{}\n",
            clock(c.start_ms),
            clock(c.end_ms),
            apply_names(&c.facts, names)
        ));
    }
    m
}

/// Summarize a meeting: bring extractions up to date (usually just the last
/// stretch), compose the notes (streamed on `meeting-summary-token`), extract
/// structured action items, and title an untitled meeting. Persists all of it.
pub fn summarize(app: &AppHandle, meeting_id: &str, model_path: &Path) -> Result<String, String> {
    let db = app.state::<Db>();
    let chunks = ensure_chunks(app, meeting_id, model_path, None)?;
    let meeting = store::get_meeting(&db, meeting_id)?.ok_or("meeting not found")?;
    if chunks.iter().all(|c| is_empty_facts(&c.facts)) && meeting.notes.trim().is_empty() {
        store::set_meeting_summary(&db, meeting_id, "")?;
        return Ok(String::new());
    }
    let material = material(&meeting, &chunks);
    let st = app.state::<AppState>();
    let _turn = LLM_TURN.lock().unwrap_or_else(|p| p.into_inner());

    let messages = [
        ("system", llm::MEETING_SUMMARY_PROMPT.to_string()),
        ("user", material.clone()),
    ];
    let emitter = app.clone();
    let summary = llm::chat(
        &st.llm,
        model_path,
        &messages,
        &Gen::default(),
        None,
        &mut |t| {
            let _ = emitter.emit("meeting-summary-token", t);
        },
    )?;
    store::set_meeting_summary(&db, meeting_id, &summary)?;

    // Structured follow-ups (grammar-constrained, so always well-formed),
    // from just the commitments the extraction tagged — a short, focused
    // input is faster and keeps small models from losing them in a long one.
    let commitments = commitments_material(&meeting, &chunks).unwrap_or_else(|| material.clone());
    match action_items(app, model_path, &commitments) {
        Ok(items) => {
            store::replace_auto_action_items(&db, meeting_id, &items)?;
        }
        Err(e) => eprintln!("[summary] action items failed: {e}"),
    }

    if meeting.title.trim().is_empty() {
        let messages = [
            ("system", llm::MEETING_TITLE_PROMPT.to_string()),
            ("user", summary.clone()),
        ];
        let gen = Gen {
            max_tokens: 24,
            ..Default::default()
        };
        if let Ok(title) = llm::chat(&st.llm, model_path, &messages, &gen, None, &mut |_| {}) {
            let title = title.lines().next().unwrap_or("").trim().trim_matches(['"', '\'', '.', '*']).trim();
            if !title.is_empty() {
                store::rename_meeting(&db, meeting_id, title)?;
            }
        }
    }
    Ok(summary)
}

/// The meeting header, the user's notes, and every fact line tagged
/// "Action:"; `None` when no chunk has tagged lines (extracted before tags).
fn commitments_material(meeting: &Meeting, chunks: &[Chunk]) -> Option<String> {
    let lines: Vec<String> = chunks
        .iter()
        .flat_map(|c| c.facts.lines())
        .filter(|l| l.to_lowercase().contains("action:"))
        .map(|l| apply_names(l.trim(), &meeting.speaker_names))
        .collect();
    let tagged_any = chunks.iter().any(|c| c.facts.to_lowercase().contains("action:"));
    if !tagged_any && meeting.notes.trim().is_empty() {
        return None;
    }
    let mut m = String::new();
    let named: Vec<String> = meeting
        .speaker_names
        .iter()
        .filter(|(_, n)| !n.trim().is_empty())
        .map(|(k, n)| format!("{} is {}", default_label(k), n.trim()))
        .collect();
    if !named.is_empty() {
        m.push_str(&format!("Speakers: {}\n", named.join("; ")));
    }
    if !meeting.notes.trim().is_empty() {
        m.push_str(&format!("\nMy own notes:\n{}\n", meeting.notes.trim()));
    }
    m.push_str("\nCommitments from the meeting:\n");
    if lines.is_empty() {
        m.push_str("(none tagged)\n");
    }
    for l in lines {
        m.push_str(&l);
        m.push('\n');
    }
    Some(m)
}

fn action_items(app: &AppHandle, model_path: &Path, material: &str) -> Result<Vec<ActionItem>, String> {
    action_items_with(&app.state::<AppState>().llm, model_path, material)
}

fn action_items_with(llm_state: &llm::LlmState, model_path: &Path, material: &str) -> Result<Vec<ActionItem>, String> {
    let messages = [
        ("system", llm::MEETING_ACTIONS_PROMPT.to_string()),
        ("user", material.to_string()),
    ];
    let v = llm::chat_json(llm_state, model_path, &messages, &llm::actions_schema(), 2000)?;
    Ok(parse_action_items(&v))
}

pub fn parse_action_items(v: &serde_json::Value) -> Vec<ActionItem> {
    let s = |x: &serde_json::Value, k: &str| x[k].as_str().unwrap_or("").trim().to_string();
    v["items"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter(|it| !s(it, "task").is_empty())
                .map(|it| ActionItem {
                    id: 0,
                    meeting_id: None,
                    text: s(it, "task"),
                    owner: s(it, "owner"),
                    due: s(it, "due"),
                    at_ms: parse_clock(s(it, "at").trim_matches(['(', ')'])),
                    done: false,
                    manual: false,
                    created_at: String::new(),
                    meeting_title: String::new(),
                })
                .collect()
        })
        .unwrap_or_default()
}

// ---- Q&A --------------------------------------------------------------------

/// Words that carry no search signal in a question.
const STOPWORDS: &[&str] = &[
    "a", "about", "all", "an", "and", "any", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "get", "got", "had", "has", "have", "he", "her",
    "him", "his", "how", "i", "if", "in", "is", "it", "its", "last", "me", "meeting", "meetings",
    "my", "of", "on", "or", "our", "out", "said", "say", "she", "so", "that", "the", "their",
    "them", "then", "there", "they", "this", "to", "told", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you", "your", "talk",
    "talked", "discuss", "discussed", "mention", "mentioned", "tell", "anything", "something",
];

fn keywords(question: &str) -> String {
    question
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .map(|w| w.trim_matches('\'').to_lowercase())
        .filter(|w| w.len() > 1 && !STOPWORDS.contains(&w.as_str()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Answer a question about one meeting. When the whole transcript fits the
/// model's budget it goes in verbatim (on the Q&A slot: the material stays
/// processed, so follow-ups only pay for the question); otherwise the notes,
/// summary and every stretch's extracted points go in, plus the transcript
/// lines that best match the question.
pub fn ask(app: &AppHandle, meeting_id: &str, model_path: &Path, question: &str) -> Result<String, String> {
    let db = app.state::<Db>();
    let st = app.state::<AppState>();
    let meeting = store::get_meeting(&db, meeting_id)?.ok_or("meeting not found")?;
    let names = &meeting.speaker_names;

    let mut head = String::new();
    if !meeting.notes.trim().is_empty() {
        head.push_str(&format!("My notes:\n{}\n\n", meeting.notes.trim()));
    }
    if !meeting.summary.trim().is_empty() {
        head.push_str(&format!("Summary:\n{}\n\n", meeting.summary.trim()));
    }
    let transcript: String = meeting
        .segments
        .iter()
        .map(|s| render_line(s, names))
        .collect::<Vec<_>>()
        .join("\n");
    let full = format!("{head}Transcript:\n{transcript}");
    let budget = llm::prompt_budget(&st.llm, model_path)?;
    let fits = llm::count_text_tokens(&st.llm, model_path, &full)? + 600 < budget;

    let (material, slot) = if fits {
        (full, Some(Slot::Qa))
    } else {
        let mut m = head;
        let chunks = store::chunks(&db, meeting_id)?;
        if !chunks.is_empty() {
            m.push_str("Points from each part of the meeting:\n");
            for c in chunks.iter().filter(|c| !is_empty_facts(&c.facts)) {
                m.push_str(&format!(
                    "### {}–{}\n{}\n",
                    clock(c.start_ms),
                    clock(c.end_ms),
                    apply_names(&c.facts, names)
                ));
            }
        }
        m.push_str("\nTranscript lines most related to the question:\n");
        m.push_str(&related_lines(&db, &meeting, question, 40)?);
        (m, None)
    };
    let messages = [
        ("system", llm::MEETING_QA_PROMPT.to_string()),
        ("user", format!("{material}\n\nQuestion: {question}")),
    ];
    let emitter = app.clone();
    let gen = Gen {
        max_tokens: 1200,
        ..Default::default()
    };
    let _turn = LLM_TURN.lock().unwrap_or_else(|p| p.into_inner());
    llm::chat(&st.llm, model_path, &messages, &gen, slot, &mut |t| {
        let _ = emitter.emit("meeting-answer-token", t);
    })
}

/// Transcript lines matching the question's keywords, each with one line of
/// context either side, merged and in timeline order.
fn related_lines(db: &Db, meeting: &Meeting, question: &str, limit: i64) -> Result<String, String> {
    let Some(q) = store::fts_query(&keywords(question), true) else {
        return Ok(String::new());
    };
    let hits = store::search_segment_ids(db, &q, Some(&meeting.id), limit)?;
    let index: HashMap<i64, usize> = meeting.segments.iter().enumerate().map(|(i, s)| (s.id, i)).collect();
    let mut keep = std::collections::BTreeSet::new();
    for (_, id, _) in hits {
        if let Some(&i) = index.get(&id) {
            for j in i.saturating_sub(1)..=(i + 1).min(meeting.segments.len().saturating_sub(1)) {
                keep.insert(j);
            }
        }
    }
    Ok(keep
        .into_iter()
        .map(|i| render_line(&meeting.segments[i], &meeting.speaker_names))
        .collect::<Vec<_>>()
        .join("\n"))
}

/// Answer a question across all meetings from full-text retrieval: the best
/// matching lines (with context) from the most relevant meetings, plus each
/// meeting's summary. Streams on `meetings-answer-token`.
pub fn ask_all(app: &AppHandle, model_path: &Path, question: &str) -> Result<String, String> {
    let db = app.state::<Db>();
    let st = app.state::<AppState>();
    let Some(q) = store::fts_query(&keywords(question), true) else {
        return Ok("Ask about something specific — a topic, a person, a decision.".into());
    };
    let hits = store::search_segment_ids(&db, &q, None, 120)?;
    // Rank meetings by how many strong hits they have (hits come best-first).
    let mut order: Vec<String> = Vec::new();
    let mut count: HashMap<String, usize> = HashMap::new();
    for (mid, _, _) in &hits {
        if !count.contains_key(mid) {
            order.push(mid.clone());
        }
        *count.entry(mid.clone()).or_default() += 1;
    }
    // Meetings whose summary/notes/title match but no transcript line does.
    for h in store::search_meetings(&db, &keywords(question))? {
        if h.kind == "meeting" && !count.contains_key(&h.meeting_id) {
            order.push(h.meeting_id.clone());
            count.insert(h.meeting_id, 0);
        }
    }
    order.truncate(6);
    if order.is_empty() {
        return Ok("I couldn't find anything about that in your meetings.".into());
    }
    let budget = llm::prompt_budget(&st.llm, model_path)?.min(24_000);
    let mut material = String::new();
    for mid in &order {
        let Some(m) = store::get_meeting(&db, mid)? else { continue };
        let date = m.started_at.get(..10).unwrap_or("");
        let title = if m.title.trim().is_empty() { "Untitled meeting" } else { m.title.trim() };
        let mut part = format!("\n=== {title} ({date}) ===\n");
        if !m.summary.trim().is_empty() {
            let s: String = m.summary.chars().take(1500).collect();
            part.push_str(&format!("Summary:\n{s}\n"));
        }
        let lines = related_lines(&db, &m, question, 12)?;
        if !lines.is_empty() {
            part.push_str(&format!("Matching transcript lines:\n{lines}\n"));
        }
        let total = llm::count_text_tokens(&st.llm, model_path, &format!("{material}{part}"))?;
        if total > budget {
            break;
        }
        material.push_str(&part);
    }
    let messages = [
        ("system", llm::MEETINGS_QA_PROMPT.to_string()),
        ("user", format!("{material}\n\nQuestion: {question}")),
    ];
    let emitter = app.clone();
    let gen = Gen {
        max_tokens: 1200,
        ..Default::default()
    };
    let _turn = LLM_TURN.lock().unwrap_or_else(|p| p.into_inner());
    llm::chat(&st.llm, model_path, &messages, &gen, None, &mut |t| {
        let _ = emitter.emit("meetings-answer-token", t);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start_ms: i64, speaker: &str, text: &str) -> Segment {
        Segment {
            id: start_ms,
            speaker: speaker.into(),
            source: if speaker == "me" { "mic" } else { "system" }.into(),
            start_ms,
            end_ms: start_ms + 2000,
            text: text.into(),
        }
    }

    #[test]
    fn plans_fixed_stretches() {
        let segs = vec![seg(10_000, "me", "a"), seg(299_000, "them:0", "b"), seg(301_000, "me", "c")];
        let p = plan(&segs);
        assert_eq!(p.len(), 2);
        assert_eq!(p[&0].len(), 2);
        assert_eq!(p[&1].len(), 1);
    }

    #[test]
    fn names_replace_default_labels_only() {
        let mut names = HashMap::new();
        names.insert("them:0".to_string(), "Priya".to_string());
        assert_eq!(
            apply_names("Speaker 1 agreed; Speaker 12 didn't (1:05)", &names),
            "Priya agreed; Speaker 12 didn't (1:05)"
        );
        assert_eq!(render_line(&seg(65_000, "them:0", " hi "), &names), "[1:05] Priya: hi");
        assert_eq!(render_line(&seg(0, "me", "yo"), &HashMap::new()), "[0:00] Me: yo");
    }

    #[test]
    fn keywords_drop_filler() {
        assert_eq!(keywords("What did we decide about the Q3 pricing?"), "decide q3 pricing");
    }

    /// A short synthetic product meeting (no real data) for eyeballing
    /// extraction, notes and action items on a real model.
    fn synthetic() -> Vec<Segment> {
        let lines: &[(i64, &str, &str)] = &[
            (5, "me", "Okay, let's start. Two things today: the login bug and the pricing page."),
            (12, "them:0", "Right. So the login bug: users on Safari get logged out every time they refresh. We traced it to the session cookie missing SameSite equals None."),
            (25, "them:1", "I can take that. It's a one-line fix but we need to test it on Safari 17 and 18."),
            (33, "me", "Great, Marcus, can you have it in by Thursday?"),
            (37, "them:1", "Thursday works."),
            (44, "them:0", "Also support has fourteen tickets about it this week, so we should post an update once it's fixed."),
            (52, "me", "I'll write the support update after the fix ships."),
            (70, "them:0", "Pricing page. Marketing wants to move the enterprise tier to contact-sales only and drop the public price."),
            (82, "them:1", "I'm not sure. Self-serve enterprise signups were about thirty percent of enterprise revenue last quarter."),
            (95, "me", "Let's not decide today. Priya, can you pull the numbers for the last two quarters so we can decide next week?"),
            (104, "them:0", "Yes, I'll have them by Monday."),
            (112, "me", "And we're still targeting March twelfth for the launch?"),
            (118, "them:0", "March twelfth, yes, as long as the Safari fix lands this week."),
        ];
        lines
            .iter()
            .enumerate()
            .map(|(i, (sec, sp, t))| Segment {
                id: i as i64 + 1,
                speaker: sp.to_string(),
                source: if *sp == "me" { "mic" } else { "system" }.into(),
                start_ms: sec * 1000,
                end_ms: sec * 1000 + 6000,
                text: t.to_string(),
            })
            .collect()
    }

    /// `LLM_MODEL=/path/model.gguf cargo test --release --lib -- --ignored synthetic_notes --nocapture`
    #[test]
    #[ignore]
    fn synthetic_notes_on_a_real_model() {
        let model = std::path::PathBuf::from(std::env::var("LLM_MODEL").expect("set LLM_MODEL"));
        let state = llm::LlmState::default();
        let segs = synthetic();
        let mut names = HashMap::new();
        names.insert("them:0".to_string(), "Priya".to_string());
        names.insert("them:1".to_string(), "Marcus".to_string());
        let t = std::time::Instant::now();
        let text = chunk_text(&plan(&segs)[&0]);
        let facts = extract_with(&state, &model, 0, CHUNK_MS, &text).unwrap();
        println!("--- facts ({:.1}s)\n{facts}", t.elapsed().as_secs_f32());
        let meeting = Meeting {
            id: "x".into(),
            title: String::new(),
            started_at: String::new(),
            ended_at: None,
            summary: String::new(),
            notes: "- safari logout bug\n- enterprise pricing?".into(),
            stt_model: String::new(),
            llm_model: String::new(),
            lang: None,
            segments: segs.clone(),
            segment_count: 0,
            speaker_names: names,
            attendees: vec![],
            marks: vec![],
            action_items: vec![],
            has_audio: false,
        };
        let chunk = Chunk { idx: 0, start_ms: 0, end_ms: CHUNK_MS, signature: String::new(), facts };
        let material = material(&meeting, &[chunk]);
        let t = std::time::Instant::now();
        let summary = llm::chat(
            &state,
            &model,
            &[("system", llm::MEETING_SUMMARY_PROMPT.to_string()), ("user", material.clone())],
            &Gen::default(),
            None,
            &mut |_| {},
        )
        .unwrap();
        println!("--- summary ({:.1}s)\n{summary}", t.elapsed().as_secs_f32());
        let t = std::time::Instant::now();
        let commitments = commitments_material(&meeting, &[Chunk { idx: 0, start_ms: 0, end_ms: CHUNK_MS, signature: String::new(), facts: material.clone() }])
            .unwrap_or_else(|| material.clone());
        let items = action_items_with(&state, &model, &commitments).unwrap();
        println!("--- action items ({:.1}s)", t.elapsed().as_secs_f32());
        for i in &items {
            println!("[ ] {} — owner: {:?}, due: {:?}, at: {:?}", i.text, i.owner, i.due, i.at_ms.map(clock));
        }
        assert!(!items.is_empty());
        assert!(summary.contains("##"));
    }

    /// Before/after timing on a real recorded meeting — numbers only, no
    /// content printed. `MEETING_DB=/copy/of/unsound.db MEETING_ID=… LLM_MODEL=…`
    #[test]
    #[ignore]
    fn real_meeting_timing() {
        let model = std::path::PathBuf::from(std::env::var("LLM_MODEL").expect("set LLM_MODEL"));
        let db = rusqlite::Connection::open(std::env::var("MEETING_DB").expect("set MEETING_DB")).unwrap();
        let id = std::env::var("MEETING_ID").expect("set MEETING_ID");
        let mut stmt = db
            .prepare("SELECT id, speaker, source, start_ms, end_ms, text FROM segments WHERE meeting_id = ?1 ORDER BY start_ms, id")
            .unwrap();
        let segs: Vec<Segment> = stmt
            .query_map([&id], |r| {
                Ok(Segment { id: r.get(0)?, speaker: r.get(1)?, source: r.get(2)?, start_ms: r.get(3)?, end_ms: r.get(4)?, text: r.get(5)? })
            })
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let state = llm::LlmState::default();
        let none = HashMap::new();
        state_warm(&state, &model);

        // Old: one pass over the whole transcript, then compose — all after stop.
        let transcript: String = segs.iter().map(|s| format!("{}: {}\n", speaker_label(&s.speaker, &none), s.text)).collect();
        let t = std::time::Instant::now();
        let old_facts = llm::chat(
            &state,
            &model,
            &[("system", llm::MEETING_EXTRACT_PROMPT.to_string()), ("user", format!("Transcript:\n\n{transcript}"))],
            &Gen::default(),
            None,
            &mut |_| {},
        )
        .unwrap();
        let old_extract = t.elapsed().as_secs_f32();
        let t = std::time::Instant::now();
        let _ = llm::chat(
            &state,
            &model,
            &[("system", llm::MEETING_SUMMARY_PROMPT.to_string()), ("user", format!("Extracted points:\n\n{old_facts}"))],
            &Gen::default(),
            None,
            &mut |_| {},
        )
        .unwrap();
        let old_compose = t.elapsed().as_secs_f32();
        println!("OLD after-stop: extract {old_extract:.1}s + compose {old_compose:.1}s = {:.1}s ({} fact lines from {} transcript tokens)",
            old_extract + old_compose, old_facts.lines().count(), llm::count_text_tokens(&state, &model, &transcript).unwrap());

        // New: per-stretch extraction (during the meeting), compose at stop.
        let planned = plan(&segs);
        let mut chunks = Vec::new();
        let t = std::time::Instant::now();
        let mut per = Vec::new();
        for (idx, ss) in &planned {
            let tc = std::time::Instant::now();
            let facts = extract_with(&state, &model, idx * CHUNK_MS, (idx + 1) * CHUNK_MS, &chunk_text(ss)).unwrap();
            per.push(tc.elapsed().as_secs_f32());
            chunks.push(Chunk { idx: *idx, start_ms: idx * CHUNK_MS, end_ms: (idx + 1) * CHUNK_MS, signature: String::new(), facts });
        }
        let live = t.elapsed().as_secs_f32();
        let meeting = Meeting {
            id,
            title: String::new(),
            started_at: String::new(),
            ended_at: None,
            summary: String::new(),
            notes: String::new(),
            stt_model: String::new(),
            llm_model: String::new(),
            lang: None,
            segments: vec![],
            segment_count: 0,
            speaker_names: none.clone(),
            attendees: vec![],
            marks: vec![],
            action_items: vec![],
            has_audio: false,
        };
        let material = material(&meeting, &chunks);
        let last = per.last().copied().unwrap_or(0.0);
        let t = std::time::Instant::now();
        let summary = llm::chat(&state, &model, &[("system", llm::MEETING_SUMMARY_PROMPT.to_string()), ("user", material.clone())], &Gen::default(), None, &mut |_| {}).unwrap();
        let compose = t.elapsed().as_secs_f32();
        let lines: Vec<&str> = summary.lines().filter(|l| !l.trim().is_empty()).collect();
        let unique: std::collections::HashSet<&str> = lines.iter().map(|l| l.trim()).collect();
        println!("NEW summary: {} tokens, {} lines ({} unique), {} bullets",
            llm::count_text_tokens(&state, &model, &summary).unwrap(), lines.len(), unique.len(),
            lines.iter().filter(|l| l.trim_start().starts_with(['*', '-'])).count());
        let t = std::time::Instant::now();
        let commitments = commitments_material(&meeting, &chunks).unwrap_or_else(|| material.clone());
        let items = action_items_with(&state, &model, &commitments).unwrap();
        let actions = t.elapsed().as_secs_f32();
        let distinct: std::collections::HashSet<String> = items.iter().map(|i| i.text.to_lowercase()).collect();
        println!("NEW items: {} ({} distinct texts), commitments input {} tokens",
            items.len(), distinct.len(), llm::count_text_tokens(&state, &model, &commitments).unwrap());
        let fact_lines: usize = chunks.iter().map(|c| c.facts.lines().count()).sum();
        println!("NEW during meeting: {} stretches, {live:.1}s total ({:.1}s max each)", chunks.len(), per.iter().cloned().fold(0.0, f32::max));
        println!("NEW after-stop: last stretch {last:.1}s + compose {compose:.1}s + action items {actions:.1}s = {:.1}s ({fact_lines} fact lines, {} action items, material {} tokens)",
            last + compose + actions, items.len(), llm::count_text_tokens(&state, &model, &material).unwrap());
    }

    fn state_warm(state: &llm::LlmState, model: &Path) {
        let _ = llm::chat(state, model, &[("user", "Say OK.".to_string())], &Gen { max_tokens: 4, ..Default::default() }, None, &mut |_| {});
    }

    #[test]
    fn parses_action_items() {
        let v = serde_json::json!({"items": [
            {"task": "Send the deck", "owner": "Me", "due": "Friday", "at": "(12:34)"},
            {"task": "  ", "owner": "", "due": "", "at": ""}
        ]});
        let items = parse_action_items(&v);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].at_ms, Some(754_000));
        assert_eq!(items[0].owner, "Me");
    }
}
