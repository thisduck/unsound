//! Meeting notes, built incrementally.
//!
//! The transcript is cut into fixed five-minute stretches ("chunks"). Each is
//! run through the extraction prompt once and cached with a signature of the
//! exact text it saw — during the meeting, as soon as a stretch is complete,
//! so by the time the meeting ends almost everything is already extracted and
//! only the final compose remains. Working in stretches reduces extraction
//! context size. Edits, re-detected speakers or a changed transcript
//! invalidate only the chunks whose text changed.
//!
//! Read-only archive Q&A is implemented in `meeting_research`.

use crate::llm::{self, Gen};
use crate::store::{self, Chunk, Db, Segment};
use crate::util::clock;
use crate::{AppState, notes_quality};
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
    if speaker.starts_with("manual:") { return format!("Participant {}", speaker.trim_start_matches("manual:")); }
    "Them".into()
}

/// "[m:ss] Speaker: text" — the transcript line format every prompt uses.
pub fn render_line(s: &Segment, names: &HashMap<String, String>) -> String {
    format!("[{}] {}: {}", clock(s.start_ms), speaker_label(&s.speaker, names), s.text.trim())
}

/// Replace default speaker labels in model output with the names set since
/// ("Speaker 2" → "Priya"), without touching "Speaker 12".
pub fn apply_names(text: &str, names: &HashMap<String, String>) -> String {
    let mut pairs: Vec<(String, String)> = names.iter()
        .filter(|(_, n)| !n.trim().is_empty())
        .map(|(k,n)| (default_label(k), n.trim().to_string()))
        .filter(|(label,_)| label != "Them").collect();
    pairs.sort_by(|a,b| b.0.len().cmp(&a.0.len()));
    let mut out = String::new();
    let mut pos = 0;
    while pos < text.len() {
        let before = text[..pos].chars().next_back();
        let matched = pairs.iter().find(|(label,_)| text[pos..].starts_with(label)
            && !before.is_some_and(|c| c.is_alphanumeric() || c == '_')
            && !text[pos+label.len()..].chars().next().is_some_and(|c| c.is_alphanumeric() || c == '_'));
        if let Some((label,name)) = matched { out.push_str(name); pos += label.len(); }
        else { let c = text[pos..].chars().next().unwrap(); out.push(c); pos += c.len_utf8(); }
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
    segs.iter().map(|s| format!("[S{}] {}", s.id, render_line(s, &none))).collect::<Vec<_>>().join("\n")
}

fn signature(model_path: &Path, text: &str) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    model_path.to_string_lossy().hash(&mut h);
    notes_quality::PROMPT.hash(&mut h);
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
            extract(app, model_path, segs)?
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

fn extract(app: &AppHandle, model_path: &Path, segments: &[&Segment]) -> Result<String, String> {
    extract_with(&app.state::<AppState>().llm, model_path, segments)
}

fn extract_with(llm_state: &llm::LlmState, model_path: &Path, segments: &[&Segment]) -> Result<String, String> {
    let messages = [
        ("system", notes_quality::PROMPT.to_string()),
        ("user", chunk_text(segments)),
    ];
    let mut messages=messages.to_vec();
    for attempt in 0..2 {
        let value=llm::chat_json(llm_state, model_path, &messages, &notes_quality::schema(segments), 4000)?;
        match notes_quality::validate(value.clone(),segments) {
            Ok(facts)=>return serde_json::to_string(&facts).map_err(|e|e.to_string()),
            Err(error) if attempt==0 => {
                messages.push(("assistant",value.to_string()));
                messages.push(("user",format!("Evidence validation failed: {error} Return a corrected complete object. Copy exact contiguous words from each cited source; do not paraphrase inside quotations. Remove facts you cannot support.")));
            }
            Err(error)=>return Err(error),
        }
    }
    unreachable!()
}

fn is_empty_facts(facts: &str) -> bool {
    notes_quality::decode(facts).facts.is_empty()
}

/// Summarize a meeting: bring extractions up to date (usually just the last
/// stretch), compose and validate notes and action items, then save them together.
/// Publishes validated text on `meeting-summary-token` and titles untitled meetings.
pub fn summarize(app: &AppHandle, meeting_id: &str, model_path: &Path) -> Result<String, String> {
    let db = app.state::<Db>();
    if crate::meeting_work::statuses(&db)?.iter().any(|w| w.meeting_id==meeting_id && (w.pending>0 || w.state=="recording")) {
        return Err("The transcript is still incomplete. Finish or retry saved audio before writing notes.".into());
    }
    let before=store::get_meeting(&db,meeting_id)?.ok_or("meeting not found")?.content_revision;
    let chunks = ensure_chunks(app, meeting_id, model_path, None)?;
    let meeting = store::get_meeting(&db, meeting_id)?.ok_or("meeting not found")?;
    if meeting.content_revision!=before { return Err("The transcript, speakers, or notes changed during extraction. Generate notes again.".into()); }
    if chunks.iter().all(|c| is_empty_facts(&c.facts)) && meeting.notes.trim().is_empty() {
        store::save_generated_notes(&db, meeting_id, "", &[],Some(meeting.content_revision))?;
        return Ok(String::new());
    }
    let facts=notes_quality::Facts { facts:chunks.iter().flat_map(|c|notes_quality::decode(&c.facts).facts).collect() };
    let st = app.state::<AppState>();
    let _turn = LLM_TURN.lock().unwrap_or_else(|p| p.into_inner());
    let (summary,items)=if facts.facts.is_empty() {
        (format!("## Your notes\n\n{}",meeting.notes.trim()),Vec::new())
    } else {
        let messages=[("system",notes_quality::COMPOSE_PROMPT.to_string()),("user",notes_quality::compose_material(&facts,&meeting))];
        let value=llm::chat_json(&st.llm,model_path,&messages,&notes_quality::compose_schema(&facts),3500)?;
        notes_quality::compose(value,&facts,&meeting)?
    };
    // Publish only after all references have been checked; incomplete JSON or
    // invalid model evidence must never replace previously saved notes.
    store::save_generated_notes(&db,meeting_id,&summary,&items,Some(meeting.content_revision))?;
    let _=app.emit("meeting-summary-token",&summary);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Meeting;

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
    fn names_do_not_replace_word_fragments_or_rewrite_other_names() {
        let names=HashMap::from([("me".into(),"Adnan".into()),("them:0".into(),"Speaker 2".into()),("them:1".into(),"Priya".into())]);
        assert_eq!(apply_names("Meeting: Me met Speaker 1 and Speaker 2. Somewhere else.",&names),"Meeting: Adnan met Speaker 2 and Priya. Somewhere else.");
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
        let facts = extract_with(&state, &model, &plan(&segs)[&0]).unwrap();
        println!("--- facts ({:.1}s)\n{facts}", t.elapsed().as_secs_f32());
        let meeting = Meeting {
            id: "x".into(),
            title: String::new(),
            started_at: String::new(),
            ended_at: None,
            summary: String::new(),
            content_revision:0, summary_stale:false,
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
        let facts=notes_quality::decode(&facts);
        let value=llm::chat_json(&state,&model,&[("system",notes_quality::COMPOSE_PROMPT.to_string()),("user",notes_quality::compose_material(&facts,&meeting))],&notes_quality::compose_schema(&facts),3500).unwrap();
        let (summary,items)=notes_quality::compose(value,&facts,&meeting).unwrap();
        println!("--- summary\n{summary}");
        assert!(summary.contains("Safari") || summary.contains("login"));
        assert!(items.iter().any(|i|i.owner=="Marcus" && i.due.contains("Thursday")),"accepted fix and deadline must survive: {:?}",serde_json::to_value(&items));
        assert!(items.iter().any(|i|i.owner=="Priya" && i.due.contains("Monday")),"accepted analysis request must survive");
        assert!(items.iter().all(|i|i.at_ms.is_some_and(|at|meeting.segments.iter().any(|s|s.start_ms==at))));
        // The same real model must not promote speculation into a commitment.
        let speculative=vec![seg(0,"me","Maybe we could move the launch to Friday, but this is only an idea. Nobody is assigned and there is no decision or deadline today.")];
        let facts=notes_quality::decode(&extract_with(&state,&model,&speculative.iter().collect::<Vec<_>>()).unwrap());
        assert!(facts.facts.iter().all(|f|f.kind!="action" && f.kind!="decision"),"speculation became a decision or action");
        // Simulate separate extraction windows: a later cancellation must remove
        // an earlier commitment from the final follow-up list.
        let mut canceled:Meeting=serde_json::from_value(serde_json::json!({"id":"cancel","startedAt":"2026-10-14","segments":[
            {"id":1,"speaker":"me","startMs":0,"text":"I will send the draft tomorrow."},
            {"id":2,"speaker":"me","startMs":600000,"text":"Cancel my plan to send the draft. We no longer need it; do not send it."}]})).unwrap();
        canceled.notes=String::new();
        let facts=notes_quality::Facts{facts:vec![
            notes_quality::Fact{kind:"action".into(),text:"Send the draft tomorrow".into(),owner:"Me".into(),due:"tomorrow".into(),evidence:vec![notes_quality::Evidence{segment_id:1,quote:canceled.segments[0].text.clone()}]},
            notes_quality::Fact{kind:"decision".into(),text:"The earlier plan to send the draft is canceled; do not send it".into(),owner:String::new(),due:String::new(),evidence:vec![notes_quality::Evidence{segment_id:2,quote:canceled.segments[1].text.clone()}]}
        ]};
        let value=llm::chat_json(&state,&model,&[("system",notes_quality::COMPOSE_PROMPT.to_string()),("user",notes_quality::compose_material(&facts,&canceled))],&notes_quality::compose_schema(&facts),2000).unwrap();
        let (_,items)=notes_quality::compose(value,&facts,&canceled).unwrap();
        assert!(items.is_empty(),"canceled commitment survived composition");
    }


}
