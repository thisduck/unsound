import { useEffect, useState } from "react";
import { api, clock, Meeting, MeetingWork, SearchHit } from "../api";
import { Icon, Spinner, groupByDay, timeOfDay } from "../ui";
import { ResearchPanel } from "./ResearchPanel";
import { elapsed, snippetParts, speakerLabel } from "./format";

interface Props {
  meetings: Meeting[];
  onOpen: (id: string, atMs?: number | null) => void;
  llmId?: string;
  idle: boolean;
  /// The search box lives in the page header; results and answers show here.
  query: string;
  /// Bumped when the user presses Return to ask the query as a question.
  askToken: number;
  onClearQuery: () => void;
  work: MeetingWork[];
  /// Meetings whose summary is being written right now.
  writing: string[];
  onPause: (id: string) => void;
  onOpenSettings?: () => void;
}

/// Group transcript hits under their meeting.
function group(hits: SearchHit[]): { meetingId: string; title: string; startedAt: string; hits: SearchHit[] }[] {
  const out: { meetingId: string; title: string; startedAt: string; hits: SearchHit[] }[] = [];
  for (const h of hits) {
    let g = out.find((x) => x.meetingId === h.meetingId);
    if (!g) {
      g = { meetingId: h.meetingId, title: h.title, startedAt: h.startedAt, hits: [] };
      out.push(g);
    }
    g.hits.push(h);
  }
  return out;
}

export function ListView({ meetings, onOpen, llmId, idle, query, askToken, onClearQuery, work, writing, onPause, onOpenSettings }: Props) {
  const [results, setResults] = useState<SearchHit[] | null>(null);
  const [asking, setAsking] = useState<{ question: string; token: number } | null>(null);

  // Full-text search as you type (debounced). Empty query → the full list.
  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setResults(null);
      return;
    }
    let active = true;
    const t = setTimeout(() => {
      api.searchMeetings(q).then((rows) => { if (active) setResults(rows); }).catch((e) => console.error("search failed", e));
    }, 180);
    return () => { active = false; clearTimeout(t); };
  }, [query]);

  useEffect(() => {
    if (askToken && query.trim()) setAsking({ question: query.trim(), token: askToken });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [askToken]);

  const status = (m: Meeting) => {
    const w = work.find((x) => x.meetingId === m.id);
    if (w?.recording) return <span className="badge danger">Recording</span>;
    if (w && w.state === "finishing") {
      return (
        <>
          <Spinner />
          <span>Finishing transcript</span>
          <button className="btn btn-quiet btn-sm" onClick={(e) => { e.stopPropagation(); onPause(m.id); }}>Pause</button>
        </>
      );
    }
    if (w && (w.state === "paused" || w.state === "error")) return <span className="badge danger">Needs attention</span>;
    if (writing.includes(m.id)) return <><Spinner /><span>Writing summary…</span></>;
    const open = (m.actionItems ?? []).filter((i) => !i.done && !i.ignored).length;
    if (open) return <span className="badge">{open} open</span>;
    if (!m.summary && m.segmentCount > 0) return <span className="faint">No summary yet</span>;
    return null;
  };

  return (
    <div className="meet-list">
      {asking && (
        <div className="card answer-card">
          <div className="answer-head">
            <Icon name="sparkle" />
            <strong>Answer</strong>
            <span className="spacer" />
            <button className="btn btn-quiet btn-icon btn-sm" aria-label="Close answer" onClick={() => setAsking(null)}>
              <Icon name="x" />
            </button>
          </div>
          <ResearchPanel
            key={asking.token}
            modelId={llmId}
            initialQuestion={asking.question}
            autoAsk
            onSource={onOpen}
            onChooseModel={onOpenSettings}
          />
        </div>
      )}

      {results !== null ? (
        <section aria-label="Search results">
          <div className="section-label">
            {results.length ? `Meetings mentioning “${query.trim()}”` : `No meetings mention “${query.trim()}”.`}
            <span className="spacer" />
            <button className="btn btn-quiet btn-sm" onClick={onClearQuery}>Clear</button>
          </div>
          {group(results).map((g) => (
            <div className="card search-group" key={g.meetingId}>
              <button className="search-group-head meet-row-title" onClick={() => onOpen(g.meetingId)}>
                {g.title || "Untitled meeting"}
                <span className="faint">{new Date(g.startedAt).toLocaleDateString([], { month: "short", day: "numeric" })}</span>
              </button>
              {g.hits.map((h, i) => (
                <button
                  key={i}
                  className="search-hit"
                  onClick={() => onOpen(h.meetingId, h.startMs ?? null)}
                  title={h.startMs != null ? "Open at this moment" : "Open"}
                >
                  <span className="search-hit-meta">
                    {h.startMs != null && <span className="mono">{clock(h.startMs)}</span>}
                    {h.speaker && <span>{speakerLabel(h.speaker, h.speakerName ? { [h.speaker]: h.speakerName } : undefined)}</span>}
                  </span>
                  <span className="search-hit-text">
                    {snippetParts(h.snippet).map((part, j) => (part.hit ? <mark key={j}>{part.text}</mark> : <span key={j}>{part.text}</span>))}
                  </span>
                </button>
              ))}
            </div>
          ))}
          {results.length > 0 && (
            <p className="page-foot">Press Return to ask this as a question instead.</p>
          )}
        </section>
      ) : meetings.length === 0 ? (
        idle && (
          <div className="empty-state">
            <Icon name="meetings" />
            <h3>No meetings yet</h3>
            <p>Record your first call, or unsound will offer to when Zoom, Meet or Teams starts.</p>
          </div>
        )
      ) : (
        groupByDay(meetings, (m) => m.startedAt).map((g) => (
          <section key={g.day} aria-label={g.day}>
            <div className="section-label">{g.day}</div>
            <div className="meet-rows">
              {g.items.map((m) => (
                <div className="meet-row" key={m.id} role="button" tabIndex={0} onClick={() => onOpen(m.id)} onKeyDown={(e) => { if (e.key === "Enter") onOpen(m.id); }}>
                  <span className="meet-row-time mono">{timeOfDay(m.startedAt)}</span>
                  <span className="meet-row-main">
                    <span className="meet-row-title">{m.title || "Untitled meeting"}</span>
                    <span className="meet-row-sub">
                      {m.summary ? summaryLine(m.summary) : `${elapsed(m.startedAt, m.endedAt)}${m.attendees?.length ? ` · ${m.attendees.join(", ")}` : ""}`}
                    </span>
                  </span>
                  <span className="meet-row-status">{status(m)}</span>
                </div>
              ))}
            </div>
          </section>
        ))
      )}
    </div>
  );
}

/// The summary's opening sentence(s), without Markdown.
function summaryLine(summary: string): string {
  const text = summary
    .split("\n")
    .filter((l) => l.trim() && !l.trim().startsWith("#"))
    .join(" ")
    .replace(/[*_`]/g, "")
    .replace(/\s*[([](?:\d{1,2}:)?\d{1,2}:\d{2}[^)\]]*[)\]]/g, "");
  return text.length > 180 ? text.slice(0, 177).trimEnd() + "…" : text;
}
