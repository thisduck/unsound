import { useEffect, useRef, useState } from "react";
import { ActionItem, api, clock, Meeting, ModelInfo, on, SearchHit } from "../api";
import { Markdown } from "../Markdown";
import { elapsed, snippetParts, speakerLabel } from "./format";

interface Props {
  meetings: Meeting[];
  onOpen: (id: string, atMs?: number | null) => void;
  llmId?: string;
  options: {
    sttModels: ModelInfo[];
    llmModels: ModelInfo[];
    embModels: ModelInfo[];
    stt?: string;
    llm?: string;
    emb?: string;
    speakers: string;
    set: (key: "stt" | "llm" | "emb" | "speakers", value: string) => void;
  };
  idle: boolean;
}

/// Group transcript hits under their meeting, meeting-level hits after.
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

export function ListView({ meetings, onOpen, llmId, options, idle }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchHit[] | null>(null);
  const [answer, setAnswer] = useState("");
  const [asking, setAsking] = useState(false);
  const askingRef = useRef(false);
  const [items, setItems] = useState<ActionItem[]>([]);
  const [showItems, setShowItems] = useState(false);

  useEffect(() => {
    api.openActionItems().then(setItems).catch(() => {});
    const sub = on.meetingsAnswerToken((chunk) => {
      if (askingRef.current) setAnswer((a) => a + chunk);
    });
    return () => {
      sub.then((un) => un());
    };
  }, [meetings]);

  // Full-text search as you type (debounced). Empty query → the full list.
  useEffect(() => {
    const q = query.trim();
    setAnswer("");
    if (!q) {
      setResults(null);
      return;
    }
    const t = setTimeout(() => {
      api.searchMeetings(q).then(setResults).catch((e) => console.error("search failed", e));
    }, 180);
    return () => clearTimeout(t);
  }, [query]);

  const askAll = async () => {
    if (!llmId || !query.trim() || asking) return;
    setAnswer("");
    setAsking(true);
    askingRef.current = true;
    try {
      setAnswer(await api.askMeetings(llmId, query.trim()));
    } catch (e) {
      setAnswer(String(e));
    } finally {
      askingRef.current = false;
      setAsking(false);
    }
  };

  const toggleItem = async (it: ActionItem) => {
    const saved = await api.saveActionItem({ ...it, done: !it.done });
    setItems((list) => list.map((x) => (x.id === it.id ? saved : x)));
  };

  const open = items.filter((i) => !i.done);
  const o = options;

  return (
    <div className="meet-list">
      <details className="meet-options">
        <summary>meeting options</summary>
        <div className="meet-options-body">
          <label>
            <span>Speech model</span>
            <select value={o.stt ?? ""} onChange={(e) => o.set("stt", e.target.value)}>
              {o.sttModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Notes model</span>
            <select value={o.llm ?? ""} onChange={(e) => o.set("llm", e.target.value)}>
              {o.llmModels.length === 0 && <option value="">none downloaded</option>}
              {o.llmModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Voice model (who's speaking)</span>
            <select value={o.emb ?? ""} onChange={(e) => o.set("emb", e.target.value)}>
              {o.embModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
          <p className="meet-options-note">
            Speakers are told apart live as the meeting runs, and voices you name are recognized next
            time. The notes model writes the summary in stretches while you talk, so it's ready moments
            after you stop — Gemma 4 or Qwen3.6 (Settings → models) write noticeably better notes than
            the small defaults.
          </p>
        </div>
      </details>

      {meetings.length > 0 && (
        <div className="meet-search-row">
          <input
            className="meet-search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && query.trim().endsWith("?")) askAll();
            }}
            placeholder="search every meeting — or ask a question and press ask"
          />
          {llmId && query.trim() && (
            <button className="quiet accent" disabled={asking} onClick={askAll}>
              {asking ? "…" : "ask"}
            </button>
          )}
        </div>
      )}

      {(answer || asking) && (
        <div className="meet-prose meet-answer">
          <Markdown text={answer} />
          {asking && <span className="caret" />}
        </div>
      )}

      {results !== null ? (
        results.length === 0 ? (
          <p className="dim meet-empty-line">no meetings match “{query.trim()}”</p>
        ) : (
          group(results).map((g) => (
            <div className="meet-row" key={g.meetingId}>
              <div className="meet-row-title" onClick={() => onOpen(g.meetingId)}>
                {g.title || "Untitled meeting"}
              </div>
              <div className="meet-row-meta">{new Date(g.startedAt).toLocaleString()}</div>
              {g.hits.map((h, i) => (
                <div
                  key={i}
                  className="meet-hit"
                  onClick={() => onOpen(h.meetingId, h.startMs ?? null)}
                  title={h.startMs != null ? "Open at this moment" : "Open"}
                >
                  {h.startMs != null && <span className="seg-time">{clock(h.startMs)}</span>}
                  {h.speaker && (
                    <span className="hit-who">
                      {speakerLabel(h.speaker, h.speakerName ? { [h.speaker]: h.speakerName } : undefined)}
                    </span>
                  )}
                  <span className="hit-text">
                    {snippetParts(h.snippet).map((part, j) =>
                      part.hit ? <mark key={j}>{part.text}</mark> : <span key={j}>{part.text}</span>,
                    )}
                  </span>
                </div>
              ))}
            </div>
          ))
        )
      ) : (
        <>
          {open.length > 0 && (
            <div className="open-items">
              <button className="quiet open-items-toggle" onClick={() => setShowItems((s) => !s)}>
                {showItems ? "▾" : "▸"} {open.length} open follow-up{open.length === 1 ? "" : "s"}
              </button>
              {showItems &&
                open.map((it) => (
                  <div className="item" key={it.id}>
                    <input type="checkbox" checked={it.done} onChange={() => toggleItem(it)} />
                    <span className="item-text">{it.text}</span>
                    {it.owner && <span className="item-chip">{it.owner}</span>}
                    {it.due && <span className="item-chip due">{it.due}</span>}
                    {it.meetingId && (
                      <button className="quiet item-src" onClick={() => onOpen(it.meetingId!, it.atMs ?? null)}>
                        {it.meetingTitle || "meeting"}
                      </button>
                    )}
                  </div>
                ))}
            </div>
          )}
          {meetings.length === 0 && idle && (
            <div className="empty">
              <p>
                your meetings land here. Start one above (or unsound will offer when Zoom, Meet or Teams
                takes your mic), and it listens to both you and the call — who said what, notes built
                around yours, action items, and anything you ask it to do. All on this machine.
              </p>
            </div>
          )}
          {meetings.map((m) => (
            <div className="meet-row" key={m.id} onClick={() => onOpen(m.id)}>
              <div className="meet-row-title">{m.title || "Untitled meeting"}</div>
              <div className="meet-row-meta">
                {new Date(m.startedAt).toLocaleString()} · {elapsed(m.startedAt, m.endedAt)} · {m.segmentCount}{" "}
                line{m.segmentCount === 1 ? "" : "s"}
                {!m.summary && m.segmentCount > 0 && " · no notes yet"}
              </div>
              {m.summary && <div className="meet-row-snippet">{summaryLine(m.summary)}</div>}
            </div>
          ))}
        </>
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
