import { useNotesSave } from "./useNotesSave";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { api, clock, formatShortcut, LiveNote, Segment, Task } from "../api";
import { Icon, Kbd, Segmented } from "../ui";
import { speakerLabel } from "./format";
import { TaskCard, mergeTask } from "./TaskCard";

interface Props {
  meetingId: string;
  title: string;
  attendees: string[];
  segments: Segment[];
  partials: Record<string, string>;
  speakerNames: Record<string, string>;
  notes: string;
  onNotes: (notes: string) => void;
  tasks: Task[];
  onTasks: (f: (t: Task[]) => Task[]) => void;
  status: { positionMs: number; pending: number; systemAudio: boolean } | null;
  wakeWord: string | null;
  commandKey: string | null;
  notesThroughMs: number;
  /// Bumped by the stage's Flag button (and ⌘M).
  flagToken?: number;
  onFlag?: () => void;
}

interface Turn {
  speaker: string;
  startMs: number;
  lines: string[];
  partial?: string;
}

/// Consecutive lines from one speaker read as one turn.
function turns(segments: Segment[], partials: Record<string, string>): Turn[] {
  const out: Turn[] = [];
  for (const s of [...segments].sort((a, b) => a.startMs - b.startMs)) {
    const last = out[out.length - 1];
    if (last && last.speaker === s.speaker) last.lines.push(s.text);
    else out.push({ speaker: s.speaker, startMs: s.startMs, lines: [s.text] });
  }
  for (const [speaker, text] of Object.entries(partials)) {
    if (!text.trim()) continue;
    const last = out[out.length - 1];
    const same = last && (last.speaker === speaker || (speaker === "them" && last.speaker.startsWith("them")));
    if (same) last.partial = text;
    else out.push({ speaker, startMs: -1, lines: [], partial: text });
  }
  return out;
}

const NOTE_GROUPS: { kind: string; label: string }[] = [
  { kind: "decision", label: "Decisions" },
  { kind: "action", label: "Action items" },
  { kind: "open_question", label: "Open questions" },
  { kind: "point", label: "Key points" },
];

/// The meeting as it happens: the transcript to follow, and beside it the
/// summary so far, requests, and one place to jot or ask.
export function LiveView(p: Props) {
  const scroller = useRef<HTMLDivElement>(null);
  const [following, setFollowing] = useState(true);
  const followingRef = useRef(true);
  const notesRef = useRef<HTMLTextAreaElement>(null);
  const [panel, setPanel] = useState<"summary" | "requests">("summary");
  const [mode, setMode] = useState<"note" | "ask">("note");
  const [ask, setAsk] = useState("");
  const [holding, setHolding] = useState(false);
  const [liveNotes, setLiveNotes] = useState<LiveNote[]>([]);
  const notesSave = useNotesSave(p.meetingId, p.onNotes);

  // Follow the transcript unless the user has scrolled up to read.
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && followingRef.current) el.scrollTop = el.scrollHeight;
  }, [p.segments, p.partials]);
  const onScroll = () => {
    const el = scroller.current;
    if (!el) return;
    const atEnd = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    followingRef.current = atEnd;
    setFollowing(atEnd);
  };
  const jumpToLive = () => {
    const el = scroller.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    followingRef.current = true;
    setFollowing(true);
  };

  // The running summary: facts written for each finished stretch.
  useEffect(() => {
    let active = true;
    api.meetingLiveNotes(p.meetingId).then((n) => { if (active) setLiveNotes(n); }).catch(() => {});
    return () => { active = false; };
  }, [p.meetingId, p.notesThroughMs]);

  const setNotes = (text: string) => {
    p.onNotes(text);
    notesSave.save(text);
  };

  /// Flag this moment: a mark (for the summary) and a timestamp in the notes.
  const mark = () => {
    const at = p.status?.positionMs ?? 0;
    api.addMark(p.meetingId, "").catch(() => {});
    setMode("note");
    const el = notesRef.current;
    const stamp = `[${clock(at)}] `;
    const text = p.notes;
    const pos = el?.selectionStart ?? text.length;
    const before = text.slice(0, pos);
    const insert = (before && !before.endsWith("\n") ? "\n" : "") + stamp;
    setNotes(before + insert + text.slice(pos));
    requestAnimationFrame(() => {
      const area = notesRef.current;
      area?.focus();
      const c = pos + insert.length;
      area?.setSelectionRange(c, c);
    });
  };
  const markRef = useRef(mark);
  markRef.current = mark;
  const lastFlag = useRef(p.flagToken ?? 0);
  useEffect(() => {
    if (p.flagToken && p.flagToken !== lastFlag.current) {
      lastFlag.current = p.flagToken;
      markRef.current();
    }
  }, [p.flagToken]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "m") {
        e.preventDefault();
        if (p.onFlag) p.onFlag();
        else markRef.current();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [p.onFlag]);

  const submitAsk = async () => {
    const text = ask.trim();
    if (!text) return;
    setAsk("");
    const t = await api.addTask(text, p.meetingId).catch(() => null);
    if (t) {
      p.onTasks((list) => mergeTask(list, t));
      setPanel("requests");
    }
  };

  const list = turns(p.segments, p.partials);
  const behind = p.status && p.status.pending > 2;
  const visibleTasks = p.tasks.filter((t) => t.status !== "dismissed");
  const readyCount = visibleTasks.filter((t) => t.status === "ready").length;

  return (
    <div className="live-body">
      <div className="live-main">
        <div className="live-transcript" ref={scroller} onScroll={onScroll}>
          <div className="transcript">
            {list.length === 0 ? (
              <p className="live-empty">Listening. The transcript appears as people speak.</p>
            ) : (
              list.map((t, i) => (
                <div className={"turn" + (t.speaker === "me" ? " me" : "")} key={i}>
                  <div className="turn-who">
                    <strong>{speakerLabel(t.speaker, p.speakerNames)}</strong>
                    {t.startMs >= 0 && <time>{clock(t.startMs)}</time>}
                  </div>
                  <p>
                    {t.lines.join(" ")}
                    {t.partial && (
                      <>
                        {t.lines.length ? " " : ""}
                        <span className="partial">{t.partial}</span>
                        <span className="caret" />
                      </>
                    )}
                  </p>
                </div>
              ))
            )}
          </div>
        </div>
        <div className="live-chips">
          {!following && (
            <button className="chip" onClick={jumpToLive}>
              <Icon name="down" /> Jump to live
            </button>
          )}
          {behind && <span className="chip" role="status">Transcript {p.status!.pending} parts behind</span>}
          {p.status && !p.status.systemAudio && <span className="chip" role="status">Your mic only</span>}
        </div>
      </div>

      <aside className="live-panel" aria-label="Summary and requests">
        <div className="live-panel-head">
          <Segmented
            label="Panel"
            value={panel}
            onChange={setPanel}
            options={[
              { value: "summary", label: "Summary" },
              { value: "requests", label: <>Requests{visibleTasks.length ? <span className={"badge" + (readyCount ? "" : " neutral")}>{visibleTasks.length}</span> : null}</> },
            ]}
          />
        </div>
        <div className="live-panel-body">
          {panel === "summary" ? (
            liveNotes.length === 0 ? (
              <p className="live-panel-empty">
                The summary builds every few minutes as people talk. Your notes and flags shape it.
              </p>
            ) : (
              <>
                {p.notesThroughMs > 0 && <p className="live-updated">Updated through {clock(p.notesThroughMs)}</p>}
                {NOTE_GROUPS.map((g) => {
                  const rows = liveNotes.filter((n) => n.kind === g.kind);
                  return rows.length ? (
                    <div className="live-note-group" key={g.kind}>
                      <h5>{g.label}</h5>
                      <ul>
                        {rows.map((n, i) => (
                          <li key={i}>
                            {n.text}
                            {(n.owner || n.due) && <span className="faint"> · {[n.owner, n.due].filter(Boolean).join(", ")}</span>}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : null;
                })}
              </>
            )
          ) : visibleTasks.length === 0 ? (
            <p className="live-panel-empty">Ask for an email, ticket or reminder. unsound drafts it from the conversation and waits for your OK.</p>
          ) : (
            <div className="stack">
              {[...visibleTasks].reverse().map((t) => (
                <TaskCard key={t.id} task={t} compact onChange={(nt) => p.onTasks((l) => mergeTask(l, nt))} />
              ))}
            </div>
          )}
        </div>

        <div className="live-composer">
          <div className="live-composer-modes" role="tablist" aria-label="Composer">
            <button role="tab" aria-selected={mode === "note"} onClick={() => setMode("note")}>Your notes</button>
            <button role="tab" aria-selected={mode === "ask"} onClick={() => setMode("ask")}>Ask unsound to…</button>
          </div>
          {mode === "note" ? (
            <>
              <textarea
                ref={notesRef}
                className="live-notes"
                aria-label="Your notes"
                value={p.notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Type what matters. ⌘M flags this moment."
              />
              {notesSave.error && (
                <div className="alert danger" role="alert">
                  <span className="alert-body">{notesSave.error}</span>
                  <div className="alert-actions"><button className="btn btn-quiet btn-sm" onClick={notesSave.flush}>Try again</button></div>
                </div>
              )}
              {notesSave.saving && <span className="visually-hidden" role="status">Saving notes</span>}
            </>
          ) : (
            <>
              <form className="live-ask" onSubmit={(e) => { e.preventDefault(); void submitAsk(); }}>
                <input
                  className="input"
                  aria-label="Ask unsound to do something"
                  value={ask}
                  onChange={(e) => setAsk(e.target.value)}
                  placeholder="Email Priya the revised tiers"
                />
                <button
                  type="button"
                  className={"btn btn-secondary btn-icon" + (holding ? " holding" : "")}
                  aria-label={holding ? "Listening" : "Hold to speak"}
                  title="Hold and speak a request"
                  onPointerDown={() => { setHolding(true); api.commandBegin(); }}
                  onPointerUp={() => { setHolding(false); api.commandEnd(); }}
                  onPointerLeave={() => { if (holding) { setHolding(false); api.commandEnd(); } }}
                >
                  <Icon name="mic" />
                </button>
              </form>
              <p className="live-hint">
                {holding ? "Listening…" : <>
                  {p.wakeWord ? <>Or say “{p.wakeWord}, …”. </> : null}
                  {p.commandKey ? <>Or hold <Kbd>{formatShortcut(p.commandKey)}</Kbd>. </> : null}
                  Works while you're muted.
                </>}
              </p>
            </>
          )}
        </div>
      </aside>
    </div>
  );
}
