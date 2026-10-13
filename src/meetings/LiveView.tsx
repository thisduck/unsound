import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { api, clock, formatShortcut, Segment, Task } from "../api";
import { speakerClass, speakerLabel } from "./format";
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
}

/// The meeting as it happens: live transcript on the left; your notes, a
/// place to ask unsound for things, and what it's working on, on the right.
export function LiveView(p: Props) {
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const notesRef = useRef<HTMLTextAreaElement>(null);
  const [ask, setAsk] = useState("");
  const [holding, setHolding] = useState(false);
  const saveTimer = useRef<number | undefined>(undefined);

  // Follow the transcript unless the user has scrolled up to read.
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [p.segments, p.partials]);
  const onScroll = () => {
    const el = scroller.current;
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  };

  const setNotes = (text: string) => {
    p.onNotes(text);
    window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => {
      api.updateMeetingNotes(p.meetingId, text).catch((e) => console.error("save notes failed", e));
    }, 700);
  };
  useEffect(() => () => window.clearTimeout(saveTimer.current), []);

  /// Flag this moment: a mark (for the summary) and a timestamp in the notes.
  const mark = async () => {
    const at = p.status?.positionMs ?? 0;
    api.addMark(p.meetingId, "").catch(() => {});
    const el = notesRef.current;
    const stamp = `[${clock(at)}] `;
    const text = p.notes;
    const pos = el?.selectionStart ?? text.length;
    const before = text.slice(0, pos);
    const insert = (before && !before.endsWith("\n") ? "\n" : "") + stamp;
    setNotes(before + insert + text.slice(pos));
    requestAnimationFrame(() => {
      el?.focus();
      const c = pos + insert.length;
      el?.setSelectionRange(c, c);
    });
  };

  const submitAsk = async () => {
    const text = ask.trim();
    if (!text) return;
    setAsk("");
    const t = await api.addTask(text, p.meetingId).catch(() => null);
    if (t) p.onTasks((list) => mergeTask(list, t));
  };

  const sorted = [...p.segments].sort((a, b) => a.startMs - b.startMs);
  const behind = p.status && p.status.pending > 2;
  const visibleTasks = p.tasks.filter((t) => t.status !== "dismissed");

  return (
    <div className="live-view">
      <div className="live-main">
        <div className="live-meta">
          <span className="live-dot" /> recording
          {p.title && <> · <strong>{p.title}</strong></>}
          {p.status && <> · {clock(p.status.positionMs)}</>}
          {behind && <> · transcript catching up ({p.status!.pending} waiting)</>}
          {p.status && !p.status.systemAudio && <> · mic only</>}
          {p.notesThroughMs > 0 && <> · notes ready through {clock(p.notesThroughMs)}</>}
        </div>
        {p.attendees.length > 0 && <div className="live-attendees">with {p.attendees.join(", ")}</div>}
        <div className="live-transcript" ref={scroller} onScroll={onScroll}>
          {sorted.length === 0 && Object.keys(p.partials).length === 0 ? (
            <p className="live-empty">listening…</p>
          ) : (
            sorted.map((s, i) => (
              <div className={"seg " + speakerClass(s.speaker)} key={i}>
                <span className="seg-who">{speakerLabel(s.speaker, p.speakerNames)}</span>
                <span className="seg-time">{clock(s.startMs)}</span>
                <span className="seg-text">{s.text}</span>
              </div>
            ))
          )}
          {Object.entries(p.partials).map(([speaker, text]) => (
            <div className={"seg seg-partial " + speakerClass(speaker)} key={"partial-" + speaker}>
              <span className="seg-who">{speakerLabel(speaker, p.speakerNames)}</span>
              <span className="seg-time">…</span>
              <span className="seg-text">{text}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="live-side">
        <section className="live-card">
          <div className="live-card-head">
            <h3>my notes</h3>
            <button
              className="quiet"
              // Keep focus in the notes: flagging mid-sentence shouldn't
              // steal the cursor (and Space must type, not re-flag).
              onMouseDown={(e) => e.preventDefault()}
              onClick={mark}
              tabIndex={-1}
              title="Flag this moment — it gets extra weight in the summary"
            >
              ⚑ mark moment
            </button>
          </div>
          <textarea
            ref={notesRef}
            className="meet-notes live-notes"
            value={p.notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="jot what matters — the summary is built around your notes"
          />
        </section>

        <section className="live-card">
          <div className="live-card-head">
            <h3>ask unsound</h3>
          </div>
          <div className="meet-ask">
            <input
              className="meet-ask-input"
              value={ask}
              onChange={(e) => setAsk(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") submitAsk();
              }}
              placeholder="e.g. create a Jira ticket for the Safari bug"
            />
            <button
              className={"quiet accent hold" + (holding ? " on" : "")}
              title="Hold and speak a request"
              onPointerDown={() => {
                setHolding(true);
                api.commandBegin();
              }}
              onPointerUp={() => {
                setHolding(false);
                api.commandEnd();
              }}
              onPointerLeave={() => {
                if (holding) {
                  setHolding(false);
                  api.commandEnd();
                }
              }}
            >
              {holding ? "listening…" : "hold to speak"}
            </button>
          </div>
          <p className="live-hint">
            {p.wakeWord ? <>or just say “{p.wakeWord}, …” </> : null}
            {p.commandKey ? <>or hold {formatShortcut(p.commandKey)}</> : null}
            {(p.wakeWord || p.commandKey) && " — works even while you're muted on the call"}
          </p>
          <div className="live-tasks">
            {visibleTasks.length === 0 ? (
              <p className="live-empty">requests you give show up here, drafted and ready to send.</p>
            ) : (
              [...visibleTasks]
                .reverse()
                .map((t) => (
                  <TaskCard key={t.id} task={t} compact onChange={(nt) => p.onTasks((l) => mergeTask(l, nt))} />
                ))
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
