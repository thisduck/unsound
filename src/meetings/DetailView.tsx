import { useEffect, useMemo, useRef, useState } from "react";
import { ActionItem, api, clock, Meeting, on, Segment, Task } from "../api";
import { Markdown } from "../Markdown";
import { elapsed, meetingMarkdown, speakerClass, speakerLabel } from "./format";
import { mergeTask, TaskCard } from "./TaskCard";
import { usePlayer } from "./usePlayer";

interface Props {
  meeting: Meeting;
  onMeeting: (m: Meeting) => void;
  onBack: () => void;
  onDelete: () => void;
  tasks: Task[];
  onTasks: (f: (t: Task[]) => Task[]) => void;
  /// The summary being written right now (streamed), if any.
  liveSummary: string | null;
  summarizing: boolean;
  onSummarize: () => void;
  llmId?: string;
  embId?: string;
  diarizeReady: boolean;
  /// Jump here once opened (from a search hit).
  focusMs?: number | null;
}

/// A single word/phrase edit, for offering "fix everywhere": the changed run
/// of words before and after.
function wordEdit(before: string, after: string): { from: string; to: string } | null {
  const a = before.split(/\s+/);
  const b = after.split(/\s+/);
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  let j = 0;
  while (j < a.length - i && j < b.length - i && a[a.length - 1 - j] === b[b.length - 1 - j]) j++;
  const from = a.slice(i, a.length - j).join(" ").replace(/[.,!?;:]+$/, "");
  const to = b.slice(i, b.length - j).join(" ").replace(/[.,!?;:]+$/, "");
  if (!from || !to || from === to) return null;
  if (from.split(" ").length > 3 || to.split(" ").length > 4) return null;
  return { from, to };
}

export function DetailView(p: Props) {
  const m = p.meeting;
  const player = usePlayer(m.hasAudio ? m.id : null);
  const [notes, setNotes] = useState(m.notes);
  const [editingSummary, setEditingSummary] = useState(false);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [asking, setAsking] = useState(false);
  const [copied, setCopied] = useState(false);
  const [reCount, setReCount] = useState("");
  const [reDetecting, setReDetecting] = useState(false);
  const [newItem, setNewItem] = useState("");
  const [fix, setFix] = useState<{ from: string; to: string; count: number } | null>(null);
  const [rememberVoice, setRememberVoice] = useState(true);
  const lineRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  useEffect(() => setNotes(m.notes), [m.id, m.notes]);

  // Stream this meeting's Q&A answer.
  const askingRef = useRef(false);
  useEffect(() => {
    const sub = on.meetingAnswerToken((chunk) => {
      if (askingRef.current) setAnswer((a) => a + chunk);
    });
    return () => {
      sub.then((un) => un());
    };
  }, []);

  const segments = useMemo(() => [...m.segments].sort((a, b) => a.startMs - b.startMs), [m.segments]);
  const playingIdx = useMemo(() => {
    if (player.atMs == null) return -1;
    let idx = -1;
    for (let i = 0; i < segments.length; i++) {
      if (segments[i].startMs <= player.atMs + 200) idx = i;
      else break;
    }
    return idx;
  }, [player.atMs, segments]);

  const scrollTo = (ms: number) => {
    const seg = segments.find((s) => s.endMs >= ms) ?? segments[segments.length - 1];
    if (seg) lineRefs.current.get(seg.id ?? -1)?.scrollIntoView({ behavior: "smooth", block: "center" });
  };
  const jump = (ms: number) => {
    scrollTo(ms);
    if (m.hasAudio) player.play(ms);
  };

  // Keep the playing line in view.
  useEffect(() => {
    if (playingIdx >= 0 && player.playing) {
      const seg = segments[playingIdx];
      lineRefs.current.get(seg.id ?? -1)?.scrollIntoView({ block: "nearest" });
    }
  }, [playingIdx, player.playing, segments]);

  useEffect(() => {
    if (p.focusMs != null) setTimeout(() => scrollTo(p.focusMs!), 50);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p.focusMs, m.id]);

  const ask = async () => {
    if (!p.llmId || !question.trim() || asking) return;
    setAnswer("");
    setAsking(true);
    askingRef.current = true;
    try {
      setAnswer(await api.askMeeting(m.id, p.llmId, question.trim()));
    } catch (e) {
      setAnswer(String(e));
    } finally {
      askingRef.current = false;
      setAsking(false);
    }
  };

  const saveSegment = async (seg: Segment, text: string) => {
    if (!seg.id || text === seg.text) return;
    try {
      await api.updateSegmentText(seg.id, text);
      p.onMeeting({ ...m, segments: m.segments.map((s) => (s.id === seg.id ? { ...s, text } : s)) });
      const edit = wordEdit(seg.text, text);
      if (edit) {
        const re = new RegExp(`\\b${edit.from.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i");
        const count = m.segments.filter((s) => s.id !== seg.id && re.test(s.text)).length;
        setFix({ ...edit, count });
      }
    } catch (e) {
      console.error("failed to save segment", e);
    }
  };

  const applyFix = async (remember: boolean) => {
    if (!fix) return;
    try {
      await api.replaceInMeeting(m.id, fix.from, fix.to, remember);
      const fresh = await api.getMeeting(m.id);
      if (fresh) p.onMeeting(fresh);
    } finally {
      setFix(null);
    }
  };

  const saveItem = async (item: ActionItem) => {
    const saved = await api.saveActionItem(item);
    const items = m.actionItems ?? [];
    p.onMeeting({
      ...m,
      actionItems: item.id ? items.map((i) => (i.id === item.id ? saved : i)) : [...items, saved],
    });
  };
  const deleteItem = async (id: number) => {
    await api.deleteActionItem(id);
    p.onMeeting({ ...m, actionItems: (m.actionItems ?? []).filter((i) => i.id !== id) });
  };

  const renameSpeaker = async (speaker: string, name: string) => {
    try {
      await api.setSpeakerName(m.id, speaker, name, rememberVoice);
      const names = { ...(m.speakerNames ?? {}) };
      if (name.trim()) names[speaker] = name.trim();
      else delete names[speaker];
      p.onMeeting({ ...m, speakerNames: names });
    } catch (e) {
      console.error("failed to rename speaker", e);
    }
  };

  const reDetect = async () => {
    setReDetecting(true);
    try {
      p.onMeeting(await api.diarizeMeeting(m.id, p.embId, reCount ? parseInt(reCount, 10) : undefined));
    } catch (e) {
      console.error(e);
    } finally {
      setReDetecting(false);
    }
  };

  const speakers = Array.from(new Set(m.segments.map((s) => s.speaker))).sort();
  const items = m.actionItems ?? [];
  const marks = m.marks ?? [];
  const summaryText = p.liveSummary ?? m.summary;
  const visibleTasks = p.tasks.filter((t) => t.status !== "dismissed" || t.error);

  return (
    <div className="meet-detail">
      <div className="meet-detail-head">
        <button className="quiet" onClick={p.onBack}>
          ← all meetings
        </button>
        <input
          className="meet-title-input"
          value={m.title}
          placeholder="Untitled meeting"
          onChange={(e) => p.onMeeting({ ...m, title: e.target.value })}
          onBlur={() => api.renameMeeting(m.id, m.title).catch((e) => console.error(e))}
        />
        <span className="spacer" />
        <button
          className="quiet"
          onClick={() => {
            navigator.clipboard.writeText(meetingMarkdown(m));
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }}
        >
          {copied ? "copied ✓" : "copy as markdown"}
        </button>
        <button className="quiet danger" onClick={p.onDelete}>
          delete
        </button>
      </div>
      <div className="meet-detail-meta">
        {new Date(m.startedAt).toLocaleString()} · {elapsed(m.startedAt, m.endedAt)}
        {m.attendees && m.attendees.length > 0 && <> · with {m.attendees.join(", ")}</>}
      </div>

      <section className="meet-section">
        <div className="meet-section-head">
          <h3>summary</h3>
          <span className="spacer" />
          {!p.summarizing && summaryText && (
            <button className="quiet" onClick={() => setEditingSummary((e) => !e)}>
              {editingSummary ? "done editing" : "edit"}
            </button>
          )}
          {p.llmId && m.segments.length > 0 && (
            <button className="quiet" disabled={p.summarizing} onClick={p.onSummarize}>
              {p.summarizing ? "writing…" : summaryText ? "regenerate" : "write notes"}
            </button>
          )}
        </div>
        {p.summarizing && !p.liveSummary ? (
          <p className="dim">finishing the last part of the meeting…</p>
        ) : !summaryText ? (
          <p className="dim">
            {m.segments.length ? "no summary yet." : "no transcript for this meeting."}
            {!p.llmId && m.segments.length > 0 && " download a language model to write notes."}
          </p>
        ) : editingSummary ? (
          <textarea
            className="meet-notes meet-summary"
            key={"sum-" + m.id}
            defaultValue={m.summary}
            onBlur={(e) => {
              const text = e.target.value;
              if (text !== m.summary) {
                api.setMeetingSummary(m.id, text).catch(console.error);
                p.onMeeting({ ...m, summary: text });
              }
            }}
          />
        ) : (
          <div className="meet-prose">
            {/* The section already says "summary"; drop the model's own heading. */}
            <Markdown text={summaryText.replace(/^\s*#{1,6}\s*summary\s*\n/i, "")} onTime={jump} />
            {p.summarizing && <span className="caret" />}
          </div>
        )}
      </section>

      <section className="meet-section">
        <h3>action items</h3>
        {items.length === 0 && <p className="dim">none yet.</p>}
        <div className="items">
          {items.map((it) => (
            <div className={"item" + (it.done ? " done" : "")} key={it.id}>
              <input type="checkbox" checked={it.done} onChange={() => saveItem({ ...it, done: !it.done })} />
              <span
                className="item-text"
                contentEditable
                suppressContentEditableWarning
                onBlur={(e) => {
                  const text = e.currentTarget.textContent?.trim() ?? "";
                  if (text && text !== it.text) saveItem({ ...it, text });
                }}
              >
                {it.text}
              </span>
              {it.owner && <span className="item-chip">{it.owner}</span>}
              {it.due && <span className="item-chip due">{it.due}</span>}
              {it.atMs != null && (
                <button className="time-link" onClick={() => jump(it.atMs!)}>
                  {clock(it.atMs)}
                </button>
              )}
              <button className="quiet item-x" onClick={() => deleteItem(it.id)} title="Remove">
                ✕
              </button>
            </div>
          ))}
        </div>
        <input
          className="item-add"
          value={newItem}
          placeholder="+ add an action item"
          onChange={(e) => setNewItem(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && newItem.trim()) {
              saveItem({
                id: 0,
                meetingId: m.id,
                text: newItem.trim(),
                owner: "",
                due: "",
                atMs: null,
                done: false,
                manual: true,
                createdAt: "",
                meetingTitle: "",
              });
              setNewItem("");
            }
          }}
        />
      </section>

      {visibleTasks.length > 0 && (
        <section className="meet-section">
          <h3>requests</h3>
          {visibleTasks.map((t) => (
            <TaskCard key={t.id} task={t} onTime={jump} onChange={(nt) => p.onTasks((l) => mergeTask(l, nt))} />
          ))}
        </section>
      )}

      {p.llmId && m.segments.length > 0 && (
        <section className="meet-section">
          <h3>ask about this meeting</h3>
          <div className="meet-ask">
            <input
              className="meet-ask-input"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") ask();
              }}
              placeholder="e.g. what did I agree to? what did Priya say about pricing?"
              disabled={asking}
            />
            <button className="quiet accent" onClick={ask} disabled={asking || !question.trim()}>
              {asking ? "…" : "ask"}
            </button>
          </div>
          {(answer || asking) && (
            <div className="meet-prose meet-answer">
              <Markdown text={answer} onTime={jump} />
              {asking && <span className="caret" />}
            </div>
          )}
        </section>
      )}

      <section className="meet-section">
        <h3>my notes</h3>
        <textarea
          className="meet-notes"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          onBlur={() => {
            if (notes !== m.notes) {
              api.updateMeetingNotes(m.id, notes).catch(console.error);
              p.onMeeting({ ...m, notes });
            }
          }}
          placeholder="jot anything down — saved with this meeting, and used by the summary"
        />
      </section>

      {m.segments.length > 0 && (
        <section className="meet-section">
          <h3>speakers</h3>
          <div className="speaker-edit">
            {speakers.map((sp) => (
              <input
                key={sp + (m.speakerNames?.[sp] ?? "")}
                className={"speaker-chip " + speakerClass(sp)}
                defaultValue={m.speakerNames?.[sp] ?? ""}
                placeholder={speakerLabel(sp)}
                onBlur={(e) => {
                  if (e.target.value.trim() !== (m.speakerNames?.[sp] ?? "")) renameSpeaker(sp, e.target.value);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                }}
              />
            ))}
          </div>
          <label className="check-row">
            <input type="checkbox" checked={rememberVoice} onChange={(e) => setRememberVoice(e.target.checked)} />
            remember voices I name, and label them automatically next time
          </label>
          {p.diarizeReady && m.hasAudio && m.segments.some((s) => s.source === "system") && (
            <div className="redetect-row">
              <span>wrong speakers?</span>
              <select value={reCount} onChange={(e) => setReCount(e.target.value)}>
                <option value="">auto</option>
                {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => (
                  <option key={n} value={String(n)}>
                    {n} {n === 1 ? "person" : "people"}
                  </option>
                ))}
              </select>
              <button className="quiet" disabled={reDetecting} onClick={reDetect}>
                {reDetecting ? "detecting…" : "re-detect speakers"}
              </button>
            </div>
          )}
        </section>
      )}

      <section className="meet-section">
        <h3>transcript</h3>
        {fix && (
          <div className="fix-bar">
            changed “{fix.from}” → “{fix.to}”.
            {fix.count > 0 && (
              <button className="quiet accent" onClick={() => applyFix(false)}>
                fix {fix.count} more in this meeting
              </button>
            )}
            <button className="quiet accent" onClick={() => applyFix(true)}>
              {fix.count > 0 ? "fix & remember" : "remember for next time"}
            </button>
            <button className="quiet" onClick={() => setFix(null)}>
              no
            </button>
          </div>
        )}
        {segments.length === 0 ? (
          <p className="dim">no transcript for this meeting</p>
        ) : (
          segments.map((s, i) => {
            const flagged = marks.some((mk) => mk.atMs >= s.startMs && mk.atMs < (segments[i + 1]?.startMs ?? Infinity));
            return (
              <div
                className={"seg " + speakerClass(s.speaker) + (i === playingIdx ? " playing" : "")}
                key={s.id ?? i}
                ref={(el) => {
                  if (el && s.id != null) lineRefs.current.set(s.id, el);
                }}
              >
                <span className="seg-who">{speakerLabel(s.speaker, m.speakerNames)}</span>
                <button
                  className="seg-time time-link"
                  onClick={() => jump(s.startMs)}
                  title={m.hasAudio ? "Play from here" : undefined}
                  disabled={!m.hasAudio}
                >
                  {flagged ? "⚑ " : ""}
                  {clock(s.startMs)}
                </button>
                <span
                  className="seg-text seg-editable"
                  contentEditable
                  suppressContentEditableWarning
                  onBlur={(e) => saveSegment(s, e.currentTarget.textContent || "")}
                >
                  {s.text}
                </span>
              </div>
            );
          })
        )}
      </section>

      {m.hasAudio && (player.atMs != null || player.loading) && (
        <div className="player">
          <button className="quiet accent" onClick={player.toggle} disabled={player.loading}>
            {player.loading ? "…" : player.playing ? "❚❚" : "▶"}
          </button>
          <span className="player-time">{player.atMs != null ? clock(player.atMs) : ""}</span>
          {player.error && <span className="task-error">{player.error}</span>}
          <span className="spacer" />
          <button className="quiet" onClick={player.stop}>
            close
          </button>
        </div>
      )}
    </div>
  );
}
