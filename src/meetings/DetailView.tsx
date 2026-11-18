import { useNotesSave } from "./useNotesSave";
import { useEffect, useMemo, useRef, useState } from "react";
import { ActionItem, api, clock, Meeting, MeetingWork, Segment, Task } from "../api";
import { ActionRow } from "./ActionRow";
import { ResearchPanel } from "./ResearchPanel";
import { Markdown } from "../Markdown";
import { elapsed, meetingMarkdown, speakerLabel } from "./format";
import { mergeTask, TaskCard } from "./TaskCard";
import { usePlayer } from "./usePlayer";
import { Icon, MoreMenu, Spinner, useConfirm } from "../ui";

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
  /// Transcription still in progress for this meeting, if any.
  work?: MeetingWork;
  onRetryWork?: () => void;
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
  const notesSave = useNotesSave(m.id, setNotes, (text) => p.onMeeting({ ...m, notes: text, summaryStale: true }));
  const [editingSummary, setEditingSummary] = useState(false);
  const [copied, setCopied] = useState(false);
  const [reCount, setReCount] = useState("");
  const [reDetecting, setReDetecting] = useState(false);
  const [newItem, setNewItem] = useState("");
  const [adding, setAdding] = useState(false);
  const [itemError, setItemError] = useState("");
  const [fix, setFix] = useState<{ from: string; to: string; count: number } | null>(null);
  const [speakerError, setSpeakerError] = useState<string | null>(null);
  const [correcting, setCorrecting] = useState<number | null>(null);
  const [correctionName, setCorrectionName] = useState("");
  const [wholeSpeaker, setWholeSpeaker] = useState(false);
  const [speakerSaving, setSpeakerSaving] = useState(false);
  const [rememberVoice, setRememberVoice] = useState(true);
  const [tab, setTab] = useState<"summary" | "transcript" | "notes">(p.focusMs != null ? "transcript" : "summary");
  const lineRefs = useRef<Map<number, HTMLDivElement>>(new Map());
  const { confirm, dialog } = useConfirm();

  useEffect(() => {
    if (notesSave.saving || notesSave.error) return;
    try { setNotes(localStorage.getItem(`unsound.notes-draft.${m.id}`) ?? m.notes); }
    catch { setNotes(m.notes); }
  }, [m.id, m.notes]);

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
    setTab("transcript");
    setTimeout(() => scrollTo(ms), 30);
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

  const saveSegment = async (seg: Segment, text: string) => {
    if (!seg.id || text === seg.text) return;
    try {
      await api.updateSegmentText(seg.id, text);
      p.onMeeting({ ...m, summaryStale: true, segments: m.segments.map((s) => (s.id === seg.id ? { ...s, text } : s)) });
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
  const renameSpeaker = async (speaker: string, name: string) => {
    try {
      setSpeakerError(null);
      const remembered = await api.setSpeakerName(m.id, speaker, name, rememberVoice);
      const updated = await api.getMeeting(m.id);
      if (updated) p.onMeeting(updated);
      if (rememberVoice && name.trim() && !remembered) setSpeakerError("Name saved. unsound needs more clear speech before it can recognize this voice.");
    } catch (e) {
      setSpeakerError(String(e));
      // Naming may have succeeded even when voice enrollment could not finish.
      const updated = await api.getMeeting(m.id).catch(() => null);
      if (updated) p.onMeeting(updated);
    }
  };

  const correctSpeaker = async (segment: Segment, target: string | null, name: string | null) => {
    if (segment.id == null) return;
    setSpeakerSaving(true); setSpeakerError(null);
    try {
      p.onMeeting(await api.correctMeetingSpeaker(m.id, segment.id, target, name, wholeSpeaker));
      setCorrecting(null); setCorrectionName(""); setWholeSpeaker(false);
    } catch (e) { setSpeakerError(String(e)); }
    finally { setSpeakerSaving(false); }
  };

  const reDetect = async () => {
    setReDetecting(true);
    try {
      p.onMeeting(await api.diarizeMeeting(m.id, p.embId, reCount ? parseInt(reCount, 10) : undefined));
    } catch (e) {
      setSpeakerError(String(e));
    } finally {
      setReDetecting(false);
    }
  };

  const speakers = Array.from(new Set(m.segments.map((s) => s.speaker))).sort();
  const items = m.actionItems ?? [];
  const marks = m.marks ?? [];
  const summaryText = p.liveSummary ?? m.summary;
  const visibleTasks = p.tasks.filter((t) => t.status !== "dismissed" || t.error);
  const canWrite = !!p.llmId && m.segments.length > 0 && !p.work;
  const canRedetect = p.diarizeReady && !!m.hasAudio && m.segments.some((s) => s.source === "system");

  const copyMarkdown = () => {
    navigator.clipboard.writeText(meetingMarkdown(m));
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  const askDelete = () =>
    confirm({
      title: `Delete “${m.title || "Untitled meeting"}”?`,
      description: "The transcript, summary, notes and audio are removed from this Mac. This can't be undone.",
      confirm: "Delete meeting",
      danger: true,
      onConfirm: p.onDelete,
    });

  const addItem = () => {
    if (!newItem.trim() || adding) return;
    setAdding(true);
    setItemError("");
    void saveItem({
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
    })
      .then(() => setNewItem(""))
      .catch((e) => setItemError(String(e)))
      .finally(() => setAdding(false));
  };

  const summaryTab = (
    <>
      {p.work && (
        <div className={"alert " + (p.work.state === "finishing" ? "" : "danger")} role="status">
          {p.work.state === "finishing" ? <Spinner /> : <Icon name="alert" />}
          <span className="alert-body">
            {p.work.state === "finishing"
              ? `Finishing the transcript${p.work.pending ? ` · ${p.work.pending} part${p.work.pending === 1 ? "" : "s"} left` : ""}. The summary is written when it's done.`
              : "Transcription stopped. Your audio is saved."}
          </span>
          {p.work.state !== "finishing" && p.onRetryWork && (
            <div className="alert-actions"><button className="btn btn-secondary btn-sm" onClick={p.onRetryWork}>Try again</button></div>
          )}
        </div>
      )}
      {m.summaryStale && m.summary && !p.summarizing && (
        <div className="alert warn" role="status">
          <Icon name="info" />
          <span className="alert-body">The transcript, speakers or your notes changed since this summary was written.</span>
          {canWrite && <div className="alert-actions"><button className="btn btn-secondary btn-sm" onClick={p.onSummarize}>Update summary</button></div>}
        </div>
      )}
      <section className="summary" aria-label="Summary">
        {p.summarizing && !p.liveSummary ? (
          <p className="summary-busy" role="status"><Spinner /> Writing the summary and checking it against the transcript…</p>
        ) : !summaryText ? (
          <div className="empty-state compact">
            {m.segments.length === 0 ? (
              <p>This meeting has no transcript.</p>
            ) : p.llmId ? (
              <>
                <p>No summary yet.</p>
                {canWrite && <button className="btn btn-primary" onClick={p.onSummarize}>Write summary</button>}
              </>
            ) : (
              <p>Writing a summary needs a summary model. Download one in Settings → Models.</p>
            )}
          </div>
        ) : editingSummary ? (
          <>
            <textarea
              className="textarea summary-editor"
              key={"sum-" + m.id}
              aria-label="Edit summary"
              defaultValue={m.summary}
              autoFocus
              onBlur={(e) => {
                const text = e.target.value;
                if (text !== m.summary) {
                  api.setMeetingSummary(m.id, text).catch(console.error);
                  p.onMeeting({ ...m, summary: text, summaryStale: false });
                }
              }}
            />
            <div className="row-actions"><button className="btn btn-secondary btn-sm" onClick={() => setEditingSummary(false)}>Done editing</button></div>
          </>
        ) : (
          <div className="summary-prose">
            {/* The tab already says "Summary"; drop the model's own heading. */}
            <Markdown text={summaryText.replace(/^\s*#{1,6}\s*summary\s*\n/i, "")} onTime={jump} />
            {p.summarizing && <span className="caret" />}
          </div>
        )}
      </section>

      <section aria-label="Action items">
        <div className="section-label">
          Action items <span className="count">{items.filter((i) => !i.done && !i.ignored).length || ""}</span>
        </div>
        <div className="card item-list">
          {items.length === 0 && <p className="item-list-empty">No action items. Add one, or they appear when the summary is written.</p>}
          {items.map((it) => (
            <ActionRow key={it.id} item={it} onSave={saveItem} onSource={it.atMs != null ? () => jump(it.atMs!) : undefined} sourceLabel={it.atMs != null ? clock(it.atMs) : undefined} />
          ))}
          <form className="item-add" onSubmit={(e) => { e.preventDefault(); addItem(); }}>
            <Icon name="check" />
            <input
              disabled={adding}
              aria-label="Add an action item"
              value={newItem}
              placeholder="Add an action item"
              onChange={(e) => setNewItem(e.target.value)}
            />
          </form>
        </div>
        {itemError && <div className="alert danger" role="alert"><span className="alert-body">{itemError}</span></div>}
      </section>

      {visibleTasks.length > 0 && (
        <section aria-label="Requests">
          <div className="section-label">Requests <span className="count">{visibleTasks.length}</span></div>
          <div className="stack">
            {visibleTasks.map((t) => (
              <TaskCard key={t.id} task={t} onTime={jump} onChange={(nt) => p.onTasks((l) => mergeTask(l, nt))} />
            ))}
          </div>
        </section>
      )}

      <section aria-label="Ask about this meeting">
        <div className="section-label">Ask about this meeting</div>
        <div className="card card-pad">
          <ResearchPanel key={m.id} modelId={p.llmId} meetingId={m.id} onSource={(_, at) => { if (at != null) jump(at); else setTab("summary"); }} />
        </div>
      </section>
    </>
  );

  const transcriptTab = (
    <>
      {m.segments.length > 0 && (
        <div className="card card-pad speakers-card">
          <div className="speakers-head">
            <Icon name="users" />
            <strong>Speakers</strong>
            <span className="faint">Name a speaker to relabel them. Click a name in the transcript to fix one line.</span>
          </div>
          <div className="speaker-chips">
            {speakers.map((sp) => (
              <input
                key={sp + (m.speakerNames?.[sp] ?? "")}
                className={"speaker-chip" + (sp === "me" ? " me" : "")}
                aria-label={`Name for ${speakerLabel(sp)}`}
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
          <div className="speakers-foot">
            <label className="check">
              <input type="checkbox" checked={rememberVoice} onChange={(e) => setRememberVoice(e.target.checked)} />
              Recognize named voices in future meetings
            </label>
            <span className="spacer" />
            {canRedetect && (
              <>
                <select className="select select-sm" aria-label="Number of people" value={reCount} onChange={(e) => setReCount(e.target.value)}>
                  <option value="">Any number of people</option>
                  {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => (
                    <option key={n} value={String(n)}>
                      {n} {n === 1 ? "person" : "people"}
                    </option>
                  ))}
                </select>
                <button className="btn btn-quiet btn-sm" disabled={reDetecting} onClick={reDetect}>
                  {reDetecting ? <><Spinner /> Detecting…</> : "Re-detect speakers"}
                </button>
              </>
            )}
          </div>
          {speakerError && <div className="alert danger" role="alert"><span className="alert-body">{speakerError}</span></div>}
        </div>
      )}
      {fix && (
        <div className="alert fix-bar" role="status">
          <Icon name="pencil" />
          <span className="alert-body">Changed “{fix.from}” to “{fix.to}”.</span>
          <div className="alert-actions">
            {fix.count > 0 && <button className="btn btn-quiet btn-sm" onClick={() => applyFix(false)}>Fix {fix.count} more</button>}
            <button className="btn btn-secondary btn-sm" onClick={() => applyFix(true)}>{fix.count > 0 ? "Fix and remember" : "Remember"}</button>
            <button className="btn btn-quiet btn-sm" onClick={() => setFix(null)}>Dismiss</button>
          </div>
        </div>
      )}
      {segments.length === 0 ? (
        <p className="empty-state compact">This meeting has no transcript.</p>
      ) : (
        <div className="transcript detail-transcript">
          {segments.map((s, i) => {
            const flagged = marks.some((mk) => mk.atMs >= s.startMs && mk.atMs < (segments[i + 1]?.startMs ?? Infinity));
            const cont = i > 0 && segments[i - 1].speaker === s.speaker && correcting !== s.id;
            return (
              <div
                className={"line" + (s.speaker === "me" ? " me" : "") + (cont ? " cont" : "") + (i === playingIdx ? " playing" : "") + (flagged ? " flagged" : "")}
                key={s.id ?? i}
                ref={(el) => {
                  if (el && s.id != null) lineRefs.current.set(s.id, el);
                }}
              >
                <div className="line-who">
                  <button className="line-speaker" title="Correct speaker" onClick={() => { setCorrecting(correcting === s.id ? null : s.id ?? null); setWholeSpeaker(false); }}>
                    {speakerLabel(s.speaker, m.speakerNames)}
                  </button>
                  <button className="line-time" onClick={() => jump(s.startMs)} title={m.hasAudio ? "Play from here" : undefined} disabled={!m.hasAudio}>
                    {flagged && <Icon name="flag" />}
                    {clock(s.startMs)}
                  </button>
                </div>
                <div className="line-body">
                  <span
                    className="line-text"
                    contentEditable
                    suppressContentEditableWarning
                    title="Click to edit"
                    onBlur={(e) => saveSegment(s, e.currentTarget.textContent || "")}
                  >
                    {s.text}
                  </span>
                  {correcting === s.id && (
                    <div className="speaker-correction">
                      <select aria-label="Assign speaker" className="select select-sm" value="" disabled={speakerSaving} onChange={(e) => { if (e.target.value) void correctSpeaker(s, e.target.value, null); }}>
                        <option value="">Choose a speaker…</option>
                        {speakers.filter((sp) => sp !== s.speaker).map((sp) => <option key={sp} value={sp}>{speakerLabel(sp, m.speakerNames)}</option>)}
                      </select>
                      <input aria-label="New speaker name" className="input" placeholder="Or type a new name" value={correctionName} onChange={(e) => setCorrectionName(e.target.value)} />
                      <button className="btn btn-secondary btn-sm" disabled={speakerSaving || !correctionName.trim()} onClick={() => void correctSpeaker(s, null, correctionName)}>Save</button>
                      <button className="btn btn-quiet btn-sm" onClick={() => setCorrecting(null)}>Cancel</button>
                      <label className="check"><input type="checkbox" checked={wholeSpeaker} onChange={(e) => setWholeSpeaker(e.target.checked)} /> Apply to all their lines</label>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </>
  );

  const notesTab = (
    <section aria-label="Your notes">
      {notesSave.error && (
        <div className="alert danger" role="alert">
          <span className="alert-body">{notesSave.error}</span>
          <div className="alert-actions"><button className="btn btn-quiet btn-sm" onClick={notesSave.flush}>Try again</button></div>
        </div>
      )}
      <textarea
        className="textarea notes-editor"
        aria-label="Your notes"
        value={notes}
        onChange={(e) => { setNotes(e.target.value); notesSave.save(e.target.value); }}
        onBlur={notesSave.flush}
        placeholder="Anything you jot here is saved with this meeting and shapes the summary."
      />
      <p className="faint notes-status" role="status">{notesSave.saving ? "Saving…" : "Saved on this Mac"}</p>
    </section>
  );

  return (
    <div className="meet-detail">
      <div className="detail-head">
        <div className="detail-title">
          <input
            className="title-input"
            aria-label="Meeting title"
            value={m.title}
            placeholder="Untitled meeting"
            onChange={(e) => p.onMeeting({ ...m, title: e.target.value })}
            onBlur={() => api.renameMeeting(m.id, m.title).catch((e) => console.error(e))}
          />
          <div className="detail-meta">
            {new Date(m.startedAt).toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" })} · {new Date(m.startedAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })} · {elapsed(m.startedAt, m.endedAt)}
            {m.attendees && m.attendees.length > 0 && <> · {m.attendees.join(", ")}</>}
          </div>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={copyMarkdown}>
          <Icon name={copied ? "check" : "copy"} />
          {copied ? "Copied" : "Copy as Markdown"}
        </button>
        <MoreMenu
          label="Meeting actions"
          items={[
            ...(canWrite ? [{ label: summaryText ? "Rewrite summary" : "Write summary", icon: "refresh" as const, onSelect: p.onSummarize, disabled: p.summarizing }] : []),
            ...(summaryText && !p.summarizing ? [{ label: "Edit summary", icon: "pencil" as const, onSelect: () => { setTab("summary"); setEditingSummary(true); } }] : []),
            ...(canRedetect ? [{ label: "Re-detect speakers…", icon: "users" as const, onSelect: () => setTab("transcript") }] : []),
            { label: "Delete meeting…", icon: "trash", danger: true, separatorBefore: true, onSelect: askDelete },
          ]}
        />
      </div>

      <div className="tabs detail-tabs" role="tablist">
        <button role="tab" aria-selected={tab === "summary"} onClick={() => setTab("summary")}>Summary</button>
        <button role="tab" aria-selected={tab === "transcript"} onClick={() => setTab("transcript")}>
          Transcript<span className="count">{m.segments.length || ""}</span>
        </button>
        <button role="tab" aria-selected={tab === "notes"} onClick={() => setTab("notes")}>Your notes</button>
      </div>

      <div className="detail-body">
        {tab === "summary" ? summaryTab : tab === "transcript" ? transcriptTab : notesTab}
      </div>

      {m.hasAudio && (player.atMs != null || player.loading) && (
        <div className="player" role="region" aria-label="Audio player">
          <button className="btn btn-primary btn-icon btn-sm" onClick={player.toggle} disabled={player.loading} aria-label={player.playing ? "Pause" : "Play"}>
            {player.loading ? <Spinner /> : <Icon name={player.playing ? "pause" : "play"} />}
          </button>
          <time>{player.atMs != null ? clock(player.atMs) : ""}</time>
          {player.error && <span className="player-error">{player.error}</span>}
          <span className="spacer" />
          <button className="btn btn-quiet btn-icon btn-sm" onClick={player.stop} aria-label="Close player">
            <Icon name="x" />
          </button>
        </div>
      )}
      {dialog}
    </div>
  );
}
