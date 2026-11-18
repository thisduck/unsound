import { listen } from "@tauri-apps/api/event";
import { ReactNode, useEffect, useRef, useState } from "react";
import { api, clock, Meeting, MeetingSettings, MeetingWork, ModelInfo, on, Segment, Task } from "./api";
import { MeetingSetup } from "./MeetingSetup";
import { DetailView } from "./meetings/DetailView";
import { ListView } from "./meetings/ListView";
import { LiveView } from "./meetings/LiveView";
import { mergeTask } from "./meetings/TaskCard";
import { Icon, Kbd } from "./ui";

type Phase = "idle" | "starting" | "recording" | "stopping";

export interface RecordingInfo {
  id: string;
  startedAt: string;
  title: string;
}

/// Elapsed time since a start, ticking.
function Elapsed({ since }: { since: string }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(t);
  }, []);
  return <time>{clock(now - new Date(since).getTime())}</time>;
}

/// Display-only description of where a meeting's transcription stands.
function workLabel(w: MeetingWork): string {
  if (w.state === "finishing") return w.pending > 0 ? `Finishing transcript · ${w.pending} part${w.pending === 1 ? "" : "s"} left` : "Finishing transcript";
  if (w.state === "paused") return "Transcription paused. Your audio is saved.";
  if (w.state === "error") return "Transcription stopped. Your audio is saved.";
  if (w.state === "complete") return "Transcript recovered";
  return "Recording";
}

interface Props {
  openRequest?: { id: string; at: number | null } | null;
  stt?: ModelInfo;
  llm?: ModelInfo;
  language: string;
  models: ModelInfo[];
  onModelsChanged: () => void;
  /// Called when a meeting starts so the app can bring the Meetings view forward
  /// (e.g. when started from the tray while another section is showing).
  onActivate?: () => void;
  /// The recording in progress (null when idle), for the app's title bar.
  onRecordingChange?: (r: RecordingInfo | null) => void;
  /// Whether the full-window live view is showing.
  onLiveVisibleChange?: (visible: boolean) => void;
  /// Bumped to bring the live view back (title bar, floating indicator).
  liveRequest?: number;
  /// Requests and reminders that need attention, shown above the list.
  needsYou?: ReactNode;
  onOpenSettings?: () => void;
}

export function Meetings({ stt, llm, language, models, onModelsChanged, onActivate, openRequest, onRecordingChange, onLiveVisibleChange, liveRequest, needsYou, onOpenSettings }: Props) {
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [selected, setSelected] = useState<Meeting | null>(null);
  const [focusMs, setFocusMs] = useState<number | null>(null);
  useEffect(() => {
    if (!openRequest) return;
    let active = true;
    api.getMeeting(openRequest.id).then(m => { if (active && m) { setSelected(m); setFocusMs(openRequest.at); } }).catch(e => { if (active) setError(String(e)); });
    return () => { active = false; };
  }, [openRequest]);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [work, setWork] = useState<MeetingWork[]>([]);
  const [warning, setWarning] = useState<string | null>(null);
  const [detected, setDetected] = useState<{ app: string } | null>(null);
  const [released, setReleased] = useState<string | null>(null);
  const [sysSupported, setSysSupported] = useState(true);
  const [summaries, setSummaries] = useState<Record<string, string>>({});
  const [tasks, setTasks] = useState<Task[]>([]);
  // Live meeting state.
  const [live, setLive] = useState<{ id: string; title: string; attendees: string[]; startedAt: string } | null>(null);
  const [showLive, setShowLive] = useState(true);
  const [flagToken, setFlagToken] = useState(0);
  const [query, setQuery] = useState("");
  const [askToken, setAskToken] = useState(0);
  const [liveSegments, setLiveSegments] = useState<Segment[]>([]);
  const [partials, setPartials] = useState<Record<string, string>>({});
  const [liveNames, setLiveNames] = useState<Record<string, string>>({});
  const [liveNotes, setLiveNotes] = useState("");
  const [status, setStatus] = useState<{ positionMs: number; pending: number; systemAudio: boolean } | null>(null);
  const [notesThrough, setNotesThrough] = useState(0);
  const [assist, setAssist] = useState<{ wake: string | null; key: string | null }>({ wake: null, key: null });
  // Meeting model and language choices live in Settings → Meetings.
  const [meetPrefs, setMeetPrefs] = useState<Partial<MeetingSettings>>({});

  const phaseRef = useRef<Phase>("idle");
  phaseRef.current = phase;
  const summaryJobs = useRef(new Set<string>());
  const summarizeAfter = useRef(new Map<string, string>());
  const summaryModelRef = useRef<string | undefined>(undefined);
  const liveIdRef = useRef<string | null>(null);
  const selectedIdRef = useRef<string | null>(null);
  selectedIdRef.current = selected?.id ?? null;
  // Once-registered listeners reach the current handlers through refs.
  const toggleRef = useRef<() => void>(() => {});
  const startRef = useRef<() => void>(() => {});

  const sttModels = models.filter((m) => m.kind === "stt" && m.downloaded);
  const llmModels = models.filter((m) => m.kind === "llm" && m.downloaded);
  const embModels = models.filter((m) => m.kind === "diarize" && m.id !== "diarize-segmentation" && m.downloaded);
  const meetStt = sttModels.find((m) => m.id === meetPrefs.speechModel) ?? stt ?? sttModels[0];
  const meetLlm = llmModels.find((m) => m.id === meetPrefs.summaryModel) ?? llm ?? llmModels[0];
  summaryModelRef.current = meetLlm?.id;
  const meetEmbId = (embModels.find((m) => m.id === meetPrefs.embeddingModel) ?? embModels.find((m) => m.id === "diarize-embedding") ?? embModels[0])?.id;
  const meetLanguage = meetPrefs.language ?? language;

  // Setup gate: a speech model, the VAD model and a voice model.
  const vadReady = !!models.find((m) => m.id === "vad-silero")?.downloaded;
  const setupNeeded = sttModels.length === 0 || !vadReady || embModels.length === 0;

  const refresh = async () => {
    try {
      setMeetings(await api.listMeetings());
    } catch (e) {
      console.error("failed to list meetings", e);
    }
  };

  const loadAssist = () =>
    api.getSettings().then((s) => {
      const a = s.assistant;
      setAssist({
        wake: a.enabled && a.listenForWakeWord && a.wakeWord.trim() ? a.wakeWord.trim() : null,
        key: s.commandKeys[0] ?? null,
      });
      setMeetPrefs(s.meetings ?? {});
      // Model choices used to be kept in this view's local storage; move them
      // into settings once so there is a single place to change them.
      const legacy = { stt: localStorage.getItem("unsound.meeting.stt"), llm: localStorage.getItem("unsound.meeting.llm"), emb: localStorage.getItem("unsound.meeting.emb") };
      if (legacy.stt || legacy.llm || legacy.emb) {
        const next = {
          ...s.meetings,
          speechModel: s.meetings.speechModel || legacy.stt || "",
          summaryModel: s.meetings.summaryModel || legacy.llm || "",
          embeddingModel: s.meetings.embeddingModel || legacy.emb || "",
        };
        api.setMeetingSettings(next).then(() => {
          for (const k of ["stt", "llm", "emb", "speakers"]) localStorage.removeItem(`unsound.meeting.${k}`);
          setMeetPrefs(next);
        }).catch(() => {});
      }
    });

  useEffect(() => {
    refresh();
    loadAssist();
    api.systemAudioSupported().then(setSysSupported).catch(() => {});
    const subs = [
      listen("action-items-changed", async () => {
        const id = selectedIdRef.current;
        if (!id) return;
        try {
          const fresh = await api.getMeeting(id);
          if (fresh) setSelected(cur => cur?.id === id ? { ...cur, actionItems: fresh.actionItems } : cur);
        } catch (e) { setError(String(e)); }
      }),
      on.meetingSummaryToken(({ meetingId, chunk }) => {
        if (summaryJobs.current.has(meetingId)) {
          setSummaries((s) => ({ ...s, [meetingId]: (s[meetingId] ?? "") + chunk }));
        }
      }),
      // Live transcript: a finalized segment supersedes that channel's partial.
      on.meetingSegments(({ meetingId, segments: segs }) => {
        if (meetingId === selectedIdRef.current) {
          api.getMeeting(meetingId).then((m) => { if (m) setSelected((cur) => cur?.id === m.id ? m : cur); }).catch(console.error);
        }
        if (meetingId !== liveIdRef.current) return;
        setLiveSegments((cur) => [...cur, ...segs]);
        setPartials((cur) => {
          const next = { ...cur };
          for (const s of segs) delete next[s.source === "mic" ? "me" : "them"];
          return next;
        });
      }),
      on.meetingPartial(({ meetingId, speaker, text }) => { if (meetingId === liveIdRef.current) setPartials((cur) => ({ ...cur, [speaker]: text })); }),
      on.meetingStatus(setStatus),
      on.meetingWarning(setWarning),
      on.meetingSpeakerNamed(({ meetingId, speaker, name }) => { if (meetingId === liveIdRef.current) setLiveNames((n) => ({ ...n, [speaker]: name })); }),
      on.meetingNotesProgress(({ meetingId, throughMs }) => {
        if (meetingId === liveIdRef.current) setNotesThrough(throughMs);
      }),
      on.meetingNotesChanged(({ meetingId, notes }) => {
        if (meetingId === liveIdRef.current) setLiveNotes(notes);
        setSelected((cur) => (cur && cur.id === meetingId ? { ...cur, notes } : cur));
      }),
      on.taskUpdated((t) => {
        const relevant = t.meetingId && (t.meetingId === liveIdRef.current || t.meetingId === selectedIdRef.current);
        if (relevant) setTasks((list) => mergeTask(list, t));
      }),
      // Tray "Start / stop meeting".
      on.meetingToggle(() => toggleRef.current()),
      // A stop from the floating indicator keeps the normal summary flow.
      on.meetingStopped((id) => {
        if (summaryModelRef.current && !summaryJobs.current.has(id)) {
          summarizeAfter.current.set(id, summaryModelRef.current);
        }
      }),
      // "Record this meeting?" prompt (or the banner below).
      on.meetingStartRequest(() => {
        setDetected(null);
        if (phaseRef.current === "idle") startRef.current();
      }),
      on.meetingDetected(({ app }) => {
        if (phaseRef.current === "idle") setDetected({ app });
      }),
      on.meetingAppReleased(({ app }) => {
        setDetected(null);
        if (phaseRef.current === "recording") setReleased(app);
      }),
      on.settingsChanged(() => loadAssist()),
    ];
    return () => {
      subs.forEach((s) => s.then((un) => un()));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let mounted = true;
    let polling = false;
    let previousStates = new Map<string, string>();
    const poll = async () => {
      if (polling) return;
      polling = true;
      try {
        const rows = await api.meetingWorkStatus();
        if (!mounted) return;
        setWork(rows);
        const selectedId = selectedIdRef.current;
        if (selectedId && rows.some((r) => r.meetingId === selectedId && (r.state === "finishing" || (r.state === "complete" && previousStates.get(r.meetingId) !== "complete")))) {
          const m = await api.getMeeting(selectedId);
          if (mounted && m) setSelected((cur) => cur?.id === m.id ? m : cur);
        }
        previousStates = new Map(rows.map((r) => [r.meetingId, r.state]));
        const id = liveIdRef.current;
        const current = rows.find((r) => r.meetingId === id);
        if (id && phaseRef.current === "recording" && current && !current.recording) {
          liveIdRef.current = null;
          setLive(null);
          setPhase("idle");
          setLiveSegments([]);
          setPartials({});
          const meeting = await api.getMeeting(id);
          if (mounted && meeting && !liveIdRef.current) setSelected(meeting);
          refresh();
        }
      } catch (e) { if (mounted) setError(`Could not read saving status: ${String(e)}`); }
      finally { polling = false; }
    };
    poll();
    const timer = window.setInterval(poll, 1000);
    return () => { mounted = false; window.clearInterval(timer); };
  }, []);

  const openMeeting = async (id: string, atMs?: number | null) => {
    try {
      const m = await api.getMeeting(id);
      if (m) {
        setSelected(m);
        setFocusMs(atMs ?? null);
        setTasks(await api.listTasks(id));
      }
    } catch (e) {
      setError(String(e));
    }
  };

  const startMeeting = async () => {
    if (phaseRef.current !== "idle") return;
    if (!meetStt) {
      setError("Download a speech model to record meetings.");
      return;
    }
    setError(null);
    setWarning(null);
    setDetected(null);
    setReleased(null);
    api.hideMeetingPrompt().catch(() => {});
    const id = crypto.randomUUID();
    liveIdRef.current = id;
    onActivate?.();
    setSelected(null);
    setLiveSegments([]);
    setPartials({});
    setLiveNames({});
    setLiveNotes("");
    setStatus(null);
    setNotesThrough(0);
    setTasks([]);
    setShowLive(true);
    phaseRef.current = "starting";
    setPhase("starting");
    const startedAt = new Date().toISOString();
    try {
      await api.createMeeting({
        id,
        title: "",
        startedAt,
        endedAt: null,
        summary: "",
        notes: "",
        sttModel: meetStt.name,
        llmModel: meetLlm?.name ?? "",
        lang: meetLanguage,
        segments: [],
        segmentCount: 0,
      });
      const m = await api.meetingStart(id, meetStt.id, meetLanguage || undefined, meetLlm?.id, meetEmbId);
      setLive({ id, title: m?.title ?? "", attendees: m?.attendees ?? [], startedAt });
      setPhase("recording");
    } catch (e) {
      setError(String(e));
      setPhase("idle");
      liveIdRef.current = null;
    }
  };

  const summarize = async (id: string, modelId = meetLlm?.id) => {
    if (!modelId || summaryJobs.current.has(id)) return;
    const pending = work.find((w) => w.meetingId === id);
    if (pending && pending.state !== "complete") { setError("The transcript is still finishing. Write the summary when it's done."); return; }
    summaryJobs.current.add(id);
    summarizeAfter.current.delete(id);
    setSummaries((s) => ({ ...s, [id]: "" }));
    try {
      const m = await api.summarizeMeeting(id, modelId);
      setSelected((cur) => (cur && cur.id === id ? { ...m, segments: m.segments } : cur));
    } catch (e) {
      setError(String(e));
    } finally {
      summaryJobs.current.delete(id);
      setSummaries((s) => {
        const next = { ...s };
        delete next[id];
        return next;
      });
      refresh();
    }
  };

  const stopMeeting = async () => {
    const id = liveIdRef.current;
    if (!id || phaseRef.current !== "recording") return;
    phaseRef.current = "stopping";
    setPhase("stopping");
    setReleased(null);
    try {
      // Capture stops promptly; durable transcription continues in the background.
      const m = await api.meetingStop(id);
      liveIdRef.current = null;
      setLive(null);
      setPhase("idle");
      if (m) {
        selectedIdRef.current = m.id;
        setSelected(m);
        api.listTasks(id).then((tasks) => {
          if (!liveIdRef.current && selectedIdRef.current === id) setTasks(tasks);
        }).catch((e) => setError(String(e)));
      }
      setLiveSegments([]);
      setPartials({});
      if (meetLlm && !summaryJobs.current.has(id)) summarizeAfter.current.set(id, meetLlm.id);
      api.meetingWorkStatus().then(setWork).catch((e) => setError(String(e)));
      refresh();
    } catch (e) {
      setError(String(e));
      // If native capture did stop before the error, the status poll releases
      // this state; otherwise keep the stop control available for a retry.
      setPhase("recording");
    }
  };

  useEffect(() => {
    // A stopped meeting keeps processing even after navigation or a new recording.
    for (const [id, modelId] of summarizeAfter.current) {
      if (work.some((w) => w.meetingId === id && w.state === "complete")) {
        void summarize(id, modelId);
      }
    }
  }, [work]);

  toggleRef.current = () => {
    if (phaseRef.current === "recording") stopMeeting();
    else if (phaseRef.current === "idle") startMeeting();
  };
  startRef.current = () => startMeeting();

  const remove = async (id: string) => {
    try {
      await api.deleteMeeting(id);
      localStorage.removeItem(`unsound.notes-draft.${id}`);
      if (selected?.id === id) setSelected(null);
      refresh();
    } catch (e) {
      setError(String(e));
    }
  };

  const busy = phase === "starting" || phase === "stopping";
  const recordingNow = (phase === "recording" || phase === "starting") && !!live;
  const liveVisible = recordingNow && showLive;
  const liveId = live?.id ?? null;
  const liveTitle = live?.title ?? "";
  const liveStartedAt = live?.startedAt ?? "";

  useEffect(() => {
    onRecordingChange?.(recordingNow && liveId ? { id: liveId, startedAt: liveStartedAt, title: liveTitle } : null);
  }, [recordingNow, liveId, liveTitle, liveStartedAt]);
  useEffect(() => { onLiveVisibleChange?.(liveVisible); }, [liveVisible]);
  useEffect(() => { if (liveRequest) setShowLive(true); }, [liveRequest]);

  const titleOf = (id: string) => meetings.find((m) => m.id === id)?.title || "Untitled meeting";
  const refreshWork = () => api.meetingWorkStatus().then(setWork).catch((e) => setError(String(e)));
  const retryWork = (id: string) => api.meetingRetry(id).then(refreshWork).catch((e) => setError(String(e)));
  const pauseWork = (id: string) => api.meetingPause(id).then(refreshWork).catch((e) => setError(String(e)));

  /// The one record/stop control. Exactly one is on screen in every state.
  const recordControl = (variant: "stage" | "card" | "bar" | "small") => {
    const recordingPhase = phase === "recording";
    const label = recordingPhase ? "Stop recording" : phase === "starting" ? "Starting…" : phase === "stopping" ? "Saving…" : variant === "small" ? "Record" : detected ? "Record this call" : "Record";
    const cls =
      variant === "stage" || variant === "card" ? "btn btn-stage" : variant === "bar" ? "btn btn-secondary" : "btn btn-secondary btn-sm";
    return (
      <button
        className={cls + " record-control" + (recordingPhase ? " live" : "")}
        data-record-control=""
        onClick={recordingPhase ? stopMeeting : startMeeting}
        disabled={busy || (setupNeeded && !recordingPhase)}
      >
        {recordingPhase ? <span className="stop-glyph" aria-hidden="true" /> : <span className="rec-glyph" aria-hidden="true" />}
        {label}
      </button>
    );
  };

  const alerts = (
    <>
      {work.filter((w) => !w.recording && (w.state === "paused" || w.state === "error" || (w.state === "complete" && w.warning))).map((w) => (
        <div key={w.meetingId} data-meeting-id={w.meetingId} className={"alert " + (w.state === "complete" ? "warn" : "danger")} role={w.state === "error" ? "alert" : "status"}>
          <Icon name="alert" />
          <div className="alert-body">
            <strong>{titleOf(w.meetingId)}</strong> · {workLabel(w)}
            {(w.error || w.warning) && <p className="alert-reason">{w.error || w.warning}</p>}
            {w.state !== "complete" && (
              <details className="alert-details">
                <summary>Details</summary>
                {w.error && w.warning && <p>{w.warning}</p>}
                <p>Mic saved through {(w.micSamples / 16000).toFixed(1)}s · call audio saved through {(w.systemSamples / 16000).toFixed(1)}s · {w.pending} part{w.pending === 1 ? "" : "s"} left</p>
              </details>
            )}
          </div>
          <div className="alert-actions">
            <button className="btn btn-quiet btn-sm" onClick={() => openMeeting(w.meetingId)}>Open</button>
            {(w.state === "paused" || w.state === "error") && <button className="btn btn-secondary btn-sm" onClick={() => retryWork(w.meetingId)}>Try again</button>}
          </div>
        </div>
      ))}
      {warning && (
        <div className="alert warn" role="status">
          <Icon name="alert" />
          <span className="alert-body">{warning}</span>
          <div className="alert-actions"><button className="btn btn-quiet btn-sm" onClick={() => setWarning(null)}>Dismiss</button></div>
        </div>
      )}
      {error && (
        <div className="alert danger" role="alert">
          <Icon name="alert" />
          <span className="alert-body">{error}</span>
          <div className="alert-actions"><button className="btn btn-quiet btn-sm" onClick={() => setError(null)}>Dismiss</button></div>
        </div>
      )}
    </>
  );

  const callEnded = released && phase === "recording" && (
    <div className="call-ended" role="status">
      <Icon name="info" />
      <span><strong>{released} left the call.</strong> Stop recording?</span>
      <span className="spacer" />
      <button className="btn btn-quiet btn-sm" onClick={() => setReleased(null)}>Keep recording</button>
      <button className="btn btn-primary btn-sm" onClick={stopMeeting}>Stop recording</button>
    </div>
  );

  // ── live: the whole window, with the recording stage on top ──
  if (liveVisible && live) {
    return (
      <div className="live-page">
        <header className="stage" data-tauri-drag-region>
          <div className="stage-clock" data-tauri-drag-region>
            <span className="rec-dot" />
            <span className="stage-timer"><Elapsed since={live.startedAt} /></span>
          </div>
          <div className="stage-title" data-tauri-drag-region>
            <strong>{live.title || "Recording"}</strong>
            <span>{live.attendees.length ? live.attendees.join(", ") : status && !status.systemAudio ? "Your mic only" : "You and the call"}</span>
          </div>
          <span className="spacer" data-tauri-drag-region />
          <button className="btn btn-stage-quiet" onClick={() => setShowLive(false)} title="Browse meetings while recording">
            <Icon name="back" /> Meetings
          </button>
          <button className="btn btn-stage-outline" onClick={() => setFlagToken((n) => n + 1)} title="Flag this moment. It gets extra weight in the summary.">
            <Icon name="flag" /> Flag <Kbd>⌘M</Kbd>
          </button>
          {recordControl("stage")}
        </header>
        {callEnded}
        {(warning || error) && <div className="live-alerts">{alerts}</div>}
        <LiveView
          meetingId={live.id}
          title={live.title}
          attendees={live.attendees}
          segments={liveSegments}
          partials={partials}
          speakerNames={liveNames}
          notes={liveNotes}
          onNotes={setLiveNotes}
          tasks={tasks}
          onTasks={setTasks}
          status={status}
          wakeWord={assist.wake}
          commandKey={assist.key}
          notesThroughMs={notesThrough}
          flagToken={flagToken}
          onFlag={() => setFlagToken((n) => n + 1)}
        />
      </div>
    );
  }

  const banner = recordingNow && live && (
    <div className="recording-banner" role="status">
      <span className="rec-dot" />
      <span className="recording-banner-text">
        <strong>Recording</strong> <Elapsed since={live.startedAt} />
        {live.title && <span className="faint"> · {live.title}</span>}
      </span>
      <span className="spacer" />
      <button className="btn btn-quiet btn-sm" onClick={() => setShowLive(true)}>Open live view</button>
      {recordControl("bar")}
    </div>
  );

  // ── a meeting ──
  if (selected) {
    return (
      <div className="page meetings">
        {banner}
        {callEnded}
        <div className="page-bar">
          <button className="btn btn-quiet btn-sm back-link" onClick={() => { setSelected(null); refresh(); }}>
            <Icon name="back" /> Meetings
          </button>
          <span className="spacer" />
          {!banner && recordControl("small")}
        </div>
        <div className="alerts">{alerts}</div>
        <DetailView
          key={selected.id}
          meeting={selected}
          onMeeting={setSelected}
          onBack={() => {
            setSelected(null);
            refresh();
          }}
          onDelete={() => remove(selected.id)}
          tasks={tasks}
          onTasks={setTasks}
          liveSummary={summaries[selected.id] ?? null}
          summarizing={summaries[selected.id] !== undefined}
          onSummarize={() => summarize(selected.id)}
          llmId={meetLlm?.id}
          embId={meetEmbId}
          diarizeReady={embModels.length > 0}
          focusMs={focusMs}
          work={work.find((w) => w.meetingId === selected.id && w.state !== "complete")}
          onRetryWork={() => retryWork(selected.id)}
        />
      </div>
    );
  }

  // ── the list ──
  return (
    <div className="page meetings">
      <div className="page-head">
        <h1 className="page-title">Meetings</h1>
        <span className="spacer" />
        {meetings.length > 0 && (
          <form className="search meet-search" role="search" onSubmit={(e) => { e.preventDefault(); if (query.trim()) setAskToken((n) => n + 1); }}>
            <Icon name="search" />
            <input
              className="input"
              aria-label="Search or ask your meetings"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Escape") setQuery(""); }}
              placeholder="Search or ask your meetings…"
            />
            <Kbd>↩ Ask</Kbd>
          </form>
        )}
      </div>
      <div className="alerts">{alerts}</div>
      {banner || (
        <div className={"record-card" + (detected && phase === "idle" ? " detected" : "")}>
          <span className="record-card-icon"><Icon name={detected ? "video" : "mic"} /></span>
          <div className="record-card-text">
            <h2>{detected && phase === "idle" ? `${detected.app} call started` : "Record a meeting"}</h2>
            <p>
              {setupNeeded
                ? "Finish the downloads below to record your first meeting."
                : detected && phase === "idle"
                  ? "Record it and get a summary when it ends."
                  : sysSupported
                    ? "unsound transcribes you and the call, tells speakers apart, and writes a summary when you stop."
                    : "This Mac can only record your mic. Recording call audio needs macOS 13 or later."}
            </p>
          </div>
          <div className="record-card-actions">
            {detected && phase === "idle" && <button className="btn btn-stage-quiet" onClick={() => setDetected(null)}>Not now</button>}
            {recordControl("card")}
          </div>
        </div>
      )}
      {setupNeeded && <MeetingSetup models={models} onModelsChanged={onModelsChanged} />}
      {needsYou}
      <ListView
        meetings={meetings}
        onOpen={openMeeting}
        llmId={meetLlm?.id}
        idle={phase === "idle"}
        query={query}
        askToken={askToken}
        onClearQuery={() => setQuery("")}
        work={work}
        writing={Object.keys(summaries)}
        onPause={pauseWork}
        onOpenSettings={onOpenSettings}
      />
    </div>
  );
}
