import { listen } from "@tauri-apps/api/event";
import { useEffect, useRef, useState } from "react";
import { api, Meeting, MeetingWork, ModelInfo, on, Segment, Task } from "./api";
import { MeetingSetup } from "./MeetingSetup";
import { DetailView } from "./meetings/DetailView";
import { ListView } from "./meetings/ListView";
import { LiveView } from "./meetings/LiveView";
import { mergeTask } from "./meetings/TaskCard";

type Phase = "idle" | "starting" | "recording" | "stopping";

function RecTimer() {
  const [secs, setSecs] = useState(0);
  useEffect(() => {
    const started = Date.now();
    const t = setInterval(() => setSecs((Date.now() - started) / 1000), 250);
    return () => clearInterval(t);
  }, []);
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return (
    <span className="pill-time">
      {String(m).padStart(2, "0")}:{String(s).padStart(2, "0")}
    </span>
  );
}

interface Props {
  openRequest?: { id: string; at: number | null } | null;
  stt?: ModelInfo;
  llm?: ModelInfo;
  language: string;
  models: ModelInfo[];
  onModelsChanged: () => void;
  /// Called when a meeting starts so the app can bring the Meetings view forward
  /// (e.g. when started from the tray while another tab is showing).
  onActivate?: () => void;
}

export function Meetings({ stt, llm, language, models, onModelsChanged, onActivate, openRequest }: Props) {
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
  const [live, setLive] = useState<{ id: string; title: string; attendees: string[] } | null>(null);
  const [liveSegments, setLiveSegments] = useState<Segment[]>([]);
  const [partials, setPartials] = useState<Record<string, string>>({});
  const [liveNames, setLiveNames] = useState<Record<string, string>>({});
  const [liveNotes, setLiveNotes] = useState("");
  const [status, setStatus] = useState<{ positionMs: number; pending: number; systemAudio: boolean } | null>(null);
  const [notesThrough, setNotesThrough] = useState(0);
  const [assist, setAssist] = useState<{ wake: string | null; key: string | null }>({ wake: null, key: null });

  // Meeting-specific model choices (persisted, independent of the dictate tab).
  const [prefs, setPrefs] = useState(() => ({
    stt: localStorage.getItem("unsound.meeting.stt") ?? "",
    llm: localStorage.getItem("unsound.meeting.llm") ?? "",
    emb: localStorage.getItem("unsound.meeting.emb") ?? "diarize-embedding",
    speakers: localStorage.getItem("unsound.meeting.speakers") ?? "",
  }));
  const setPref = (key: keyof typeof prefs, value: string) => {
    localStorage.setItem(`unsound.meeting.${key}`, value);
    setPrefs((p) => ({ ...p, [key]: value }));
  };

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
  const meetStt = sttModels.find((m) => m.id === prefs.stt) ?? stt ?? sttModels[0];
  const meetLlm = llmModels.find((m) => m.id === prefs.llm) ?? llm ?? llmModels[0];
  summaryModelRef.current = meetLlm?.id;
  const meetEmbId = (embModels.find((m) => m.id === prefs.emb) ?? embModels[0])?.id;

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
      // Keep the backend's idea of the notes model in step with the picker
      // (the assistant falls back to it).
      if (meetLlm && s.meetings.summaryModel !== meetLlm.id) {
        api.setMeetingSettings({ ...s.meetings, summaryModel: meetLlm.id }).catch(() => {});
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
      setError("download a speech model first (settings → models)");
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
    phaseRef.current = "starting";
    setPhase("starting");
    try {
      await api.createMeeting({
        id,
        title: "",
        startedAt: new Date().toISOString(),
        endedAt: null,
        summary: "",
        notes: "",
        sttModel: meetStt.name,
        llmModel: meetLlm?.name ?? "",
        lang: language,
        segments: [],
        segmentCount: 0,
      });
      const m = await api.meetingStart(id, meetStt.id, language || undefined, meetLlm?.id, meetEmbId);
      setLive({ id, title: m?.title ?? "", attendees: m?.attendees ?? [] });
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
    if (pending && pending.state !== "complete") { setError("Finish transcription before writing the summary."); return; }
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
  const pillLabel =
    phase === "recording"
      ? "end meeting"
      : phase === "starting"
        ? "starting…"
        : phase === "stopping"
          ? "wrapping up…"
          : "start a meeting";

  return (
    <div className="meetings">
      <div className="stage">
        <button
          className={"pill" + (phase === "recording" ? " live" : "")}
          onClick={phase === "recording" ? stopMeeting : startMeeting}
          disabled={busy || setupNeeded}
        >
          <span className="pill-dot" />
          {pillLabel}
          {phase === "recording" && <RecTimer />}
        </button>
      </div>

      {setupNeeded && !selected && !live && <MeetingSetup models={models} onModelsChanged={onModelsChanged} />}
      {Object.keys(summaries).map((id) => (
        <div key={id} data-meeting-id={id} className="meet-hint" role="status">
          {meetings.find((m) => m.id === id)?.title || "Meeting"}: writing notes…
        </div>
      ))}
      {work.filter((w) => w.state !== "complete" || w.warning).map((w) => (
        <div key={w.meetingId} data-meeting-id={w.meetingId} className={"meet-hint " + (w.error ? "meet-error" : "meet-warning")} role={w.error ? "alert" : "status"}>
          <div>
            <strong>{meetings.find((m) => m.id === w.meetingId)?.title || "Meeting"}: </strong>
            {w.state === "recording" ? "recording" : w.state === "finishing" ? "finishing transcript" : w.state === "complete" ? "transcript recovered" : w.state === "paused" ? "transcription paused" : "needs attention"}
            {` · ${w.pending} audio ranges remaining · mic saved through ${(w.micSamples / 16000).toFixed(1)}s · system saved through ${(w.systemSamples / 16000).toFixed(1)}s`}
            {w.error && <p>{w.error}</p>}
            {w.warning && <p>{w.warning}</p>}
          </div>
          {!w.recording && <button className="quiet" onClick={() => openMeeting(w.meetingId)}>open</button>}
          {!w.recording && w.state === "finishing" && <button className="quiet" onClick={() => api.meetingPause(w.meetingId).then(() => api.meetingWorkStatus()).then(setWork).catch((e) => setError(String(e)))}>pause</button>}
          {!w.recording && (w.state === "paused" || w.state === "error") && <button className="quiet" onClick={() => api.meetingRetry(w.meetingId).then(() => api.meetingWorkStatus()).then(setWork).catch((e) => setError(String(e)))}>retry saved audio</button>}
        </div>
      ))}
      {!sysSupported && (
        <div className="meet-hint">system audio needs macOS 13+ — meetings will capture your mic only</div>
      )}
      {detected && phase === "idle" && (
        <div className="meet-hint meet-banner">
          {detected.app} is using your mic.
          <button className="quiet accent" onClick={startMeeting}>
            record this meeting
          </button>
          <button className="quiet" onClick={() => setDetected(null)}>
            not now
          </button>
        </div>
      )}
      {released && phase === "recording" && (
        <div className="meet-hint meet-banner">
          {released} let go of the mic — did the call end?
          <button className="quiet accent" onClick={stopMeeting}>
            end meeting
          </button>
          <button className="quiet" onClick={() => setReleased(null)}>
            keep recording
          </button>
        </div>
      )}
      {warning && (
        <div className="meet-hint meet-warning">
          {warning}
          <button className="quiet" onClick={() => setWarning(null)}>
            dismiss
          </button>
        </div>
      )}
      {error && (
        <div className="meet-hint meet-error">
          {error}
          <button className="quiet" onClick={() => setError(null)}>
            dismiss
          </button>
        </div>
      )}

      {(phase === "recording" || phase === "starting") && live ? (
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
        />
      ) : selected ? (
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
        />
      ) : null}
      {/* Preserve the archive conversation while opening a cited meeting. */}
      <div hidden={Boolean(selected) || ((phase === "recording" || phase === "starting") && Boolean(live))}>
        <ListView
          meetings={meetings}
          onOpen={openMeeting}
          llmId={meetLlm?.id}
          idle={phase === "idle"}
          options={{
            sttModels,
            llmModels,
            embModels,
            stt: meetStt?.id,
            llm: meetLlm?.id,
            emb: meetEmbId,
            speakers: prefs.speakers,
            set: setPref,
          }}
        />
      </div>
    </div>
  );
}
