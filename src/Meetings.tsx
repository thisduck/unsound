import { useEffect, useRef, useState } from "react";
import { api, on, formatDuration, Meeting, ModelInfo, SearchHit, Segment } from "./api";
import { MeetingSetup } from "./MeetingSetup";

type Phase = "idle" | "recording" | "transcribing" | "summarizing" | "diarizing";

/// How a segment's speaker is shown: "You" for the mic, "Speaker N" for a
/// diarized participant (`them:0` → Speaker 1), else "Them".
function speakerLabel(speaker: string): string {
  if (speaker === "me") return "You";
  const m = speaker.match(/^them:(\d+)$/);
  if (m) return `Speaker ${Number(m[1]) + 1}`;
  return "Them";
}

/// Stable-ish color class per speaker so the transcript is easy to scan.
function speakerClass(speaker: string): string {
  if (speaker === "me") return "seg-me";
  const m = speaker.match(/^them:(\d+)$/);
  if (m) return `seg-spk${Number(m[1]) % 6}`;
  return "seg-them";
}

/// A concise meeting title derived from the summary's opening sentence.
function deriveTitle(summary: string): string {
  const lines = summary.split("\n");
  const idx = lines.findIndex((l) => /^#+\s*summary/i.test(l.trim()));
  const start = idx >= 0 ? idx + 1 : 0;
  const text = lines
    .slice(start)
    .map((l) => l.trim())
    .find((l) => l.length > 0 && !l.startsWith("#"));
  if (!text) return "";
  const sentence = (text.split(/(?<=[.!?])\s/)[0] ?? text).replace(/[*_`]/g, "").trim();
  return sentence.length > 60 ? sentence.slice(0, 57).trimEnd() + "…" : sentence;
}

function elapsed(startedAt: string, endedAt?: string | null): string {
  const start = new Date(startedAt).getTime();
  const end = endedAt ? new Date(endedAt).getTime() : Date.now();
  return formatDuration(Math.max(0, (end - start) / 1000));
}

function mmss(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/// The whole meeting as shareable Markdown.
function meetingMarkdown(m: Meeting): string {
  const out: string[] = [`# ${m.title || "Untitled meeting"}`, ""];
  out.push(`_${new Date(m.startedAt).toLocaleString()} · ${elapsed(m.startedAt, m.endedAt)}_`, "");
  if (m.summary.trim()) out.push(m.summary.trim(), "");
  if (m.notes.trim()) out.push("## My notes", "", m.notes.trim(), "");
  if (m.segments.length) {
    out.push("## Transcript", "");
    for (const s of [...m.segments].sort((a, b) => a.startMs - b.startMs)) {
      out.push(`**${speakerLabel(s.speaker)}** (${mmss(s.startMs)}): ${s.text}`);
    }
  }
  return out.join("\n");
}

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
  stt?: ModelInfo;
  llm?: ModelInfo;
  language: string;
  models: ModelInfo[];
  onModelsChanged: () => void;
  /// Called when a meeting starts so the app can bring the Meetings view forward
  /// (e.g. when started from the tray while another tab is showing).
  onActivate?: () => void;
}

export function Meetings({ stt, llm, language, models, onModelsChanged, onActivate }: Props) {
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [selected, setSelected] = useState<Meeting | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sysSupported, setSysSupported] = useState(true);
  const [liveSummary, setLiveSummary] = useState("");
  const [notes, setNotes] = useState("");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchHit[] | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [asking, setAsking] = useState(false);
  const [testing, setTesting] = useState(false);
  const [copied, setCopied] = useState(false);
  const [liveSegments, setLiveSegments] = useState<Segment[]>([]);
  // Tentative in-progress transcript per channel ("me"/"them"), shown dimmed.
  const [partials, setPartials] = useState<Record<string, string>>({});
  // Meeting-specific model choices (persisted, independent of the dictate tab).
  const [meetingSttId, setMeetingSttId] = useState(
    () => localStorage.getItem("unsound.meeting.stt") ?? "",
  );
  const [meetingEmbId, setMeetingEmbId] = useState(
    () => localStorage.getItem("unsound.meeting.emb") ?? "diarize-embedding",
  );
  const [meetingLlmId, setMeetingLlmId] = useState(
    () => localStorage.getItem("unsound.meeting.llm") ?? "",
  );
  const [meetingSpeakers, setMeetingSpeakers] = useState(
    () => localStorage.getItem("unsound.meeting.speakers") ?? "",
  );
  const persist = (key: string, value: string, set: (v: string) => void) => {
    localStorage.setItem(key, value);
    set(value);
  };
  const activeIdRef = useRef<string | null>(null);
  const summaryRef = useRef(false);
  const answerRef = useRef(false);
  const phaseRef = useRef<Phase>("idle");
  phaseRef.current = phase;
  // The tray-toggle listener is registered once; route it through a ref so it
  // always sees the current phase and handlers.
  const toggleRef = useRef<() => void>(() => {});

  // Downloaded models available to meetings, and the resolved choices.
  const sttModels = models.filter((m) => m.kind === "stt" && m.downloaded);
  const llmModels = models.filter((m) => m.kind === "llm" && m.downloaded);
  const embModels = models.filter(
    (m) => m.kind === "diarize" && m.id !== "diarize-segmentation" && m.downloaded,
  );
  const meetStt = sttModels.find((m) => m.id === meetingSttId) ?? stt ?? sttModels[0];
  const meetLlm = llmModels.find((m) => m.id === meetingLlmId) ?? llm ?? llmModels[0];
  const meetEmbId = (embModels.find((m) => m.id === meetingEmbId) ?? embModels[0])?.id;
  const numSpeakers = meetingSpeakers ? parseInt(meetingSpeakers, 10) : undefined;

  // Setup gate: meetings need a speech model, the VAD model, the segmentation
  // model, and at least one speaker-embedding model.
  const sttReady = sttModels.length > 0;
  const vadReady = !!models.find((m) => m.id === "vad-silero")?.downloaded;
  const segReady = !!models.find((m) => m.id === "diarize-segmentation")?.downloaded;
  const diarizeReady = segReady && embModels.length > 0;
  const setupNeeded = !sttReady || !vadReady || !diarizeReady;

  const refresh = async () => {
    try {
      setMeetings(await api.listMeetings());
    } catch (e) {
      console.error("failed to list meetings", e);
    }
  };

  useEffect(() => {
    refresh();
    api.systemAudioSupported().then(setSysSupported).catch(() => {});
    const sub = on.meetingSummaryToken((chunk) => {
      if (summaryRef.current) setLiveSummary((s) => s + chunk);
    });
    const sub2 = on.meetingAnswerToken((chunk) => {
      if (answerRef.current) setAnswer((a) => a + chunk);
    });
    // Live transcript: segments finalized by the backend as the meeting runs.
    // A finalized segment supersedes that channel's partial.
    const sub3 = on.meetingSegments((segs) => {
      setLiveSegments((cur) => [...cur, ...segs]);
      setPartials((cur) => {
        const next = { ...cur };
        for (const s of segs) delete next[s.speaker];
        return next;
      });
    });
    const sub5 = on.meetingPartial(({ speaker, text }) =>
      setPartials((cur) => ({ ...cur, [speaker]: text })),
    );
    // Tray "Start / stop meeting".
    const sub4 = on.meetingToggle(() => toggleRef.current());
    return () => {
      sub.then((un) => un());
      sub2.then((un) => un());
      sub3.then((un) => un());
      sub4.then((un) => un());
      sub5.then((un) => un());
    };
  }, []);

  // Cross-meeting search (debounced). Empty query → show the full list.
  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setResults(null);
      return;
    }
    const t = setTimeout(async () => {
      try {
        setResults(await api.searchMeetings(q));
      } catch (e) {
        console.error("search failed", e);
      }
    }, 200);
    return () => clearTimeout(t);
  }, [query]);

  const openMeeting = async (id: string) => {
    try {
      const m = await api.getMeeting(id);
      if (m) {
        setSelected(m);
        setNotes(m.notes);
        setLiveSummary("");
        setQuestion("");
        setAnswer("");
      }
    } catch (e) {
      setError(String(e));
    }
  };

  const ask = async () => {
    if (!selected || !meetLlm || !question.trim() || asking) return;
    setAnswer("");
    setAsking(true);
    answerRef.current = true;
    try {
      const a = await api.askMeeting(selected.id, meetLlm.id, question.trim());
      setAnswer(a);
    } catch (e) {
      setAnswer(String(e));
    } finally {
      answerRef.current = false;
      setAsking(false);
    }
  };

  const startMeeting = async () => {
    if (!meetStt) {
      setError("download a speech model first (settings → models)");
      return;
    }
    setError(null);
    const id = crypto.randomUUID();
    const startedAt = new Date().toISOString();
    activeIdRef.current = id;
    onActivate?.();
    setSelected(null);
    setLiveSegments([]);
    setPartials({});
    setLiveSummary("");
    setPhase("recording");
    setStatus(
      sysSupported
        ? "recording you + the meeting — transcribing live…"
        : "recording your mic — transcribing live…",
    );
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
        lang: language,
        segments: [],
        segmentCount: 0,
      });
      // The backend now owns capture + rolling transcription; it emits
      // `meeting-segments` as they finalize.
      await api.meetingStart(id, meetStt.id, language || undefined);
    } catch (e) {
      setError(String(e));
      setPhase("idle");
    }
  };

  const stopMeeting = async () => {
    const id = activeIdRef.current;
    if (!id) return;
    try {
      setPhase("transcribing");
      setStatus("wrapping up…");
      // Transcription already happened live; this just flushes the tail.
      await api.meetingStop();
      let m = await api.getMeeting(id);
      if (m) {
        setSelected(m);
        setNotes(m.notes);
      }
      // Tell the remote speakers apart (Speaker 1/2/…) before summarizing, so
      // the summary can attribute points to the right person.
      const hasThem = !!m && m.segments.some((s) => s.source === "system");
      if (diarizeReady && hasThem) {
        setPhase("diarizing");
        setStatus("detecting speakers…");
        try {
          m = await api.diarizeMeeting(id, meetEmbId, numSpeakers);
          setSelected(m);
          setNotes(m.notes);
        } catch (e) {
          console.error("diarization failed", e);
        }
      }
      if (meetLlm && m && m.segments.length > 0) {
        setPhase("summarizing");
        setStatus("summarizing…");
        setLiveSummary("");
        summaryRef.current = true;
        const summary = await api.summarizeMeeting(id, meetLlm.id);
        summaryRef.current = false;
        setSelected((cur) => (cur ? { ...cur, summary } : cur));
        // Give an untitled meeting a name from its summary.
        if (m && !m.title.trim() && summary.trim()) {
          const title = deriveTitle(summary);
          if (title) {
            api.renameMeeting(id, title).catch((e) => console.error(e));
            setSelected((cur) => (cur ? { ...cur, title } : cur));
          }
        }
      }
      setPhase("idle");
      setStatus("meeting saved");
      setLiveSegments([]);
      setPartials({});
      refresh();
    } catch (e) {
      summaryRef.current = false;
      setError(String(e));
      setPhase("idle");
    }
  };

  // Always points at the current handlers so the once-registered tray listener
  // never sees stale state.
  toggleRef.current = () => {
    if (phaseRef.current === "recording") stopMeeting();
    else if (phaseRef.current === "idle") startMeeting();
  };

  const saveNotes = async () => {
    if (!selected) return;
    try {
      await api.updateMeetingNotes(selected.id, notes);
      setSelected({ ...selected, notes });
    } catch (e) {
      console.error("failed to save notes", e);
    }
  };

  // Diagnostic: capture 8s of system audio to a WAV and open it, so capture can
  // be verified by ear — completely independent of Whisper.
  const runTest = async () => {
    if (testing || phase !== "idle") return;
    setTesting(true);
    setError(null);
    try {
      await api.startSystemCapture();
      setStatus("capturing system audio for 8s — play a YouTube video or your call now…");
      await new Promise((r) => setTimeout(r, 8000));
      await api.stopSystemCapture();
      const r = await api.saveSystemCaptureWav();
      if (r.peak < 0.01) {
        setStatus(
          `no sound captured (peak ${r.peak.toFixed(4)}) — the system channel isn't receiving audio. file: ${r.path}`,
        );
      } else {
        setStatus(
          `captured ${r.durationSecs.toFixed(1)}s · peak ${r.peak.toFixed(3)} · rms ${r.rms.toFixed(3)} — opening it so you can listen`,
        );
        try {
          const { openPath } = await import("@tauri-apps/plugin-opener");
          await openPath(r.path);
        } catch (e) {
          console.error("could not open the wav", e);
          setStatus((s) => s + ` (open it yourself: ${r.path})`);
        }
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setTesting(false);
    }
  };

  const remove = async (id: string) => {
    try {
      await api.deleteMeeting(id);
      if (selected?.id === id) setSelected(null);
      refresh();
    } catch (e) {
      setError(String(e));
    }
  };

  const busy = phase === "transcribing" || phase === "summarizing" || phase === "diarizing";
  const pillLabel =
    phase === "recording"
      ? "end meeting"
      : phase === "transcribing"
        ? "wrapping up…"
        : phase === "diarizing"
          ? "detecting speakers…"
          : phase === "summarizing"
            ? "summarizing…"
            : "start a meeting";

  if (setupNeeded) {
    return (
      <div className="meetings">
        <MeetingSetup models={models} onModelsChanged={onModelsChanged} />
      </div>
    );
  }

  return (
    <div className="meetings">
      <div className="stage">
        <button
          className={"pill" + (phase === "recording" ? " live" : "")}
          onClick={phase === "recording" ? stopMeeting : startMeeting}
          disabled={busy}
        >
          <span className="pill-dot" />
          {pillLabel}
          {phase === "recording" && <RecTimer />}
        </button>
      </div>

      {!sysSupported && (
        <div className="meet-hint">
          system audio needs macOS 13+ — meetings will capture your mic only
        </div>
      )}
      {sysSupported && phase === "idle" && !selected && (
        <div className="meet-hint">
          <button className="quiet" onClick={runTest} disabled={testing}>
            {testing ? "capturing 8s…" : "🔧 test system audio (saves a WAV to listen to)"}
          </button>
        </div>
      )}
      {status && (phase !== "idle" || testing || !selected) && (
        <div className="meet-hint">{status}</div>
      )}
      {error && (
        <div className="meet-hint meet-error">
          {error}
          <button className="quiet" onClick={() => setError(null)}>
            dismiss
          </button>
        </div>
      )}

      {phase === "recording" ? (
        <div className="meet-detail">
          <div className="meet-detail-meta">
            recording · {liveSegments.length} line{liveSegments.length === 1 ? "" : "s"} so far
          </div>
          <section className="meet-section">
            <h3>live transcript</h3>
            {liveSegments.length === 0 && Object.keys(partials).length === 0 ? (
              <p className="dim">listening…</p>
            ) : (
              [...liveSegments]
                .sort((a, b) => a.startMs - b.startMs)
                .map((s, i) => (
                  <div className={"seg " + speakerClass(s.speaker)} key={i}>
                    <span className="seg-who">{speakerLabel(s.speaker)}</span>
                    <span className="seg-time">{mmss(s.startMs)}</span>
                    <span className="seg-text">{s.text}</span>
                  </div>
                ))
            )}
            {Object.entries(partials).map(([speaker, text]) => (
              <div className={"seg seg-partial " + speakerClass(speaker)} key={"partial-" + speaker}>
                <span className="seg-who">{speakerLabel(speaker)}</span>
                <span className="seg-time">…</span>
                <span className="seg-text">{text}</span>
              </div>
            ))}
          </section>
        </div>
      ) : selected ? (
        <div className="meet-detail">
          <div className="meet-detail-head">
            <button
              className="quiet"
              onClick={() => {
                setSelected(null);
                refresh();
              }}
            >
              ← all meetings
            </button>
            <input
              className="meet-title-input"
              value={selected.title}
              placeholder="Untitled meeting"
              onChange={(e) => setSelected({ ...selected, title: e.target.value })}
              onBlur={() =>
                api.renameMeeting(selected.id, selected.title).catch((e) => console.error(e))
              }
            />
            <span className="spacer" />
            <button
              className="quiet"
              onClick={() => {
                navigator.clipboard.writeText(meetingMarkdown(selected));
                setCopied(true);
                setTimeout(() => setCopied(false), 1500);
              }}
            >
              {copied ? "copied ✓" : "copy"}
            </button>
            <button className="quiet danger" onClick={() => remove(selected.id)}>
              delete
            </button>
          </div>
          <div className="meet-detail-meta">
            {new Date(selected.startedAt).toLocaleString()} ·{" "}
            {elapsed(selected.startedAt, selected.endedAt)}
          </div>

          {(selected.summary || liveSummary) && (
            <section className="meet-section">
              <h3>summary</h3>
              <div className="prose meet-prose">
                {liveSummary || selected.summary}
                {phase === "summarizing" && <span className="caret" />}
              </div>
            </section>
          )}

          {meetLlm && selected.segments.length > 0 && (
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
                  placeholder="e.g. what did I agree to? what are the action items?"
                  disabled={asking}
                />
                <button
                  className="quiet accent"
                  onClick={ask}
                  disabled={asking || !question.trim()}
                >
                  {asking ? "…" : "ask"}
                </button>
              </div>
              {(answer || asking) && (
                <div className="prose meet-prose meet-answer">
                  {answer}
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
              onBlur={saveNotes}
              placeholder="jot anything down — saved with this meeting"
            />
          </section>

          <section className="meet-section">
            <h3>transcript</h3>
            {selected.segments.length === 0 ? (
              <p className="dim">no transcript for this meeting</p>
            ) : (
              [...selected.segments]
                .sort((a, b) => a.startMs - b.startMs)
                .map((s, i) => (
                  <div className={"seg " + speakerClass(s.speaker)} key={s.id ?? i}>
                    <span className="seg-who">{speakerLabel(s.speaker)}</span>
                    <span className="seg-time">{mmss(s.startMs)}</span>
                    <span className="seg-text">{s.text}</span>
                  </div>
                ))
            )}
          </section>
        </div>
      ) : (
        <div className="meet-list">
          <details className="meet-options">
            <summary>meeting options</summary>
            <div className="meet-options-body">
              <label>
                <span>Speech model</span>
                <select
                  value={meetStt?.id ?? ""}
                  onChange={(e) => persist("unsound.meeting.stt", e.target.value, setMeetingSttId)}
                >
                  {sttModels.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Speaker detection</span>
                <select
                  value={meetEmbId ?? ""}
                  onChange={(e) => persist("unsound.meeting.emb", e.target.value, setMeetingEmbId)}
                >
                  {embModels.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>People on the call (besides you)</span>
                <select
                  value={meetingSpeakers}
                  onChange={(e) =>
                    persist("unsound.meeting.speakers", e.target.value, setMeetingSpeakers)
                  }
                >
                  <option value="">Auto-detect</option>
                  {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => (
                    <option key={n} value={String(n)}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Summary model</span>
                <select
                  value={meetLlm?.id ?? ""}
                  onChange={(e) => persist("unsound.meeting.llm", e.target.value, setMeetingLlmId)}
                >
                  {llmModels.length === 0 && <option value="">none downloaded</option>}
                  {llmModels.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </select>
              </label>
              <p className="meet-options-note">
                Getting too many speakers? Auto-detect tends to over-split — set the exact number of
                people on the call, or try the TitaNet speaker model (download it in setup).
              </p>
            </div>
          </details>

          {meetings.length > 0 && (
            <input
              className="meet-search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="search across all meetings…"
            />
          )}

          {results !== null ? (
            results.length === 0 ? (
              <p className="dim meet-empty-line">no meetings match “{query.trim()}”</p>
            ) : (
              results.map((h) => (
                <div className="meet-row" key={h.meetingId} onClick={() => openMeeting(h.meetingId)}>
                  <div className="meet-row-title">{h.title || "Untitled meeting"}</div>
                  <div className="meet-row-meta">{new Date(h.startedAt).toLocaleString()}</div>
                  {h.snippet && <div className="meet-row-snippet">…{h.snippet}…</div>}
                </div>
              ))
            )
          ) : (
            <>
              {meetings.length === 0 && phase === "idle" && (
                <div className="empty">
                  <p>
                    your meetings land here. Start one above, open Meet or Zoom, and unsound
                    listens to both you and the call — then writes up who said what and a summary.
                    All on this machine.
                  </p>
                </div>
              )}
              {meetings.map((m) => (
                <div className="meet-row" key={m.id} onClick={() => openMeeting(m.id)}>
                  <div className="meet-row-title">{m.title || "Untitled meeting"}</div>
                  <div className="meet-row-meta">
                    {new Date(m.startedAt).toLocaleString()} · {elapsed(m.startedAt, m.endedAt)} ·{" "}
                    {m.segmentCount} line{m.segmentCount === 1 ? "" : "s"}
                  </div>
                  {m.summary && <div className="meet-row-snippet">{m.summary.slice(0, 160)}</div>}
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}
