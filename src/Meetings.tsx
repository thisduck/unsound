import { useEffect, useRef, useState } from "react";
import { api, on, formatDuration, Meeting, ModelInfo, SearchHit } from "./api";

type Phase = "idle" | "recording" | "transcribing" | "summarizing";

function elapsed(startedAt: string, endedAt?: string | null): string {
  const start = new Date(startedAt).getTime();
  const end = endedAt ? new Date(endedAt).getTime() : Date.now();
  return formatDuration(Math.max(0, (end - start) / 1000));
}

function mmss(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
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
}

export function Meetings({ stt, llm, language }: Props) {
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
  const activeIdRef = useRef<string | null>(null);
  const sysStartedRef = useRef(false);
  const summaryRef = useRef(false);
  const answerRef = useRef(false);

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
    return () => {
      sub.then((un) => un());
      sub2.then((un) => un());
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
    if (!selected || !llm || !question.trim() || asking) return;
    setAnswer("");
    setAsking(true);
    answerRef.current = true;
    try {
      const a = await api.askMeeting(selected.id, llm.id, question.trim());
      setAnswer(a);
    } catch (e) {
      setAnswer(String(e));
    } finally {
      answerRef.current = false;
      setAsking(false);
    }
  };

  const startMeeting = async () => {
    if (!stt) {
      setError("download a speech model first (settings → models)");
      return;
    }
    setError(null);
    const id = crypto.randomUUID();
    const startedAt = new Date().toISOString();
    activeIdRef.current = id;
    try {
      await api.createMeeting({
        id,
        title: "",
        startedAt,
        endedAt: null,
        summary: "",
        notes: "",
        sttModel: stt.name,
        llmModel: llm?.name ?? "",
        lang: language,
        segments: [],
        segmentCount: 0,
      });
      await api.startRecording();
      sysStartedRef.current = false;
      if (sysSupported) {
        try {
          await api.startSystemCapture();
          sysStartedRef.current = true;
        } catch (e) {
          console.error("system capture failed to start", e);
        }
      }
      setSelected(null);
      setLiveSummary("");
      setPhase("recording");
      setStatus(
        sysStartedRef.current ? "recording you + the meeting…" : "recording your mic…",
      );
    } catch (e) {
      setError(String(e));
      setPhase("idle");
    }
  };

  const stopMeeting = async () => {
    const id = activeIdRef.current;
    if (!id || !stt) return;
    try {
      await api.stopRecording().catch(() => {});
      if (sysStartedRef.current) await api.stopSystemCapture().catch(() => {});
      setPhase("transcribing");
      setStatus("transcribing the meeting…");
      const m = await api.transcribeMeeting(id, stt.id, language || undefined);
      setSelected(m);
      setNotes(m.notes);
      if (llm) {
        setPhase("summarizing");
        setStatus("summarizing…");
        setLiveSummary("");
        summaryRef.current = true;
        const summary = await api.summarizeMeeting(id, llm.id);
        summaryRef.current = false;
        setSelected((cur) => (cur ? { ...cur, summary } : cur));
      }
      setPhase("idle");
      setStatus("meeting saved");
      refresh();
    } catch (e) {
      summaryRef.current = false;
      setError(String(e));
      setPhase("idle");
    }
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

  const remove = async (id: string) => {
    try {
      await api.deleteMeeting(id);
      if (selected?.id === id) setSelected(null);
      refresh();
    } catch (e) {
      setError(String(e));
    }
  };

  const busy = phase === "transcribing" || phase === "summarizing";
  const pillLabel =
    phase === "recording"
      ? "end meeting"
      : phase === "transcribing"
        ? "transcribing…"
        : phase === "summarizing"
          ? "summarizing…"
          : "start a meeting";

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
      {status && phase !== "idle" && <div className="meet-hint">{status}</div>}
      {error && (
        <div className="meet-hint meet-error">
          {error}
          <button className="quiet" onClick={() => setError(null)}>
            dismiss
          </button>
        </div>
      )}

      {selected ? (
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

          {llm && selected.segments.length > 0 && (
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
                  <div
                    className={"seg " + (s.speaker === "me" ? "seg-me" : "seg-them")}
                    key={s.id ?? i}
                  >
                    <span className="seg-who">{s.speaker === "me" ? "You" : "Them"}</span>
                    <span className="seg-time">{mmss(s.startMs)}</span>
                    <span className="seg-text">{s.text}</span>
                  </div>
                ))
            )}
          </section>
        </div>
      ) : (
        <div className="meet-list">
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
