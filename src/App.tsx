import { useCallback, useEffect, useRef, useState } from "react";
import { api, on, ModelInfo } from "./api";
import { ModelManager } from "./ModelManager";
import "./App.css";

type Phase = "idle" | "recording" | "transcribing" | "cleaning";

const OLD_DEFAULT_PROMPT =
  "You clean up raw speech-to-text transcripts. Fix punctuation, capitalization and obvious transcription errors, remove filler words (um, uh, you know), and break the text into paragraphs where natural. Preserve the speaker's wording and meaning; do not summarize or add anything. Output only the cleaned text.";

function useLocalStorage(key: string, initial: string) {
  const [value, setValue] = useState(() => localStorage.getItem(key) ?? initial);
  useEffect(() => {
    localStorage.setItem(key, value);
  }, [key, value]);
  return [value, setValue] as const;
}

function Meter({ level }: { level: number }) {
  const segments = 26;
  const lit = Math.min(segments, Math.round(Math.sqrt(Math.min(level / 0.25, 1)) * segments));
  return (
    <div className="meter" aria-hidden>
      {Array.from({ length: segments }, (_, i) => (
        <span
          key={i}
          className={
            "meter-seg" +
            (i < lit ? " lit" : "") +
            (i >= segments - 4 ? " hot" : i >= segments - 9 ? " warm" : "")
          }
        />
      ))}
    </div>
  );
}

function Timer({ running }: { running: boolean }) {
  const [secs, setSecs] = useState(0);
  useEffect(() => {
    if (!running) return;
    setSecs(0);
    const started = Date.now();
    const t = setInterval(() => setSecs((Date.now() - started) / 1000), 100);
    return () => clearInterval(t);
  }, [running]);
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  const d = Math.floor((secs % 1) * 10);
  return (
    <div className="timer">
      {String(m).padStart(2, "0")}:{String(s).padStart(2, "0")}
      <span className="timer-tenths">.{d}</span>
    </div>
  );
}

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [phase, setPhase] = useState<Phase>("idle");
  const [level, setLevel] = useState(0);
  const [duration, setDuration] = useState<number | null>(null);
  const [transcript, setTranscript] = useState("");
  const [cleaned, setCleaned] = useState("");
  const [status, setStatus] = useState("ready — fully offline");
  const [error, setError] = useState<string | null>(null);
  const [managerOpen, setManagerOpen] = useState(false);
  const [promptOpen, setPromptOpen] = useState(false);
  // Only a user-edited prompt is persisted; empty means "use the backend default".
  const [prompt, setPrompt] = useLocalStorage("unsound.prompt", "");
  const [defaultPrompt, setDefaultPrompt] = useState("");
  const [sttId, setSttId] = useLocalStorage("unsound.stt", "");
  const [llmId, setLlmId] = useLocalStorage("unsound.llm", "");
  const cleaningRef = useRef(false);

  const refreshModels = useCallback(async () => {
    const list = await api.listModels();
    setModels(list);
    return list;
  }, []);

  useEffect(() => {
    refreshModels();
    api.defaultCleanupPrompt().then((p) => {
      setDefaultPrompt(p);
      // Early builds persisted the default itself; clear it so updated
      // defaults from the backend take effect.
      setPrompt((cur) => (cur.trim() === OLD_DEFAULT_PROMPT ? "" : cur));
    });
    const subs = [
      on.audioLevel(setLevel),
      on.llmToken((chunk) => {
        if (cleaningRef.current) setCleaned((c) => c + chunk);
      }),
      on.downloadDone(() => refreshModels()),
    ];
    return () => {
      subs.forEach((p) => p.then((un) => un()));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const sttModels = models.filter((m) => m.kind === "stt" && m.downloaded);
  const llmModels = models.filter((m) => m.kind === "llm" && m.downloaded);
  const stt = sttModels.find((m) => m.id === sttId) ?? sttModels[0];
  const llm = llmModels.find((m) => m.id === llmId) ?? llmModels[0];

  const fail = (e: unknown) => {
    setError(String(e));
    setPhase("idle");
  };

  const transcribeNow = useCallback(
    async (modelId: string, modelName: string): Promise<string | null> => {
      setPhase("transcribing");
      setStatus(`transcribing with ${modelName}…`);
      setError(null);
      try {
        const text = await api.transcribe(modelId);
        setTranscript(text);
        setPhase("idle");
        setStatus("transcript ready — refine it, or swap models and re-run");
        return text;
      } catch (e) {
        setError(String(e));
        setPhase("idle");
        return null;
      }
    },
    [],
  );

  const runCleanup = async (text: string) => {
    if (!llm || !text) return;
    setPhase("cleaning");
    setCleaned("");
    cleaningRef.current = true;
    setStatus(`refining with ${llm.name}…`);
    setError(null);
    try {
      const result = await api.cleanupText(llm.id, text, prompt || undefined);
      cleaningRef.current = false;
      setCleaned(result);
      setStatus("refined — copy it out, or swap models and re-run");
    } catch (e) {
      cleaningRef.current = false;
      setError(String(e));
    }
    setPhase("idle");
  };

  const toggleRecord = async () => {
    setError(null);
    if (phase === "recording") {
      try {
        const res = await api.stopRecording();
        setDuration(res.durationSecs);
        setLevel(0);
        if (stt) {
          const text = await transcribeNow(stt.id, stt.name);
          // Full pipeline on stop: transcript flows straight into cleanup.
          if (text && llm) await runCleanup(text);
          else if (text && !llm)
            setStatus("transcript ready — download a cleanup model to refine");
        } else {
          setPhase("idle");
          setStatus("recorded — download a speech model to transcribe");
        }
      } catch (e) {
        fail(e);
      }
    } else if (phase === "idle") {
      try {
        await api.startRecording();
        setTranscript("");
        setCleaned("");
        setDuration(null);
        setPhase("recording");
        setStatus("recording — everything stays on this machine");
      } catch (e) {
        fail(e);
      }
    }
  };

  const refine = async () => {
    if (phase !== "idle") return;
    await runCleanup(transcript);
  };

  const copy = (text: string, what: string) => {
    navigator.clipboard.writeText(text);
    setStatus(`${what} copied to clipboard`);
  };

  const busy = phase === "transcribing" || phase === "cleaning";

  return (
    <div className="shell">
      <header className="masthead">
        <div className="wordmark">
          <span className="wordmark-un">un</span>sound
        </div>
        <div className="masthead-note">local dictation &amp; cleanup · no cloud</div>
        <button className="btn ghost" onClick={() => setManagerOpen(true)}>
          models
        </button>
      </header>

      <main className="deck">
        <section className="transport panel">
          <div className="panel-label">01 · record</div>
          <button
            className={"rec-btn" + (phase === "recording" ? " live" : "")}
            onClick={toggleRecord}
            disabled={busy}
            title={phase === "recording" ? "Stop" : "Record"}
          >
            <span className="rec-core" />
          </button>
          <Timer running={phase === "recording"} />
          <Meter level={phase === "recording" ? level : 0} />
          {duration !== null && phase !== "recording" && (
            <div className="take-info">last take · {duration.toFixed(1)}s</div>
          )}

          <div className="selectors">
            <label className="selector">
              <span className="selector-label">speech model</span>
              <select
                value={stt?.id ?? ""}
                onChange={(e) => setSttId(e.target.value)}
                disabled={sttModels.length === 0}
              >
                {sttModels.length === 0 && <option value="">none downloaded</option>}
                {sttModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="selector">
              <span className="selector-label">cleanup model</span>
              <select
                value={llm?.id ?? ""}
                onChange={(e) => setLlmId(e.target.value)}
                disabled={llmModels.length === 0}
              >
                {llmModels.length === 0 && <option value="">none downloaded</option>}
                {llmModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </label>
            {(sttModels.length === 0 || llmModels.length === 0) && (
              <button className="btn accent" onClick={() => setManagerOpen(true)}>
                get models →
              </button>
            )}
          </div>
        </section>

        <section className="panel output">
          <div className="panel-head">
            <div className="panel-label">02 · raw transcript</div>
            <div className="panel-actions">
              {transcript && stt && (
                <button
                  className="btn ghost"
                  disabled={busy}
                  onClick={() => transcribeNow(stt.id, stt.name)}
                  title="Re-run speech-to-text on the last take with the selected model"
                >
                  re-run
                </button>
              )}
              {transcript && (
                <button className="btn ghost" onClick={() => copy(transcript, "transcript")}>
                  copy
                </button>
              )}
            </div>
          </div>
          <div className={"text-well" + (phase === "transcribing" ? " working" : "")}>
            {phase === "transcribing" ? (
              "listening back…"
            ) : (
              transcript || <span className="placeholder">record a take to see its transcript</span>
            )}
          </div>
        </section>

        <section className="panel output">
          <div className="panel-head">
            <div className="panel-label">03 · refined</div>
            <div className="panel-actions">
              <button className="btn ghost" onClick={() => setPromptOpen((o) => !o)}>
                {promptOpen ? "hide prompt" : "edit prompt"}
              </button>
              {cleaned && (
                <button className="btn ghost" onClick={() => copy(cleaned, "refined text")}>
                  copy
                </button>
              )}
              <button className="btn accent" onClick={refine} disabled={!transcript || !llm || busy}>
                {phase === "cleaning" ? "refining…" : "refine ↦"}
              </button>
            </div>
          </div>
          {promptOpen && (
            <textarea
              className="prompt-editor"
              value={prompt || defaultPrompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={10}
              spellCheck={false}
            />
          )}
          <div className={"text-well refined" + (phase === "cleaning" ? " working" : "")}>
            {cleaned ? (
              <>
                {cleaned}
                {phase === "cleaning" && <span className="caret" />}
              </>
            ) : phase === "cleaning" ? (
              <span className="caret" />
            ) : (
              <span className="placeholder">refined text will stream in here</span>
            )}
          </div>
        </section>
      </main>

      <footer className="statusbar">
        <span className={"status-dot" + (phase === "recording" ? " live" : "")} />
        <span className="status-text">{error ?? status}</span>
        {error && (
          <button className="btn ghost" onClick={() => setError(null)}>
            dismiss
          </button>
        )}
      </footer>

      {managerOpen && (
        <ModelManager models={models} onClose={() => setManagerOpen(false)} onChanged={refreshModels} />
      )}
    </div>
  );
}
