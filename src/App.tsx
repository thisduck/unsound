import { useCallback, useEffect, useRef, useState } from "react";
import { api, on, ModelInfo } from "./api";
import { MicPicker } from "./sections";
import { SettingsSheet } from "./SettingsSheet";
import { Onboarding } from "./Onboarding";
import { HistoryDrawer, Take } from "./HistoryDrawer";
import "./App.css";

type Phase = "idle" | "recording" | "transcribing" | "cleaning";

const OLD_DEFAULT_PROMPT =
  "You clean up raw speech-to-text transcripts. Fix punctuation, capitalization and obvious transcription errors, remove filler words (um, uh, you know), and break the text into paragraphs where natural. Preserve the speaker's wording and meaning; do not summarize or add anything. Output only the cleaned text.";

const HISTORY_KEY = "unsound.history";
const HISTORY_CAP = 200;

function useLocalStorage(key: string, initial: string) {
  const [value, setValue] = useState(() => localStorage.getItem(key) ?? initial);
  useEffect(() => {
    localStorage.setItem(key, value);
  }, [key, value]);
  return [value, setValue] as const;
}

function loadHistory(): Take[] {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]");
  } catch {
    return [];
  }
}

function Timer({ running }: { running: boolean }) {
  const [secs, setSecs] = useState(0);
  useEffect(() => {
    if (!running) {
      setSecs(0);
      return;
    }
    const started = Date.now();
    const t = setInterval(() => setSecs((Date.now() - started) / 1000), 250);
    return () => clearInterval(t);
  }, [running]);
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return (
    <span className="pill-time">
      {String(m).padStart(2, "0")}:{String(s).padStart(2, "0")}
    </span>
  );
}

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [phase, setPhase] = useState<Phase>("idle");
  const [level, setLevel] = useState(0);
  const [transcript, setTranscript] = useState("");
  const [cleaned, setCleaned] = useState("");
  const [status, setStatus] = useState("ready — fully offline");
  const [error, setError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [rawOpen, setRawOpen] = useState(false);
  const [onboarding, setOnboarding] = useState(() => !localStorage.getItem("unsound.onboarded"));
  const [prompt, setPrompt] = useLocalStorage("unsound.prompt", "");
  const [defaultPrompt, setDefaultPrompt] = useState("");
  const [sttId, setSttId] = useLocalStorage("unsound.stt", "");
  const [llmId, setLlmId] = useLocalStorage("unsound.llm", "");
  const [history, setHistory] = useState<Take[]>(loadHistory);
  const cleaningRef = useRef(false);
  const takeRef = useRef<string | null>(null);

  useEffect(() => {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  }, [history]);

  const refreshModels = useCallback(async () => {
    setModels(await api.listModels());
  }, []);

  useEffect(() => {
    refreshModels();
    api.defaultCleanupPrompt().then((p) => {
      setDefaultPrompt(p);
      setPrompt((cur) => (cur.trim() === OLD_DEFAULT_PROMPT ? "" : cur));
    });
    const subs = [
      on.audioLevel(setLevel),
      on.llmToken((chunk) => {
        if (cleaningRef.current) setCleaned((c) => c + chunk);
      }),
      on.downloadDone(() => refreshModels()),
      on.hotkeyToggle(() => hotkeyRef.current()),
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

  const upsertTake = (patch: Partial<Omit<Take, "id" | "at">>) => {
    const id = takeRef.current;
    if (!id) return;
    setHistory((h) => {
      const i = h.findIndex((t) => t.id === id);
      if (i >= 0) {
        const copy = [...h];
        copy[i] = { ...copy[i], ...patch };
        return copy;
      }
      const take: Take = {
        id,
        at: new Date().toISOString(),
        raw: "",
        refined: "",
        sttModel: "",
        llmModel: "",
        ...patch,
      };
      return [take, ...h].slice(0, HISTORY_CAP);
    });
  };

  const fail = (e: unknown) => {
    setError(String(e));
    setPhase("idle");
  };

  const transcribeNow = async (model: ModelInfo): Promise<string | null> => {
    setPhase("transcribing");
    setStatus(`transcribing with ${model.name}…`);
    setError(null);
    try {
      const text = await api.transcribe(model.id);
      setTranscript(text);
      upsertTake({ raw: text, sttModel: model.name });
      setPhase("idle");
      setStatus("transcript ready");
      return text;
    } catch (e) {
      setError(String(e));
      setPhase("idle");
      return null;
    }
  };

  const runCleanup = async (text: string): Promise<string | null> => {
    if (!llm || !text) return null;
    setPhase("cleaning");
    setCleaned("");
    cleaningRef.current = true;
    setStatus(`refining with ${llm.name}…`);
    setError(null);
    try {
      const result = await api.cleanupText(llm.id, text, prompt || undefined);
      cleaningRef.current = false;
      setCleaned(result);
      upsertTake({ refined: result, llmModel: llm.name });
      setPhase("idle");
      setStatus("refined — copy it out, or swap models and re-run");
      return result;
    } catch (e) {
      cleaningRef.current = false;
      setError(String(e));
      setPhase("idle");
      return null;
    }
  };

  const toggleRecord = async (fromHotkey = false) => {
    setError(null);
    if (phase === "recording") {
      try {
        await api.stopRecording();
        setLevel(0);
        if (!stt) {
          setPhase("idle");
          setStatus("recorded — download a speech model to transcribe");
          return;
        }
        takeRef.current = crypto.randomUUID();
        const text = await transcribeNow(stt);
        if (!text) return;
        const refined = llm ? await runCleanup(text) : null;
        if (!llm) setStatus("transcript ready — download a cleanup model to refine");
        if (fromHotkey) {
          const out = refined ?? text;
          try {
            await api.deliverText(out);
            setStatus("delivered to the frontmost app — also on your clipboard");
          } catch (e) {
            setError(String(e));
          }
        }
      } catch (e) {
        fail(e);
      }
    } else if (phase === "idle") {
      try {
        await api.startRecording();
        setTranscript("");
        setCleaned("");
        setRawOpen(false);
        setPhase("recording");
        setStatus(
          fromHotkey
            ? "recording — press the shortcut again to stop"
            : "recording — everything stays on this machine",
        );
      } catch (e) {
        fail(e);
      }
    }
  };

  // The hotkey listener is registered once; route it through a ref so it
  // always sees the current phase and model selection.
  const hotkeyRef = useRef<() => void>(() => {});
  hotkeyRef.current = () => {
    if (phase === "idle" || phase === "recording") toggleRecord(true);
  };

  const copy = (text: string, what: string) => {
    navigator.clipboard.writeText(text);
    setStatus(`${what} copied to clipboard`);
  };

  const loadTake = (take: Take) => {
    setTranscript(take.raw);
    setCleaned(take.refined);
    setHistoryOpen(false);
    setRawOpen(!take.refined);
    setStatus("loaded from history");
  };

  const busy = phase === "transcribing" || phase === "cleaning";
  const heroText = cleaned || "";
  const pillLabel =
    phase === "recording"
      ? "listening — tap to stop"
      : phase === "transcribing"
        ? "transcribing…"
        : phase === "cleaning"
          ? "refining…"
          : "tap to speak";

  return (
    <div className="shell">
      <header className="head">
        <span className="mark">unsound</span>
        <span className="spacer" />
        <button className="quiet" onClick={() => setHistoryOpen(true)}>
          history
        </button>
        <button className="quiet" onClick={() => setSettingsOpen(true)}>
          settings
        </button>
      </header>

      <div className="stage">
        <button
          className={"pill" + (phase === "recording" ? " live" : "")}
          onClick={() => toggleRecord()}
          disabled={busy}
        >
          <span
            className="pill-dot"
            style={
              phase === "recording"
                ? { boxShadow: `0 0 0 ${3 + Math.min(level * 60, 14)}px rgba(232,168,124,0.16)` }
                : undefined
            }
          />
          {pillLabel}
          <Timer running={phase === "recording"} />
        </button>
      </div>

      <main className="body">
        {(sttModels.length === 0 || llmModels.length === 0) && phase === "idle" && !transcript ? (
          <div className="empty">
            <p>
              unsound listens, transcribes with a local Whisper model, and tidies the words with a
              local LLM. Nothing leaves this machine.
            </p>
            <button className="quiet accent" onClick={() => setSettingsOpen(true)}>
              download models to get started →
            </button>
          </div>
        ) : (
          <>
            {heroText || phase === "cleaning" ? (
              <>
                <div className="hero-tools">
                  {cleaned && (
                    <>
                      <button className="quiet" onClick={() => copy(cleaned, "refined text")}>
                        copy
                      </button>
                      <button
                        className="quiet"
                        disabled={busy || !transcript}
                        onClick={() => runCleanup(transcript)}
                      >
                        re-refine
                      </button>
                    </>
                  )}
                </div>
                <div className="prose">
                  {heroText}
                  {phase === "cleaning" && <span className="caret" />}
                </div>
              </>
            ) : (
              transcript && (
                <div className="prose dim">
                  {transcript}
                  {llm && phase === "idle" && (
                    <div className="hero-tools" style={{ marginTop: 16 }}>
                      <button className="quiet accent" onClick={() => runCleanup(transcript)}>
                        refine ↦
                      </button>
                      <button className="quiet" onClick={() => copy(transcript, "transcript")}>
                        copy
                      </button>
                    </div>
                  )}
                </div>
              )
            )}

            {transcript && cleaned && (
              <div className="raw-block">
                <button className="raw-toggle" onClick={() => setRawOpen((o) => !o)}>
                  {rawOpen ? "▾" : "▸"} raw transcript
                </button>
                {rawOpen && (
                  <div className="raw-body">
                    <div className="raw-text">{transcript}</div>
                    <div className="hero-tools">
                      <button className="quiet" onClick={() => copy(transcript, "transcript")}>
                        copy
                      </button>
                      {stt && (
                        <button className="quiet" disabled={busy} onClick={() => transcribeNow(stt)}>
                          re-transcribe
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </main>

      <footer className="foot">
        <span className={"foot-dot" + (phase === "recording" ? " live" : "")} />
        <span className="foot-text">{error ?? status}</span>
        {error && (
          <button className="quiet" onClick={() => setError(null)}>
            dismiss
          </button>
        )}
        <MicPicker className="chip-select mic-chip" />
      </footer>

      {settingsOpen && (
        <SettingsSheet
          models={models}
          sttId={stt?.id ?? ""}
          llmId={llm?.id ?? ""}
          onSttChange={setSttId}
          onLlmChange={setLlmId}
          prompt={prompt}
          defaultPrompt={defaultPrompt}
          onPromptChange={setPrompt}
          onClose={() => setSettingsOpen(false)}
          onChanged={refreshModels}
          onReplayOnboarding={() => {
            setSettingsOpen(false);
            setOnboarding(true);
          }}
        />
      )}
      {onboarding && (
        <Onboarding
          models={models}
          onChanged={refreshModels}
          onDone={() => {
            localStorage.setItem("unsound.onboarded", "1");
            setOnboarding(false);
            refreshModels();
          }}
        />
      )}
      {historyOpen && (
        <HistoryDrawer
          takes={history}
          onClose={() => setHistoryOpen(false)}
          onLoad={loadTake}
          onDelete={(id) => setHistory((h) => h.filter((t) => t.id !== id))}
          onClear={() => setHistory([])}
          onCopy={copy}
        />
      )}
    </div>
  );
}
