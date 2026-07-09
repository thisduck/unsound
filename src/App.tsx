import { useCallback, useEffect, useRef, useState } from "react";
import { api, on, formatBytes, formatDuration, LANGUAGES, ModelInfo, Style } from "./api";
import { CorrectableText } from "./Correctable";
import { useLevelHistory, Wave } from "./Wave";
import { MicPicker } from "./sections";
import { SettingsSheet } from "./SettingsSheet";
import { Onboarding } from "./Onboarding";
import { HistoryDrawer, Take } from "./HistoryDrawer";
import { Meetings } from "./Meetings";
import "./App.css";

type Phase = "idle" | "recording" | "transcribing" | "cleaning";

const OLD_DEFAULT_PROMPT =
  "You clean up raw speech-to-text transcripts. Fix punctuation, capitalization and obvious transcription errors, remove filler words (um, uh, you know), and break the text into paragraphs where natural. Preserve the speaker's wording and meaning; do not summarize or add anything. Output only the cleaned text.";

const HISTORY_KEY = "unsound.history";
const MIGRATED_KEY = "unsound.history.migrated";
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
  const [transcript, setTranscript] = useState("");
  const [cleaned, setCleaned] = useState("");
  // Live caption shown while dictating (raw, tentative); cleaned text lands at stop.
  const [liveText, setLiveText] = useState("");
  const [status, setStatus] = useState("ready — fully offline");
  const [error, setError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [view, setView] = useState<"dictate" | "meetings">("dictate");
  const [rawOpen, setRawOpen] = useState(false);
  const [onboarding, setOnboarding] = useState(() => !localStorage.getItem("unsound.onboarded"));
  const [prompt, setPrompt] = useLocalStorage("unsound.prompt", "");
  const [defaultPrompt, setDefaultPrompt] = useState("");
  const [sttId, setSttId] = useLocalStorage("unsound.stt", "");
  const [llmId, setLlmId] = useLocalStorage("unsound.llm", "");
  // Spoken language (what Whisper listens for) vs. output language (what the
  // cleanup model translates into; "" keeps the transcript's language).
  const [language, setLanguage] = useLocalStorage("unsound.lang", "");
  const [outLang, setOutLang] = useLocalStorage("unsound.outlang", "");
  const [textScale, setTextScale] = useLocalStorage("unsound.textscale", "1");
  const [history, setHistory] = useState<Take[]>([]);
  const [stylesList, setStylesList] = useState<Style[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [styleId, setStyleId] = useState("");
  const styleIdRef = useRef("");
  const outLangRef = useRef("");
  outLangRef.current = outLang;
  const cleaningRef = useRef(false);
  const takeRef = useRef<string | null>(null);
  // Event handlers race React state: phaseRef is always current, and a
  // release that lands while the mic is still starting queues the stop.
  const phaseRef = useRef<Phase>("idle");
  const startingRef = useRef(false);
  const stopQueuedRef = useRef(false);

  const changePhase = (p: Phase) => {
    phaseRef.current = p;
    setPhase(p);
  };

  // Keep a ref to the latest history so upsertTake can merge + persist a full
  // record without a stale closure.
  const historyRef = useRef<Take[]>([]);
  historyRef.current = history;

  // History now lives in SQLite. Migrate the old localStorage copy once, then
  // hydrate from the database.
  useEffect(() => {
    (async () => {
      try {
        if (!localStorage.getItem(MIGRATED_KEY)) {
          const old = loadHistory();
          if (old.length) await api.importTakes(old);
          localStorage.setItem(MIGRATED_KEY, "1");
        }
        setHistory(await api.listTakes());
      } catch (e) {
        console.error("failed to load history from database", e);
      }
    })();
  }, []);

  useEffect(() => {
    document.documentElement.style.setProperty("--text-scale", textScale);
  }, [textScale]);

  const refreshModels = useCallback(async () => {
    setModels(await api.listModels());
  }, []);

  const chooseStyle = (id: string) => {
    styleIdRef.current = id;
    setStyleId(id);
  };

  const refreshStyles = useCallback(async (adoptDefault = false) => {
    const s = await api.getSettings();
    setStylesList(s.styles);
    if (adoptDefault || !s.styles.some((st) => st.id === styleIdRef.current)) {
      styleIdRef.current = s.defaultStyle;
      setStyleId(s.defaultStyle);
    }
  }, []);

  useEffect(() => {
    refreshModels();
    refreshStyles(true);
    api.defaultCleanupPrompt().then((p) => {
      setDefaultPrompt(p);
      // The prompt field used to hold the whole system prompt; now it holds
      // only user additions. Clear any stored copy of a full base prompt so
      // it isn't appended to itself.
      setPrompt((cur) => {
        const t = cur.trim();
        return t === OLD_DEFAULT_PROMPT || t === p.trim() ? "" : cur;
      });
    });
    const subs = [
      on.settingsChanged(() => refreshStyles()),
      on.llmToken((chunk) => {
        if (cleaningRef.current) setCleaned((c) => c + chunk);
      }),
      on.downloadDone(() => refreshModels()),
      on.hotkeyToggle(() => hotkeyRef.current()),
      on.pttDown(() => pttRef.current("down")),
      on.pttUp(() => pttRef.current("up")),
      on.pttCancel(() => pttRef.current("cancel")),
      on.dictationLive((t) => {
        if (phaseRef.current === "recording") setLiveText(t);
      }),
    ];
    // Window drag-and-drop for audio files.
    const dropUnlisten = import("@tauri-apps/api/webview").then(({ getCurrentWebview }) =>
      getCurrentWebview().onDragDropEvent((event) => {
        if (event.payload.type === "over" || event.payload.type === "enter") setDragOver(true);
        else if (event.payload.type === "leave") setDragOver(false);
        else if (event.payload.type === "drop") {
          setDragOver(false);
          const path = event.payload.paths?.[0];
          if (path) dropRef.current(path);
        }
      }),
    );

    return () => {
      subs.forEach((p) => p.then((un) => un()));
      dropUnlisten.then((un) => un());
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
    const existing = historyRef.current.find((t) => t.id === id);
    const take: Take = existing
      ? { ...existing, ...patch }
      : {
          id,
          at: new Date().toISOString(),
          raw: "",
          refined: "",
          sttModel: "",
          llmModel: "",
          ...patch,
        };
    setHistory((prev) => {
      const i = prev.findIndex((t) => t.id === id);
      if (i >= 0) {
        const copy = [...prev];
        copy[i] = { ...copy[i], ...patch };
        return copy;
      }
      return [take, ...prev].slice(0, HISTORY_CAP);
    });
    // Persist the full merged record to SQLite (fire-and-forget).
    api.saveTake(take).catch((e) => console.error("failed to save take", e));
  };

  const fail = (e: unknown) => {
    setError(String(e));
    changePhase("idle");
  };

  const transcribeNow = async (model: ModelInfo): Promise<string | null> => {
    changePhase("transcribing");
    setStatus(`transcribing with ${model.name}…`);
    setError(null);
    try {
      const text = await api.transcribe(model.id, language || undefined);
      setTranscript(text);
      upsertTake({ raw: text, sttModel: model.name, lang: language });
      changePhase("idle");
      setStatus("transcript ready");
      return text;
    } catch (e) {
      setError(String(e));
      changePhase("idle");
      return null;
    }
  };

  // Decode + transcribe an uploaded/dropped audio file, then refine like a
  // mic take. Handles the whole pipeline; runs the file through the same
  // transcript → cleanup path.
  const transcribeFromFile = async (path: string) => {
    if (busy || phaseRef.current !== "idle") return;
    if (!stt) {
      setError("download a speech model first (settings → models)");
      return;
    }
    const name = path.split("/").pop() || "file";
    changePhase("transcribing");
    setTranscript("");
    setCleaned("");
    setRawOpen(false);
    setStatus(`decoding ${name}…`);
    setError(null);
    takeRef.current = crypto.randomUUID();
    // A one-shot listener shows the file's size and length once decoding
    // finishes, so long files don't look stuck.
    const infoSub = on.fileInfo(({ sizeBytes, durationSecs }) => {
      setStatus(
        `transcribing ${name} · ${formatDuration(durationSecs)} · ${formatBytes(sizeBytes)}…`,
      );
    });
    try {
      const text = await api.transcribeFile(path, stt.id, language || undefined);
      setTranscript(text);
      upsertTake({ raw: text, sttModel: stt.name, app: name, lang: language });
      changePhase("idle");
      setStatus(`transcribed ${name}`);
      if (llm) await runCleanup(text);
    } catch (e) {
      setError(String(e));
      changePhase("idle");
    } finally {
      infoSub.then((un) => un());
    }
  };

  const pickFile = async () => {
    if (busy || phaseRef.current !== "idle") return;
    const { open } = await import("@tauri-apps/plugin-dialog");
    // No extension filter — the real format is sniffed from the bytes, and
    // WhatsApp files often carry a wrong extension (e.g. a voice note saved
    // as .jpeg), so any file must be selectable.
    const path = await open({ multiple: false });
    if (typeof path === "string") transcribeFromFile(path);
  };

  const runCleanup = async (
    text: string,
    styleOverride?: string,
    langOverride?: string,
  ): Promise<string | null> => {
    if (!llm || !text) return null;
    const useStyle = styleOverride !== undefined ? styleOverride : styleIdRef.current;
    const useLangCode = langOverride !== undefined ? langOverride : outLangRef.current;
    const transliterate = useLangCode === "translit";
    const targetName = transliterate
      ? undefined
      : LANGUAGES.find((l) => l.code === useLangCode && l.code !== "")?.name;
    changePhase("cleaning");
    setCleaned("");
    cleaningRef.current = true;
    setStatus(
      transliterate
        ? "refining → romanized…"
        : targetName
          ? `refining → ${targetName}…`
          : `refining with ${llm.name}…`,
    );
    setError(null);
    try {
      const result = await api.cleanupText(
        llm.id,
        text,
        prompt || undefined,
        useStyle || undefined,
        targetName,
        transliterate,
      );
      cleaningRef.current = false;
      setCleaned(result);
      upsertTake({ refined: result, llmModel: llm.name });
      changePhase("idle");
      setStatus("refined — copy it out, or swap models and re-run");
      return result;
    } catch (e) {
      cleaningRef.current = false;
      setError(String(e));
      changePhase("idle");
      return null;
    }
  };

  const overlayOff = () => {
    api.emitOverlayState("hidden");
    api.setOverlay(false).catch(() => {});
  };

  const toggleRecord = async (fromHotkey = false) => {
    setError(null);
    if (phaseRef.current === "recording") {
      try {
        const res = await api.stopRecording();
        api.stopLiveDictation().catch(() => {});
        if (fromHotkey) api.emitOverlayState("processing");
        if (res.durationSecs < 0.35) {
          changePhase("idle");
          setStatus("take too short — hold a moment longer");
          return;
        }
        if (!stt) {
          changePhase("idle");
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
            const target = await api.deliverText(out);
            if (target) upsertTake({ app: target });
            setStatus(`typed into ${target || "the frontmost app"}`);
          } catch (e) {
            setError(String(e));
          }
        }
      } catch (e) {
        fail(e);
      } finally {
        if (fromHotkey) overlayOff();
      }
    } else if (phaseRef.current === "idle" && !startingRef.current) {
      startingRef.current = true;
      stopQueuedRef.current = false;
      try {
        await api.startRecording();
        setTranscript("");
        setCleaned("");
        setLiveText("");
        setRawOpen(false);
        changePhase("recording");
        // Live captions for in-app takes (window visible); the hotkey flow uses
        // the overlay and isn't looking at the window, so skip it there.
        if (!fromHotkey && stt) {
          api.startLiveDictation(stt.id, language || undefined).catch(() => {});
        }
        if (fromHotkey) {
          await api.setOverlay(true).catch(() => {});
          api.emitOverlayState("recording");
        }
        setStatus(
          fromHotkey
            ? "recording — press the shortcut again to stop"
            : "recording — everything stays on this machine",
        );
      } catch (e) {
        fail(e);
      } finally {
        startingRef.current = false;
      }
      // The release arrived while the mic was still starting; honor it now.
      if (stopQueuedRef.current && (phaseRef.current as Phase) === "recording") {
        stopQueuedRef.current = false;
        await toggleRecord(fromHotkey);
      }
    }
  };

  // The hotkey and drop listeners are registered once; route them through
  // refs so they always see the current phase and model selection.
  const hotkeyRef = useRef<() => void>(() => {});
  hotkeyRef.current = () => {
    if (startingRef.current) stopQueuedRef.current = true;
    else if (phaseRef.current === "idle" || phaseRef.current === "recording") toggleRecord(true);
  };

  const dropRef = useRef<(path: string) => void>(() => {});
  dropRef.current = (path: string) => transcribeFromFile(path);

  const pttRef = useRef<(what: "down" | "up" | "cancel") => void>(() => {});
  pttRef.current = (what) => {
    if (what === "down") {
      if (phaseRef.current === "idle" && !startingRef.current) toggleRecord(true);
    } else if (what === "up") {
      if (startingRef.current) stopQueuedRef.current = true;
      else if (phaseRef.current === "recording") toggleRecord(true);
    } else if (what === "cancel" && phaseRef.current === "recording") {
      // The held key turned out to be part of a bigger combo; discard.
      api.stopRecording().catch(() => {});
      api.stopLiveDictation().catch(() => {});
      overlayOff();
      changePhase("idle");
      setStatus("cancelled");
    }
  };

  const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const correctIn = (source: "raw" | "refined") => async (from: string, to: string) => {
    try {
      await api.addCorrection(from, to);
    } catch (e) {
      setError(String(e));
      return;
    }
    const fix = (s: string) => s.replace(new RegExp(`\\b${escapeRe(from)}\\b`, "g"), to);
    if (source === "raw") {
      const next = fix(transcript);
      setTranscript(next);
      upsertTake({ raw: next });
    } else {
      const next = fix(cleaned);
      setCleaned(next);
      upsertTake({ refined: next });
    }
    setStatus(`learned: "${from}" → "${to}" — future takes will hear it right`);
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
    // Restore the spoken language this take was recorded in.
    if (take.lang !== undefined) setLanguage(take.lang);
    setStatus("loaded from history");
  };

  const busy = phase === "transcribing" || phase === "cleaning";
  const waveBars = useLevelHistory(phase === "recording");
  const heroText = cleaned || "";

  // Switching style re-renders the current take immediately.
  const styleSwitcher = stylesList.length > 0 && (
    <select
      className="chip-select style-chip"
      value={styleId}
      disabled={busy}
      onChange={(e) => {
        chooseStyle(e.target.value);
        if (transcript && phaseRef.current === "idle") runCleanup(transcript, e.target.value);
      }}
      title="Writing style"
    >
      <option value="">no style</option>
      {stylesList.map((s) => (
        <option key={s.id} value={s.id}>
          ✍ {s.name}
        </option>
      ))}
    </select>
  );

  // Output-language switcher — re-renders the current take when changed.
  const outLangSwitcher = (
    <select
      className="chip-select outlang-chip"
      value={outLang}
      disabled={busy}
      onChange={(e) => {
        setOutLang(e.target.value);
        if (transcript && phaseRef.current === "idle") runCleanup(transcript, undefined, e.target.value);
      }}
      title="Output language — keep the original, or translate"
    >
      <option value="">keep language</option>
      <option value="translit">transliterate (Latin script)</option>
      {LANGUAGES.filter((l) => l.code !== "").map((l) => (
        <option key={l.code} value={l.code}>
          → {l.name}
        </option>
      ))}
    </select>
  );
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
        <img src="/unsound.svg" alt="" className="mark-logo" />
        <span className="mark">unsound</span>
        <nav className="nav">
          <button
            className={"nav-tab" + (view === "dictate" ? " on" : "")}
            onClick={() => setView("dictate")}
          >
            dictate
          </button>
          <button
            className={"nav-tab" + (view === "meetings" ? " on" : "")}
            onClick={() => setView("meetings")}
          >
            meetings<span className="beta-badge">beta</span>
          </button>
        </nav>
        <span className="spacer" />
        {view === "dictate" && (
          <button className="quiet" onClick={() => setHistoryOpen(true)}>
            history
          </button>
        )}
        <button className="quiet" onClick={() => setSettingsOpen(true)}>
          settings
        </button>
      </header>

      {/* Meetings stays mounted (hidden when inactive) so the tray "Start /
          stop meeting" toggle works from any tab or with the window hidden. */}
      <div className={"view-wrap" + (view === "meetings" ? "" : " hidden")}>
        <Meetings
          stt={stt}
          llm={llm}
          language={language}
          models={models}
          onModelsChanged={refreshModels}
          onActivate={() => setView("meetings")}
        />
      </div>

      {view === "dictate" && (
        <>
          {/* dictation view below */}

      <div className="stage">
        <button
          className={"pill" + (phase === "recording" ? " live" : "")}
          onClick={() => toggleRecord()}
          disabled={busy}
        >
          {phase === "recording" ? (
            <Wave bars={waveBars} className="pill-wave" />
          ) : (
            <span className="pill-dot" />
          )}
          {pillLabel}
          <Timer running={phase === "recording"} />
        </button>
      </div>

      <div className="stage-utils">
        <button
          className="util-link"
          onClick={pickFile}
          disabled={busy || phase === "recording"}
          title="Transcribe an audio file (or drop one on the window)"
        >
          ↑ upload a file
        </button>
        <span className="util-sep">·</span>
        <span className="util-lang" title="Spoken language — what Whisper listens for">
          <select
            className="util-select"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            disabled={busy}
          >
            {LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>
                {l.code === "" ? "auto-detect" : `speaking ${l.name}`}
              </option>
            ))}
          </select>
        </span>
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
        ) : phase === "recording" ? (
          <div className="prose dim">
            {liveText || <span className="placeholder">listening… your words will appear here</span>}
            <span className="caret" />
          </div>
        ) : (
          <>
            {heroText || phase === "cleaning" ? (
              <>
                <div className="hero-tools">
                  {styleSwitcher}
                  {outLangSwitcher}
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
                  {phase === "cleaning" ? (
                    heroText
                  ) : (
                    <CorrectableText text={heroText} onCorrect={correctIn("refined")} />
                  )}
                  {phase === "cleaning" && <span className="caret" />}
                </div>
              </>
            ) : (
              transcript && (
                <div className="prose dim">
                  <CorrectableText text={transcript} onCorrect={correctIn("raw")} />
                  {llm && phase === "idle" && (
                    <div className="hero-tools" style={{ marginTop: 16 }}>
                      {styleSwitcher}
                      {outLangSwitcher}
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
                    <div className="raw-text">
                      <CorrectableText text={transcript} onCorrect={correctIn("raw")} />
                    </div>
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
        </>
      )}

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
          textScale={textScale}
          onTextScaleChange={setTextScale}
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
          onDelete={(id) => {
            setHistory((h) => h.filter((t) => t.id !== id));
            api.deleteTake(id).catch((e) => console.error("failed to delete take", e));
          }}
          onClear={() => {
            setHistory([]);
            api.clearTakes().catch((e) => console.error("failed to clear history", e));
          }}
          onCopy={copy}
        />
      )}
      {dragOver && (
        <div className="drop-veil">
          <div className="drop-veil-inner">drop an audio file to transcribe</div>
        </div>
      )}
    </div>
  );
}
