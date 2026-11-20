import { useCallback, useEffect, useRef, useState } from "react";
import { api, on, formatBytes, formatDuration, LANGUAGES, ModelInfo, Style } from "./api";
import { useLevelHistory } from "./Wave";
import { SettingsSheet, type SettingsSection } from "./SettingsSheet";
import { Onboarding } from "./Onboarding";
import { PermissionsCheck, useOpenPermissions } from "./Permissions";
import { Dictations, type Take } from "./Dictations";
import { Inbox, NeedsYou, useInbox } from "./Inbox";
import { Meetings, type RecordingInfo } from "./Meetings";
import { createDictationQueue, type DictationState } from "./dictationQueue";
import { UpdateCenter } from "./UpdateCenter";
import { Icon, Logo } from "./ui";
import { languageNote, migrateSttId } from "./speech";

type Phase = "idle" | "recording" | "transcribing" | "cleaning";

interface DictationTake {
  id: string;
  at: string;
  fromHotkey: boolean;
  stt?: ModelInfo;
  llm?: ModelInfo;
  language: string;
  prompt: string;
  style: string;
  outLang: string;
}

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

/// What an output-language code asks of cleanup: romanize, or translate into
/// a named language ("" keeps the transcript's own language).
function outputLang(code: string): { transliterate: boolean; targetName?: string } {
  const transliterate = code === "translit";
  const targetName = transliterate
    ? undefined
    : LANGUAGES.find((l) => l.code === code && l.code !== "")?.name;
  return { transliterate, targetName };
}

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [workPhase, setWorkPhase] = useState<Phase>("idle");
  const [dictationState, setDictationState] = useState<DictationState<DictationTake>>({ active: null, capturing: false, pending: 0 });
  const phase: Phase = dictationState.active ? "recording" : workPhase;
  const [transcript, setTranscript] = useState("");
  const [cleaned, setCleaned] = useState("");
  // Live caption shown while dictating (raw, tentative); cleaned text lands at stop.
  const [liveText, setLiveText] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState<SettingsSection | null>(null);
  const [view, setView] = useState<"meetings" | "dictations" | "inbox">("meetings");
  const [reminderTarget, setReminderTarget] = useState<{ id: string; at: number | null } | null>(null);
  const [recording, setRecording] = useState<RecordingInfo | null>(null);
  const [liveVisible, setLiveVisible] = useState(false);
  const [liveRequest, setLiveRequest] = useState(0);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [shortcut, setShortcut] = useState<string | null>(null);
  const inbox = useInbox();
  const [rawOpen, setRawOpen] = useState(false);
  const [onboarding, setOnboarding] = useState(() => !localStorage.getItem("unsound.onboarded"));
  // Only people already set up at launch get the missing-permission check.
  const [onboardedAtLaunch] = useState(() => !!localStorage.getItem("unsound.onboarded"));
  useOpenPermissions(useCallback(() => setSettingsOpen("permissions"), []));
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
  const cleaningRef = useRef<string | null>(null);
  const takeRef = useRef<string | null>(null);
  // Capture and processing have independent lifetimes. Refs also keep native
  // shortcut events current before React has rendered the latest state.
  const phaseRef = useRef<Phase>("idle");
  const workPhaseRef = useRef<Phase>("idle");
  const dictationStateRef = useRef(dictationState);
  const hotkeyTakesRef = useRef(new Set<string>());
  const overlayStateRef = useRef("hidden");
  const overlayTailRef = useRef(Promise.resolve());

  const changePhase = (p: Phase) => {
    workPhaseRef.current = p;
    phaseRef.current = dictationStateRef.current.active ? "recording" : p;
    setWorkPhase(p);
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
    setShortcut(s.handsFree[0] || s.pushToTalk[0] || null);
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
      on.llmToken(({ takeId, chunk }) => {
        if (takeId && takeId === cleaningRef.current && takeId === takeRef.current) setCleaned((c) => c + chunk);
      }),
      on.downloadDone(() => refreshModels()),
      on.hotkeyToggle(() => hotkeyRef.current()),
      on.pttDown(() => pttRef.current("down")),
      on.pttUp(() => pttRef.current("up")),
      on.pttCancel(() => pttRef.current("cancel")),
      on.dictationLive(({ takeId, text }) => {
        if (phaseRef.current === "recording" && takeRef.current === takeId) setLiveText(text);
      }),
      // The floating recording indicator asks for the live meeting.
      on.navigate((target) => {
        if (target === "live") {
          setView("meetings");
          setLiveRequest((n) => n + 1);
        }
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

  // A removed speech model (the old CPU Parakeet) becomes its replacement,
  // which is a new download; until then the first installed model stands in.
  useEffect(() => {
    const next = migrateSttId(sttId);
    if (next === sttId || models.length === 0) return;
    setSttId(next);
    const m = models.find((x) => x.id === next);
    if (m && !m.downloaded) {
      setError(`${m.name} now runs on the GPU and needs a new download (${formatBytes(m.sizeBytes)}) in Settings › Models.`);
    }
  }, [sttId, models]);

  // Load the speech model ahead of the first take, so it isn't the slow one.
  useEffect(() => {
    if (stt) api.warmStt(stt.id).catch(() => {});
  }, [stt?.id]);

  // Likewise the cleanup model, with the instructions every take shares
  // already processed. Debounced: the prompt additions change as they're typed.
  useEffect(() => {
    if (!llm) return;
    const { transliterate, targetName } = outputLang(outLang);
    const t = setTimeout(() => {
      api
        .warmLlm(llm.id, prompt || undefined, styleId || undefined, targetName, transliterate)
        .catch(() => {});
    }, 800);
    return () => clearTimeout(t);
  }, [llm?.id, prompt, styleId, outLang]);

  const upsertTake = (patch: Partial<Omit<Take, "id" | "at">>, id = takeRef.current, at = new Date().toISOString()) => {
    if (!id) return;
    const existing = historyRef.current.find((t) => t.id === id);
    const take: Take = existing
      ? { ...existing, ...patch }
      : {
          id,
          at,
          raw: "",
          refined: "",
          sttModel: "",
          llmModel: "",
          ...patch,
        };
    // Merge synchronously: two pipeline stages can finish before React renders.
    const next = historyRef.current.filter((t) => t.id !== id).concat(take)
      .sort((a, b) => b.at.localeCompare(a.at)).slice(0, HISTORY_CAP);
    historyRef.current = next;
    setHistory(next);
    // Persist the full merged record to SQLite (fire-and-forget).
    api.saveTake(take).catch((e) => console.error("failed to save take", e));
  };

  const transcribeNow = async (model: ModelInfo): Promise<string | null> => {
    if (dictationStateRef.current.capturing || dictationStateRef.current.pending) return null;
    changePhase("transcribing");
    setStatus(`Transcribing with ${model.name}…`);
    setError(null);
    try {
      const text = await api.transcribe(model.id, language || undefined);
      setTranscript(text);
      upsertTake({ raw: text, sttModel: model.name, lang: language });
      changePhase("idle");
      setStatus("Transcript ready");
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
    if (dictationStateRef.current.capturing || dictationStateRef.current.pending || phaseRef.current !== "idle") return;
    if (!stt) {
      setError("Download a speech model first (Settings → Models).");
      return;
    }
    const name = path.split("/").pop() || "file";
    changePhase("transcribing");
    setTranscript("");
    setCleaned("");
    setRawOpen(false);
    setStatus(`Reading ${name}…`);
    setError(null);
    takeRef.current = crypto.randomUUID();
    setCurrentId(takeRef.current);
    // A one-shot listener shows the file's size and length once decoding
    // finishes, so long files don't look stuck.
    const infoSub = on.fileInfo(({ sizeBytes, durationSecs }) => {
      setStatus(
        `Transcribing ${name} · ${formatDuration(durationSecs)} · ${formatBytes(sizeBytes)}…`,
      );
    });
    try {
      const text = await api.transcribeFile(path, stt.id, language || undefined);
      setTranscript(text);
      upsertTake({ raw: text, sttModel: stt.name, app: name, lang: language });
      changePhase("idle");
      setStatus(`Transcribed ${name}`);
      if (llm) await runCleanup(text);
    } catch (e) {
      setError(String(e));
      changePhase("idle");
    } finally {
      infoSub.then((un) => un());
    }
  };

  const pickFile = async () => {
    if (dictationStateRef.current.capturing || dictationStateRef.current.pending || phaseRef.current !== "idle") return;
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
    if (dictationStateRef.current.capturing || dictationStateRef.current.pending) return null;
    if (!llm || !text) return null;
    const useStyle = styleOverride !== undefined ? styleOverride : styleIdRef.current;
    const useLangCode = langOverride !== undefined ? langOverride : outLangRef.current;
    const { transliterate, targetName } = outputLang(useLangCode);
    changePhase("cleaning");
    setCleaned("");
    cleaningRef.current = takeRef.current;
    setStatus(
      transliterate
        ? "Cleaning up in Latin script…"
        : targetName
          ? `Translating to ${targetName}…`
          : `Cleaning up with ${llm.name}…`,
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
        takeRef.current ?? undefined,
      );
      cleaningRef.current = null;
      setCleaned(result);
      upsertTake({ refined: result, llmModel: llm.name });
      changePhase("idle");
      setStatus("Cleaned up");
      return result;
    } catch (e) {
      cleaningRef.current = null;
      setError(String(e));
      changePhase("idle");
      return null;
    }
  };

  const queueRef = useRef<ReturnType<typeof createDictationQueue<DictationTake, true>> | null>(null);
  if (!queueRef.current) queueRef.current = createDictationQueue<DictationTake, true>({
    changed: (state) => {
      dictationStateRef.current = state;
      setDictationState(state);
      phaseRef.current = state.active ? "recording" : workPhaseRef.current;
      const overlay = state.active?.fromHotkey ? "recording"
        : hotkeyTakesRef.current.size > 0 ? "processing" : "hidden";
      const overlayKey = `${overlay}:${state.active?.id ?? ""}`;
      if (overlayKey !== overlayStateRef.current) {
        overlayStateRef.current = overlayKey;
        // Keep native visibility and events ordered, including a fast restart.
        overlayTailRef.current = overlayTailRef.current.then(async () => {
          await api.setOverlay(overlay !== "hidden");
          await api.emitOverlayState(overlay, state.active?.id);
        }).catch(() => {});
      }
    },
    start: async (take) => {
      await api.startRecording();
      // Await startup so a quick release cannot stop a worker before it starts.
      if (take.stt) await api.startLiveDictation(take.id, take.stt.id, take.language || undefined).catch(() => {});
    },
    finish: async (take, cancelled) => {
      const keep = !cancelled && !!take.stt;
      const res = await api.stopRecording(keep ? take.id : undefined);
      if (cancelled || !take.stt || res.durationSecs < 0.35) {
        if (keep) await api.discardRecording(take.id);
        hotkeyTakesRef.current.delete(take.id);
        if (takeRef.current === take.id) {
          changePhase("idle");
          setStatus(cancelled ? "Cancelled" : res.durationSecs < 0.35
            ? "Too short to transcribe. Hold the shortcut a little longer."
            : "Recorded. Download a speech model to transcribe it.");
        }
        return null;
      }
      return true;
    },
    process: async (take) => {
      const current = () => takeRef.current === take.id;
      const model = take.stt!;
      try {
        if (current()) {
          changePhase("transcribing");
          setStatus(`Transcribing with ${model.name}…`);
        }
        const text = await api.transcribe(model.id, take.language || undefined, false, take.id);
        upsertTake({ raw: text, sttModel: model.name, lang: take.language }, take.id, take.at);
        if (current()) setTranscript(text);
        let refined: string | null = null;
        if (text && take.llm) {
          const { transliterate, targetName } = outputLang(take.outLang);
          if (current()) {
            changePhase("cleaning");
            setCleaned("");
            setStatus(`Cleaning up with ${take.llm.name}…`);
          }
          cleaningRef.current = take.id;
          try {
            refined = await api.cleanupText(take.llm.id, text, take.prompt || undefined,
              take.style || undefined, targetName, transliterate, take.id);
            upsertTake({ refined, llmModel: take.llm.name }, take.id, take.at);
            if (current()) setCleaned(refined);
          } catch (e) {
            // Preserve the raw transcript and deliver it if cleanup fails.
            setError(String(e));
          } finally {
            cleaningRef.current = null;
          }
        }
        if (take.fromHotkey && text) {
          const target = await api.deliverText(refined ?? text);
          if (target) upsertTake({ app: target }, take.id, take.at);
          if (current()) setStatus(`Typed into ${target || "the front app"}`);
        } else if (current()) {
          setStatus(text ? "Transcript ready" : "No speech detected");
        }
      } finally {
        hotkeyTakesRef.current.delete(take.id);
        if (current()) changePhase("idle");
      }
    },
    failed: (take, e) => {
      hotkeyTakesRef.current.delete(take.id);
      setError(String(e));
      if (takeRef.current === take.id) changePhase("idle");
    },
  });

  const startTake = (fromHotkey: boolean) => {
    const queue = queueRef.current!;
    if (queue.state().active) return;
    // File imports and manual re-runs still own the editor until they finish.
    if (!queue.state().pending && !queue.state().capturing && phaseRef.current !== "idle") return;
    const take: DictationTake = {
      id: crypto.randomUUID(), at: new Date().toISOString(), fromHotkey,
      stt, llm, language, prompt, style: styleIdRef.current, outLang: outLangRef.current,
    };
    takeRef.current = take.id;
    setCurrentId(take.id);
    if (fromHotkey) hotkeyTakesRef.current.add(take.id);
    setError(null);
    setTranscript("");
    setCleaned("");
    setLiveText("");
    setRawOpen(false);
    changePhase("idle");
    setStatus(fromHotkey ? "Recording. Press the shortcut again to stop." : "Recording on this Mac");
    queue.start(take);
  };

  const stopTake = (cancelled = false) => {
    if (!queueRef.current!.state().active) return;
    changePhase(cancelled ? "idle" : "transcribing");
    setStatus(cancelled ? "Cancelled" : "Processing. You can dictate again.");
    queueRef.current!.stop(cancelled);
  };

  const toggleRecord = (fromHotkey = false) => {
    if (queueRef.current!.state().active) stopTake();
    else startTake(fromHotkey);
  };

  // Native listeners are registered once and route through current handlers.
  const hotkeyRef = useRef<() => void>(() => {});
  hotkeyRef.current = () => toggleRecord(true);
  const dropRef = useRef<(path: string) => void>(() => {});
  dropRef.current = (path: string) => transcribeFromFile(path);
  const pttRef = useRef<(what: "down" | "up" | "cancel") => void>(() => {});
  pttRef.current = (what) => {
    if (what === "down") startTake(true);
    else stopTake(what === "cancel");
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
    setStatus(`Added to dictionary: “${from}” → “${to}”`);
  };

  const copy = (text: string, what: string) => {
    navigator.clipboard.writeText(text);
    setStatus(`${what} copied`);
  };

  const loadTake = (take: Take) => {
    if (dictationStateRef.current.capturing || dictationStateRef.current.pending || phaseRef.current !== "idle") return;
    takeRef.current = take.id;
    setCurrentId(take.id);
    setTranscript(take.raw);
    setCleaned(take.refined);
    setRawOpen(!take.refined);
    // Restore the spoken language this take was recorded in.
    if (take.lang !== undefined) setLanguage(take.lang);
    setStatus("");
  };

  const busy = dictationState.capturing || dictationState.pending > 0 || phase === "transcribing" || phase === "cleaning";
  const manualBusy = busy && !dictationState.capturing && dictationState.pending === 0;
  const waveBars = useLevelHistory(phase === "recording");

  // Switching style re-renders the current take immediately.
  const styleSwitcher = stylesList.length > 0 && (
    <select
      className="select select-sm"
      aria-label="Writing style"
      value={styleId}
      disabled={busy}
      onChange={(e) => {
        chooseStyle(e.target.value);
        if (transcript && phaseRef.current === "idle") runCleanup(transcript, e.target.value);
      }}
      title="Writing style"
    >
      <option value="">No style</option>
      {stylesList.map((s) => (
        <option key={s.id} value={s.id}>
          {s.name}
        </option>
      ))}
    </select>
  );

  // Output-language switcher — re-renders the current take when changed.
  const outLangSwitcher = (
    <select
      className="select select-sm"
      aria-label="Output language"
      value={outLang}
      disabled={busy}
      onChange={(e) => {
        setOutLang(e.target.value);
        if (transcript && phaseRef.current === "idle") runCleanup(transcript, undefined, e.target.value);
      }}
      title="Keep the original language, or translate"
    >
      <option value="">Keep language</option>
      <option value="translit">Latin script</option>
      {LANGUAGES.filter((l) => l.code !== "").map((l) => (
        <option key={l.code} value={l.code}>
          Translate to {l.name}
        </option>
      ))}
    </select>
  );
  // Opening Dictations shows the newest one rather than an empty reader.
  useEffect(() => {
    if (view === "dictations" && !currentId && history.length && phaseRef.current === "idle" && !dictationStateRef.current.pending) loadTake(history[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, history.length]);

  const modelsReady = sttModels.length > 0 && llmModels.length > 0;
  const tabs = [
    { id: "meetings" as const, label: "Meetings" },
    { id: "dictations" as const, label: "Dictations" },
    { id: "inbox" as const, label: "Inbox", count: inbox.attention },
  ];
  const openSource = (id: string, at?: number | null) => {
    setView("meetings");
    setReminderTarget({ id, at: at ?? null });
  };
  // The live meeting takes the whole window, with its own stage bar.
  const focusMode = view === "meetings" && liveVisible;

  return (
    <div className={"shell" + (focusMode ? " focus" : "")}>
      {!focusMode && (
        <header className="topbar" data-tauri-drag-region>
          <div className="topbar-brand" data-tauri-drag-region>
            <Logo size={20} />
            <span data-tauri-drag-region>unsound</span>
          </div>
          <nav className="topbar-nav segmented" aria-label="Sections">
            {tabs.map((t) => (
              <button key={t.id} aria-pressed={view === t.id} onClick={() => setView(t.id)}>
                {t.label}
                {!!t.count && <span className="badge">{t.count}</span>}
              </button>
            ))}
          </nav>
          <div className="topbar-end">
            {recording && (
              <button
                className="topbar-rec"
                onClick={() => {
                  setView("meetings");
                  setLiveRequest((n) => n + 1);
                }}
                title="Open the live meeting"
              >
                <span className="rec-dot" />
                Recording
                <RecClock since={recording.startedAt} />
              </button>
            )}
            <UpdateCenter busy={phase !== "idle" || dictationState.pending > 0} />
            <button className="btn btn-quiet btn-icon" onClick={() => setSettingsOpen("general")} aria-label="Settings" title="Settings">
              <Icon name="settings" />
            </button>
          </div>
        </header>
      )}

      {/* Meetings stays mounted (hidden when inactive) so the tray "Record a
          meeting" toggle works from any section or with the window hidden. */}
      <main className={"view" + (view === "meetings" ? "" : " hidden")}>
        <Meetings
          stt={stt}
          llm={llm}
          language={language}
          models={models}
          onModelsChanged={refreshModels}
          openRequest={reminderTarget}
          onActivate={() => setView("meetings")}
          onRecordingChange={setRecording}
          onLiveVisibleChange={setLiveVisible}
          liveRequest={liveRequest}
          needsYou={<NeedsYou data={inbox} onOpenInbox={() => setView("inbox")} />}
          onOpenSettings={() => setSettingsOpen("meetings")}
        />
      </main>

      {view === "inbox" && (
        <main className="view">
          <Inbox data={inbox} onSource={openSource} />
        </main>
      )}

      {view === "dictations" && (
        <main className="view">
          <Dictations
            history={history}
            currentId={currentId}
            onSelect={loadTake}
            onDelete={(id) => {
              setHistory((h) => h.filter((t) => t.id !== id));
              if (id === currentId) {
                setCurrentId(null);
                setTranscript("");
                setCleaned("");
              }
              api.deleteTake(id).catch((e) => console.error("failed to delete dictation", e));
            }}
            phase={phase}
            pending={dictationState.pending}
            liveText={liveText}
            waveBars={waveBars}
            transcript={transcript}
            cleaned={cleaned}
            rawOpen={rawOpen}
            onRawOpen={setRawOpen}
            status={status}
            error={error}
            onDismissError={() => setError(null)}
            busy={busy}
            manualBusy={manualBusy}
            modelsReady={modelsReady}
            hasLlm={!!llm}
            hasStt={!!stt}
            shortcut={shortcut}
            language={language}
            onLanguage={setLanguage}
            languageNote={languageNote(stt, language, models)}
            styleSwitcher={styleSwitcher}
            outLangSwitcher={outLangSwitcher}
            onToggleRecord={() => toggleRecord()}
            onPickFile={pickFile}
            onCleanUp={() => runCleanup(transcript)}
            onRetranscribe={() => stt && transcribeNow(stt)}
            onCopy={copy}
            onCorrect={correctIn}
            onOpenModels={() => setSettingsOpen("models")}
          />
        </main>
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
          onClose={() => setSettingsOpen(null)}
          onChanged={refreshModels}
          onReplayOnboarding={() => {
            setSettingsOpen(null);
            setOnboarding(true);
          }}
          initialSection={settingsOpen}
          dictationCount={history.length}
          onClearDictations={() => {
            setHistory([]);
            setCurrentId(null);
            setTranscript("");
            setCleaned("");
            api.clearTakes().catch((e) => console.error("failed to clear dictations", e));
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
      {onboardedAtLaunch && !onboarding && <PermissionsCheck />}
      {dragOver && (
        <div className="drop-veil">
          <div className="drop-veil-inner">
            <Icon name="upload" />
            Drop an audio file to transcribe it
          </div>
        </div>
      )}
    </div>
  );
}

/// Elapsed time since an ISO start, ticking.
function RecClock({ since }: { since: string }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, []);
  const secs = Math.max(0, Math.floor((now - new Date(since).getTime()) / 1000));
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const ss = String(secs % 60).padStart(2, "0");
  return <time>{h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`}</time>;
}
