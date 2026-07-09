import { useEffect, useState } from "react";
import { api, DictEntry, formatBytes, formatShortcut, ModelInfo, ModelKind, on, Style } from "./api";

/* ── global shortcuts (hands-free + push-to-talk, multiple each) ── */

type ShortcutMode = "handsFree" | "pushToTalk";

function comboFromEvent(e: KeyboardEvent): string | null {
  const mods = [
    e.metaKey ? "cmd" : null,
    e.ctrlKey ? "ctrl" : null,
    e.altKey ? "alt" : null,
    e.shiftKey ? "shift" : null,
  ].filter(Boolean) as string[];

  let key: string | null = null;
  if (e.code.startsWith("Key")) key = e.code.slice(3);
  else if (e.code.startsWith("Digit")) key = e.code.slice(5);
  else if (e.code === "Space") key = "space";
  else if (e.code === "Backspace") key = "backspace";
  else if (e.code === "Delete") key = "delete";
  else if (e.code === "Home") key = "home";
  else if (e.code === "End") key = "end";
  else if (/^F\d{1,2}$/.test(e.code)) key = e.code;
  if (!key) return null; // still holding only modifiers

  // A bare letter/digit would hijack normal typing; F-keys are fine alone.
  if (mods.length === 0 && !/^F\d{1,2}$/.test(key)) return null;
  return [...mods, key.toLowerCase()].join("+");
}

function partialFromEvent(e: KeyboardEvent): string {
  return [
    e.metaKey ? "cmd" : null,
    e.ctrlKey ? "ctrl" : null,
    e.altKey ? "alt" : null,
    e.shiftKey ? "shift" : null,
  ]
    .filter(Boolean)
    .join("+");
}

export function ShortcutsSection({ onError }: { onError: (msg: string | null) => void }) {
  const [handsFree, setHandsFree] = useState<string[]>([]);
  const [pushToTalk, setPushToTalk] = useState<string[]>([]);
  const [loaded, setLoaded] = useState(false);
  // Which binding is being recorded: mode + index (-1 appends a new one).
  const [capturing, setCapturing] = useState<{ mode: ShortcutMode; index: number } | null>(null);
  // Keys currently held, shown live while capturing.
  const [held, setHeld] = useState("");
  const [nativeCapture, setNativeCapture] = useState(false);

  useEffect(() => {
    api.getSettings().then((s) => {
      setHandsFree(s.handsFree);
      setPushToTalk(s.pushToTalk);
      setLoaded(true);
    });
  }, []);

  const save = (hf: string[], ptt: string[]) => {
    onError(null);
    api
      .setShortcuts(hf, ptt)
      .then(() => {
        setHandsFree(hf);
        setPushToTalk(ptt);
      })
      .catch((err) => onError(String(err)));
  };

  const commit = (combo: string) => {
    if (!capturing) return;
    const { mode, index } = capturing;
    const current = mode === "handsFree" ? handsFree : pushToTalk;
    const next = index === -1 ? [...current, combo] : current.map((c, i) => (i === index ? combo : c));
    setCapturing(null);
    setHeld("");
    save(mode === "handsFree" ? next : handsFree, mode === "pushToTalk" ? next : pushToTalk);
  };

  const beginCapture = async (mode: ShortcutMode, index: number) => {
    setHeld("");
    setCapturing({ mode, index });
    // The native event tap sees every key incl. fn; falls back to webview
    // key events (no fn) when Accessibility isn't granted.
    setNativeCapture(await api.startShortcutCapture());
  };

  const endCapture = () => {
    api.cancelShortcutCapture();
    setCapturing(null);
    setHeld("");
  };

  // Native capture: live updates + commit come from the Rust listener.
  useEffect(() => {
    if (!capturing || !nativeCapture) return;
    const subs = [
      on.captureUpdate(setHeld),
      on.captureCommit((combo) => commit(combo)),
      on.captureCancel(() => {
        setCapturing(null);
        setHeld("");
      }),
    ];
    return () => {
      subs.forEach((p) => p.then((un) => un()));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [capturing, nativeCapture, handsFree, pushToTalk]);

  // Webview fallback: modifier keydowns give the live display, a full combo commits.
  useEffect(() => {
    if (!capturing || nativeCapture) return;
    const onKey = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.key === "Escape") {
        endCapture();
        return;
      }
      const combo = comboFromEvent(e);
      if (combo) commit(combo);
      else setHeld(partialFromEvent(e));
    };
    const onUp = (e: KeyboardEvent) => setHeld(partialFromEvent(e));
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("keyup", onUp, true);
    return () => {
      window.removeEventListener("keydown", onKey, true);
      window.removeEventListener("keyup", onUp, true);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [capturing, nativeCapture, handsFree, pushToTalk]);

  const liveChip = (
    <span className="capture-hint">
      {held ? <span className="shortcut-keys live">{formatShortcut(held)}</span> : "press keys…"}
      <button className="quiet" onClick={endCapture} title="Cancel (esc)">
        esc
      </button>
    </span>
  );

  const block = (mode: ShortcutMode, title: string, desc: string, combos: string[]) => (
    <div className="row shortcut-block">
      <div className="row-info">
        <div className="row-name">{title}</div>
        <div className="row-desc">{desc}</div>
      </div>
      <div className="shortcut-list">
        {combos.map((combo, i) => (
          <div className="shortcut-item" key={`${combo}-${i}`}>
            {capturing?.mode === mode && capturing.index === i ? (
              liveChip
            ) : (
              <span className="shortcut-keys">{formatShortcut(combo)}</span>
            )}
            <button className="quiet" title="Change" onClick={() => beginCapture(mode, i)}>
              ✎
            </button>
            <button
              className="quiet"
              title="Remove"
              onClick={() =>
                save(
                  mode === "handsFree" ? combos.filter((_, j) => j !== i) : handsFree,
                  mode === "pushToTalk" ? combos.filter((_, j) => j !== i) : pushToTalk,
                )
              }
            >
              ✕
            </button>
          </div>
        ))}
        {capturing?.mode === mode && capturing.index === -1 ? (
          <div className="shortcut-item">{liveChip}</div>
        ) : (
          <button className="quiet accent" onClick={() => beginCapture(mode, -1)}>
            + add
          </button>
        )}
      </div>
    </div>
  );

  if (!loaded) return null;
  return (
    <>
      {block(
        "handsFree",
        "hands-free",
        "press once to start recording, press again to stop",
        handsFree,
      )}
      {block(
        "pushToTalk",
        "push to talk",
        "hold to say something short; release to finish",
        pushToTalk,
      )}
      <div className="sheet-hint" style={{ display: "block", marginTop: 6 }}>
        fn-based shortcuts (bare fn, fn Space…) need the Accessibility permission; other combos
        (⌘⇧Space, ⌥⌫, ⇧Home…) work without it. release modifier-only combos to set them.
      </div>
    </>
  );
}

/* ── writing styles ──────────────────────────────────────────────── */

export function StylesSection({ onError }: { onError: (msg: string | null) => void }) {
  const [styles, setStyles] = useState<Style[]>([]);
  const [defaultStyle, setDefaultStyle] = useState("");
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    api.getSettings().then((s) => {
      setStyles(s.styles);
      setDefaultStyle(s.defaultStyle);
      setLoaded(true);
    });
  }, []);

  const persist = (nextStyles: Style[], nextDefault: string) => {
    onError(null);
    setStyles(nextStyles);
    setDefaultStyle(nextDefault);
    api.setStyles(nextStyles, nextDefault).catch((e) => onError(String(e)));
  };

  // Text edits update local state on each keystroke and persist on blur,
  // so settings.json isn't rewritten per character.
  const updateLocal = (next: Style[]) => setStyles(next);
  const persistCurrent = () => persist(styles, defaultStyle);

  const patchStyle = (id: string, patch: Partial<Style>) =>
    styles.map((s) => (s.id === id ? { ...s, ...patch } : s));

  const addStyle = () => {
    const style: Style = {
      id: crypto.randomUUID(),
      name: "new style",
      notes: "",
      lowercase: false,
      samples: [""],
    };
    persist([...styles, style], defaultStyle || style.id);
  };

  const removeStyle = (id: string) => {
    const next = styles.filter((s) => s.id !== id);
    persist(next, defaultStyle === id ? (next[0]?.id ?? "") : defaultStyle);
  };

  if (!loaded) return null;
  return (
    <>
      {styles.map((style) => (
        <div className="style-card" key={style.id}>
          <div className="style-head">
            <input
              className="style-name"
              value={style.name}
              onChange={(e) => updateLocal(patchStyle(style.id, { name: e.target.value }))}
              onBlur={persistCurrent}
              spellCheck={false}
            />
            <label className="style-default">
              <input
                type="radio"
                name="default-style"
                checked={defaultStyle === style.id}
                onChange={() => persist(styles, style.id)}
              />
              default
            </label>
            <label className="style-default" title="Applied in code — always exact, unlike model behavior">
              <input
                type="checkbox"
                checked={style.lowercase ?? false}
                onChange={(e) =>
                  persist(patchStyle(style.id, { lowercase: e.target.checked }), defaultStyle)
                }
              />
              lowercase
            </label>
            <span className="spacer" />
            <button className="quiet danger" onClick={() => removeStyle(style.id)}>
              delete style
            </button>
          </div>
          <input
            className="style-notes"
            placeholder="style rules (optional) — e.g. all lowercase, even 'i'; short sentences; no emoji"
            value={style.notes ?? ""}
            onChange={(e) => updateLocal(patchStyle(style.id, { notes: e.target.value }))}
            onBlur={persistCurrent}
            spellCheck={false}
          />
          {style.samples.map((sample, i) => (
            <div className="style-sample" key={i}>
              <textarea
                rows={3}
                placeholder="paste a block of your writing in this style…"
                value={sample}
                onChange={(e) =>
                  updateLocal(
                    patchStyle(style.id, {
                      samples: style.samples.map((s, j) => (j === i ? e.target.value : s)),
                    }),
                  )
                }
                onBlur={persistCurrent}
                spellCheck={false}
              />
              <button
                className="quiet"
                title="Remove sample"
                onClick={() =>
                  persist(
                    patchStyle(style.id, { samples: style.samples.filter((_, j) => j !== i) }),
                    defaultStyle,
                  )
                }
              >
                ✕
              </button>
            </div>
          ))}
          <button
            className="quiet accent"
            onClick={() => persist(patchStyle(style.id, { samples: [...style.samples, ""] }), defaultStyle)}
          >
            + add sample
          </button>
        </div>
      ))}
      <button className="quiet add-btn" onClick={addStyle}>
        + add a style
      </button>
      {styles.length === 0 && (
        <div className="sheet-hint" style={{ display: "block", marginTop: 8 }}>
          a style is a name plus a few blocks of your own writing; refined text is rendered to
          match how those samples are written
        </div>
      )}
    </>
  );
}

/* ── personal dictionary ─────────────────────────────────────────── */

export function DictionarySection({ onError }: { onError: (msg: string | null) => void }) {
  const [entries, setEntries] = useState<DictEntry[]>([]);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");

  const load = () => api.getSettings().then((s) => setEntries(s.dictionary));
  useEffect(() => {
    load();
  }, []);

  const remove = (i: number) => {
    const next = entries.filter((_, j) => j !== i);
    api.setDictionary(next).then(() => setEntries(next)).catch((e) => onError(String(e)));
  };

  const add = () => {
    onError(null);
    api
      .addCorrection(from, to)
      .then(() => {
        setFrom("");
        setTo("");
        load();
      })
      .catch((e) => onError(String(e)));
  };

  return (
    <>
      {entries.map((e, i) => (
        <div className="row" key={`${e.from}-${i}`}>
          <div className="row-info">
            <div className="row-name">
              “{e.from}” <span className="dict-arrow">→</span> “{e.to}”
            </div>
          </div>
          <div className="row-action">
            <button className="quiet" onClick={() => remove(i)}>
              remove
            </button>
          </div>
        </div>
      ))}
      <div className="add-form">
        <div className="add-row">
          <input
            placeholder="what whisper hears"
            value={from}
            onChange={(e) => setFrom(e.target.value)}
            spellCheck={false}
          />
          <input
            placeholder="what you mean"
            value={to}
            onChange={(e) => setTo(e.target.value)}
            spellCheck={false}
          />
          <button className="quiet accent" disabled={!from.trim() || !to.trim()} onClick={add}>
            add
          </button>
        </div>
      </div>
      {entries.length === 0 && (
        <div className="sheet-hint" style={{ display: "block", marginTop: 8 }}>
          click any word in a transcript to correct it — corrections land here, bias recognition
          toward your vocabulary, and teach the cleanup model
        </div>
      )}
    </>
  );
}

/* ── microphone picker ───────────────────────────────────────────── */

export function MicPicker({ className = "chip-select" }: { className?: string }) {
  const [devices, setDevices] = useState<string[]>([]);
  const [selected, setSelected] = useState("");

  const refresh = () => {
    api.listMicrophones().then(setDevices);
    api.getSettings().then((s) => setSelected(s.micDevice));
  };

  useEffect(() => {
    refresh();
    // Follow changes made from the tray menu (and vice versa).
    const sub = on.settingsChanged(refresh);
    return () => {
      sub.then((un) => un());
    };
  }, []);

  const choose = (device: string) => {
    setSelected(device);
    api.setMicrophone(device).catch(() => refresh());
  };

  return (
    <select
      className={className}
      value={selected}
      onChange={(e) => choose(e.target.value)}
      onMouseDown={refresh}
      title="Microphone"
    >
      <option value="">🎙 system default</option>
      {devices.map((d) => (
        <option key={d} value={d}>
          🎙 {d}
        </option>
      ))}
    </select>
  );
}

/* ── system permissions ──────────────────────────────────────────── */

export function PermissionsSection({ onError }: { onError: (msg: string | null) => void }) {
  const [accessibility, setAccessibility] = useState<boolean | null>(null);
  const [micState, setMicState] = useState<"unknown" | "checking" | "ok" | "failed">("unknown");

  const refresh = () => api.permissionStatus().then((p) => setAccessibility(p.accessibility));

  useEffect(() => {
    refresh();
    // The user may grant access in System Settings while this view is open.
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
  }, []);

  const testMic = async () => {
    setMicState("checking");
    onError(null);
    try {
      await api.requestMicrophone();
      setMicState("ok");
    } catch (e) {
      setMicState("failed");
      onError(String(e));
    }
  };

  const askAccessibility = async () => {
    onError(null);
    const granted = await api.requestAccessibility();
    setAccessibility(granted);
  };

  return (
    <>
      <div className="row">
        <div className="row-info">
          <div className="row-name">
            microphone{" "}
            {micState === "ok" && <span className="granted">✓ working</span>}
            {micState === "failed" && <span className="denied">not available</span>}
          </div>
          <div className="row-desc">needed to record; macOS asks the first time</div>
        </div>
        <div className="row-action">
          <button className="quiet accent" onClick={testMic} disabled={micState === "checking"}>
            {micState === "checking" ? "listening…" : "test"}
          </button>
        </div>
      </div>
      <div className="row">
        <div className="row-info">
          <div className="row-name">
            auto-paste (Accessibility){" "}
            {accessibility === true && <span className="granted">✓ granted</span>}
            {accessibility === false && <span className="denied">not granted</span>}
          </div>
          <div className="row-desc">
            lets unsound type the refined text into the app you're using — the clipboard is never
            touched
          </div>
        </div>
        <div className="row-action">
          {accessibility === false && (
            <button className="quiet accent" onClick={askAccessibility}>
              allow…
            </button>
          )}
        </div>
      </div>
    </>
  );
}

/* ── model library (download list) ───────────────────────────────── */

interface Progress {
  downloaded: number;
  total: number;
}

export function useDownloads(onError: (msg: string | null) => void) {
  const [progress, setProgress] = useState<Record<string, Progress>>({});

  useEffect(() => {
    const subs = [
      on.downloadProgress((p) =>
        setProgress((prev) => ({ ...prev, [p.id]: { downloaded: p.downloaded, total: p.total } })),
      ),
      on.downloadDone((d) => {
        setProgress((prev) => {
          const next = { ...prev };
          delete next[d.id];
          return next;
        });
        if (!d.ok && d.error) onError(d.error);
      }),
    ];
    return () => {
      subs.forEach((p) => p.then((un) => un()));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const start = (id: string) => {
    onError(null);
    setProgress((prev) => ({ ...prev, [id]: { downloaded: 0, total: 0 } }));
    // Progress and completion arrive via events; errors surface there too.
    api.downloadModel(id).catch((e) => onError(String(e)));
  };

  return { progress, start };
}

export function ModelRow({
  model,
  progress,
  onDownload,
  onRemove,
}: {
  model: ModelInfo;
  progress?: Progress;
  onDownload: (id: string) => void;
  onRemove?: (id: string) => void;
}) {
  const pct =
    progress && progress.total > 0 ? Math.round((progress.downloaded / progress.total) * 100) : null;
  return (
    <div className="row">
      <div className="row-info">
        <div className="row-name">
          {model.name}
          {model.recommended && <span className="tag">recommended</span>}
          {model.custom && <span className="tag">custom</span>}
        </div>
        <div className="row-desc">
          {formatBytes(model.sizeBytes)}
          {model.languages && <> · {model.languages}</>} · {model.description}
        </div>
      </div>
      <div className="row-action">
        {progress ? (
          <div className="progress">
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${pct ?? 2}%` }} />
            </div>
            <span className="progress-label">{pct !== null ? `${pct}%` : "…"}</span>
          </div>
        ) : model.downloaded ? (
          <>
            <span className="installed">● installed</span>
            {onRemove && (
              <button className="quiet" onClick={() => onRemove(model.id)}>
                remove
              </button>
            )}
          </>
        ) : (
          <button className="quiet accent" onClick={() => onDownload(model.id)}>
            download
          </button>
        )}
      </div>
    </div>
  );
}

export function ModelLibrary({
  models,
  onChanged,
  onError,
  compact = false,
}: {
  models: ModelInfo[];
  onChanged: () => void;
  onError: (msg: string | null) => void;
  compact?: boolean;
}) {
  const { progress, start } = useDownloads(onError);

  const remove = async (id: string) => {
    onError(null);
    try {
      await api.deleteModel(id);
      onChanged();
    } catch (e) {
      onError(String(e));
    }
  };

  const section = (kind: ModelKind, title: string, hint: string) => (
    <section className="sheet-section">
      <div className="sheet-section-head">
        <h3>{title}</h3>
        <span className="sheet-hint">{hint}</span>
      </div>
      {models
        .filter((m) => m.kind === kind)
        .map((m) => (
          <ModelRow
            key={m.id}
            model={m}
            progress={progress[m.id]}
            onDownload={start}
            onRemove={compact ? undefined : remove}
          />
        ))}
    </section>
  );

  return (
    <>
      {section("stt", "voice → text", "whisper.cpp models; pick English-only if that's all you speak")}
      {section("llm", "text cleanup", "any llama.cpp GGUF instruct model")}
      {!compact && section("diarize", "meeting speakers", "who-said-what models; a stronger embedding separates similar voices better")}
      {!compact && section("vad", "meeting voice detection", "chunks meeting audio on pauses")}
    </>
  );
}
