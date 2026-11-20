import { ReactNode, useEffect, useId, useRef, useState } from "react";
import { api, DictEntry, formatBytes, formatShortcut, ModelInfo, ModelKind, on, Style } from "./api";
import { Icon, useConfirm } from "./ui";

/* ── layout helpers shared by settings, onboarding and meeting setup ── */

/// A titled group of settings: small heading, optional hint, then content
/// (usually a PrefCard).
export function PrefGroup({
  title,
  hint,
  action,
  children,
}: {
  title: string;
  hint?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="pref-group">
      <div className="pref-group-head">
        <h3 className="section-label">{title}</h3>
        {action}
      </div>
      {hint && <p className="pref-group-hint">{hint}</p>}
      {children}
    </section>
  );
}

export function PrefCard({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={"card pref-card " + className}>{children}</div>;
}

/// Label + description on the left, control on the right.
export function PrefRow({
  label,
  desc,
  htmlFor,
  children,
  className = "",
}: {
  label?: ReactNode;
  desc?: ReactNode;
  htmlFor?: string;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <div className={"pref-row " + className}>
      {(label || desc) && (
        <div className="pref-text">
          {label &&
            (htmlFor ? (
              <label className="pref-label" htmlFor={htmlFor}>
                {label}
              </label>
            ) : (
              <div className="pref-label">{label}</div>
            ))}
          {desc && <div className="pref-desc">{desc}</div>}
        </div>
      )}
      {children && <div className="pref-control">{children}</div>}
    </div>
  );
}

/// A whole-row checkbox: the text is its label, the box sits on the right.
export function PrefCheck({
  label,
  desc,
  checked,
  disabled,
  onChange,
}: {
  label: ReactNode;
  desc?: ReactNode;
  checked: boolean;
  disabled?: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className={"pref-row pref-check" + (disabled ? " disabled" : "")}>
      <span className="pref-text">
        <span className="pref-label">{label}</span>
        {desc && <span className="pref-desc">{desc}</span>}
      </span>
      <span className="pref-control">
        <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      </span>
    </label>
  );
}

/* ── global shortcuts (hands-free + push-to-talk, multiple each) ── */

type ShortcutMode = "handsFree" | "pushToTalk" | "command";

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

const SHORTCUT_LABELS: Record<ShortcutMode, string> = {
  handsFree: "Hands-free",
  pushToTalk: "Push to talk",
  command: "Ask unsound",
};

export function ShortcutsSection({ onError }: { onError: (msg: string | null) => void }) {
  const [handsFree, setHandsFree] = useState<string[]>([]);
  const [pushToTalk, setPushToTalk] = useState<string[]>([]);
  const [commandKeys, setCommandKeys] = useState<string[]>([]);
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
      setCommandKeys(s.commandKeys);
      setLoaded(true);
    });
  }, []);

  const saveCommand = (keys: string[]) => {
    onError(null);
    api
      .setCommandKeys(keys)
      .then(() => setCommandKeys(keys))
      .catch((err) => onError(String(err)));
  };

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
    const current = mode === "handsFree" ? handsFree : mode === "pushToTalk" ? pushToTalk : commandKeys;
    const next = index === -1 ? [...current, combo] : current.map((c, i) => (i === index ? combo : c));
    setCapturing(null);
    setHeld("");
    if (mode === "command") saveCommand(next);
    else save(mode === "handsFree" ? next : handsFree, mode === "pushToTalk" ? next : pushToTalk);
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
    // Escape cancels the capture natively; keep it from also closing the
    // surrounding sheet.
    const swallowEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") e.preventDefault();
    };
    window.addEventListener("keydown", swallowEscape, true);
    return () => {
      subs.forEach((p) => p.then((un) => un()));
      window.removeEventListener("keydown", swallowEscape, true);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [capturing, nativeCapture, handsFree, pushToTalk, commandKeys]);

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
  }, [capturing, nativeCapture, handsFree, pushToTalk, commandKeys]);

  const liveChip = (
    <span className="shortcut-capture" role="status">
      {held ? (
        <span className="shortcut-keys live">{formatShortcut(held)}</span>
      ) : (
        <span className="shortcut-keys live empty">Press keys…</span>
      )}
      <button className="btn btn-quiet btn-sm" onClick={endCapture}>
        Cancel
      </button>
    </span>
  );

  const block = (mode: ShortcutMode, desc: string, combos: string[]) => {
    const label = SHORTCUT_LABELS[mode];
    return (
      <PrefRow label={label} desc={desc} className="shortcut-row">
        <div className="shortcut-list">
          {combos.map((combo, i) => (
            <div className="shortcut-item" key={`${combo}-${i}`}>
              {capturing?.mode === mode && capturing.index === i ? (
                liveChip
              ) : (
                <>
                  <span className="shortcut-keys">{formatShortcut(combo)}</span>
                  <button
                    className="btn btn-quiet btn-icon btn-sm"
                    aria-label={`Change ${label} shortcut ${formatShortcut(combo)}`}
                    title="Change"
                    onClick={() => beginCapture(mode, i)}
                  >
                    <Icon name="pencil" />
                  </button>
                  <button
                    className="btn btn-quiet btn-icon btn-sm"
                    aria-label={`Remove ${label} shortcut ${formatShortcut(combo)}`}
                    title="Remove"
                    onClick={() =>
                      mode === "command"
                        ? saveCommand(combos.filter((_, j) => j !== i))
                        : save(
                            mode === "handsFree" ? combos.filter((_, j) => j !== i) : handsFree,
                            mode === "pushToTalk" ? combos.filter((_, j) => j !== i) : pushToTalk,
                          )
                    }
                  >
                    <Icon name="x" />
                  </button>
                </>
              )}
            </div>
          ))}
          {capturing?.mode === mode && capturing.index === -1 ? (
            <div className="shortcut-item">{liveChip}</div>
          ) : (
            <button
              className="btn btn-secondary btn-sm"
              aria-label={`Add ${label} shortcut`}
              onClick={() => beginCapture(mode, -1)}
            >
              {combos.length ? "Add" : "Set shortcut"}
            </button>
          )}
        </div>
      </PrefRow>
    );
  };

  if (!loaded) return null;
  return (
    <>
      <PrefCard>
        {block("handsFree", "Press once to start, again to stop.", handsFree)}
        {block("pushToTalk", "Hold while you speak. Release to finish.", pushToTalk)}
        {block(
          "command",
          "Hold and say a request, like “draft an email to Priya about the launch”. While recording a call, it uses the conversation.",
          commandKeys,
        )}
      </PrefCard>
      <p className="pref-note">
        Shortcuts with fn need Accessibility access. Other combos, like ⌘⇧Space, work without it. For a
        modifier-only shortcut, release the keys to set it.
      </p>
    </>
  );
}

/* ── writing styles ──────────────────────────────────────────────── */

export function StylesSection({ onError }: { onError: (msg: string | null) => void }) {
  const [styles, setStyles] = useState<Style[]>([]);
  const [defaultStyle, setDefaultStyle] = useState("");
  const [loaded, setLoaded] = useState(false);
  const { confirm, dialog } = useConfirm();

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
      name: "New style",
      notes: "",
      lowercase: false,
      samples: [""],
    };
    persist([...styles, style], defaultStyle || style.id);
  };

  const removeStyle = (style: Style) =>
    confirm({
      title: `Delete “${style.name || "Untitled"}”?`,
      description: "Its rules and writing samples are removed. This can't be undone.",
      confirm: "Delete style",
      danger: true,
      onConfirm: () => {
        const next = styles.filter((s) => s.id !== style.id);
        persist(next, defaultStyle === style.id ? (next[0]?.id ?? "") : defaultStyle);
      },
    });

  const removeSample = (style: Style, i: number) => {
    const run = () =>
      persist(patchStyle(style.id, { samples: style.samples.filter((_, j) => j !== i) }), defaultStyle);
    if (!style.samples[i]?.trim()) return run();
    confirm({
      title: "Remove this sample?",
      description: "The text is removed from the style. This can't be undone.",
      confirm: "Remove",
      danger: true,
      onConfirm: run,
    });
  };

  if (!loaded) return null;
  return (
    <div className="style-list">
      {styles.map((style) => (
        <div className="card style-card" key={style.id}>
          <div className="style-head">
            <input
              className="input style-name"
              aria-label="Style name"
              value={style.name}
              onChange={(e) => updateLocal(patchStyle(style.id, { name: e.target.value }))}
              onBlur={persistCurrent}
              spellCheck={false}
            />
            <label className="check">
              <input
                type="radio"
                name="default-style"
                checked={defaultStyle === style.id}
                onChange={() => persist(styles, style.id)}
              />
              Default
            </label>
            <label className="check" title="Applied exactly, after cleanup">
              <input
                type="checkbox"
                checked={style.lowercase ?? false}
                onChange={(e) => persist(patchStyle(style.id, { lowercase: e.target.checked }), defaultStyle)}
              />
              All lowercase
            </label>
            <span className="spacer" />
            <button
              className="btn btn-quiet btn-sm danger"
              aria-label={`Delete style ${style.name}`}
              onClick={() => removeStyle(style)}
            >
              Delete
            </button>
          </div>
          <input
            className="input style-notes"
            aria-label={`Rules for ${style.name}`}
            placeholder="Rules (optional), like “short sentences, no emoji”"
            value={style.notes ?? ""}
            onChange={(e) => updateLocal(patchStyle(style.id, { notes: e.target.value }))}
            onBlur={persistCurrent}
            spellCheck={false}
          />
          {style.samples.map((sample, i) => (
            <div className="style-sample" key={i}>
              <textarea
                className="textarea"
                rows={3}
                aria-label={`Writing sample ${i + 1}`}
                placeholder="Paste something you wrote in this style…"
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
                className="btn btn-quiet btn-icon btn-sm"
                aria-label={`Remove sample ${i + 1}`}
                title="Remove sample"
                onClick={() => removeSample(style, i)}
              >
                <Icon name="x" />
              </button>
            </div>
          ))}
          <div>
            <button
              className="btn btn-quiet btn-sm"
              onClick={() => persist(patchStyle(style.id, { samples: [...style.samples, ""] }), defaultStyle)}
            >
              Add sample
            </button>
          </div>
        </div>
      ))}
      {styles.length === 0 && (
        <PrefCard>
          <div className="pref-empty">
            A style is a name and a few samples of your writing. Cleaned-up text is written to match them.
          </div>
        </PrefCard>
      )}
      <div>
        <button className="btn btn-secondary" onClick={addStyle}>
          Add style
        </button>
      </div>
      {dialog}
    </div>
  );
}

/* ── personal dictionary ─────────────────────────────────────────── */

export function DictionarySection({ onError }: { onError: (msg: string | null) => void }) {
  const [entries, setEntries] = useState<DictEntry[]>([]);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const { confirm, dialog } = useConfirm();
  const id = useId();

  const load = () => api.getSettings().then((s) => setEntries(s.dictionary));
  useEffect(() => {
    load();
  }, []);

  const remove = (i: number) =>
    confirm({
      title: `Remove “${entries[i].from}”?`,
      description: "unsound stops correcting it.",
      confirm: "Remove",
      danger: true,
      onConfirm: () => {
        const next = entries.filter((_, j) => j !== i);
        return api
          .setDictionary(next)
          .then(() => setEntries(next))
          .catch((e) => onError(String(e)));
      },
    });

  const add = () => {
    if (!from.trim() || !to.trim()) return;
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
    <PrefCard>
      {entries.length === 0 && (
        <div className="pref-empty">
          No corrections yet. Click a word in any transcript to correct it, and it's added here.
        </div>
      )}
      {entries.map((e, i) => (
        <PrefRow
          key={`${e.from}-${i}`}
          label={
            <span className="dict-pair">
              <span>{e.from}</span>
              <Icon name="chevronRight" className="dict-arrow" />
              <span>{e.to}</span>
            </span>
          }
        >
          <button className="btn btn-quiet btn-sm" aria-label={`Remove ${e.from}`} onClick={() => remove(i)}>
            Remove
          </button>
        </PrefRow>
      ))}
      <form
        className="pref-row dict-add"
        onSubmit={(ev) => {
          ev.preventDefault();
          add();
        }}
      >
        <label className="visually-hidden" htmlFor={`${id}-from`}>
          What it hears
        </label>
        <input
          id={`${id}-from`}
          className="input"
          placeholder="What it hears"
          value={from}
          onChange={(e) => setFrom(e.target.value)}
          spellCheck={false}
        />
        <label className="visually-hidden" htmlFor={`${id}-to`}>
          What you mean
        </label>
        <input
          id={`${id}-to`}
          className="input"
          placeholder="What you mean"
          value={to}
          onChange={(e) => setTo(e.target.value)}
          spellCheck={false}
        />
        <button type="submit" className="btn btn-secondary" disabled={!from.trim() || !to.trim()}>
          Add
        </button>
      </form>
      {dialog}
    </PrefCard>
  );
}

/* ── microphone picker ───────────────────────────────────────────── */

export function MicPicker({ className = "select" }: { className?: string }) {
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
      aria-label="Microphone"
    >
      <option value="">System default</option>
      {devices.map((d) => (
        <option key={d} value={d}>
          {d}
        </option>
      ))}
    </select>
  );
}

/* ── model library (download list) ───────────────────────────────── */

interface Progress {
  downloaded: number;
  total: number;
}

export function useDownloads(
  onError: (msg: string | null) => void,
  onDone?: (id: string, ok: boolean) => void,
) {
  const [progress, setProgress] = useState<Record<string, Progress>>({});
  // Listeners subscribe once; read the latest callbacks through refs.
  const cb = useRef({ onError, onDone });
  cb.current = { onError, onDone };

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
        if (!d.ok && d.error) cb.current.onError(d.error);
        cb.current.onDone?.(d.id, d.ok);
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
    <div className="pref-row model-row">
      <div className="pref-text">
        <div className="pref-label">
          {model.name}
          {model.downloaded ? (
            <span className="badge ok">Installed</span>
          ) : (
            model.recommended && <span className="badge">Recommended</span>
          )}
          {model.custom && <span className="badge neutral">Custom</span>}
        </div>
        <div className="pref-desc">
          {model.description}
          {model.languages && <> · {model.languages}</>} · {formatBytes(model.sizeBytes)}
        </div>
      </div>
      <div className="pref-control">
        {progress ? (
          <div
            className="progress"
            role="progressbar"
            aria-label={`Downloading ${model.name}`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={pct ?? undefined}
          >
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${pct ?? 2}%` }} />
            </div>
            <span className="progress-label">{pct !== null ? `${pct}%` : "Starting…"}</span>
          </div>
        ) : model.downloaded ? (
          onRemove && (
            <button
              className="btn btn-quiet btn-sm danger"
              aria-label={`Delete ${model.name}`}
              onClick={() => onRemove(model.id)}
            >
              Delete
            </button>
          )
        ) : (
          <button
            className="btn btn-secondary btn-sm"
            aria-label={`Download ${model.name}`}
            onClick={() => onDownload(model.id)}
          >
            <Icon name="download" />
            Download
          </button>
        )}
      </div>
    </div>
  );
}

const LIBRARY_GROUPS: { kind: ModelKind; title: string; hint: string; full?: boolean }[] = [
  { kind: "stt", title: "Speech models", hint: "Turn speech into text. English-only versions are a little more accurate in English." },
  { kind: "llm", title: "Cleanup models", hint: "Clean up dictations and write summaries. Any GGUF instruct model works." },
  { kind: "diarize", title: "Speaker detection", hint: "Tells voices apart in recordings. Stronger models separate similar voices better.", full: true },
  { kind: "vad", title: "Voice activity detection", hint: "Splits recordings at pauses.", full: true },
];

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
  const { confirm, dialog } = useConfirm();

  const remove = (id: string) => {
    const m = models.find((x) => x.id === id);
    confirm({
      title: `Delete ${m?.name ?? "this model"}?`,
      description: `This frees ${m ? formatBytes(m.sizeBytes) : "space"} on this Mac. You can download it again later.`,
      confirm: "Delete",
      danger: true,
      onConfirm: async () => {
        onError(null);
        try {
          await api.deleteModel(id);
          onChanged();
        } catch (e) {
          onError(String(e));
        }
      },
    });
  };

  return (
    <>
      {LIBRARY_GROUPS.filter((g) => !compact || !g.full).map((g) => {
        const list = models.filter((m) => m.kind === g.kind);
        if (list.length === 0) return null;
        return (
          <PrefGroup key={g.kind} title={g.title} hint={compact ? undefined : g.hint}>
            <PrefCard>
              {list.map((m) => (
                <ModelRow
                  key={m.id}
                  model={m}
                  progress={progress[m.id]}
                  onDownload={start}
                  onRemove={compact ? undefined : remove}
                />
              ))}
            </PrefCard>
          </PrefGroup>
        );
      })}
      {dialog}
    </>
  );
}
