import { useEffect, useState } from "react";
import { api, formatBytes, formatShortcut, ModelInfo, ModelKind, on } from "./api";

/* ── global shortcut ─────────────────────────────────────────────── */

export function ShortcutSection({ onError }: { onError: (msg: string | null) => void }) {
  const [shortcut, setShortcut] = useState<string | null>(null);
  const [capturing, setCapturing] = useState(false);

  useEffect(() => {
    api.getSettings().then((s) => setShortcut(s.shortcut));
  }, []);

  useEffect(() => {
    if (!capturing) return;
    const onKey = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.key === "Escape") {
        setCapturing(false);
        return;
      }
      const mods = [
        e.metaKey ? "cmd" : null,
        e.ctrlKey ? "ctrl" : null,
        e.altKey ? "alt" : null,
        e.shiftKey ? "shift" : null,
      ].filter(Boolean) as string[];
      if (mods.length === 0) return; // a bare key is not a global shortcut

      let key: string | null = null;
      if (e.code.startsWith("Key")) key = e.code.slice(3);
      else if (e.code.startsWith("Digit")) key = e.code.slice(5);
      else if (e.code === "Space") key = "space";
      else if (/^F\d{1,2}$/.test(e.code)) key = e.code;
      if (!key) return; // still holding only modifiers

      const combo = [...mods, key.toLowerCase()].join("+");
      setCapturing(false);
      onError(null);
      api
        .setShortcut(combo)
        .then(() => setShortcut(combo))
        .catch((err) => onError(String(err)));
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [capturing, onError]);

  const disable = () => {
    onError(null);
    api
      .setShortcut("")
      .then(() => setShortcut(""))
      .catch((err) => onError(String(err)));
  };

  return (
    <div className="row">
      <div className="row-info">
        <div className="row-name">
          {capturing ? (
            <span className="capture-hint">press a key combination… (esc to cancel)</span>
          ) : (
            <span className="shortcut-keys">{shortcut === null ? "…" : formatShortcut(shortcut)}</span>
          )}
        </div>
        <div className="row-desc">
          press once to start recording — in any app — and again to stop; the refined text is
          pasted where your cursor is
        </div>
      </div>
      <div className="row-action">
        <button className="quiet accent" onClick={() => setCapturing(true)} disabled={capturing}>
          change
        </button>
        {shortcut && (
          <button className="quiet" onClick={disable}>
            disable
          </button>
        )}
      </div>
    </div>
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
            lets unsound press ⌘V for you so shortcut dictation lands in the app you're using;
            without it the text still goes to your clipboard
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
    </>
  );
}
