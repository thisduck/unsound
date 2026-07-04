import { useEffect, useState } from "react";
import { api, formatBytes, formatShortcut, ModelInfo, ModelKind, on } from "./api";

interface Props {
  models: ModelInfo[];
  onClose: () => void;
  onChanged: () => void;
}

interface Progress {
  downloaded: number;
  total: number;
}

function ShortcutSettings({ onError }: { onError: (msg: string | null) => void }) {
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
    <section className="sheet-section">
      <div className="sheet-section-head">
        <h3>global shortcut</h3>
        <span className="sheet-hint">record from any app; refined text is pasted where you are</span>
      </div>
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
            press once to start recording, again to stop — the refined text lands in the frontmost
            app (macOS will ask for Accessibility permission the first time)
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
    </section>
  );
}

export function ModelManager({ models, onClose, onChanged }: Props) {
  const [progress, setProgress] = useState<Record<string, Progress>>({});
  const [err, setErr] = useState<string | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  const [customName, setCustomName] = useState("");
  const [customUrl, setCustomUrl] = useState("");
  const [customKind, setCustomKind] = useState<ModelKind>("llm");

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
        if (!d.ok && d.error) setErr(d.error);
      }),
    ];
    return () => {
      subs.forEach((p) => p.then((un) => un()));
    };
  }, []);

  const download = (id: string) => {
    setErr(null);
    setProgress((prev) => ({ ...prev, [id]: { downloaded: 0, total: 0 } }));
    // Resolution (progress + refresh) is driven by events; errors surface there too.
    api.downloadModel(id).catch((e) => setErr(String(e)));
  };

  const remove = async (id: string) => {
    setErr(null);
    try {
      await api.deleteModel(id);
      onChanged();
    } catch (e) {
      setErr(String(e));
    }
  };

  const addCustom = async () => {
    setErr(null);
    try {
      await api.addCustomModel(
        customName.trim() || customUrl.split("/").pop() || "custom",
        customKind,
        customUrl.trim(),
      );
      setCustomName("");
      setCustomUrl("");
      setAddOpen(false);
      onChanged();
    } catch (e) {
      setErr(String(e));
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
        .map((m) => {
          const p = progress[m.id];
          const pct = p && p.total > 0 ? Math.round((p.downloaded / p.total) * 100) : null;
          return (
            <div className="row" key={m.id}>
              <div className="row-info">
                <div className="row-name">
                  {m.name}
                  {m.custom && <span className="tag">custom</span>}
                </div>
                <div className="row-desc">
                  {formatBytes(m.sizeBytes)} · {m.description}
                </div>
              </div>
              <div className="row-action">
                {p ? (
                  <div className="progress">
                    <div className="progress-track">
                      <div className="progress-fill" style={{ width: `${pct ?? 2}%` }} />
                    </div>
                    <span className="progress-label">{pct !== null ? `${pct}%` : "…"}</span>
                  </div>
                ) : m.downloaded ? (
                  <>
                    <span className="installed">● installed</span>
                    <button className="quiet" onClick={() => remove(m.id)}>
                      remove
                    </button>
                  </>
                ) : (
                  <button className="quiet accent" onClick={() => download(m.id)}>
                    download
                  </button>
                )}
              </div>
            </div>
          );
        })}
    </section>
  );

  return (
    <div className="sheet-overlay" onClick={onClose}>
      <div className="sheet" onClick={(e) => e.stopPropagation()}>
        <header className="sheet-head">
          <h2>models &amp; settings</h2>
          <span className="sheet-note">downloads are the only time unsound touches the network</span>
          <button className="quiet" onClick={onClose}>
            close ✕
          </button>
        </header>

        <div className="sheet-body">
          <ShortcutSettings onError={setErr} />
          {section("stt", "speech → text", "whisper.cpp GGML models")}
          {section("llm", "text cleanup", "any llama.cpp GGUF instruct model")}

          <section className="sheet-section">
            {addOpen ? (
              <div className="add-form">
                <div className="add-row">
                  <select value={customKind} onChange={(e) => setCustomKind(e.target.value as ModelKind)}>
                    <option value="stt">speech (GGML)</option>
                    <option value="llm">cleanup (GGUF)</option>
                  </select>
                  <input
                    placeholder="display name (optional)"
                    value={customName}
                    onChange={(e) => setCustomName(e.target.value)}
                  />
                </div>
                <input
                  placeholder="direct download URL, e.g. https://huggingface.co/…/resolve/main/model.gguf"
                  value={customUrl}
                  onChange={(e) => setCustomUrl(e.target.value)}
                />
                <div className="add-row">
                  <button className="quiet accent" disabled={!customUrl.trim()} onClick={addCustom}>
                    add model
                  </button>
                  <button className="quiet" onClick={() => setAddOpen(false)}>
                    cancel
                  </button>
                </div>
              </div>
            ) : (
              <button className="quiet add-btn" onClick={() => setAddOpen(true)}>
                + add a custom model by URL
              </button>
            )}
          </section>

          {err && <div className="sheet-error">{err}</div>}
        </div>
      </div>
    </div>
  );
}
