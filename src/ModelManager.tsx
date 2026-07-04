import { useEffect, useState } from "react";
import { api, formatBytes, ModelInfo, ModelKind, on } from "./api";

interface Props {
  models: ModelInfo[];
  onClose: () => void;
  onChanged: () => void;
}

interface Progress {
  downloaded: number;
  total: number;
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
      await api.addCustomModel(customName.trim() || customUrl.split("/").pop() || "custom", customKind, customUrl.trim());
      setCustomName("");
      setCustomUrl("");
      setAddOpen(false);
      onChanged();
    } catch (e) {
      setErr(String(e));
    }
  };

  const section = (kind: ModelKind, title: string, hint: string) => (
    <section className="mm-section">
      <div className="mm-section-head">
        <h3>{title}</h3>
        <span className="mm-hint">{hint}</span>
      </div>
      {models
        .filter((m) => m.kind === kind)
        .map((m) => {
          const p = progress[m.id];
          const pct = p && p.total > 0 ? Math.round((p.downloaded / p.total) * 100) : null;
          return (
            <div className="mm-row" key={m.id}>
              <div className="mm-row-info">
                <div className="mm-name">
                  {m.name}
                  {m.custom && <span className="mm-tag">custom</span>}
                </div>
                <div className="mm-desc">
                  {formatBytes(m.sizeBytes)} · {m.description}
                </div>
              </div>
              <div className="mm-row-action">
                {p ? (
                  <div className="mm-progress">
                    <div className="mm-progress-track">
                      <div
                        className="mm-progress-fill"
                        style={{ width: `${pct ?? 2}%` }}
                      />
                    </div>
                    <span className="mm-progress-label">
                      {pct !== null ? `${pct}%` : "…"}
                    </span>
                  </div>
                ) : m.downloaded ? (
                  <>
                    <span className="mm-installed">● installed</span>
                    <button className="btn ghost" onClick={() => remove(m.id)}>
                      remove
                    </button>
                  </>
                ) : (
                  <button className="btn accent" onClick={() => download(m.id)}>
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
    <div className="mm-overlay" onClick={onClose}>
      <div className="mm-sheet" onClick={(e) => e.stopPropagation()}>
        <header className="mm-head">
          <h2>model library</h2>
          <div className="mm-head-note">
            downloads come from Hugging Face — the only time this app touches the network
          </div>
          <button className="btn ghost" onClick={onClose}>
            close ✕
          </button>
        </header>

        <div className="mm-body">
          {section("stt", "speech → text", "whisper.cpp GGML models")}
          {section("llm", "text cleanup", "any llama.cpp GGUF instruct model")}

          <section className="mm-section">
            {addOpen ? (
              <div className="mm-add-form">
                <div className="mm-add-row">
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
                <div className="mm-add-row">
                  <button className="btn accent" disabled={!customUrl.trim()} onClick={addCustom}>
                    add model
                  </button>
                  <button className="btn ghost" onClick={() => setAddOpen(false)}>
                    cancel
                  </button>
                </div>
              </div>
            ) : (
              <button className="btn ghost mm-add-btn" onClick={() => setAddOpen(true)}>
                + add a custom model by URL
              </button>
            )}
          </section>

          {err && <div className="mm-error">{err}</div>}
        </div>
      </div>
    </div>
  );
}
