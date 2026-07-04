import { useState } from "react";
import { api, ModelInfo, ModelKind } from "./api";
import { MicPicker, ModelLibrary, PermissionsSection, ShortcutSection } from "./sections";

interface Props {
  models: ModelInfo[];
  sttId: string;
  llmId: string;
  onSttChange: (id: string) => void;
  onLlmChange: (id: string) => void;
  prompt: string;
  defaultPrompt: string;
  onPromptChange: (p: string) => void;
  onClose: () => void;
  onChanged: () => void;
  onReplayOnboarding: () => void;
}

export function SettingsSheet({
  models,
  sttId,
  llmId,
  onSttChange,
  onLlmChange,
  prompt,
  defaultPrompt,
  onPromptChange,
  onClose,
  onChanged,
  onReplayOnboarding,
}: Props) {
  const [err, setErr] = useState<string | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  const [customName, setCustomName] = useState("");
  const [customUrl, setCustomUrl] = useState("");
  const [customKind, setCustomKind] = useState<ModelKind>("llm");

  const sttModels = models.filter((m) => m.kind === "stt" && m.downloaded);
  const llmModels = models.filter((m) => m.kind === "llm" && m.downloaded);

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

  return (
    <div className="sheet-overlay" onClick={onClose}>
      <div className="sheet" onClick={(e) => e.stopPropagation()}>
        <header className="sheet-head">
          <h2>settings</h2>
          <span className="sheet-note">downloads are the only time unsound touches the network</span>
          <button className="quiet" onClick={onClose}>
            close ✕
          </button>
        </header>

        <div className="sheet-body">
          <section className="sheet-section">
            <div className="sheet-section-head">
              <h3>active models</h3>
              <span className="sheet-hint">which installed models each stage uses</span>
            </div>
            <div className="row">
              <div className="row-info">
                <div className="row-name">voice → text</div>
              </div>
              <div className="row-action">
                <select
                  className="chip-select"
                  value={sttId}
                  onChange={(e) => onSttChange(e.target.value)}
                  disabled={sttModels.length === 0}
                >
                  {sttModels.length === 0 && <option value="">none installed</option>}
                  {sttModels.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="row">
              <div className="row-info">
                <div className="row-name">text cleanup</div>
              </div>
              <div className="row-action">
                <select
                  className="chip-select"
                  value={llmId}
                  onChange={(e) => onLlmChange(e.target.value)}
                  disabled={llmModels.length === 0}
                >
                  {llmModels.length === 0 && <option value="">none installed</option>}
                  {llmModels.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </section>

          <section className="sheet-section">
            <div className="sheet-section-head">
              <h3>microphone</h3>
              <span className="sheet-hint">also switchable from the main window and the menu bar</span>
            </div>
            <div className="row">
              <div className="row-info">
                <div className="row-name">input source</div>
              </div>
              <div className="row-action">
                <MicPicker />
              </div>
            </div>
          </section>

          <section className="sheet-section">
            <div className="sheet-section-head">
              <h3>global shortcut</h3>
              <span className="sheet-hint">dictate into any app on this Mac</span>
            </div>
            <ShortcutSection onError={setErr} />
          </section>

          <section className="sheet-section">
            <div className="sheet-section-head">
              <h3>permissions</h3>
              <span className="sheet-hint">what macOS needs to allow</span>
            </div>
            <PermissionsSection onError={setErr} />
          </section>

          <section className="sheet-section">
            <div className="sheet-section-head">
              <h3>cleanup prompt</h3>
              <span className="sheet-hint">instructions the text model follows; clear it to use the default</span>
            </div>
            <textarea
              className="prompt-editor"
              value={prompt || defaultPrompt}
              onChange={(e) => onPromptChange(e.target.value)}
              rows={10}
              spellCheck={false}
            />
          </section>

          <ModelLibrary models={models} onChanged={onChanged} onError={setErr} />

          <section className="sheet-section">
            {addOpen ? (
              <div className="add-form">
                <div className="add-row">
                  <select value={customKind} onChange={(e) => setCustomKind(e.target.value as ModelKind)}>
                    <option value="stt">voice (GGML)</option>
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

          <section className="sheet-section" style={{ textAlign: "center" }}>
            <button className="quiet" onClick={onReplayOnboarding}>
              show the welcome guide again
            </button>
          </section>

          {err && <div className="sheet-error">{err}</div>}
        </div>
      </div>
    </div>
  );
}
