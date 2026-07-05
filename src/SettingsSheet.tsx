import { useState } from "react";
import { api, ModelInfo, ModelKind } from "./api";
import { DictionarySection, MicPicker, ModelLibrary, PermissionsSection, ShortcutsSection, StylesSection } from "./sections";

type Section =
  | "models"
  | "shortcuts"
  | "styles"
  | "dictionary"
  | "microphone"
  | "prompt"
  | "accessibility"
  | "permissions";

const SECTIONS: { id: Section; label: string }[] = [
  { id: "shortcuts", label: "shortcuts" },
  { id: "styles", label: "styles" },
  { id: "dictionary", label: "dictionary" },
  { id: "models", label: "models" },
  { id: "microphone", label: "microphone" },
  { id: "prompt", label: "prompt" },
  { id: "accessibility", label: "accessibility" },
  { id: "permissions", label: "permissions" },
];

const TEXT_SIZES: { label: string; value: string }[] = [
  { label: "Small", value: "0.9" },
  { label: "Default", value: "1" },
  { label: "Large", value: "1.2" },
  { label: "Larger", value: "1.4" },
  { label: "Largest", value: "1.6" },
];

interface Props {
  models: ModelInfo[];
  sttId: string;
  llmId: string;
  onSttChange: (id: string) => void;
  onLlmChange: (id: string) => void;
  prompt: string;
  defaultPrompt: string;
  onPromptChange: (p: string) => void;
  textScale: string;
  onTextScaleChange: (v: string) => void;
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
  textScale,
  onTextScaleChange,
  onClose,
  onChanged,
  onReplayOnboarding,
}: Props) {
  const [section, setSection] = useState<Section>("shortcuts");
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

  const activePicker = (
    label: string,
    value: string,
    onChange: (id: string) => void,
    options: ModelInfo[],
  ) => (
    <div className="row">
      <div className="row-info">
        <div className="row-name">{label}</div>
      </div>
      <div className="row-action">
        <select
          className="chip-select"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={options.length === 0}
        >
          {options.length === 0 && <option value="">none installed</option>}
          {options.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name}
            </option>
          ))}
        </select>
      </div>
    </div>
  );

  const body = () => {
    switch (section) {
      case "models":
        return (
          <>
            <section className="sheet-section">
              <div className="sheet-section-head">
                <h3>active models</h3>
                <span className="sheet-hint">which installed model each stage uses</span>
              </div>
              {activePicker("voice → text", sttId, onSttChange, sttModels)}
              {activePicker("text cleanup", llmId, onLlmChange, llmModels)}
            </section>
            <ModelLibrary models={models} onChanged={onChanged} onError={setErr} />
            <section className="sheet-section">
              {addOpen ? (
                <div className="add-form">
                  <div className="add-row">
                    <select
                      value={customKind}
                      onChange={(e) => setCustomKind(e.target.value as ModelKind)}
                    >
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
          </>
        );
      case "shortcuts":
        return (
          <section className="sheet-section">
            <div className="sheet-section-head">
              <h3>global shortcuts</h3>
              <span className="sheet-hint">dictate into any app; the text lands at your cursor</span>
            </div>
            <ShortcutsSection onError={setErr} />
          </section>
        );
      case "styles":
        return (
          <section className="sheet-section">
            <div className="sheet-section-head">
              <h3>writing styles</h3>
              <span className="sheet-hint">refined text imitates the samples of the active style</span>
            </div>
            <StylesSection onError={setErr} />
          </section>
        );
      case "dictionary":
        return (
          <section className="sheet-section">
            <div className="sheet-section-head">
              <h3>personal dictionary</h3>
              <span className="sheet-hint">corrections that bias recognition and cleanup</span>
            </div>
            <DictionarySection onError={setErr} />
          </section>
        );
      case "microphone":
        return (
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
        );
      case "prompt":
        return (
          <>
            <section className="sheet-section">
              <div className="sheet-section-head">
                <h3>system prompt</h3>
                <span className="sheet-hint">what the cleanup model is told — fixed</span>
              </div>
              <textarea className="prompt-editor readonly" value={defaultPrompt} readOnly rows={12} />
            </section>
            <section className="sheet-section">
              <div className="sheet-section-head">
                <h3>your additions</h3>
                <span className="sheet-hint">appended to the system prompt on every refine</span>
              </div>
              <textarea
                className="prompt-editor"
                placeholder="e.g. never use em dashes; spell it “colour”; keep numbers as digits"
                value={prompt}
                onChange={(e) => onPromptChange(e.target.value)}
                rows={6}
                spellCheck={false}
              />
            </section>
          </>
        );
      case "accessibility":
        return (
          <section className="sheet-section">
            <div className="sheet-section-head">
              <h3>text size</h3>
              <span className="sheet-hint">scales the transcript and refined text</span>
            </div>
            <div className="size-row">
              {TEXT_SIZES.map((s) => (
                <button
                  key={s.value}
                  className={"size-btn" + (textScale === s.value ? " on" : "")}
                  onClick={() => onTextScaleChange(s.value)}
                >
                  {s.label}
                </button>
              ))}
            </div>
            <div
              className="size-preview"
              style={{ fontSize: `calc(15.5px * ${textScale})` }}
            >
              The quick brown fox jumps over the lazy dog.
            </div>
          </section>
        );
      case "permissions":
        return (
          <section className="sheet-section">
            <div className="sheet-section-head">
              <h3>permissions</h3>
              <span className="sheet-hint">what macOS needs to allow</span>
            </div>
            <PermissionsSection onError={setErr} />
          </section>
        );
    }
  };

  return (
    <div className="sheet-overlay" onClick={onClose}>
      <div className="sheet settings" onClick={(e) => e.stopPropagation()}>
        <nav className="settings-nav">
          <div className="settings-nav-title">settings</div>
          {SECTIONS.map((s) => (
            <button
              key={s.id}
              className={"settings-nav-item" + (section === s.id ? " on" : "")}
              onClick={() => {
                setErr(null);
                setSection(s.id);
              }}
            >
              {s.label}
            </button>
          ))}
          <div className="spacer" />
          <button className="settings-nav-item" onClick={onReplayOnboarding}>
            welcome guide
          </button>
        </nav>
        <div className="settings-pane">
          <header className="sheet-head">
            <h2>{SECTIONS.find((s) => s.id === section)?.label}</h2>
            <span className="sheet-note">downloads are the only time unsound touches the network</span>
            <button className="quiet" onClick={onClose}>
              close ✕
            </button>
          </header>
          <div className="sheet-body">
            {body()}
            {err && <div className="sheet-error">{err}</div>}
          </div>
        </div>
      </div>
    </div>
  );
}
