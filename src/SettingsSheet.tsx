import { useCallback, useEffect, useId, useRef, useState } from "react";
import { getVersion } from "@tauri-apps/api/app";
import { openUpdates } from "./UpdateCenter";
import { api, ModelInfo, ModelKind } from "./api";
import {
  DictionarySection,
  MicPicker,
  ModelLibrary,
  PrefCard,
  PrefCheck,
  PrefGroup,
  PrefRow,
  ShortcutsSection,
  StylesSection,
} from "./sections";
import { AssistantSection, MeetingsSection } from "./AssistantSettings";
import { PermissionsPane, useOpenPermissions } from "./Permissions";
import { Icon, IconName, Segmented, useConfirm } from "./ui";

export type SettingsSection = "general" | "dictation" | "meetings" | "assistant" | "models" | "permissions";

const SECTIONS: { id: SettingsSection; label: string; icon: IconName; desc: string }[] = [
  {
    id: "general",
    label: "General",
    icon: "settings",
    desc: "Shortcuts, microphone, text size and updates.",
  },
  {
    id: "dictation",
    label: "Dictation",
    icon: "mic",
    desc: "How your speech becomes clean text.",
  },
  {
    id: "meetings",
    label: "Meetings",
    icon: "meetings",
    desc: "Models and options for recording calls.",
  },
  {
    id: "assistant",
    label: "Assistant",
    icon: "sparkle",
    desc: "What unsound does when you ask it to. Connected tools use the network only as you allow.",
  },
  {
    id: "models",
    label: "Models",
    icon: "download",
    desc: "Download and manage the models that run on this Mac.",
  },
  {
    id: "permissions",
    label: "Permissions",
    icon: "shield",
    desc: "What unsound can use on this Mac, and why.",
  },
];

const TEXT_SIZES: { label: string; value: string }[] = [
  { label: "Small", value: "0.9" },
  { label: "Default", value: "1" },
  { label: "Large", value: "1.2" },
  { label: "Larger", value: "1.4" },
  { label: "Largest", value: "1.6" },
];

const AUTO_UPDATE_KEY = "unsound.auto-check-updates";

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
  initialSection?: SettingsSection;
  dictationCount: number;
  onClearDictations: () => void;
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
  initialSection = "general",
  dictationCount,
  onClearDictations,
}: Props) {
  const [section, setSection] = useState<SettingsSection>(initialSection);
  const [err, setErr] = useState<string | null>(null);
  const titleId = useId();
  const bodyRef = useRef<HTMLDivElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  // Escape closes, unless something inside (a confirm dialog, shortcut
  // capture) already handled it.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !e.defaultPrevented) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    dialogRef.current?.focus();
  }, []);

  const go = (id: SettingsSection) => {
    setErr(null);
    setSection(id);
    bodyRef.current?.scrollTo(0, 0);
  };
  useOpenPermissions(useCallback(() => go("permissions"), []));

  const current = SECTIONS.find((s) => s.id === section)!;

  const body = () => {
    switch (section) {
      case "general":
        return (
          <GeneralPane
            textScale={textScale}
            onTextScaleChange={onTextScaleChange}
            onReplayOnboarding={onReplayOnboarding}
            onError={setErr}
          />
        );
      case "dictation":
        return (
          <DictationPane
            models={models}
            sttId={sttId}
            llmId={llmId}
            onSttChange={onSttChange}
            onLlmChange={onLlmChange}
            prompt={prompt}
            defaultPrompt={defaultPrompt}
            onPromptChange={onPromptChange}
            dictationCount={dictationCount}
            onClearDictations={onClearDictations}
            onOpenModels={() => go("models")}
            onError={setErr}
          />
        );
      case "meetings":
        return <MeetingsSection models={models} onError={setErr} />;
      case "assistant":
        return <AssistantSection models={models} onError={setErr} onOpenShortcuts={() => go("general")} />;
      case "models":
        return <ModelsPane models={models} onChanged={onChanged} onError={setErr} />;
      case "permissions":
        return <PermissionsPane onError={setErr} />;
    }
  };

  return (
    <div className="settings-overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div
        className="settings"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        ref={dialogRef}
      >
        <nav className="settings-nav" aria-label="Settings sections">
          <div className="settings-nav-title">Settings</div>
          {SECTIONS.map((s) => (
            <button
              key={s.id}
              className="settings-nav-item"
              aria-current={section === s.id ? "page" : undefined}
              onClick={() => go(s.id)}
            >
              <Icon name={s.icon} />
              {s.label}
            </button>
          ))}
        </nav>
        <div className="settings-pane">
          <header className="settings-head">
            <div className="settings-head-text">
              <h2 className="page-title" id={titleId}>
                {current.label}
              </h2>
              <p className="settings-desc">{current.desc}</p>
            </div>
            <button className="btn btn-quiet btn-icon" aria-label="Close settings" title="Close" onClick={onClose}>
              <Icon name="x" />
            </button>
          </header>
          <div className="settings-body" ref={bodyRef}>
            {err && (
              <div className="alert danger" role="alert">
                <Icon name="alert" />
                <div className="alert-body">{err}</div>
                <div className="alert-actions">
                  <button className="btn btn-quiet btn-icon btn-sm" aria-label="Dismiss" onClick={() => setErr(null)}>
                    <Icon name="x" />
                  </button>
                </div>
              </div>
            )}
            {body()}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── General ─────────────────────────────────────────────────────── */

function GeneralPane({
  textScale,
  onTextScaleChange,
  onReplayOnboarding,
  onError,
}: {
  textScale: string;
  onTextScaleChange: (v: string) => void;
  onReplayOnboarding: () => void;
  onError: (msg: string | null) => void;
}) {
  const [version, setVersion] = useState<string | null>(null);
  const [autoCheck, setAutoCheck] = useState(() => {
    try {
      return localStorage.getItem(AUTO_UPDATE_KEY) !== "false";
    } catch {
      return true;
    }
  });

  useEffect(() => {
    getVersion()
      .then(setVersion)
      .catch(() => setVersion(null));
  }, []);

  const setAuto = (on: boolean) => {
    setAutoCheck(on);
    try {
      localStorage.setItem(AUTO_UPDATE_KEY, on ? "true" : "false");
    } catch {
      /* storage unavailable; the toggle still applies for this session */
    }
  };

  return (
    <>
      <PrefGroup title="Shortcuts" hint="Work in any app. The text is typed at your cursor.">
        <ShortcutsSection onError={onError} />
      </PrefGroup>

      <PrefGroup title="Input">
        <PrefCard>
          <PrefRow label="Microphone" desc="You can also switch it from the menu bar.">
            <MicPicker />
          </PrefRow>
        </PrefCard>
      </PrefGroup>

      <PrefGroup title="Text size" hint="Scales transcripts, summaries and dictations.">
        <PrefCard>
          <div className="pref-row text-size">
            <Segmented
              label="Text size"
              value={TEXT_SIZES.some((s) => s.value === textScale) ? textScale : "1"}
              options={TEXT_SIZES}
              onChange={onTextScaleChange}
            />
            <p className="text-size-preview" style={{ fontSize: `calc(15px * ${textScale})` }}>
              The quick brown fox jumps over the lazy dog.
            </p>
          </div>
        </PrefCard>
      </PrefGroup>

      <PrefGroup title="Updates">
        <PrefCard>
          <PrefRow
            label="unsound"
            desc={version ? `Version ${version}` : "Checking version…"}
          >
            <button className="btn btn-secondary btn-sm" onClick={openUpdates}>
              Check for updates…
            </button>
          </PrefRow>
          <PrefCheck
            label="Check for updates automatically"
            desc="Contacts GitHub to look for new versions. You choose when to install."
            checked={autoCheck}
            onChange={setAuto}
          />
        </PrefCard>
      </PrefGroup>

      <PrefGroup title="Help">
        <PrefCard>
          <PrefRow label="Welcome guide" desc="Walk through setup again.">
            <button className="btn btn-secondary btn-sm" onClick={onReplayOnboarding}>
              Show welcome guide
            </button>
          </PrefRow>
        </PrefCard>
      </PrefGroup>
    </>
  );
}

/* ── Dictation ───────────────────────────────────────────────────── */

function DictationPane({
  models,
  sttId,
  llmId,
  onSttChange,
  onLlmChange,
  prompt,
  defaultPrompt,
  onPromptChange,
  dictationCount,
  onClearDictations,
  onOpenModels,
  onError,
}: {
  models: ModelInfo[];
  sttId: string;
  llmId: string;
  onSttChange: (id: string) => void;
  onLlmChange: (id: string) => void;
  prompt: string;
  defaultPrompt: string;
  onPromptChange: (p: string) => void;
  dictationCount: number;
  onClearDictations: () => void;
  onOpenModels: () => void;
  onError: (msg: string | null) => void;
}) {
  const { confirm, dialog } = useConfirm();
  const promptId = useId();
  const sttModels = models.filter((m) => m.kind === "stt" && m.downloaded);
  const llmModels = models.filter((m) => m.kind === "llm" && m.downloaded);

  const activePicker = (
    label: string,
    desc: string,
    value: string,
    onChange: (id: string) => void,
    options: ModelInfo[],
  ) => (
    <PrefRow label={label} desc={desc}>
      {options.length === 0 ? (
        <button className="btn btn-secondary btn-sm" onClick={onOpenModels}>
          Download a model
        </button>
      ) : (
        <select className="select" aria-label={label} value={value} onChange={(e) => onChange(e.target.value)}>
          {!options.some((m) => m.id === value) && <option value="">Choose…</option>}
          {options.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name}
            </option>
          ))}
        </select>
      )}
    </PrefRow>
  );

  return (
    <>
      <PrefGroup title="Models">
        <PrefCard>
          {activePicker("Speech model", "Turns your speech into text.", sttId, onSttChange, sttModels)}
          {activePicker("Cleanup model", "Fixes punctuation and removes filler words.", llmId, onLlmChange, llmModels)}
        </PrefCard>
      </PrefGroup>

      <PrefGroup title="Writing styles" hint="Cleaned-up text matches the samples in your default style.">
        <StylesSection onError={onError} />
      </PrefGroup>

      <PrefGroup title="Dictionary" hint="Names and words unsound should get right. Used by both models.">
        <DictionarySection onError={onError} />
      </PrefGroup>

      <PrefGroup title="Custom instructions">
        <label className="visually-hidden" htmlFor={promptId}>
          Custom instructions
        </label>
        <textarea
          id={promptId}
          className="textarea"
          placeholder="Added to every clean up. For example: never use em dashes; spell it “colour”; keep numbers as digits."
          value={prompt}
          onChange={(e) => onPromptChange(e.target.value)}
          rows={5}
          spellCheck={false}
        />
        <details className="pref-disclosure">
          <summary>
            <Icon name="chevronRight" />
            View built-in instructions
          </summary>
          <pre className="builtin-prompt">{defaultPrompt}</pre>
        </details>
      </PrefGroup>

      <PrefGroup title="Data">
        <PrefCard>
          <PrefRow
            label="Dictations"
            desc={`${dictationCount} ${dictationCount === 1 ? "dictation" : "dictations"} saved on this Mac.`}
          >
            <button
              className="btn btn-quiet btn-sm danger"
              disabled={dictationCount === 0}
              onClick={() =>
                confirm({
                  title: `Delete all ${dictationCount} dictations?`,
                  description: "They're removed from this Mac. This can't be undone.",
                  confirm: "Delete all",
                  danger: true,
                  onConfirm: onClearDictations,
                })
              }
            >
              Delete all dictations…
            </button>
          </PrefRow>
        </PrefCard>
      </PrefGroup>
      {dialog}
    </>
  );
}

/* ── Models ──────────────────────────────────────────────────────── */

function ModelsPane({
  models,
  onChanged,
  onError,
}: {
  models: ModelInfo[];
  onChanged: () => void;
  onError: (msg: string | null) => void;
}) {
  const [addOpen, setAddOpen] = useState(false);
  const [customName, setCustomName] = useState("");
  const [customUrl, setCustomUrl] = useState("");
  const [customKind, setCustomKind] = useState<ModelKind>("llm");
  const id = useId();

  const addCustom = async () => {
    if (!customUrl.trim()) return;
    onError(null);
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
      onError(String(e));
    }
  };

  return (
    <>
      <ModelLibrary models={models} onChanged={onChanged} onError={onError} />
      <PrefGroup title="Custom models">
        {addOpen ? (
          <form
            className="card add-model"
            onSubmit={(e) => {
              e.preventDefault();
              addCustom();
            }}
          >
            <div className="add-model-grid">
              <label className="field">
                <span className="field-label">Type</span>
                <select
                  className="select"
                  value={customKind}
                  onChange={(e) => setCustomKind(e.target.value as ModelKind)}
                >
                  <option value="stt">Speech model (GGML)</option>
                  <option value="llm">Cleanup model (GGUF)</option>
                </select>
              </label>
              <label className="field">
                <span className="field-label">Name (optional)</span>
                <input className="input" value={customName} onChange={(e) => setCustomName(e.target.value)} />
              </label>
            </div>
            <label className="field" htmlFor={`${id}-url`}>
              <span className="field-label">Download URL</span>
            </label>
            <input
              id={`${id}-url`}
              className="input"
              placeholder="https://huggingface.co/…/resolve/main/model.gguf"
              value={customUrl}
              onChange={(e) => setCustomUrl(e.target.value)}
              autoFocus
            />
            <div className="add-model-actions">
              <button type="button" className="btn btn-quiet" onClick={() => setAddOpen(false)}>
                Cancel
              </button>
              <button type="submit" className="btn btn-primary" disabled={!customUrl.trim()}>
                Add model
              </button>
            </div>
          </form>
        ) : (
          <div>
            <button className="btn btn-secondary" onClick={() => setAddOpen(true)}>
              Add model from URL…
            </button>
          </div>
        )}
      </PrefGroup>
    </>
  );
}
