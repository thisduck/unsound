import { useMemo, useState } from "react";
import { formatBytes, ModelInfo } from "./api";
import { ModelLibrary, ModelRow, PermissionsSection, ShortcutsSection, useDownloads } from "./sections";

interface Props {
  models: ModelInfo[];
  onChanged: () => void;
  onDone: () => void;
}

export function Onboarding({ models, onChanged, onDone }: Props) {
  const [step, setStep] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [browse, setBrowse] = useState(false);
  const { progress, start } = useDownloads(setErr);

  const recommended = useMemo(() => models.filter((m) => m.recommended), [models]);
  const recommendedSize = recommended.reduce((n, m) => n + m.sizeBytes, 0);
  const sttReady = models.some((m) => m.kind === "stt" && m.downloaded);
  const llmReady = models.some((m) => m.kind === "llm" && m.downloaded);
  const ready = sttReady && llmReady;
  const downloading = recommended.some((m) => progress[m.id]);

  const downloadRecommended = () => {
    setErr(null);
    recommended.filter((m) => !m.downloaded).forEach((m) => start(m.id));
  };

  const steps = [
    /* ── 0 · private by design ── */
    <div className="ob-step" key="welcome">
      <div className="ob-mark">unsound</div>
      <h1 className="ob-title">your voice, into clean text.<br />entirely on this Mac.</h1>
      <p className="ob-text">
        Speak — unsound writes it down, then quietly tidies it up: punctuation, paragraphs, the
        "um"s and false starts gone.
      </p>
      <p className="ob-text">
        Everything happens on your machine. No cloud, no account, no internet — nothing you say
        ever leaves this computer. The one exception: downloading the models that do the work,
        which happens once, next.
      </p>
      <div className="ob-actions">
        <button className="ob-primary" onClick={() => setStep(1)}>
          set up →
        </button>
      </div>
    </div>,

    /* ── 1 · models ── */
    <div className="ob-step" key="models">
      <h1 className="ob-title">two small models do the work</h1>
      <p className="ob-text">
        A <b>voice model</b> (Whisper) turns speech into words — most understand ~99 languages,
        and there are English-only versions that are a touch sharper if that's all you speak. A{" "}
        <b>text model</b> cleans the transcript — language coverage varies by model, so if you
        dictate in other languages, pick one that speaks them (Qwen covers ~29).
      </p>
      {!ready && (
        <div className="ob-recommended">
          {recommended.map((m) => (
            <ModelRow key={m.id} model={m} progress={progress[m.id]} onDownload={start} />
          ))}
        </div>
      )}
      <div className="ob-actions">
        {ready ? (
          <button className="ob-primary" onClick={() => setStep(2)}>
            models ready →
          </button>
        ) : (
          <button className="ob-primary" onClick={downloadRecommended} disabled={downloading}>
            {downloading ? "downloading…" : `download recommended (${formatBytes(recommendedSize)})`}
          </button>
        )}
        {!ready && (
          <button className="quiet" onClick={() => setBrowse((b) => !b)}>
            {browse ? "hide the full list" : "or choose your own"}
          </button>
        )}
      </div>
      {browse && !ready && (
        <div className="ob-library">
          <ModelLibrary models={models} onChanged={onChanged} onError={setErr} compact />
        </div>
      )}
      {err && <div className="sheet-error">{err}</div>}
    </div>,

    /* ── 2 · shortcut & permissions ── */
    <div className="ob-step" key="shortcut">
      <h1 className="ob-title">dictate into any app</h1>
      <p className="ob-text">
        Two ways to talk, in any app: <b>hands-free</b> — press the shortcut to start, again to
        stop — or <b>push to talk</b> — hold it, speak, let go. Either way the cleaned-up text is
        typed right where your cursor is. macOS needs your okay for two things below.
      </p>
      <ShortcutsSection onError={setErr} />
      <PermissionsSection onError={setErr} />
      <div className="ob-actions">
        <button className="ob-primary" onClick={onDone}>
          start using unsound →
        </button>
      </div>
      {err && <div className="sheet-error">{err}</div>}
    </div>,
  ];

  return (
    <div className="ob-overlay">
      <div className="ob-frame">
        {steps[step]}
        <div className="ob-foot">
          <div className="ob-dots">
            {steps.map((_, i) => (
              <button
                key={i}
                className={"ob-dot" + (i === step ? " on" : "")}
                onClick={() => setStep(i)}
                aria-label={`step ${i + 1}`}
              />
            ))}
          </div>
          <button className="quiet" onClick={onDone}>
            skip — set up later in settings
          </button>
        </div>
      </div>
    </div>
  );
}
