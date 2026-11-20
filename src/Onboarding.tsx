import { useMemo, useState } from "react";
import { formatBytes, ModelInfo } from "./api";
import { PermissionsStep } from "./Permissions";
import { ModelLibrary, ModelRow, ShortcutsSection, useDownloads } from "./sections";
import { Icon, Logo } from "./ui";

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
  const missing = recommended.filter((m) => !m.downloaded);
  const missingSize = missing.reduce((n, m) => n + m.sizeBytes, 0);
  const totalSize = recommended.reduce((n, m) => n + m.sizeBytes, 0);
  const sttReady = models.some((m) => m.kind === "stt" && m.downloaded);
  const llmReady = models.some((m) => m.kind === "llm" && m.downloaded);
  const ready = sttReady && llmReady;
  const downloading = recommended.some((m) => progress[m.id]);

  const downloadRecommended = () => {
    setErr(null);
    missing.forEach((m) => start(m.id));
  };

  const error = err && (
    <div className="alert danger" role="alert">
      <Icon name="alert" />
      <div className="alert-body">{err}</div>
    </div>
  );

  const steps = [
    /* 1 · welcome */
    <div className="ob-step ob-welcome" key="welcome">
      <Logo size={56} />
      <h1 className="ob-title">Welcome to unsound</h1>
      <p className="ob-lede">Dictate into any app and take notes on your calls. Everything runs on this Mac.</p>
      <p className="ob-text">Setup takes about two minutes.</p>
      <div className="ob-actions">
        <button className="btn btn-primary ob-primary" onClick={() => setStep(1)} autoFocus>
          Get started
        </button>
      </div>
    </div>,

    /* 2 · models */
    <div className="ob-step" key="models">
      <h1 className="ob-title">Download models</h1>
      <p className="ob-text">One-time downloads. Nothing you say leaves this Mac.</p>
      <div className="card pref-card ob-models">
        {recommended.map((m) => (
          <ModelRow key={m.id} model={m} progress={progress[m.id]} onDownload={start} />
        ))}
        <div className="ob-total">
          <span>Total</span>
          <span className="tabular">{formatBytes(totalSize)}</span>
        </div>
      </div>
      <p className="ob-note">
        Meeting notes also need speaker detection and voice activity detection. You can download those later from
        the Meetings screen.
      </p>
      <div className="ob-actions">
        {ready && !downloading ? (
          <button className="btn btn-primary ob-primary" onClick={() => setStep(2)}>
            Continue
          </button>
        ) : (
          <button
            className="btn btn-primary ob-primary"
            onClick={downloadRecommended}
            disabled={downloading || missing.length === 0}
          >
            {downloading ? "Downloading…" : `Download ${formatBytes(missingSize)}`}
          </button>
        )}
        <button className="btn btn-quiet" onClick={() => setBrowse((b) => !b)} aria-expanded={browse}>
          {browse ? "Hide model list" : "Choose models myself"}
        </button>
      </div>
      {browse && (
        <div className="ob-library">
          <ModelLibrary models={models} onChanged={onChanged} onError={setErr} compact />
        </div>
      )}
      {error}
    </div>,

    /* 3 · permissions */
    <PermissionsStep key="permissions" onNext={() => setStep(3)} onError={setErr}>
      {error}
    </PermissionsStep>,

    /* 4 · shortcut */
    <div className="ob-step" key="shortcut">
      <h1 className="ob-title">Dictate anywhere</h1>
      <p className="ob-text">
        Press your shortcut in any app, speak, then press it again. The cleaned-up text is typed at your cursor.
      </p>
      <ShortcutsSection onError={setErr} />
      <p className="ob-note">For calls, unsound offers to record when Zoom, Meet or Teams starts.</p>
      <div className="ob-actions">
        <button className="btn btn-primary ob-primary" onClick={onDone}>
          Start using unsound
        </button>
      </div>
      {error}
    </div>,
  ];

  return (
    <div className="ob-overlay">
      <div className="ob-frame" role="dialog" aria-modal="true" aria-label="Welcome to unsound">
        <div className="ob-body">{steps[step]}</div>
        <div className="ob-foot">
          <div className="ob-dots" role="group" aria-label="Setup steps">
            {steps.map((_, i) => (
              <button
                key={i}
                className="ob-dot"
                aria-current={i === step ? "step" : undefined}
                onClick={() => {
                  setErr(null);
                  setStep(i);
                }}
                aria-label={`Step ${i + 1} of ${steps.length}`}
              />
            ))}
          </div>
          {step < steps.length - 1 && (
            <button className="btn btn-quiet" onClick={onDone}>
              Skip for now
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
