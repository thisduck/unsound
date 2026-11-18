import { useState } from "react";
import { formatBytes, ModelInfo } from "./api";
import { ModelRow, useDownloads } from "./sections";
import { Icon } from "./ui";

interface Props {
  models: ModelInfo[];
  onModelsChanged: () => void;
}

export function MeetingSetup({ models, onModelsChanged }: Props) {
  const [err, setErr] = useState<string | null>(null);
  // A finished download changes what's installed.
  const { progress, start } = useDownloads(setErr, () => onModelsChanged());

  const recStt =
    models.find((m) => m.kind === "stt" && m.recommended) ?? models.find((m) => m.kind === "stt");
  const recLlm =
    models.find((m) => m.kind === "llm" && m.recommended) ?? models.find((m) => m.kind === "llm");
  const sttModel = models.find((m) => m.kind === "stt" && m.downloaded) ?? recStt;
  const llmModel = models.find((m) => m.kind === "llm" && m.downloaded) ?? recLlm;
  const vad = models.find((m) => m.id === "vad-silero");
  // CAM++ ("diarize-embedding") is fast and the most accurate on real calls.
  const embs = models
    .filter((m) => m.kind === "diarize" && m.id !== "diarize-segmentation")
    .sort((a, b) => Number(b.id === "diarize-embedding") - Number(a.id === "diarize-embedding"));
  const installedEmb = embs.find((m) => m.downloaded);
  const primaryEmb = installedEmb ?? embs[0];
  const otherEmbs = embs.filter((m) => m.id !== primaryEmb?.id);
  const assistant =
    models.find((m) => m.id === "gemma-4-12b" && m.downloaded) ??
    models.find((m) => m.id === "gemma-4-26b-a4b" && m.downloaded) ??
    models.find((m) => m.id === "gemma-4-12b");

  const required = [sttModel, vad, primaryEmb].filter(Boolean) as ModelInfo[];
  const missing = required.filter((m) => !m.downloaded && !progress[m.id]);
  const missingSize = missing.reduce((n, m) => n + m.sizeBytes, 0);
  const anyRequiredDownloading = required.some((m) => progress[m.id]);

  const row = (m: ModelInfo) => <ModelRow key={m.id} model={m} progress={progress[m.id]} onDownload={start} />;

  return (
    <section className="card meet-setup" aria-labelledby="meet-setup-title">
      <header className="meet-setup-head">
        <div>
          <h2 id="meet-setup-title" className="meet-setup-title">
            Set up meeting notes
          </h2>
          <p className="meet-setup-desc">
            One-time downloads. Recording a call needs a speech model, voice activity detection and speaker
            detection.
          </p>
        </div>
        {missing.length > 0 && (
          <button className="btn btn-primary" onClick={() => missing.forEach((m) => start(m.id))}>
            <Icon name="download" />
            Download all required ({formatBytes(missingSize)})
          </button>
        )}
        {missing.length === 0 && anyRequiredDownloading && (
          <span className="meet-setup-status" role="status">
            Downloading…
          </span>
        )}
      </header>

      {err && (
        <div className="alert danger" role="alert">
          <Icon name="alert" />
          <div className="alert-body">{err}</div>
        </div>
      )}

      <h3 className="section-label">Required</h3>
      <div className="meet-setup-rows">
        {sttModel && row(sttModel)}
        {vad && row(vad)}
        {primaryEmb && row(primaryEmb)}
      </div>
      {otherEmbs.length > 0 && (
        <details className="pref-disclosure">
          <summary>
            <Icon name="chevronRight" />
            Other speaker models
          </summary>
          <div className="meet-setup-rows">{otherEmbs.map(row)}</div>
        </details>
      )}

      <h3 className="section-label">Recommended</h3>
      <p className="meet-setup-hint">Writes the summary and action items, and handles requests.</p>
      <div className="meet-setup-rows">
        {llmModel && row(llmModel)}
        {assistant && assistant.id !== llmModel?.id && row(assistant)}
      </div>
    </section>
  );
}
