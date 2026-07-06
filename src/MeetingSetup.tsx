import { useState } from "react";
import { api, on, formatBytes, ModelInfo } from "./api";

function ModelRow({ model, onDone }: { model: ModelInfo; onDone: () => void }) {
  const [pct, setPct] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const download = async () => {
    setErr(null);
    setPct(0);
    const unsub = await on.downloadProgress((p) => {
      if (p.id === model.id && p.total) setPct(p.downloaded / p.total);
    });
    try {
      await api.downloadModel(model.id);
      onDone();
    } catch (e) {
      setErr(String(e));
    } finally {
      unsub();
      setPct(null);
    }
  };

  return (
    <div className="setup-row">
      <div className="setup-row-main">
        <div className="setup-row-name">{model.name}</div>
        <div className="setup-row-desc">
          {model.description} · {formatBytes(model.sizeBytes)}
        </div>
        {err && <div className="setup-row-err">{err}</div>}
      </div>
      {model.downloaded ? (
        <span className="setup-ok">✓ ready</span>
      ) : pct !== null ? (
        <span className="setup-pct">{Math.round(pct * 100)}%</span>
      ) : (
        <button className="quiet accent" onClick={download}>
          download
        </button>
      )}
    </div>
  );
}

interface Props {
  models: ModelInfo[];
  onModelsChanged: () => void;
}

export function MeetingSetup({ models, onModelsChanged }: Props) {
  const recStt =
    models.find((m) => m.kind === "stt" && m.recommended) ??
    models.find((m) => m.kind === "stt");
  const recLlm =
    models.find((m) => m.kind === "llm" && m.recommended) ??
    models.find((m) => m.kind === "llm");
  const sttModel = models.find((m) => m.kind === "stt" && m.downloaded) ?? recStt;
  const llmModel = models.find((m) => m.kind === "llm" && m.downloaded) ?? recLlm;
  const seg = models.find((m) => m.id === "diarize-segmentation");
  const embs = models.filter((m) => m.kind === "diarize" && m.id !== "diarize-segmentation");

  return (
    <div className="meet-setup">
      <h2>set up meetings</h2>
      <p className="setup-intro">
        Meetings transcribe both you and the call, tell the other speakers apart, and write a
        summary — entirely on your machine. These are one-time downloads.
      </p>

      <div className="setup-group-label">Required</div>
      {sttModel && <ModelRow model={sttModel} onDone={onModelsChanged} />}
      {seg && <ModelRow model={seg} onDone={onModelsChanged} />}

      <div className="setup-group-label">
        Speaker detection — get at least one (you can switch in meeting options)
      </div>
      {embs.map((m) => (
        <ModelRow key={m.id} model={m} onDone={onModelsChanged} />
      ))}

      <div className="setup-group-label">Recommended — for summaries &amp; Q&amp;A</div>
      {llmModel && <ModelRow model={llmModel} onDone={onModelsChanged} />}
    </div>
  );
}
