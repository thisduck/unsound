import { ReactNode } from "react";
import { formatShortcut, LANGUAGES } from "./api";
import { CorrectableText } from "./Correctable";
import { Wave } from "./Wave";
import { Icon, Kbd, MoreMenu, Spinner, groupByDay, timeOfDay } from "./ui";

export interface Take {
  id: string;
  at: string;
  raw: string;
  refined: string;
  sttModel: string;
  llmModel: string;
  app?: string;
  /// Spoken-language code this dictation was recorded in ("" = auto).
  lang?: string;
}

type Phase = "idle" | "recording" | "transcribing" | "cleaning";

interface Props {
  history: Take[];
  currentId: string | null;
  onSelect: (take: Take) => void;
  onDelete: (id: string) => void;
  phase: Phase;
  pending: number;
  liveText: string;
  waveBars: number[];
  transcript: string;
  cleaned: string;
  rawOpen: boolean;
  onRawOpen: (open: boolean) => void;
  status: string;
  error: string | null;
  onDismissError: () => void;
  busy: boolean;
  manualBusy: boolean;
  modelsReady: boolean;
  hasLlm: boolean;
  hasStt: boolean;
  shortcut: string | null;
  language: string;
  onLanguage: (code: string) => void;
  /// How the speech model handles the spoken language, when it can't simply.
  languageNote: { tone: "info" | "warn"; text: string } | null;
  styleSwitcher: ReactNode;
  outLangSwitcher: ReactNode;
  onToggleRecord: () => void;
  onPickFile: () => void;
  onCleanUp: () => void;
  onRetranscribe: () => void;
  onCopy: (text: string, what: string) => void;
  onCorrect: (source: "raw" | "refined") => (from: string, to: string) => void;
  onOpenModels: () => void;
}

/// Dictations: everything you've said, newest first, with the selected one
/// open on the right. Dictating itself mostly happens from the shortcut.
export function Dictations(p: Props) {
  const recording = p.phase === "recording";
  const current = p.history.find((t) => t.id === p.currentId);
  const groups = groupByDay(p.history, (t) => t.at);
  const recordLabel = recording ? "Stop" : p.pending > 0 ? "Dictate again" : "Dictate";
  const hasText = !!(p.cleaned || p.transcript);

  const detail = () => {
    if (!p.modelsReady && !recording && !p.transcript) {
      return (
        <div className="empty-state">
          <Icon name="download" />
          <h3>Dictation needs two downloads</h3>
          <p>A speech model turns your voice into text, and a cleanup model tidies it. Everything runs on this Mac.</p>
          <button className="btn btn-primary" onClick={p.onOpenModels}>Download models</button>
        </div>
      );
    }
    if (recording) {
      return (
        <div className="dict-body">
          <div className="dict-meta">
            <span className="rec-dot" /> Listening
          </div>
          <div className="dict-text live">
            {p.liveText || <span className="faint">Your words appear here as you speak.</span>}
            <span className="caret" />
          </div>
        </div>
      );
    }
    if (!hasText && p.phase === "idle") {
      return p.history.length === 0 ? (
        <div className="empty-state">
          <Icon name="mic" />
          <h3>No dictations yet</h3>
          <p>
            Press {p.shortcut ? <Kbd>{formatShortcut(p.shortcut)}</Kbd> : "your shortcut"} in any app, speak, then press it again.
            The cleaned-up text is typed where your cursor is.
          </p>
        </div>
      ) : (
        <div className="empty-state">
          <p>Choose a dictation to read, copy or clean up again.</p>
        </div>
      );
    }
    const working = p.phase === "transcribing" || p.phase === "cleaning";
    return (
      <div className="dict-body">
        <div className="dict-meta">
          {working && <Spinner />}
          {current ? (
            <>
              <span>{timeOfDay(current.at)}</span>
              {current.app && <span>· Typed into {current.app}</span>}
            </>
          ) : (
            <span>{p.phase === "transcribing" ? "Transcribing…" : p.phase === "cleaning" ? "Cleaning up…" : "New dictation"}</span>
          )}
        </div>
        <div className={"dict-text" + (p.cleaned || p.phase === "cleaning" ? "" : " original")}>
          {p.phase === "cleaning" ? (
            <>
              {p.cleaned}
              <span className="caret" />
            </>
          ) : p.cleaned ? (
            <CorrectableText text={p.cleaned} onCorrect={p.onCorrect("refined")} />
          ) : (
            <CorrectableText text={p.transcript} onCorrect={p.onCorrect("raw")} />
          )}
        </div>
        {!working && (
          <div className="dict-tools">
            {p.styleSwitcher}
            {p.outLangSwitcher}
            <span className="spacer" />
            {p.hasLlm && p.transcript && (
              <button className="btn btn-quiet btn-sm" disabled={p.busy} onClick={p.onCleanUp}>
                {p.cleaned ? "Clean up again" : "Clean up"}
              </button>
            )}
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => (p.cleaned ? p.onCopy(p.cleaned, "Text") : p.onCopy(p.transcript, "Original"))}
            >
              <Icon name="copy" />
              Copy
            </button>
          </div>
        )}
        {p.transcript && p.cleaned && (
          <div className="dict-original">
            <button className="disclosure" aria-expanded={p.rawOpen} onClick={() => p.onRawOpen(!p.rawOpen)}>
              <Icon name={p.rawOpen ? "chevronDown" : "chevronRight"} />
              Original
            </button>
            {p.rawOpen && (
              <div className="dict-original-body">
                <div className="dict-original-text">
                  <CorrectableText text={p.transcript} onCorrect={p.onCorrect("raw")} />
                </div>
                <div className="dict-tools">
                  <button className="btn btn-quiet btn-sm" onClick={() => p.onCopy(p.transcript, "Original")}>
                    Copy original
                  </button>
                  {p.hasStt && (
                    <button className="btn btn-quiet btn-sm" disabled={p.busy} onClick={p.onRetranscribe}>
                      Transcribe again
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="page dictations">
      <div className="page-head">
        <h1 className="page-title">Dictations</h1>
        <span className="spacer" />
        <select
          className="select select-sm"
          value={p.language}
          onChange={(e) => p.onLanguage(e.target.value)}
          disabled={p.busy}
          aria-label="Spoken language"
          title="Spoken language"
        >
          {LANGUAGES.map((l) => (
            <option key={l.code} value={l.code}>
              {l.code === "" ? "Speaking: Auto-detect" : `Speaking: ${l.name}`}
            </option>
          ))}
        </select>
        <button
          className="btn btn-quiet"
          onClick={p.onPickFile}
          disabled={p.busy || recording}
          title="Transcribe an audio file (or drop one on the window)"
        >
          <Icon name="upload" />
          Transcribe a file
        </button>
        <button
          className={"btn " + (recording ? "btn-secondary dict-stop" : "btn-secondary")}
          onClick={p.onToggleRecord}
          disabled={p.manualBusy}
        >
          {recording ? <Wave bars={p.waveBars.slice(-8)} className="dict-wave" /> : <Icon name="mic" />}
          {recordLabel}
        </button>
      </div>
      {p.shortcut && (
        <p className="page-hint">
          Press <Kbd>{formatShortcut(p.shortcut)}</Kbd> in any app to dictate. The text is typed where your cursor is.
        </p>
      )}
      {p.languageNote && (
        <div className={"alert " + p.languageNote.tone} role="status">
          <Icon name={p.languageNote.tone === "warn" ? "alert" : "info"} />
          <span className="alert-body">{p.languageNote.text}</span>
        </div>
      )}

      <div className="card dict-split">
        <div className="dict-list" role="list" aria-label="Dictations">
          {p.history.length === 0 && <p className="dict-list-empty">Your dictations show up here.</p>}
          {groups.map((g) => (
            <div key={g.day} role="presentation">
              <div className="list-day">{g.day}</div>
              {g.items.map((t) => (
                <div
                  key={t.id}
                  role="listitem"
                  className={"dict-item" + (t.id === p.currentId ? " selected" : "")}
                >
                  <button className="dict-item-main" onClick={() => p.onSelect(t)} disabled={p.busy}>
                    <span className="dict-item-meta">
                      <span className="mono">{timeOfDay(t.at)}</span>
                      {t.app && <span>{t.app}</span>}
                    </span>
                    <span className="dict-item-text">{t.refined || t.raw || "Empty dictation"}</span>
                  </button>
                  <span className="dict-item-more">
                    <MoreMenu
                      label="Dictation actions"
                      items={[
                        ...(t.refined ? [{ label: "Copy text", icon: "copy" as const, onSelect: () => p.onCopy(t.refined, "Text") }] : []),
                        ...(t.raw ? [{ label: "Copy original", icon: "copy" as const, onSelect: () => p.onCopy(t.raw, "Original") }] : []),
                        { label: "Delete", icon: "trash", danger: true, separatorBefore: true, onSelect: () => p.onDelete(t.id) },
                      ]}
                    />
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
        <div className="dict-detail">{detail()}</div>
      </div>

      {(p.error || p.status) && (
        <div className={"status-line" + (p.error ? " error" : "")} role={p.error ? "alert" : "status"}>
          {p.error ? <Icon name="alert" /> : null}
          <span>{p.error ?? p.status}</span>
          {p.error && (
            <button className="btn btn-quiet btn-sm" onClick={p.onDismissError}>
              Dismiss
            </button>
          )}
        </div>
      )}
    </div>
  );
}
