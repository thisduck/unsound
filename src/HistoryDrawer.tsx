export interface Take {
  id: string;
  at: string;
  raw: string;
  refined: string;
  sttModel: string;
  llmModel: string;
  app?: string;
}

interface Props {
  takes: Take[];
  onClose: () => void;
  onLoad: (take: Take) => void;
  onDelete: (id: string) => void;
  onClear: () => void;
  onCopy: (text: string, what: string) => void;
}

function timeLabel(iso: string): string {
  const d = new Date(iso);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (sameDay) return time;
  return `${d.toLocaleDateString([], { month: "short", day: "numeric" })} · ${time}`;
}

export function HistoryDrawer({ takes, onClose, onLoad, onDelete, onClear, onCopy }: Props) {
  return (
    <div className="sheet-overlay" onClick={onClose}>
      <div className="sheet" onClick={(e) => e.stopPropagation()}>
        <header className="sheet-head">
          <h2>history</h2>
          <span className="sheet-note">
            {takes.length === 0 ? "no takes yet" : `${takes.length} take${takes.length === 1 ? "" : "s"} · stored only on this machine`}
          </span>
          {takes.length > 0 && (
            <button className="quiet" onClick={onClear}>
              clear all
            </button>
          )}
          <button className="quiet" onClick={onClose}>
            close ✕
          </button>
        </header>

        <div className="sheet-body">
          {takes.length === 0 && (
            <div className="empty" style={{ padding: "40px 0" }}>
              <p>finished recordings land here — raw and refined, ready to copy.</p>
            </div>
          )}
          {takes.map((t) => (
            <div className="take" key={t.id}>
              <div className="take-head">
                <span className="take-time">{timeLabel(t.at)}</span>
                <span className="take-models">
                  {[t.sttModel, t.llmModel].filter(Boolean).join(" → ")}
                  {t.app ? ` · typed into ${t.app}` : ""}
                </span>
              </div>
              <div className="take-text" onClick={() => onLoad(t)} title="Open this take">
                {t.refined || t.raw || <span className="placeholder">(empty)</span>}
              </div>
              <div className="take-actions">
                {t.refined && (
                  <button className="quiet" onClick={() => onCopy(t.refined, "refined text")}>
                    copy refined
                  </button>
                )}
                {t.raw && (
                  <button className="quiet" onClick={() => onCopy(t.raw, "transcript")}>
                    copy raw
                  </button>
                )}
                <span className="spacer" />
                <button className="quiet danger" onClick={() => onDelete(t.id)}>
                  delete
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
