import { useState } from "react";
import { isPermissionGranted, requestPermission } from "@tauri-apps/plugin-notification";
import { ActionItem } from "../api";
import { Icon, MoreMenu } from "../ui";

function localInput(ms?: number | null) {
  if (!ms) return "";
  const d = new Date(ms);
  return new Date(ms - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
}

function whenLabel(ms: number): string {
  const d = new Date(ms);
  const sameDay = d.toDateString() === new Date().toDateString();
  return sameDay
    ? d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
    : d.toLocaleString([], { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

/// One action item: check it off, or edit, remind, snooze, ignore.
export function ActionRow({
  item,
  onSave,
  onSource,
  sourceLabel,
}: {
  item: ActionItem;
  onSave: (item: ActionItem) => Promise<unknown>;
  onSource?: () => void;
  /// Text for the source link; defaults to the meeting title.
  sourceLabel?: string;
}) {
  const [editing, setEditing] = useState<null | "edit" | "remind">(null);
  const [draft, setDraft] = useState(item);
  const [when, setWhen] = useState(localInput(item.remindAt));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const closed = item.done || item.ignored;
  const due = !!item.remindAt && !closed && item.remindAt <= Date.now();

  const save = async (next: ActionItem) => {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      // Permission is requested only in response to the user's reminder action.
      if (next.remindAt && next.remindAt !== item.remindAt) {
        try {
          if (!(await isPermissionGranted()) && (await requestPermission()) !== "granted")
            setNotice("Reminder saved. System notifications are off, so it shows here only.");
        } catch {
          setNotice("Reminder saved. System notifications aren't available.");
        }
      }
      await onSave(next);
      setEditing(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const open = (mode: "edit" | "remind") => {
    setDraft(item);
    setWhen(localInput(item.remindAt));
    setEditing(editing === mode ? null : mode);
  };

  return (
    <div className={"action-row follow-up" + (item.done ? " is-done" : "") + (item.ignored ? " is-ignored" : "")}>
      <div className="action-main">
        <button
          className={"check-box" + (item.done ? " on" : "")}
          role="checkbox"
          aria-checked={item.done}
          aria-label={item.done ? "Mark as not done" : "Mark done"}
          disabled={busy}
          onClick={() => void save(item.done ? { ...item, done: false, ignored: false } : { ...item, done: true, ignored: false, remindAt: null })}
        >
          {item.done && <Icon name="check" stroke={3} />}
        </button>
        <div className="action-text">
          <span className="action-title">{item.text}</span>
          <span className="action-meta">
            {item.owner && <span>{item.owner}</span>}
            {item.due && <span>{item.due}</span>}
            {item.remindAt && !closed && (
              <span className={due ? "due" : ""}>
                <Icon name="bell" />
                {due ? "Due " : ""}
                {whenLabel(item.remindAt)}
              </span>
            )}
            {item.ignored && <span>Ignored</span>}
            {onSource && (
              <button className="action-source" onClick={onSource}>
                {sourceLabel ?? item.meetingTitle ?? "Source"}
              </button>
            )}
          </span>
        </div>
        <div className="action-controls">
          {due && (
            <button className="btn btn-quiet btn-sm" disabled={busy} onClick={() => void save({ ...item, remindAt: Date.now() + 3600000 })}>
              Snooze 1 hour
            </button>
          )}
          {!closed && (
            <button className="btn btn-quiet btn-sm action-remind" disabled={busy} onClick={() => open("remind")} aria-label="Remind me">
              <Icon name="bell" />
            </button>
          )}
          <MoreMenu
            label="Action item options"
            items={
              closed
                ? [{ label: "Reopen", icon: "refresh", onSelect: () => void save({ ...item, done: false, ignored: false }) }]
                : [
                    { label: "Edit", icon: "pencil", onSelect: () => open("edit") },
                    { label: "Remind me…", icon: "bell", onSelect: () => open("remind") },
                    { label: "Ignore", icon: "x", separatorBefore: true, onSelect: () => void save({ ...item, ignored: true, done: false, remindAt: null }) },
                  ]
            }
          />
        </div>
      </div>

      {editing && (
        <form
          className="action-editor"
          onSubmit={(e) => {
            e.preventDefault();
            const remindAt = when ? new Date(when).getTime() : null;
            if (remindAt != null && (!Number.isFinite(remindAt) || remindAt <= Date.now())) {
              setError("Choose a time in the future.");
              return;
            }
            void save({ ...(editing === "edit" ? draft : item), remindAt });
          }}
        >
          {editing === "edit" && (
            <>
              <label>
                <span>Action</span>
                <input className="input" required value={draft.text} onChange={(e) => setDraft({ ...draft, text: e.target.value })} />
              </label>
              <label>
                <span>Owner</span>
                <input className="input" value={draft.owner} onChange={(e) => setDraft({ ...draft, owner: e.target.value })} />
              </label>
              <label>
                <span>Deadline</span>
                <input className="input" value={draft.due} placeholder="Friday, after the review…" onChange={(e) => setDraft({ ...draft, due: e.target.value })} />
              </label>
            </>
          )}
          <label>
            <span>Remind me</span>
            <input className="input" type="datetime-local" value={when} onChange={(e) => setWhen(e.target.value)} />
          </label>
          <p className="action-editor-note">Reminders need unsound running. The menu bar is enough.</p>
          <div className="action-editor-buttons">
            <button className="btn btn-primary btn-sm" disabled={busy}>Save</button>
            {when && <button className="btn btn-quiet btn-sm" type="button" onClick={() => setWhen("")}>Clear reminder</button>}
            <button className="btn btn-quiet btn-sm" type="button" onClick={() => setEditing(null)}>Cancel</button>
          </div>
        </form>
      )}
      {(error || item.reminderError) && <p className="action-error" role="alert">{error || item.reminderError}</p>}
      {notice && <p className="action-notice" role="status">{notice}</p>}
    </div>
  );
}
