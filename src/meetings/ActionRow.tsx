import { useState } from "react";
import { isPermissionGranted, requestPermission } from "@tauri-apps/plugin-notification";
import { ActionItem } from "../api";

function localInput(ms?: number | null) {
  if (!ms) return "";
  const d = new Date(ms);
  return new Date(ms - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
}
export function ActionRow({ item, onSave, onSource }: { item: ActionItem; onSave: (item: ActionItem) => Promise<unknown>; onSource?: () => void }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(item);
  const [when, setWhen] = useState(localInput(item.remindAt));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const closed = item.done || item.ignored;
  const save = async (next: ActionItem) => {
    setBusy(true); setError(""); setNotice("");
    try {
      // Permission is requested only in response to the user's reminder action.
      if (next.remindAt && next.remindAt !== item.remindAt) {
        try {
          if (!(await isPermissionGranted()) && await requestPermission() !== "granted") setNotice("Reminder saved in unsound. System notifications are disabled.");
        } catch { setNotice("Reminder saved in unsound. System notifications are unavailable."); }
      }
      await onSave(next); setEditing(false);
    } catch (e) { setError(String(e)); }
    finally { setBusy(false); }
  };
  return <div className={`follow-up ${item.done ? "is-done" : ""} ${item.ignored ? "is-ignored" : ""}`}>
    <div className="item">
      <span className="item-text">{item.text}</span>
      {item.owner && <span className="item-chip">{item.owner}</span>}
      {item.due && <span className="item-chip due">{item.due}</span>}
      {onSource && <button className="quiet item-src" onClick={onSource}>{item.meetingTitle || "source"}</button>}
    </div>
    {item.remindAt && !closed && <p className="meet-hint">{item.remindAt <= Date.now() ? "Reminder due: " : "Remind me: "}{new Date(item.remindAt).toLocaleString()}</p>}
    <div className="follow-up-controls">
      {closed ? <><span className="dim">{item.done ? "Done" : "Ignored"}</span><button className="quiet" disabled={busy} onClick={() => void save({ ...item, done: false, ignored: false })}>reopen</button></> : <>
        <button className="quiet" disabled={busy} onClick={() => void save({ ...item, done: true, ignored: false, remindAt: null })}>mark done</button>
        <button className="quiet" disabled={busy} onClick={() => void save({ ...item, ignored: true, done: false, remindAt: null })}>ignore</button>
        <button className="quiet" disabled={busy} onClick={() => { setDraft(item); setWhen(localInput(item.remindAt)); setEditing(!editing); }}>edit / remind me</button>
        {item.remindAt && item.remindAt <= Date.now() && <button className="quiet" disabled={busy} onClick={() => void save({ ...item, remindAt: Date.now() + 3600000 })}>snooze 1 hour</button>}
      </>}
    </div>
    {editing && <form className="follow-up-editor" onSubmit={e => { e.preventDefault(); const remindAt = when ? new Date(when).getTime() : null; if (remindAt != null && (!Number.isFinite(remindAt) || remindAt <= Date.now())) { setError("Choose a future reminder time."); return; } void save({ ...draft, remindAt }); }}>
      <label>Action<input required value={draft.text} onChange={e => setDraft({ ...draft, text: e.target.value })} /></label>
      <label>Owner<input value={draft.owner} onChange={e => setDraft({ ...draft, owner: e.target.value })} /></label>
      <label>Deadline<input value={draft.due} placeholder="e.g. Friday, after the review" onChange={e => setDraft({ ...draft, due: e.target.value })} /></label>
      <label>Reminder (your local time)<input type="datetime-local" value={when} onChange={e => setWhen(e.target.value)} /></label>
      <p className="meet-hint">Reminders run while unsound is open, including in the menu bar. If you quit, overdue reminders appear next time you open it.</p>
      <div><button className="quiet accent" disabled={busy}>save</button><button className="quiet" type="button" onClick={() => setWhen("")}>clear reminder</button><button className="quiet" type="button" onClick={() => setEditing(false)}>cancel</button></div>
    </form>}
    {(error || item.reminderError) && <p className="meet-error" role="alert">{error || item.reminderError}</p>}
    {notice && <p className="meet-hint" role="status">{notice}</p>}
  </div>;
}
