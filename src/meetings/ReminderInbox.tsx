import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { ActionItem, api } from "../api";
import { ActionRow } from "./ActionRow";

export function ReminderInbox({ onSource }: { onSource: (id: string, at?: number | null) => void }) {
  const [items, setItems] = useState<ActionItem[]>([]);
  const [expanded, setExpanded] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    const load = () => api.dueActionReminders().then(rows => { if (active) { setItems(rows); setError(""); } }).catch(e => { if (active) setError(String(e)); });
    void load(); const timer = setInterval(load, 15000); const sub = listen("action-items-changed", load);
    return () => { active = false; clearInterval(timer); void sub.then(un => un()); };
  }, []);
  if (!items.length && !error) return null;
  return <aside className="reminder-inbox" aria-label="Due reminders">
    <button className="quiet accent" onClick={() => setExpanded(!expanded)}>{expanded ? "▾" : "▸"} {items.length} due reminder{items.length === 1 ? "" : "s"}</button>
    {error && <p className="meet-error" role="alert">Could not check reminders: {error}</p>}
    {expanded && items.map(item => <ActionRow key={item.id} item={item} onSource={item.meetingId ? () => onSource(item.meetingId!, item.atMs) : undefined} onSave={async next => { await api.saveActionItem(next); setItems(await api.dueActionReminders()); }} />)}
  </aside>;
}
