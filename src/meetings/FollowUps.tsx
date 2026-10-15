import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { ActionItem, api } from "../api";
import { ActionRow } from "./ActionRow";

export function FollowUps({ onSource, refreshKey }: { onSource: (id: string, at?: number | null) => void; refreshKey: unknown }) {
  const [items, setItems] = useState<ActionItem[]>([]);
  const [filter, setFilter] = useState("open");
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState(false);
  useEffect(() => {
    let active = true;
    const load = () => api.listActionItems().then(rows => { if (active) { setItems(rows); setError(""); } }).catch(e => { if (active) setError(String(e)); });
    void load(); const timer = setInterval(load, 30000);
    const sub = listen("action-items-changed", load);
    return () => { active = false; clearInterval(timer); void sub.then(un => un()); };
  }, [refreshKey]);
  const open = items.filter(i => !i.done && !i.ignored);
  const visible = items.filter(i => filter === "all" || (filter === "done" ? i.done : filter === "ignored" ? i.ignored : !i.done && !i.ignored));
  return <section className="open-items">
    <button className="quiet open-items-toggle" onClick={() => setExpanded(!expanded)}>{expanded ? "▾" : "▸"} follow-ups · {open.length} open</button>
    {error && <p className="meet-error" role="alert">{error}</p>}
    {expanded && <><div className="follow-up-filters" aria-label="Filter follow-ups">{["open", "done", "ignored", "all"].map(f => <button className={"quiet" + (filter === f ? " accent" : "")} aria-pressed={filter === f} key={f} onClick={() => setFilter(f)}>{f}</button>)}</div>
      {!visible.length && <p className="dim">No {filter === "all" ? "" : filter} follow-ups.</p>}
      {visible.map(item => <ActionRow key={item.id} item={item} onSource={item.meetingId ? () => onSource(item.meetingId!, item.atMs) : undefined} onSave={async next => { const saved = await api.saveActionItem(next); setItems(rows => rows.map(i => i.id === saved.id ? saved : i)); }} />)}
    </>}
  </section>;
}
