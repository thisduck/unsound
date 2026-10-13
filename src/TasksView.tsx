import { useEffect, useState } from "react";
import { api, on, Task } from "./api";
import { mergeTask, TaskCard } from "./meetings/TaskCard";

/// Everything you've asked unsound to do — in meetings or with the command
/// key any time — grouped by what needs you.
export function TasksView({ onCount }: { onCount?: (ready: number) => void }) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [ask, setAsk] = useState("");
  const [status, setStatus] = useState<{ ok: boolean; model?: string; error?: string } | null>(null);

  useEffect(() => {
    api.listTasks().then(setTasks).catch(() => {});
    api.assistantStatus().then(setStatus).catch(() => {});
    const subs = [
      on.taskUpdated((t) => setTasks((list) => mergeTask(list, t))),
      on.settingsChanged(() => api.assistantStatus().then(setStatus).catch(() => {})),
    ];
    return () => {
      subs.forEach((s) => s.then((un) => un()));
    };
  }, []);

  const ready = tasks.filter((t) => t.status === "ready");
  useEffect(() => onCount?.(ready.length), [ready.length, onCount]);

  const submit = async () => {
    const text = ask.trim();
    if (!text) return;
    setAsk("");
    const t = await api.addTask(text).catch(() => null);
    if (t) setTasks((list) => mergeTask(list, t));
  };

  const byNewest = (a: Task, b: Task) => b.createdAt.localeCompare(a.createdAt);
  const groups: [string, Task[]][] = [
    ["needs your ok", ready],
    ["in progress", tasks.filter((t) => ["heard", "planning", "drafting", "running", "waiting"].includes(t.status))],
    ["didn't work", tasks.filter((t) => t.status === "failed")],
    ["done", tasks.filter((t) => t.status === "done")],
  ];
  const update = (t: Task) => setTasks((list) => mergeTask(list, t));

  return (
    <div className="tasks-view">
      <div className="meet-ask tasks-ask">
        <input
          className="meet-ask-input"
          value={ask}
          onChange={(e) => setAsk(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
          }}
          placeholder="ask unsound to do something — draft an email, remind me, create a ticket…"
        />
        <button className="quiet accent" onClick={submit} disabled={!ask.trim()}>
          ask
        </button>
      </div>
      {status && (
        <p className={"tasks-status" + (status.ok ? "" : " meet-error")}>
          {status.ok ? `thinking with ${status.model}` : status.error}
        </p>
      )}
      {tasks.length === 0 && (
        <div className="empty">
          <p>
            requests land here. In a meeting, say “unsound, create a Jira ticket for that” or hold your
            command key and speak — any time, even with the window hidden. unsound drafts it from the
            conversation; nothing is sent until you say so.
          </p>
        </div>
      )}
      {groups.map(([label, list]) =>
        list.length === 0 ? null : (
          <section className="meet-section" key={label}>
            <h3>{label}</h3>
            {[...list].sort(byNewest).map((t) => (
              <TaskCard key={t.id} task={t} compact={t.status !== "ready"} onChange={update} />
            ))}
          </section>
        ),
      )}
    </div>
  );
}
