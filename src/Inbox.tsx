import { useCallback, useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { ActionItem, api, on, Task } from "./api";
import { ActionRow } from "./meetings/ActionRow";
import { ACTION_LABELS, mergeTask, TaskCard } from "./meetings/TaskCard";
import { Icon, IconName, Segmented, Spinner } from "./ui";

const WORKING = ["heard", "planning", "drafting", "running", "waiting"];

export interface InboxData {
  tasks: Task[];
  items: ActionItem[];
  due: ActionItem[];
  error: string;
  updateTask: (t: Task) => void;
  saveItem: (item: ActionItem) => Promise<void>;
  /// Requests ready to send plus reminders that are due: the badge count.
  attention: number;
}

/// Everything that can need you: requests unsound drafted, action items
/// from meetings, and reminders that are due. Loaded once for the app.
export function useInbox(): InboxData {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [items, setItems] = useState<ActionItem[]>([]);
  const [due, setDue] = useState<ActionItem[]>([]);
  const [error, setError] = useState("");

  const loadItems = useCallback(async () => {
    try {
      const [all, dueNow] = await Promise.all([api.listActionItems(), api.dueActionReminders()]);
      setItems(all);
      setDue(dueNow);
      setError("");
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    api.listTasks().then(setTasks).catch(() => {});
    void loadItems();
    const timer = setInterval(loadItems, 15000);
    const subs = [on.taskUpdated((t) => setTasks((list) => mergeTask(list, t))), listen("action-items-changed", () => void loadItems())];
    return () => {
      clearInterval(timer);
      subs.forEach((s) => s.then((un) => un()));
    };
  }, [loadItems]);

  const updateTask = useCallback((t: Task) => setTasks((list) => mergeTask(list, t)), []);
  const saveItem = useCallback(
    async (item: ActionItem) => {
      await api.saveActionItem(item);
      await loadItems();
    },
    [loadItems],
  );
  const ready = tasks.filter((t) => t.status === "ready").length;
  return { tasks, items, due, error, updateTask, saveItem, attention: ready + due.length };
}

export function taskIcon(t: Task): IconName {
  switch (t.action) {
    case "email":
      return "mail";
    case "jira_issue":
      return "ticket";
    case "reminder":
      return "bell";
    case "calendar_event":
      return "calendar";
    case "slack_message":
      return "message";
    case "browser":
      return "globe";
    case "note":
      return "note";
    default:
      return "sparkle";
  }
}

export function taskTitle(t: Task): string {
  return t.log?.[0]?.title || t.instruction || t.utterance;
}

/// The short "needs you" strip at the top of Meetings: ready requests and
/// due reminders, each with its one next step.
export function NeedsYou({ data, onOpenInbox }: { data: InboxData; onOpenInbox: () => void }) {
  const ready = data.tasks.filter((t) => t.status === "ready");
  const rows = ready.length + data.due.length;
  if (!rows) return null;
  return (
    <section aria-label="Needs you">
      <div className="section-label">
        Needs you <span className="count">{rows}</span>
        <span className="spacer" />
        <button className="btn btn-quiet btn-sm" onClick={onOpenInbox}>
          Open Inbox
        </button>
      </div>
      <div className="card rows">
        {ready.slice(0, 3).map((t) => (
          <div className="list-row" key={t.id}>
            <span className="row-icon accent">
              <Icon name={taskIcon(t)} />
            </span>
            <span className="row-text">
              <span className="row-title">{taskTitle(t)}</span>
              <span className="row-sub">{(t.action ? ACTION_LABELS[t.action] : "Request")} draft is ready</span>
            </span>
            <span className="row-end">
              <button className="btn btn-secondary btn-sm" onClick={onOpenInbox}>
                Review
              </button>
            </span>
          </div>
        ))}
        {data.due.slice(0, 3).map((item) => (
          <div className="list-row" key={item.id}>
            <span className="row-icon">
              <Icon name="clock" />
            </span>
            <span className="row-text">
              <span className="row-title">{item.text}</span>
              <span className="row-sub">Reminder due{item.meetingTitle ? ` · ${item.meetingTitle}` : ""}</span>
            </span>
            <span className="row-end">
              <button className="btn btn-quiet btn-sm" onClick={() => void data.saveItem({ ...item, remindAt: Date.now() + 3600000 })}>
                Snooze 1 hour
              </button>
              <button className="btn btn-secondary btn-sm" onClick={() => void data.saveItem({ ...item, done: true, ignored: false, remindAt: null })}>
                Mark done
              </button>
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

type ItemFilter = "open" | "done" | "ignored" | "all";

/// Inbox: requests to send, reminders that are due, and action items from
/// meetings, with a composer to ask unsound for something new.
export function Inbox({ data, onSource }: { data: InboxData; onSource: (meetingId: string, atMs?: number | null) => void }) {
  const [ask, setAsk] = useState("");
  const [filter, setFilter] = useState<ItemFilter>("open");
  const [showDone, setShowDone] = useState(false);
  const [assistant, setAssistant] = useState<{ ok: boolean; error?: string } | null>(null);

  useEffect(() => {
    api.assistantStatus().then(setAssistant).catch(() => {});
    const sub = on.settingsChanged(() => api.assistantStatus().then(setAssistant).catch(() => {}));
    return () => {
      sub.then((un) => un());
    };
  }, []);

  const submit = async () => {
    const text = ask.trim();
    if (!text) return;
    setAsk("");
    const t = await api.addTask(text).catch(() => null);
    if (t) data.updateTask(t);
  };

  const byNewest = (a: Task, b: Task) => b.createdAt.localeCompare(a.createdAt);
  const requests = data.tasks
    .filter((t) => t.status === "ready" || t.status === "failed" || WORKING.includes(t.status))
    .sort((a, b) => Number(b.status === "ready") - Number(a.status === "ready") || byNewest(a, b));
  const finished = data.tasks.filter((t) => t.status === "done" || t.status === "dismissed").sort(byNewest);
  const dueIds = new Set(data.due.map((d) => d.id));
  const items = data.items.filter(
    (i) =>
      !dueIds.has(i.id) &&
      (filter === "all" || (filter === "done" ? i.done : filter === "ignored" ? i.ignored : !i.done && !i.ignored)),
  );
  const openCount = data.items.filter((i) => !i.done && !i.ignored).length;
  const empty = !requests.length && !data.due.length && !data.items.length && !finished.length;

  return (
    <div className="page inbox">
      <div className="page-head">
        <h1 className="page-title">Inbox</h1>
      </div>

      <form
        className="composer card"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <Icon name="sparkle" className="composer-icon" />
        <input
          className="composer-input"
          aria-label="Ask unsound to do something"
          value={ask}
          onChange={(e) => setAsk(e.target.value)}
          placeholder="Ask unsound to… draft an email, set a reminder, file a ticket"
        />
        <button className="btn btn-primary btn-sm" disabled={!ask.trim()}>
          Ask
        </button>
      </form>
      {assistant && !assistant.ok && assistant.error && (
        <div className="alert warn inbox-alert" role="status">
          <Icon name="alert" />
          <span className="alert-body">{assistant.error}</span>
        </div>
      )}
      {data.error && (
        <div className="alert danger inbox-alert" role="alert">
          <Icon name="alert" />
          <span className="alert-body">Couldn't load action items: {data.error}</span>
        </div>
      )}

      {empty && (
        <div className="empty-state">
          <Icon name="inbox" />
          <h3>You're all caught up</h3>
          <p>
            Action items from your meetings and anything you ask unsound to do show up here. Nothing is sent until you
            approve it.
          </p>
        </div>
      )}

      {requests.length > 0 && (
        <section aria-label="Requests">
          <div className="section-label">
            Requests <span className="count">{requests.length}</span>
          </div>
          <div className="stack">
            {requests.map((t) => (
              <TaskCard key={t.id} task={t} compact={t.status !== "ready"} onChange={data.updateTask} />
            ))}
          </div>
        </section>
      )}

      {data.due.length > 0 && (
        <section aria-label="Due">
          <div className="section-label">
            Due <span className="count">{data.due.length}</span>
          </div>
          <div className="card item-list">
            {data.due.map((item) => (
              <ActionRow
                key={item.id}
                item={item}
                onSave={data.saveItem}
                onSource={item.meetingId ? () => onSource(item.meetingId!, item.atMs) : undefined}
              />
            ))}
          </div>
        </section>
      )}

      {data.items.length > 0 && (
        <section aria-label="Action items">
          <div className="section-label">
            Action items <span className="count">{openCount} open</span>
            <span className="spacer" />
            <Segmented
              label="Filter action items"
              value={filter}
              onChange={setFilter}
              options={[
                { value: "open", label: "Open" },
                { value: "done", label: "Done" },
                { value: "ignored", label: "Ignored" },
                { value: "all", label: "All" },
              ]}
            />
          </div>
          <div className="card item-list">
            {items.length === 0 ? (
              <p className="item-list-empty">No {filter === "all" ? "" : filter} action items.</p>
            ) : (
              items.map((item) => (
                <ActionRow
                  key={item.id}
                  item={item}
                  onSave={data.saveItem}
                  onSource={item.meetingId ? () => onSource(item.meetingId!, item.atMs) : undefined}
                />
              ))
            )}
          </div>
        </section>
      )}

      {finished.length > 0 && (
        <section aria-label="Done and dismissed">
          <button className="disclosure section-disclosure" aria-expanded={showDone} onClick={() => setShowDone(!showDone)}>
            <Icon name={showDone ? "chevronDown" : "chevronRight"} />
            Done and dismissed <span className="count">{finished.length}</span>
          </button>
          {showDone && (
            <div className="stack">
              {finished.map((t) => (
                <TaskCard key={t.id} task={t} compact onChange={data.updateTask} />
              ))}
            </div>
          )}
        </section>
      )}
      {requests.some((t) => WORKING.includes(t.status)) && (
        <p className="page-foot">
          <Spinner /> unsound is drafting. You can keep working; drafts wait for your approval.
        </p>
      )}
    </div>
  );
}
