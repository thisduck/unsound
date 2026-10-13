import { useEffect, useRef, useState } from "react";
import { api, clock, Task, TaskAction } from "../api";

export const ACTION_LABELS: Record<TaskAction, string> = {
  email: "email",
  jira_issue: "Jira ticket",
  reminder: "reminder",
  calendar_event: "calendar event",
  slack_message: "Slack message",
  note: "note",
  text: "text",
  browser: "website",
  delegate: "hand-off",
};

/// What the primary button says for each kind of draft.
const RUN_LABELS: Record<TaskAction, string> = {
  email: "Open in Mail",
  jira_issue: "Create in Jira",
  reminder: "Add to Reminders",
  calendar_event: "Add to Calendar",
  slack_message: "Copy for Slack",
  note: "Add to notes",
  text: "Copy",
  browser: "Do it in my browser",
  delegate: "Run it",
};

const STATUS_LABELS: Record<string, string> = {
  heard: "heard",
  planning: "understanding…",
  drafting: "drafting…",
  waiting: "after the meeting",
  ready: "ready for you",
  running: "working…",
  done: "done",
  failed: "failed",
  dismissed: "dismissed",
};

type Field = { key: string; label: string; kind?: "text" | "area" | "list" | "select"; options?: string[] };

const FIELDS: Record<TaskAction, Field[]> = {
  email: [
    { key: "to", label: "to", kind: "list" },
    { key: "cc", label: "cc", kind: "list" },
    { key: "subject", label: "subject" },
    { key: "body", label: "body", kind: "area" },
  ],
  jira_issue: [
    { key: "summary", label: "summary" },
    { key: "issue_type", label: "type", kind: "select", options: ["Task", "Bug", "Story", "Epic"] },
    { key: "priority", label: "priority", kind: "select", options: ["", "Highest", "High", "Medium", "Low"] },
    { key: "assignee", label: "assignee" },
    { key: "project", label: "project" },
    { key: "labels", label: "labels", kind: "list" },
    { key: "description", label: "description", kind: "area" },
  ],
  reminder: [
    { key: "title", label: "remind me to" },
    { key: "due", label: "when" },
    { key: "notes", label: "notes", kind: "area" },
  ],
  calendar_event: [
    { key: "title", label: "title" },
    { key: "start", label: "starts" },
    { key: "end", label: "ends" },
    { key: "attendees", label: "invite", kind: "list" },
    { key: "location", label: "where" },
    { key: "notes", label: "notes", kind: "area" },
  ],
  slack_message: [
    { key: "channel", label: "to" },
    { key: "text", label: "message", kind: "area" },
  ],
  note: [{ key: "text", label: "note", kind: "area" }],
  browser: [
    { key: "site", label: "start at" },
    { key: "steps", label: "what to do", kind: "area" },
  ],
  text: [
    { key: "title", label: "title" },
    { key: "body", label: "text", kind: "area" },
  ],
  delegate: [
    { key: "runner", label: "using", kind: "select", options: ["claude_code", "mcp"] },
    { key: "prompt", label: "instructions", kind: "area" },
  ],
};

/// A plain-text version of a draft, for the clipboard.
function draftText(action: TaskAction, d: Record<string, unknown>): string {
  const s = (k: string) => String(d?.[k] ?? "");
  const l = (k: string) => (Array.isArray(d?.[k]) ? (d[k] as string[]).join(", ") : s(k));
  switch (action) {
    case "email":
      return `To: ${l("to")}${l("cc") ? `\nCc: ${l("cc")}` : ""}\nSubject: ${s("subject")}\n\n${s("body")}`;
    case "jira_issue":
      return `${s("summary")}\n\n${s("description")}`;
    case "reminder":
      return `${s("title")}${s("due") ? ` (${s("due")})` : ""}${s("notes") ? `\n${s("notes")}` : ""}`;
    case "calendar_event":
      return `${s("title")}\n${s("start")} – ${s("end")}${l("attendees") ? `\nWith: ${l("attendees")}` : ""}\n${s("notes")}`;
    case "slack_message":
      return s("text");
    case "note":
      return s("text");
    case "delegate":
      return s("prompt");
    case "browser":
      return `${s("site")}\n\n${s("steps")}`.trim();
    default:
      return `${s("title")}\n\n${s("body")}`.trim();
  }
}

interface Props {
  task: Task;
  compact?: boolean;
  onTime?: (ms: number) => void;
  onChange?: (t: Task) => void;
}

/// One request to the assistant: what was asked, where it is in the pipeline,
/// and — once drafted — an editable draft with a button that actually does it.
export function TaskCard({ task, compact, onTime, onChange }: Props) {
  const [draft, setDraft] = useState<Record<string, unknown>>(task.draft ?? {});
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  // Open when it needs the user; finished ones start folded.
  const [open, setOpen] = useState(
    task.status === "ready" || task.status === "failed" || (!compact && !["done", "dismissed"].includes(task.status)),
  );
  const dirty = useRef(false);

  // Take the backend's draft unless the user is mid-edit.
  useEffect(() => {
    if (!dirty.current) setDraft(task.draft ?? {});
  }, [task.draft]);
  useEffect(() => {
    if (task.status === "ready") setOpen(true);
  }, [task.status]);

  const action = (task.action || "text") as TaskAction;
  const title = task.log?.[0]?.title || task.instruction || task.utterance;
  const tool: string | null = task.log?.[0]?.tool ?? null;
  const browserRun = (task.log ?? []).some((l) => l.step === "browser");
  const working = ["heard", "planning", "drafting", "running"].includes(task.status);

  const set = (key: string, value: unknown) => {
    dirty.current = true;
    setDraft((d) => ({ ...d, [key]: value }));
  };
  const persist = async () => {
    if (!dirty.current) return;
    dirty.current = false;
    try {
      onChange?.(await api.updateTaskDraft(task.id, draft));
    } catch (e) {
      console.error("save draft failed", e);
    }
  };
  const act = async (fn: () => Promise<Task | void>) => {
    setBusy(true);
    try {
      const t = await fn();
      if (t) onChange?.(t);
    } catch (e) {
      console.error(e);
    } finally {
      setBusy(false);
    }
  };
  const copy = () => {
    navigator.clipboard.writeText(draftText(action, draft));
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  const openUrl = async (url: string) => {
    const { openUrl } = await import("@tauri-apps/plugin-opener");
    openUrl(url).catch(() => {});
  };

  return (
    <div className={`task task-${task.status}` + (compact ? " compact" : "")}>
      <div className="task-head" onClick={() => setOpen((o) => !o)}>
        <span className="task-kind">{tool ?? (task.action ? ACTION_LABELS[action] : "request")}</span>
        <span className="task-title">{title}</span>
        {browserRun && <span className="task-chip">in your browser</span>}
        <span className={`task-status s-${task.status}`}>
          {working && <span className="task-spin" />}
          {STATUS_LABELS[task.status] ?? task.status}
        </span>
      </div>
      {open && (
        <div className="task-body">
          <div className="task-said">
            “{task.utterance}”
            {task.atMs != null && onTime && (
              <button className="time-link" onClick={() => onTime(task.atMs!)}>
                {clock(task.atMs)}
              </button>
            )}
            {task.spanStartMs != null && task.spanEndMs != null && onTime && (
              <span className="task-span">
                · about{" "}
                <button className="time-link" onClick={() => onTime(task.spanStartMs!)}>
                  {clock(task.spanStartMs)}
                </button>
                –{clock(task.spanEndMs)}
              </span>
            )}
          </div>

          {task.status === "waiting" && (
            <p className="dim task-note">I'll draft this with the whole meeting as context when it ends.</p>
          )}
          {task.error && <p className="task-error">{task.error}</p>}

          {(task.status === "ready" || task.status === "failed" || task.status === "done") &&
            task.draft && (
              <div className="task-fields">
                {FIELDS[action].map((f) => {
                  const v = draft?.[f.key];
                  const value = Array.isArray(v) ? v.join(", ") : String(v ?? "");
                  const readOnly = task.status === "done";
                  return (
                    <label key={f.key} className={"task-field" + (f.kind === "area" ? " area" : "")}>
                      <span>{f.label}</span>
                      {f.kind === "area" ? (
                        <textarea
                          value={value}
                          readOnly={readOnly}
                          rows={Math.min(14, Math.max(3, value.split("\n").length + 1))}
                          onChange={(e) => set(f.key, e.target.value)}
                          onBlur={persist}
                        />
                      ) : f.kind === "select" ? (
                        <select
                          value={value}
                          disabled={readOnly}
                          onChange={(e) => {
                            set(f.key, e.target.value);
                          }}
                          onBlur={persist}
                        >
                          {[...new Set([value, ...(f.options ?? [])])].map((o) => (
                            <option key={o} value={o}>
                              {o === "claude_code" ? "Claude Code" : o === "mcp" ? "MCP tools" : o || "—"}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <input
                          value={value}
                          readOnly={readOnly}
                          onChange={(e) =>
                            set(
                              f.key,
                              f.kind === "list"
                                ? e.target.value.split(",").map((x) => x.trimStart())
                                : e.target.value,
                            )
                          }
                          onBlur={() => {
                            if (f.kind === "list")
                              set(
                                f.key,
                                ((draft?.[f.key] as string[]) ?? []).map((x) => x.trim()).filter(Boolean),
                              );
                            persist();
                          }}
                        />
                      )}
                    </label>
                  );
                })}
              </div>
            )}

          {task.status === "done" && task.result?.message && (
            <p className="task-result">
              ✓ {task.result.message}
              {task.result.url && (
                <button className="quiet accent" onClick={() => openUrl(task.result!.url!)}>
                  open
                </button>
              )}
            </p>
          )}

          {Array.isArray(task.log) && task.log.some((l) => l.tool) && (
            <details className="task-log">
              <summary>steps</summary>
              {task.log
                .filter((l) => l.tool)
                .map((l, i) => (
                  <div key={i} className={l.ok ? "" : "task-error"}>
                    {l.step}. {l.tool} {l.ok ? "✓" : "✕"} <span className="dim">{l.result}</span>
                  </div>
                ))}
            </details>
          )}

          <div className="task-actions">
            {task.status === "ready" && (
              <button
                className="quiet accent"
                disabled={busy}
                onClick={() => act(async () => {
                  dirty.current = false;
                  return api.approveTask(task.id, draft);
                })}
              >
                {RUN_LABELS[action]}
              </button>
            )}
            {task.draft && action !== "note" && (
              <button className="quiet" onClick={copy}>
                {copied ? "copied ✓" : "copy"}
              </button>
            )}
            {(task.status === "ready" || task.status === "done") && (
              <button className="quiet" disabled={busy} onClick={() => act(() => api.redraftTask(task.id))}>
                redraft
              </button>
            )}
            {(task.status === "failed" || task.status === "dismissed") && (
              <button className="quiet" disabled={busy} onClick={() => act(() => api.retryTask(task.id))}>
                retry
              </button>
            )}
            {task.status !== "done" && task.status !== "dismissed" && !working && (
              <button className="quiet" disabled={busy} onClick={() => act(() => api.dismissTask(task.id))}>
                dismiss
              </button>
            )}
            {(task.status === "dismissed" || task.status === "done" || task.status === "failed") && (
              <button
                className="quiet danger"
                disabled={busy}
                onClick={() =>
                  act(async () => {
                    await api.deleteTask(task.id);
                    onChange?.({ ...task, status: "dismissed", id: `deleted:${task.id}` });
                  })
                }
              >
                delete
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/// Keep a task list current from `task-updated` events.
export function mergeTask(list: Task[], t: Task): Task[] {
  if (t.id.startsWith("deleted:")) {
    const id = t.id.slice("deleted:".length);
    return list.filter((x) => x.id !== id);
  }
  const i = list.findIndex((x) => x.id === t.id);
  if (i === -1) return [...list, t];
  const next = list.slice();
  next[i] = t;
  return next;
}
