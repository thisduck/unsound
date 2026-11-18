import { useEffect, useState } from "react";
import {
  api,
  AssistantSettings,
  CalAccess,
  formatShortcut,
  LANGUAGES,
  McpServer,
  MeetingSettings,
  ModelInfo,
  TaskAction,
  ToolGuide,
} from "./api";
import { ACTION_LABELS } from "./meetings/TaskCard";
import { PrefCard, PrefCheck, PrefGroup, PrefRow } from "./sections";
import { Icon, useConfirm } from "./ui";

const JIRA_TOKEN = "jira-api-token";
const ENDPOINT_KEY = "assistant-endpoint-key";
const OPENAI_KEY = "openai-api-key";
const ANTHROPIC_KEY = "anthropic-api-key";

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/// OpenAI-compatible providers, by base URL.
const PROVIDERS: { name: string; url: string; local?: boolean }[] = [
  { name: "Fireworks", url: "https://api.fireworks.ai/inference/v1" },
  { name: "Groq", url: "https://api.groq.com/openai/v1" },
  { name: "Together", url: "https://api.together.xyz/v1" },
  { name: "OpenRouter", url: "https://openrouter.ai/api/v1" },
  { name: "DeepSeek", url: "https://api.deepseek.com/v1" },
  { name: "Mistral", url: "https://api.mistral.ai/v1" },
  { name: "Ollama (this Mac)", url: "http://localhost:11434/v1", local: true },
  { name: "LM Studio (this Mac)", url: "http://localhost:1234/v1", local: true },
  { name: "llama-server (this Mac)", url: "http://localhost:8080/v1", local: true },
];

type Status = { ok: boolean; model?: string; error?: string; cloud?: boolean } | null;

/* ── which model thinks ──────────────────────────────────────────── */

function ModelSection({
  s,
  llms,
  recommended,
  status,
  patch,
  onError,
}: {
  s: AssistantSettings;
  llms: ModelInfo[];
  recommended: ModelInfo[];
  status: Status;
  patch: (p: Partial<AssistantSettings>) => void;
  onError: (msg: string | null) => void;
}) {
  const backend = s.backend === "endpoint" ? "compatible" : s.backend || "local";
  const [clis, setClis] = useState<{ claude: string | null; codex: string | null }>({ claude: null, codex: null });
  const [hasKey, setHasKey] = useState<Record<string, boolean>>({});
  const [keyInput, setKeyInput] = useState("");
  const [cloudModels, setCloudModels] = useState<string[] | null>(null);
  const [listing, setListing] = useState<string | null>(null);

  useEffect(() => {
    api.subscriptionClis().then(setClis).catch(() => {});
    for (const k of [OPENAI_KEY, ANTHROPIC_KEY, ENDPOINT_KEY]) {
      api.hasSecret(k).then((v) => setHasKey((h) => ({ ...h, [k]: v }))).catch(() => {});
    }
  }, []);
  useEffect(() => {
    setCloudModels(null);
    setListing(null);
    setKeyInput("");
  }, [backend]);

  const keyName = backend === "openai" ? OPENAI_KEY : backend === "anthropic" ? ANTHROPIC_KEY : ENDPOINT_KEY;
  const saveKey = async () => {
    if (!keyInput.trim()) return;
    try {
      await api.setSecret(keyName, keyInput.trim());
      setHasKey((h) => ({ ...h, [keyName]: true }));
      setKeyInput("");
      patch({}); // refresh status
    } catch (e) {
      onError(String(e));
    }
  };
  const fetchModels = async () => {
    setListing("Loading…");
    try {
      const ids = await api.listCloudModels(backend, backend === "compatible" ? s.endpointUrl : undefined);
      setCloudModels(ids);
      setListing(ids.length ? null : "The provider didn't list any models.");
    } catch (e) {
      setListing(String(e));
    }
  };
  const modelValue = backend === "compatible" ? s.endpointModel : s.cloudModel;
  const setModel = (m: string) => patch(backend === "compatible" ? { endpointModel: m } : { cloudModel: m });

  return (
    <PrefGroup title="Model">
      {status && (
        <div className={"alert " + (status.ok ? "" : "warn")}>
          <Icon name={status.ok ? "check" : "alert"} />
          <div className="alert-body">{status.ok ? `Using ${status.model}` : status.error}</div>
        </div>
      )}
      <PrefCard>
        <PrefRow label="Runs on" desc="Understands your requests, writes drafts and works in the browser.">
          <select
            className="select"
            aria-label="Runs on"
            value={backend}
            onChange={(e) => patch({ backend: e.target.value })}
          >
            <option value="local">This Mac (downloaded model)</option>
            <option value="claude" disabled={!clis.claude}>
              Your Claude subscription{clis.claude ? "" : " (needs Claude Code)"}
            </option>
            <option value="codex" disabled={!clis.codex}>
              Your ChatGPT subscription{clis.codex ? "" : " (needs Codex CLI)"}
            </option>
            <option value="anthropic">Anthropic API key</option>
            <option value="openai">OpenAI API key</option>
            <option value="compatible">Another provider (Fireworks, Groq, Ollama…)</option>
          </select>
        </PrefRow>

        {backend === "local" && (
          <PrefRow
            label="Local model"
            desc={
              recommended.some((m) => !m.downloaded)
                ? `Bigger models write better drafts. Try ${recommended
                    .slice(0, 3)
                    .map((m) => m.name)
                    .join(", ")} from Models.`
                : "Bigger models write better drafts."
            }
          >
            <select
              className="select"
              aria-label="Local model"
              value={s.localModel}
              onChange={(e) => patch({ localModel: e.target.value })}
            >
              <option value="">Automatic (best downloaded)</option>
              {llms.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </PrefRow>
        )}

        {(backend === "claude" || backend === "codex") && (
          <PrefRow
            label="Model"
            desc={
              backend === "claude"
                ? "Uses the account Claude Code is signed in to. Each step takes a few seconds longer than with an API key."
                : "Uses the account the Codex CLI is signed in to. Each step takes a few seconds longer than with an API key."
            }
          >
            {backend === "claude" ? (
              <select
                className="select"
                aria-label="Claude model"
                value={s.cliModel}
                onChange={(e) => patch({ cliModel: e.target.value })}
              >
                <option value="">Default</option>
                <option value="sonnet">Sonnet</option>
                <option value="opus">Opus</option>
                <option value="haiku">Haiku (fastest)</option>
              </select>
            ) : (
              <input
                className="input"
                aria-label="Codex model"
                defaultValue={s.cliModel}
                placeholder="Default"
                onBlur={(e) => patch({ cliModel: e.target.value.trim() })}
              />
            )}
          </PrefRow>
        )}

        {backend === "compatible" && (
          <PrefRow label="Provider" desc="Any OpenAI-compatible server." className="stack">
            <select
              className="select"
              aria-label="Provider"
              value={PROVIDERS.find((p) => p.url === s.endpointUrl)?.url ?? ""}
              onChange={(e) => e.target.value && patch({ endpointUrl: e.target.value })}
            >
              <option value="">Custom URL</option>
              {PROVIDERS.map((p) => (
                <option key={p.url} value={p.url}>
                  {p.name}
                </option>
              ))}
            </select>
            <input
              className="input wide"
              aria-label="Provider URL"
              key={s.endpointUrl}
              defaultValue={s.endpointUrl}
              placeholder="https://…/v1"
              onBlur={(e) => e.target.value.trim() !== s.endpointUrl && patch({ endpointUrl: e.target.value.trim() })}
            />
          </PrefRow>
        )}

        {(backend === "anthropic" || backend === "openai" || backend === "compatible") && (
          <>
            <PrefRow
              label="API key"
              desc={
                hasKey[keyName]
                  ? "Saved in your Keychain."
                  : backend === "compatible"
                    ? "Not needed for servers on this Mac."
                    : backend === "anthropic"
                      ? "Create one at platform.claude.com, under API keys."
                      : "Create one at platform.openai.com, under API keys."
              }
            >
              <input
                className="input wide"
                type="password"
                aria-label="API key"
                value={keyInput}
                placeholder={hasKey[keyName] ? "••••••••" : "Paste key"}
                onChange={(e) => setKeyInput(e.target.value)}
                onBlur={saveKey}
                onKeyDown={(e) => e.key === "Enter" && saveKey()}
              />
            </PrefRow>
            <PrefRow
              label="Model"
              desc={
                listing ??
                (backend === "anthropic" ? "Default: claude-opus-5-5" : "Pick from the provider's list, or type an ID.")
              }
            >
              {cloudModels && cloudModels.length > 0 ? (
                <select
                  className="select"
                  aria-label="Model"
                  value={modelValue}
                  onChange={(e) => setModel(e.target.value)}
                >
                  <option value="">Choose…</option>
                  {[...new Set([modelValue, ...cloudModels].filter(Boolean))].map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  className="input"
                  aria-label="Model ID"
                  key={backend + modelValue}
                  defaultValue={modelValue}
                  placeholder={backend === "anthropic" ? "claude-opus-5-5" : "Model ID"}
                  onBlur={(e) => e.target.value.trim() !== modelValue && setModel(e.target.value.trim())}
                />
              )}
              <button className="btn btn-secondary btn-sm" onClick={fetchModels}>
                List models
              </button>
            </PrefRow>
          </>
        )}
      </PrefCard>

      {status?.cloud && (
        <div className="alert warn">
          <Icon name="info" />
          <div className="alert-body">
            Requests, drafts and the relevant part of the transcript are sent to this provider. Transcripts and
            summaries are still made on this Mac.
          </div>
        </div>
      )}
    </PrefGroup>
  );
}

/// Local models that fill structured drafts reliably, best first.
const RECOMMENDED = ["gemma-4-26b-a4b", "qwen3.6-35b-a3b", "gpt-oss-20b", "gemma-4-12b", "gemma-4-e4b"];

/// Starting points for common tools; the user fills in their own rules.
const TEMPLATES: Omit<ToolGuide, "id" | "enabled">[] = [
  {
    name: "Jira",
    when: "creating or updating Jira tickets — bugs, tasks, stories",
    site: "https://YOURTEAM.atlassian.net",
    guidance: "If no project is mentioned, use project ___.\nAssign it to whoever took it on; otherwise leave it unassigned.\nAdd the label from-meeting.",
    mode: "auto",
    leaveFinalStep: false,
  },
  {
    name: "Linear",
    when: "creating or updating Linear issues",
    site: "https://linear.app/YOURTEAM",
    guidance: "Default team: ___. New issues go to Triage unless a cycle is mentioned.",
    mode: "browser",
    leaveFinalStep: false,
  },
  {
    name: "Notion",
    when: "adding notes, docs or database items in Notion",
    site: "https://www.notion.so/",
    guidance: "Meeting follow-ups go in the ___ database, titled with today's date.",
    mode: "browser",
    leaveFinalStep: false,
  },
  {
    name: "GitHub issues",
    when: "filing GitHub issues",
    site: "https://github.com/ORG/REPO/issues",
    guidance: "Use the bug template for bugs. Add the needs-triage label.",
    mode: "browser",
    leaveFinalStep: false,
  },
  {
    name: "HubSpot",
    when: "updating deals, contacts and call notes in HubSpot",
    site: "https://app.hubspot.com",
    guidance: "Log call notes on the deal, not the contact. Never change deal stage unless told to.",
    mode: "browser",
    leaveFinalStep: true,
  },
  {
    name: "Gmail",
    when: "drafting emails (in Gmail instead of the Mail app)",
    site: "https://mail.google.com",
    guidance: "Create a draft reply in the right thread when there is one.",
    mode: "browser",
    leaveFinalStep: true,
  },
];

/* ── assistant ───────────────────────────────────────────────────── */

export function AssistantSection({
  models,
  onError,
  onOpenShortcuts,
}: {
  models: ModelInfo[];
  onError: (msg: string | null) => void;
  onOpenShortcuts: () => void;
}) {
  const [s, setS] = useState<AssistantSettings | null>(null);
  const [commandKey, setCommandKey] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>(null);
  const [hasJiraToken, setHasJiraToken] = useState(false);
  const [jiraToken, setJiraToken] = useState("");
  const [jiraResult, setJiraResult] = useState<string | null>(null);
  const [projects, setProjects] = useState<{ key: string; name: string }[]>([]);
  const [claudePath, setClaudePath] = useState<string | null | undefined>(undefined);
  const [mcpJson, setMcpJson] = useState("");
  const [probe, setProbe] = useState<Record<number, string>>({});
  const [browserTest, setBrowserTest] = useState<string | null>(null);
  const { confirm, dialog } = useConfirm();

  const load = () =>
    api.getSettings().then((st) => {
      setS(st.assistant);
      setCommandKey(st.commandKeys[0] ?? null);
    });

  useEffect(() => {
    load();
    api.assistantStatus().then(setStatus).catch(() => {});
    api.hasSecret(JIRA_TOKEN).then(setHasJiraToken).catch(() => {});
    api.claudeCodePath().then(setClaudePath).catch(() => setClaudePath(null));
  }, []);

  const save = (next: AssistantSettings) => {
    onError(null);
    setS(next);
    api
      .setAssistantSettings(next)
      .then(() => api.assistantStatus().then(setStatus))
      .catch((e) => onError(String(e)));
  };

  if (!s) return null;
  const llms = models.filter((m) => m.kind === "llm" && m.downloaded);
  const recommended = RECOMMENDED.map((id) => models.find((m) => m.id === id)).filter(Boolean) as ModelInfo[];
  const patch = (p: Partial<AssistantSettings>) => save({ ...s, ...p });
  const patchJira = (p: Partial<AssistantSettings["jira"]>) => save({ ...s, jira: { ...s.jira, ...p } });
  const setServer = (i: number, p: Partial<McpServer>) =>
    patch({ mcpServers: s.mcpServers.map((m, j) => (j === i ? { ...m, ...p } : m)) });
  const setGuide = (id: string, p: Partial<ToolGuide>) =>
    patch({ guides: s.guides.map((g) => (g.id === id ? { ...g, ...p } : g)) });
  const addGuide = (t?: Omit<ToolGuide, "id" | "enabled">) =>
    patch({
      guides: [
        ...s.guides,
        {
          id: crypto.randomUUID(),
          enabled: true,
          ...(t ?? { name: "", when: "", site: "", guidance: "", mode: "browser", leaveFinalStep: false }),
        },
      ],
    });
  const removeGuide = (g: ToolGuide) =>
    confirm({
      title: `Remove ${g.name || "this tool"}?`,
      description: "Its rules are deleted. This can't be undone.",
      confirm: "Remove",
      danger: true,
      onConfirm: () => patch({ guides: s.guides.filter((x) => x.id !== g.id) }),
    });
  const removeServer = (i: number) =>
    confirm({
      title: `Remove ${s.mcpServers[i].name || "this server"}?`,
      description: "The assistant can no longer use its tools.",
      confirm: "Remove",
      danger: true,
      onConfirm: () => patch({ mcpServers: s.mcpServers.filter((_, j) => j !== i) }),
    });

  const saveSecret = async (key: string, value: string, done: (v: boolean) => void) => {
    try {
      await api.setSecret(key, value);
      done(!!value.trim());
    } catch (e) {
      onError(String(e));
    }
  };

  return (
    <>
      <PrefGroup
        title="Requests"
        hint={
          <>
            Ask unsound to… “create a Jira ticket from what we just discussed”, “draft an email to Priya about the
            launch”. It drafts from the conversation, and nothing is sent until you approve it.
          </>
        }
      >
        <PrefCard>
          <PrefCheck label="Turn on requests" checked={s.enabled} onChange={(v) => patch({ enabled: v })} />
          <PrefCheck
            label="Listen for its name while recording"
            desc="Only your mic is checked, so others on the call can't trigger it."
            checked={s.listenForWakeWord}
            disabled={!s.enabled}
            onChange={(v) => patch({ listenForWakeWord: v })}
          />
          <PrefRow label="Name to listen for" desc="Say it at the start of a sentence, then the request.">
            <input
              className="input"
              aria-label="Name to listen for"
              defaultValue={s.wakeWord}
              onBlur={(e) => e.target.value.trim() && patch({ wakeWord: e.target.value.trim() })}
            />
          </PrefRow>
          <PrefRow label="Ask unsound shortcut" desc="Hold it and speak a request, any time.">
            <span className="shortcut-keys">{commandKey ? formatShortcut(commandKey) : "Not set"}</span>
            <button className="btn btn-secondary btn-sm" onClick={onOpenShortcuts}>
              {commandKey ? "Change" : "Set shortcut"}
            </button>
          </PrefRow>
        </PrefCard>
      </PrefGroup>

      <ModelSection s={s} llms={llms} recommended={recommended} status={status} patch={patch} onError={onError} />

      <PrefGroup title="About you" hint="Included with every draft: who you are, your team, how you sign off.">
        <textarea
          className="textarea"
          aria-label="About you"
          defaultValue={s.aboutMe}
          placeholder="I'm Sam, design lead at Acme. Sign emails “Sam”. Our sprint ends on Fridays."
          onBlur={(e) => e.target.value !== s.aboutMe && patch({ aboutMe: e.target.value })}
        />
      </PrefGroup>

      <PrefGroup
        title="Your tools"
        hint="Describe the tools you use in your own words. unsound picks one by what it's for, follows your rules, and can do the work in your browser, where you're already signed in."
      >
        {s.guides.map((g) => (
          <div className="card guide-card" key={g.id}>
            <div className="guide-row">
              <input
                type="checkbox"
                aria-label={`Use ${g.name || "this tool"}`}
                checked={g.enabled}
                onChange={(e) => setGuide(g.id, { enabled: e.target.checked })}
              />
              <input
                className="input guide-name"
                aria-label="Tool name"
                defaultValue={g.name}
                placeholder="Name, like Jira"
                onBlur={(e) => setGuide(g.id, { name: e.target.value.trim() })}
              />
              <select
                className="select"
                aria-label="How to do it"
                value={g.mode}
                onChange={(e) => setGuide(g.id, { mode: e.target.value })}
              >
                <option value="browser">Do it in my browser</option>
                <option value="auto">Built-in if available, else my browser</option>
                <option value="draft">Only draft it</option>
              </select>
              <button
                className="btn btn-quiet btn-icon btn-sm danger"
                aria-label={`Remove ${g.name || "tool"}`}
                title="Remove"
                onClick={() => removeGuide(g)}
              >
                <Icon name="trash" />
              </button>
            </div>
            <input
              className="input"
              aria-label="Use it for"
              defaultValue={g.when}
              placeholder="Use it for… like creating or updating Jira tickets"
              onBlur={(e) => setGuide(g.id, { when: e.target.value.trim() })}
            />
            <input
              className="input"
              aria-label="Start at URL"
              defaultValue={g.site}
              placeholder="Start at (URL)"
              onBlur={(e) => setGuide(g.id, { site: e.target.value.trim() })}
            />
            <textarea
              className="textarea"
              aria-label="Your rules"
              defaultValue={g.guidance}
              placeholder="Your rules, like “If no project is mentioned, use ENG. Always add the label from-meeting.”"
              onBlur={(e) => e.target.value !== g.guidance && setGuide(g.id, { guidance: e.target.value })}
            />
            {g.mode !== "draft" && (
              <label className="check">
                <input
                  type="checkbox"
                  checked={g.leaveFinalStep}
                  onChange={(e) => setGuide(g.id, { leaveFinalStep: e.target.checked })}
                />
                Fill it in, but leave the final click (Create, Send, Save) to me
              </label>
            )}
          </div>
        ))}
        <div className="guide-templates">
          <button className="btn btn-secondary btn-sm" onClick={() => addGuide()}>
            Add tool
          </button>
          {TEMPLATES.filter((t) => !s.guides.some((g) => g.name === t.name)).map((t) => (
            <button key={t.name} className="chip" onClick={() => addGuide(t)} aria-label={`Add ${t.name}`}>
              {t.name}
            </button>
          ))}
        </div>
      </PrefGroup>

      <PrefGroup
        title="Browser"
        hint="How website work gets done, in your own Chrome and signed in as you. It only runs after you approve a draft."
      >
        <PrefCard>
          <label className="pref-row pref-radio">
            <input
              type="radio"
              name="assistant-browser"
              checked={s.browser === "chrome"}
              disabled={!claudePath}
              onChange={() => patch({ browser: "chrome" })}
            />
            <span className="pref-text">
              <span className="pref-label">
                Claude in Chrome <span className="badge">Recommended</span>
              </span>
              <span className="pref-desc">
                {claudePath
                  ? "Most reliable. Runs through Claude Code with your claude.ai account. The task and approved content are sent to Claude."
                  : "Needs Claude Code (claude.com/claude-code) and the Claude in Chrome extension."}
              </span>
            </span>
          </label>
          <label className="pref-row pref-radio">
            <input
              type="radio"
              name="assistant-browser"
              checked={s.browser === "local"}
              onChange={() => patch({ browser: "local" })}
            />
            <span className="pref-text">
              <span className="pref-label">Assistant model with a browser connector</span>
              <span className="pref-desc">
                Works with the model above. A local model stays on this Mac but is less reliable on complex sites.
                Default connector: Playwright MCP, with the Playwright MCP Bridge extension.
              </span>
            </span>
          </label>
          <label className="pref-row pref-radio">
            <input
              type="radio"
              name="assistant-browser"
              checked={s.browser === "off" || !s.browser}
              onChange={() => patch({ browser: "off" })}
            />
            <span className="pref-text">
              <span className="pref-label">Off</span>
            </span>
          </label>
          {s.browser === "chrome" && (
            <PrefRow label="Claude model" desc="Sonnet is quicker and cheaper for clicking through sites.">
              <select
                className="select"
                aria-label="Claude model"
                value={s.claudeModel}
                onChange={(e) => patch({ claudeModel: e.target.value })}
              >
                <option value="">Default</option>
                <option value="sonnet">Sonnet</option>
                <option value="opus">Opus</option>
                <option value="haiku">Haiku</option>
              </select>
            </PrefRow>
          )}
          {s.browser === "local" && (
            <PrefRow
              label="Browser connector"
              className="stack"
              desc={
                <>
                  Any MCP browser server, like <code>npx -y @playwright/mcp@latest --extension</code> or{" "}
                  <code>npx -y chrome-devtools-mcp@latest --autoConnect</code>
                </>
              }
            >
              <input
                className="input wide mono"
                aria-label="Browser connector command"
                defaultValue={[s.browserServer.command, ...s.browserServer.args].join(" ")}
                onBlur={(e) => {
                  const [command, ...args] = e.target.value.trim().split(/\s+/);
                  patch({ browserServer: { ...s.browserServer, command: command ?? "", args, url: "" } });
                }}
              />
            </PrefRow>
          )}
          {s.browser !== "off" && s.browser && (
            <PrefRow label="Try it" desc={browserTest ?? "Opens Wikipedia in your browser and reads the page."}>
              <button
                className="btn btn-secondary btn-sm"
                onClick={async () => {
                  setBrowserTest("Opening Wikipedia in your browser…");
                  try {
                    setBrowserTest(`Works. Read: “${(await api.browserTest()).trim()}”`);
                  } catch (e) {
                    setBrowserTest(String(e));
                  }
                }}
              >
                Test browser
              </button>
            </PrefRow>
          )}
        </PrefCard>
      </PrefGroup>

      <PrefGroup
        title="Run without asking"
        hint="Drafts are always made right away. These kinds also run as soon as they're drafted. Browser work always waits for you."
      >
        <PrefCard className="check-grid">
          {(Object.keys(ACTION_LABELS) as TaskAction[]).map((a) => (
            <label key={a} className="check">
              <input
                type="checkbox"
                checked={s.autoRun.includes(a)}
                onChange={(e) =>
                  patch({ autoRun: e.target.checked ? [...s.autoRun, a] : s.autoRun.filter((x) => x !== a) })
                }
              />
              {cap(ACTION_LABELS[a])}
            </label>
          ))}
        </PrefCard>
      </PrefGroup>

      <PrefGroup
        title="Jira API (optional)"
        hint="Only needed to create tickets without the browser. Without it, unsound opens Jira's create page with the ticket on your clipboard."
      >
        <PrefCard>
          <PrefRow label="Site">
            <input
              className="input wide"
              aria-label="Jira site"
              placeholder="https://yourteam.atlassian.net"
              defaultValue={s.jira.site}
              onBlur={(e) => patchJira({ site: e.target.value.trim() })}
            />
          </PrefRow>
          <PrefRow label="Atlassian email">
            <input
              className="input wide"
              type="email"
              aria-label="Atlassian email"
              defaultValue={s.jira.email}
              onBlur={(e) => patchJira({ email: e.target.value.trim() })}
            />
          </PrefRow>
          <PrefRow
            label="API token"
            desc={hasJiraToken ? "Saved in your Keychain." : "Create one at id.atlassian.com, under Security."}
          >
            <input
              className="input wide"
              type="password"
              aria-label="Jira API token"
              value={jiraToken}
              placeholder={hasJiraToken ? "••••••••" : "Paste token"}
              onChange={(e) => setJiraToken(e.target.value)}
              onBlur={() => {
                if (jiraToken) saveSecret(JIRA_TOKEN, jiraToken, setHasJiraToken).then(() => setJiraToken(""));
              }}
            />
          </PrefRow>
          <PrefRow label="Default project">
            {projects.length > 0 ? (
              <select
                className="select"
                aria-label="Default project"
                value={s.jira.defaultProject}
                onChange={(e) => patchJira({ defaultProject: e.target.value })}
              >
                <option value="">None</option>
                {projects.map((p) => (
                  <option key={p.key} value={p.key}>
                    {p.key} · {p.name}
                  </option>
                ))}
              </select>
            ) : (
              <input
                className="input narrow"
                aria-label="Default project key"
                placeholder="ENG"
                defaultValue={s.jira.defaultProject}
                onBlur={(e) => patchJira({ defaultProject: e.target.value.trim().toUpperCase() })}
              />
            )}
            <select
              className="select"
              aria-label="Default issue type"
              value={s.jira.defaultIssueType || "Task"}
              onChange={(e) => patchJira({ defaultIssueType: e.target.value })}
            >
              {["Task", "Bug", "Story"].map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </PrefRow>
          <PrefRow label="Connection" desc={jiraResult ?? "Checks your site, email and token."}>
            <button
              className="btn btn-secondary btn-sm"
              onClick={async () => {
                setJiraResult("Checking…");
                try {
                  const r = await api.jiraTest();
                  setProjects(r.projects);
                  setJiraResult(`Connected as ${r.user} · ${r.projects.length} projects`);
                } catch (e) {
                  setJiraResult(String(e));
                }
              }}
            >
              Test connection
            </button>
          </PrefRow>
        </PrefCard>
      </PrefGroup>

      <PrefGroup
        title="MCP servers"
        hint={
          <>
            Give the assistant more tools, like Notion, Linear, Slack or GitHub. Each step runs after you approve and
            is shown on the request. Servers that need sign-in work through <code>npx mcp-remote &lt;url&gt;</code>.
          </>
        }
        action={
          <button
            className="btn btn-quiet btn-sm"
            onClick={() =>
              patch({
                mcpServers: [
                  ...s.mcpServers,
                  { name: "", command: "", args: [], env: {}, url: "", headers: {}, enabled: true },
                ],
              })
            }
          >
            Add server
          </button>
        }
      >
        <PrefCard>
          {s.mcpServers.length === 0 && <div className="pref-empty">No servers yet.</div>}
          {s.mcpServers.map((m, i) => (
            <div className="pref-row mcp-row" key={i}>
              <input
                type="checkbox"
                aria-label={`Use ${m.name || "server"}`}
                checked={m.enabled}
                onChange={(e) => setServer(i, { enabled: e.target.checked })}
              />
              <input
                className="input mcp-name"
                aria-label="Server name"
                defaultValue={m.name}
                placeholder="Name"
                onBlur={(e) => setServer(i, { name: e.target.value.trim() })}
              />
              <input
                className="input mono mcp-cmd"
                aria-label="Command or URL"
                defaultValue={m.url || [m.command, ...m.args].join(" ")}
                placeholder="Command and arguments, or https://… URL"
                onBlur={(e) => {
                  const v = e.target.value.trim();
                  if (/^https?:\/\//.test(v)) setServer(i, { url: v, command: "", args: [] });
                  else {
                    const [command, ...args] = v.split(/\s+/);
                    setServer(i, { command: command ?? "", args, url: "" });
                  }
                }}
              />
              <button
                className="btn btn-secondary btn-sm"
                onClick={async () => {
                  setProbe((p) => ({ ...p, [i]: "Connecting…" }));
                  try {
                    const tools = await api.mcpProbe(s.mcpServers[i]);
                    setProbe((p) => ({
                      ...p,
                      [i]: `${tools.length} tools: ${tools
                        .slice(0, 6)
                        .map((t) => t.name)
                        .join(", ")}${tools.length > 6 ? "…" : ""}`,
                    }));
                  } catch (e) {
                    setProbe((p) => ({ ...p, [i]: String(e) }));
                  }
                }}
              >
                Test
              </button>
              <button
                className="btn btn-quiet btn-icon btn-sm danger"
                aria-label={`Remove ${m.name || "server"}`}
                title="Remove"
                onClick={() => removeServer(i)}
              >
                <Icon name="trash" />
              </button>
              {probe[i] && <div className="mcp-probe">{probe[i]}</div>}
            </div>
          ))}
        </PrefCard>
        <details className="pref-disclosure">
          <summary>
            <Icon name="chevronRight" />
            Import from a Claude or Cursor config
          </summary>
          <textarea
            className="textarea mono"
            aria-label="MCP config JSON"
            value={mcpJson}
            onChange={(e) => setMcpJson(e.target.value)}
            placeholder='{"mcpServers": {"notion": {"command": "npx", "args": ["-y", "@notionhq/notion-mcp-server"]}}}'
          />
          <div>
            <button
              className="btn btn-secondary btn-sm"
              disabled={!mcpJson.trim()}
              onClick={async () => {
                try {
                  const parsed = await api.mcpParse(mcpJson);
                  patch({ mcpServers: [...s.mcpServers, ...parsed] });
                  setMcpJson("");
                } catch (e) {
                  onError(String(e));
                }
              }}
            >
              Import servers
            </button>
          </div>
        </details>
      </PrefGroup>

      <PrefGroup title="Claude Code">
        <PrefCard>
          <PrefCheck
            label="Hand open-ended requests to Claude Code"
            desc={
              claudePath === undefined
                ? "Looking for Claude Code…"
                : claudePath
                  ? `Uses your Claude account, MCP servers and browser tools. Found at ${claudePath}. Only the request and relevant context are sent, after you approve.`
                  : "Claude Code isn't installed (claude.com/claude-code)."
            }
            checked={s.claudeCode}
            disabled={!claudePath}
            onChange={(v) => patch({ claudeCode: v })}
          />
          {s.claudeCode && (
            <PrefRow
              label="Allowed tools"
              className="stack"
              desc="Comma-separated, like mcp__atlassian, mcp__claude-in-chrome, WebFetch. Tools not listed are refused."
            >
              <input
                className="input wide mono"
                aria-label="Allowed tools"
                defaultValue={s.claudeAllowedTools.join(", ")}
                onBlur={(e) =>
                  patch({
                    claudeAllowedTools: e.target.value
                      .split(",")
                      .map((x) => x.trim())
                      .filter(Boolean),
                  })
                }
              />
            </PrefRow>
          )}
        </PrefCard>
      </PrefGroup>
      {dialog}
    </>
  );
}

/* ── meetings ────────────────────────────────────────────────────── */

export function MeetingsSection({ models, onError }: { models: ModelInfo[]; onError: (msg: string | null) => void }) {
  const [s, setS] = useState<MeetingSettings | null>(null);
  const [cal, setCal] = useState<CalAccess | null>(null);
  const [voices, setVoices] = useState<[string, number][]>([]);
  const { confirm, dialog } = useConfirm();

  useEffect(() => {
    api.getSettings().then((st) => setS(st.meetings));
    api.calendarAccess().then(setCal).catch(() => {});
    api.listVoices().then(setVoices).catch(() => {});
  }, []);

  const save = (next: MeetingSettings) => {
    onError(null);
    setS(next);
    api.setMeetingSettings(next).catch((e) => onError(String(e)));
  };

  if (!s) return null;
  const stts = models.filter((m) => m.kind === "stt" && m.downloaded);
  const llms = models.filter((m) => m.kind === "llm" && m.downloaded);
  const embs = models.filter((m) => m.kind === "diarize" && m.id !== "diarize-segmentation" && m.downloaded);
  // "" means the default speaker model (CAM++).
  const embValue = s.embeddingModel || "diarize-embedding";
  const langValue = s.language === null ? "same" : s.language === "" ? "auto" : s.language;

  const forget = (name: string) =>
    confirm({
      title: `Forget ${name}'s voice?`,
      description: "Past recordings keep their labels. New recordings won't recognize them.",
      confirm: "Forget",
      danger: true,
      onConfirm: () =>
        api
          .deleteVoice(name)
          .then(() => setVoices((v) => v.filter(([x]) => x !== name)))
          .catch((e) => onError(String(e))),
    });

  return (
    <>
      <PrefGroup title="Models" hint="Download more in Models.">
        <PrefCard>
          <PrefRow label="Speech model" desc="Turns the call into a transcript.">
            <select
              className="select"
              aria-label="Meeting speech model"
              value={s.speechModel}
              onChange={(e) => save({ ...s, speechModel: e.target.value })}
            >
              <option value="">Same as dictation</option>
              {stts.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
              {s.speechModel && !stts.some((m) => m.id === s.speechModel) && (
                <option value={s.speechModel}>{s.speechModel} (not installed)</option>
              )}
            </select>
          </PrefRow>
          <PrefRow label="Spoken language" desc="Pick one if auto-detect gets it wrong.">
            <select
              className="select"
              aria-label="Meeting spoken language"
              value={langValue}
              onChange={(e) => {
                const v = e.target.value;
                save({ ...s, language: v === "same" ? null : v === "auto" ? "" : v });
              }}
            >
              <option value="same">Same as dictation</option>
              <option value="auto">Auto-detect</option>
              {LANGUAGES.filter((l) => l.code !== "").map((l) => (
                <option key={l.code} value={l.code}>
                  {l.name}
                </option>
              ))}
            </select>
          </PrefRow>
          <PrefRow label="Summary model" desc="Writes the summary and action items.">
            <select
              className="select"
              aria-label="Summary model"
              value={s.summaryModel}
              onChange={(e) => save({ ...s, summaryModel: e.target.value })}
            >
              <option value="">Automatic</option>
              {llms.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
              {s.summaryModel && !llms.some((m) => m.id === s.summaryModel) && (
                <option value={s.summaryModel}>{s.summaryModel} (not installed)</option>
              )}
            </select>
          </PrefRow>
          <PrefRow label="Speaker detection" desc="Tells the other people on the call apart.">
            <select
              className="select"
              aria-label="Speaker detection model"
              value={embValue}
              disabled={embs.length === 0}
              onChange={(e) => save({ ...s, embeddingModel: e.target.value })}
            >
              {embs.length === 0 && <option value={embValue}>None installed</option>}
              {embs.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
              {embs.length > 0 && !embs.some((m) => m.id === embValue) && (
                <option value={embValue}>{embValue} (not installed)</option>
              )}
            </select>
          </PrefRow>
        </PrefCard>
      </PrefGroup>

      <PrefGroup title="Recording">
        <PrefCard>
          <PrefCheck
            label="Offer to record calls"
            desc="When Zoom, Meet, Teams or another supported call starts. Browser calls need the meeting tab in front and Accessibility access. Needs macOS 14.2 or later."
            checked={s.detectMeetings}
            onChange={(v) => save({ ...s, detectMeetings: v })}
          />
          <PrefCheck
            label="Use my calendar"
            desc="Names recordings and attendees from the current event."
            checked={s.useCalendar && cal === "granted"}
            onChange={async (checked) => {
              if (!checked) return save({ ...s, useCalendar: false });
              let access = cal;
              if (access !== "granted") {
                const ok = await api.calendarRequest().catch(() => false);
                access = ok ? "granted" : await api.calendarAccess();
                setCal(access);
              }
              save({ ...s, useCalendar: access === "granted" });
            }}
          />
        </PrefCard>
        {cal === "denied" && (
          <div className="alert warn">
            <Icon name="alert" />
            <div className="alert-body">
              Calendar access is off. Allow unsound in System Settings › Privacy &amp; Security › Calendars.
            </div>
          </div>
        )}
      </PrefGroup>

      <PrefGroup
        title="Remembered voices"
        hint="People you name in a recording are recognized by voice and labeled next time."
      >
        <PrefCard>
          {voices.length === 0 ? (
            <div className="pref-empty">No remembered voices yet.</div>
          ) : (
            voices.map(([name, n]) => (
              <PrefRow key={name} label={name} desc={`Learned from ${n} ${n === 1 ? "line" : "lines"}`}>
                <button
                  className="btn btn-quiet btn-sm danger"
                  aria-label={`Forget ${name}`}
                  onClick={() => forget(name)}
                >
                  Forget
                </button>
              </PrefRow>
            ))
          )}
        </PrefCard>
      </PrefGroup>
      {dialog}
    </>
  );
}
