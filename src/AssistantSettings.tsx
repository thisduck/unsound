import { useEffect, useState } from "react";
import {
  api,
  AssistantSettings,
  CalAccess,
  formatShortcut,
  McpServer,
  MeetingSettings,
  ModelInfo,
  TaskAction,
  ToolGuide,
} from "./api";
import { ACTION_LABELS } from "./meetings/TaskCard";

const JIRA_TOKEN = "jira-api-token";
const ENDPOINT_KEY = "assistant-endpoint-key";
const OPENAI_KEY = "openai-api-key";
const ANTHROPIC_KEY = "anthropic-api-key";

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
  status: { ok: boolean; model?: string; error?: string; cloud?: boolean } | null;
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
    setListing("loading…");
    try {
      const ids = await api.listCloudModels(backend, backend === "compatible" ? s.endpointUrl : undefined);
      setCloudModels(ids);
      setListing(ids.length ? null : "no models listed");
    } catch (e) {
      setListing(String(e));
    }
  };
  const modelValue = backend === "compatible" ? s.endpointModel : s.cloudModel;
  const setModel = (m: string) => patch(backend === "compatible" ? { endpointModel: m } : { cloudModel: m });

  return (
    <section className="sheet-section">
      <div className="sheet-section-head">
        <h3>model</h3>
      </div>
      {status && (
        <p className={"sheet-hint block" + (status.ok ? "" : " warn")}>
          {status.ok ? `now using: ${status.model}` : status.error}
        </p>
      )}
      <div className="row">
        <div className="row-info">
          <div className="row-name">thinks with</div>
          <div className="row-desc">understands your requests, drafts, and drives the browser</div>
        </div>
        <div className="row-action">
          <select className="chip-select" value={backend} onChange={(e) => patch({ backend: e.target.value })}>
            <option value="local">this Mac — a downloaded model</option>
            <option value="claude" disabled={!clis.claude}>
              my Claude subscription{clis.claude ? "" : " (needs Claude Code)"}
            </option>
            <option value="codex" disabled={!clis.codex}>
              my ChatGPT subscription{clis.codex ? "" : " (needs Codex CLI)"}
            </option>
            <option value="anthropic">Anthropic API (key)</option>
            <option value="openai">OpenAI API (key)</option>
            <option value="compatible">another provider (Fireworks, Groq, Ollama…)</option>
          </select>
        </div>
      </div>

      {backend === "local" && (
        <>
          <div className="row">
            <div className="row-info">
              <div className="row-name">local model</div>
              <div className="row-desc">bigger models understand requests and write drafts much better</div>
            </div>
            <div className="row-action">
              <select className="chip-select" value={s.localModel} onChange={(e) => patch({ localModel: e.target.value })}>
                <option value="">automatic (best downloaded)</option>
                {llms.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
          {recommended.some((m) => !m.downloaded) && (
            <p className="sheet-hint block">
              recommended: {recommended.map((m) => `${m.name}${m.downloaded ? " ✓" : ""}`).join(" · ")} — download in
              Settings → models (E4B for 8–16 GB, 12B for 16–32 GB, 26B-A4B for 32 GB+).
            </p>
          )}
        </>
      )}

      {(backend === "claude" || backend === "codex") && (
        <>
          <div className="row">
            <div className="row-info">
              <div className="row-name">model</div>
              <div className="row-desc">
                {backend === "claude"
                  ? "uses the Claude account Claude Code is signed into — no API key"
                  : "uses the ChatGPT account the Codex CLI is signed into — no API key"}
              </div>
            </div>
            <div className="row-action">
              {backend === "claude" ? (
                <select className="chip-select" value={s.cliModel} onChange={(e) => patch({ cliModel: e.target.value })}>
                  <option value="">its default</option>
                  <option value="sonnet">Sonnet</option>
                  <option value="opus">Opus</option>
                  <option value="haiku">Haiku (fastest)</option>
                </select>
              ) : (
                <input className="text-input" defaultValue={s.cliModel} placeholder="its default" onBlur={(e) => patch({ cliModel: e.target.value.trim() })} />
              )}
            </div>
          </div>
          <p className="sheet-hint block">
            each step starts the CLI, so it's a few seconds slower per step than an API key.
          </p>
        </>
      )}

      {backend === "compatible" && (
        <div className="row">
          <div className="row-info">
            <div className="row-name">provider</div>
            <div className="row-desc">any OpenAI-compatible server</div>
          </div>
          <div className="row-action">
            <select
              className="chip-select"
              value={PROVIDERS.find((p) => p.url === s.endpointUrl)?.url ?? ""}
              onChange={(e) => e.target.value && patch({ endpointUrl: e.target.value })}
            >
              <option value="">custom URL</option>
              {PROVIDERS.map((p) => (
                <option key={p.url} value={p.url}>
                  {p.name}
                </option>
              ))}
            </select>
            <input
              className="text-input wide"
              key={s.endpointUrl}
              defaultValue={s.endpointUrl}
              placeholder="https://…/v1"
              onBlur={(e) => e.target.value.trim() !== s.endpointUrl && patch({ endpointUrl: e.target.value.trim() })}
            />
          </div>
        </div>
      )}

      {(backend === "anthropic" || backend === "openai" || backend === "compatible") && (
        <>
          <div className="row">
            <div className="row-info">
              <div className="row-name">API key</div>
              <div className="row-desc">
                {hasKey[keyName]
                  ? "saved in your Keychain"
                  : backend === "compatible"
                    ? "not needed for servers on this Mac"
                    : backend === "anthropic"
                      ? "from platform.claude.com → API keys"
                      : "from platform.openai.com → API keys"}
              </div>
            </div>
            <div className="row-action">
              <input
                className="text-input wide"
                type="password"
                value={keyInput}
                placeholder={hasKey[keyName] ? "••••••••" : "paste key"}
                onChange={(e) => setKeyInput(e.target.value)}
                onBlur={saveKey}
                onKeyDown={(e) => e.key === "Enter" && saveKey()}
              />
            </div>
          </div>
          <div className="row">
            <div className="row-info">
              <div className="row-name">model</div>
              <div className="row-desc">
                {listing ?? (backend === "anthropic" ? "default: claude-opus-5-5" : "pick from the provider's list, or type an id")}
              </div>
            </div>
            <div className="row-action">
              {cloudModels && cloudModels.length > 0 ? (
                <select className="chip-select" value={modelValue} onChange={(e) => setModel(e.target.value)}>
                  <option value="">—</option>
                  {[...new Set([modelValue, ...cloudModels].filter(Boolean))].map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  className="text-input wide"
                  key={backend + modelValue}
                  defaultValue={modelValue}
                  placeholder={backend === "anthropic" ? "claude-opus-5-5" : "model id"}
                  onBlur={(e) => e.target.value.trim() !== modelValue && setModel(e.target.value.trim())}
                />
              )}
              <button className="quiet accent" onClick={fetchModels}>
                list models
              </button>
            </div>
          </div>
        </>
      )}

      {status?.cloud && (
        <p className="sheet-hint block warn">
          This sends your requests, drafts and the relevant part of the meeting transcript to the provider. Meeting
          notes and transcription stay on this Mac.
        </p>
      )}
    </section>
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
  const [status, setStatus] = useState<{ ok: boolean; model?: string; error?: string; cloud?: boolean } | null>(null);
  const [hasJiraToken, setHasJiraToken] = useState(false);
  const [jiraToken, setJiraToken] = useState("");
  const [jiraResult, setJiraResult] = useState<string | null>(null);
  const [projects, setProjects] = useState<{ key: string; name: string }[]>([]);
  const [claudePath, setClaudePath] = useState<string | null | undefined>(undefined);
  const [mcpJson, setMcpJson] = useState("");
  const [probe, setProbe] = useState<Record<number, string>>({});
  const [browserTest, setBrowserTest] = useState<string | null>(null);

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
      <section className="sheet-section">
        <div className="sheet-section-head">
          <h3>requests</h3>
        </div>
        <p className="sheet-hint block">
          Ask unsound to do things during a meeting — “unsound, create a Jira ticket from what we just
          discussed”, “…draft an email to Priya about the launch”, “…remind me to send the deck after
          the call”. It drafts from the conversation; nothing leaves your Mac until you approve it.
        </p>
        <label className="check-row">
          <input type="checkbox" checked={s.enabled} onChange={(e) => patch({ enabled: e.target.checked })} />
          assistant on
        </label>
        <label className="check-row">
          <input
            type="checkbox"
            checked={s.listenForWakeWord}
            disabled={!s.enabled}
            onChange={(e) => patch({ listenForWakeWord: e.target.checked })}
          />
          listen for its name during meetings (your mic only — others on the call can't trigger it)
        </label>
        <div className="row">
          <div className="row-info">
            <div className="row-name">name to listen for</div>
            <div className="row-desc">say it at the start of a sentence, then the request</div>
          </div>
          <div className="row-action">
            <input
              className="text-input"
              defaultValue={s.wakeWord}
              onBlur={(e) => e.target.value.trim() && patch({ wakeWord: e.target.value.trim() })}
            />
          </div>
        </div>
        <div className="row">
          <div className="row-info">
            <div className="row-name">command key</div>
            <div className="row-desc">hold it and speak a request — in a meeting or any time</div>
          </div>
          <div className="row-action">
            <span className="shortcut-keys">{commandKey ? formatShortcut(commandKey) : "not set"}</span>
            <button className="quiet accent" onClick={onOpenShortcuts}>
              {commandKey ? "change" : "set one"}
            </button>
          </div>
        </div>
      </section>

      <ModelSection s={s} llms={llms} recommended={recommended} status={status} patch={patch} onError={onError} />
      <section className="sheet-section">
        <div className="sheet-section-head">
          <h3>about you</h3>
        </div>
        <p className="sheet-hint block">included with every draft — who you are, your team, how you sign off.</p>
        <textarea
          className="meet-notes"
          defaultValue={s.aboutMe}
          placeholder="e.g. I'm Adnan, eng lead on the web team at Acme. Sign emails “Adnan”. Our sprint ends Fridays."
          onBlur={(e) => e.target.value !== s.aboutMe && patch({ aboutMe: e.target.value })}
        />
      </section>

      <section className="sheet-section">
        <div className="sheet-section-head">
          <h3>your tools</h3>
        </div>
        <p className="sheet-hint block">
          Describe the tools you use in your own words — no integration needed. unsound picks one by
          “use it for”, follows your rules when drafting, and (in browser mode) does the work on the site in
          your browser, where you're already signed in.
        </p>
        {s.guides.map((g) => (
          <div className="guide" key={g.id}>
            <div className="guide-row">
              <input type="checkbox" checked={g.enabled} onChange={(e) => setGuide(g.id, { enabled: e.target.checked })} />
              <input className="text-input" defaultValue={g.name} placeholder="name, e.g. Jira" onBlur={(e) => setGuide(g.id, { name: e.target.value.trim() })} />
              <select className="chip-select" value={g.mode} onChange={(e) => setGuide(g.id, { mode: e.target.value })}>
                <option value="browser">do it in my browser</option>
                <option value="auto">built-in if there is one, else my browser</option>
                <option value="draft">just draft it for me</option>
              </select>
              <button className="quiet danger" onClick={() => patch({ guides: s.guides.filter((x) => x.id !== g.id) })}>
                ✕
              </button>
            </div>
            <div className="guide-row">
              <input className="text-input" defaultValue={g.when} placeholder="use it for… e.g. creating or updating Jira tickets" onBlur={(e) => setGuide(g.id, { when: e.target.value.trim() })} />
            </div>
            <div className="guide-row">
              <input className="text-input" defaultValue={g.site} placeholder="start at (URL)" onBlur={(e) => setGuide(g.id, { site: e.target.value.trim() })} />
            </div>
            <textarea
              className="meet-notes"
              defaultValue={g.guidance}
              placeholder="your rules — e.g. If no project is mentioned, use BLAH. Always add the label from-meeting."
              onBlur={(e) => e.target.value !== g.guidance && setGuide(g.id, { guidance: e.target.value })}
            />
            {g.mode !== "draft" && (
              <label className="check-row">
                <input type="checkbox" checked={g.leaveFinalStep} onChange={(e) => setGuide(g.id, { leaveFinalStep: e.target.checked })} />
                fill it in, but leave the final click (Create / Send / Save) to me
              </label>
            )}
          </div>
        ))}
        <div className="guide-templates">
          <button className="quiet accent" onClick={() => addGuide()}>
            + add a tool
          </button>
          {TEMPLATES.filter((t) => !s.guides.some((g) => g.name === t.name)).map((t) => (
            <button key={t.name} className="quiet" onClick={() => addGuide(t)}>
              + {t.name}
            </button>
          ))}
        </div>
      </section>

      <section className="sheet-section">
        <div className="sheet-section-head">
          <h3>browser</h3>
        </div>
        <p className="sheet-hint block">
          How website work gets done — in your own Chrome, signed in as you. It only runs after you approve a
          draft, and each run is shown on the task.
        </p>
        <div className="engine-pick">
          <label>
            <input type="radio" checked={s.browser === "chrome"} disabled={!claudePath} onChange={() => patch({ browser: "chrome" })} />
            Claude in Chrome, through Claude Code (recommended)
          </label>
          <div className="row-desc">
            {claudePath
              ? "Most reliable. Uses your claude.ai sign-in and the Claude in Chrome extension; the task and its approved content are sent to Claude."
              : "Needs Claude Code (claude.com/claude-code) and the Claude in Chrome extension."}
          </div>
          <label>
            <input type="radio" checked={s.browser === "local"} onChange={() => patch({ browser: "local" })} />
            the assistant's model (above) + a browser connector
          </label>
          <div className="row-desc">
            Works with any model you pick above — a local one stays on this Mac but is less sure-footed on
            complex sites. Default connector: Playwright MCP — install the “Playwright MCP Bridge” Chrome
            extension once.
          </div>
          <label>
            <input type="radio" checked={s.browser === "off" || !s.browser} onChange={() => patch({ browser: "off" })} />
            off
          </label>
        </div>
        {s.browser === "chrome" && (
          <div className="row">
            <div className="row-info">
              <div className="row-name">Claude model</div>
              <div className="row-desc">Sonnet is quicker and cheaper for clicking through sites</div>
            </div>
            <div className="row-action">
              <select className="chip-select" value={s.claudeModel} onChange={(e) => patch({ claudeModel: e.target.value })}>
                <option value="">your default</option>
                <option value="sonnet">Sonnet</option>
                <option value="opus">Opus</option>
                <option value="haiku">Haiku</option>
              </select>
            </div>
          </div>
        )}
        {s.browser === "local" && (
          <div className="row">
            <div className="row-info">
              <div className="row-name">browser connector</div>
              <div className="row-desc">
                any MCP browser server — e.g. <code>npx -y @playwright/mcp@latest --extension</code> or{" "}
                <code>npx -y chrome-devtools-mcp@latest --autoConnect</code> (turn on chrome://inspect/#remote-debugging)
              </div>
            </div>
            <div className="row-action">
              <input
                className="text-input wide"
                defaultValue={[s.browserServer.command, ...s.browserServer.args].join(" ")}
                onBlur={(e) => {
                  const [command, ...args] = e.target.value.trim().split(/\s+/);
                  patch({ browserServer: { ...s.browserServer, command: command ?? "", args, url: "" } });
                }}
              />
            </div>
          </div>
        )}
        {s.browser !== "off" && s.browser && (
          <div className="row">
            <div className="row-info">
              <div className="row-desc">{browserTest}</div>
            </div>
            <div className="row-action">
              <button
                className="quiet accent"
                onClick={async () => {
                  setBrowserTest("opening Wikipedia in your browser…");
                  try {
                    setBrowserTest(`works — read: “${(await api.browserTest()).trim()}”`);
                  } catch (e) {
                    setBrowserTest(String(e));
                  }
                }}
              >
                test it
              </button>
            </div>
          </div>
        )}
      </section>

      <section className="sheet-section">
        <div className="sheet-section-head">
          <h3>do these without asking</h3>
        </div>
        <p className="sheet-hint block">
          drafts are always made right away; these kinds also run as soon as they're drafted. Anything done in
          your browser always waits for you.
        </p>
        <div className="check-grid">
          {(Object.keys(ACTION_LABELS) as TaskAction[]).map((a) => (
            <label key={a} className="check-row">
              <input
                type="checkbox"
                checked={s.autoRun.includes(a)}
                onChange={(e) =>
                  patch({ autoRun: e.target.checked ? [...s.autoRun, a] : s.autoRun.filter((x) => x !== a) })
                }
              />
              {ACTION_LABELS[a]}
            </label>
          ))}
        </div>
      </section>

      <section className="sheet-section">
        <div className="sheet-section-head">
          <h3>jira api (optional)</h3>
        </div>
        <p className="sheet-hint block">
          Only if you want tickets created without the browser. Otherwise a Jira tool above (or browser use)
          handles it.
        </p>
        <div className="row">
          <div className="row-info">
            <div className="row-name">site</div>
          </div>
          <div className="row-action">
            <input className="text-input wide" placeholder="https://yourteam.atlassian.net" defaultValue={s.jira.site} onBlur={(e) => patchJira({ site: e.target.value.trim() })} />
          </div>
        </div>
        <div className="row">
          <div className="row-info">
            <div className="row-name">your Atlassian email</div>
          </div>
          <div className="row-action">
            <input className="text-input wide" defaultValue={s.jira.email} onBlur={(e) => patchJira({ email: e.target.value.trim() })} />
          </div>
        </div>
        <div className="row">
          <div className="row-info">
            <div className="row-name">API token</div>
            <div className="row-desc">
              {hasJiraToken ? "saved in your Keychain" : "create one at id.atlassian.com → Security → API tokens"}
            </div>
          </div>
          <div className="row-action">
            <input
              className="text-input wide"
              type="password"
              value={jiraToken}
              placeholder={hasJiraToken ? "••••••••" : ""}
              onChange={(e) => setJiraToken(e.target.value)}
              onBlur={() => {
                if (jiraToken) saveSecret(JIRA_TOKEN, jiraToken, setHasJiraToken).then(() => setJiraToken(""));
              }}
            />
          </div>
        </div>
        <div className="row">
          <div className="row-info">
            <div className="row-name">default project</div>
          </div>
          <div className="row-action">
            {projects.length > 0 ? (
              <select className="chip-select" value={s.jira.defaultProject} onChange={(e) => patchJira({ defaultProject: e.target.value })}>
                <option value="">—</option>
                {projects.map((p) => (
                  <option key={p.key} value={p.key}>
                    {p.key} · {p.name}
                  </option>
                ))}
              </select>
            ) : (
              <input className="text-input" placeholder="ENG" defaultValue={s.jira.defaultProject} onBlur={(e) => patchJira({ defaultProject: e.target.value.trim().toUpperCase() })} />
            )}
            <select className="chip-select" value={s.jira.defaultIssueType || "Task"} onChange={(e) => patchJira({ defaultIssueType: e.target.value })}>
              {["Task", "Bug", "Story"].map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </div>
        </div>
        <div className="row">
          <div className="row-info">
            <div className="row-desc">{jiraResult}</div>
          </div>
          <div className="row-action">
            <button
              className="quiet accent"
              onClick={async () => {
                setJiraResult("checking…");
                try {
                  const r = await api.jiraTest();
                  setProjects(r.projects);
                  setJiraResult(`connected as ${r.user} · ${r.projects.length} projects`);
                } catch (e) {
                  setJiraResult(String(e));
                }
              }}
            >
              test connection
            </button>
          </div>
        </div>
        <p className="sheet-hint block">
          without a token or browser use, unsound opens Jira's create page with the ticket on your clipboard.
        </p>
      </section>

      <section className="sheet-section">
        <div className="sheet-section-head">
          <h3>tools (MCP)</h3>
        </div>
        <p className="sheet-hint block">
          Give the assistant other tools — Notion, Linear, Slack, GitHub, a browser — through MCP servers.
          Requests that need them run step by step after you approve, and each step is shown. Remote
          servers that need sign-in work through <code>npx mcp-remote &lt;url&gt;</code>.
        </p>
        {s.mcpServers.map((m, i) => (
          <div className="mcp-row" key={i}>
            <label className="check-row">
              <input type="checkbox" checked={m.enabled} onChange={(e) => setServer(i, { enabled: e.target.checked })} />
            </label>
            <input className="text-input" defaultValue={m.name} placeholder="name" onBlur={(e) => setServer(i, { name: e.target.value.trim() })} />
            <input
              className="text-input wide"
              defaultValue={m.url || [m.command, ...m.args].join(" ")}
              placeholder="command and args, or https://… URL"
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
              className="quiet"
              onClick={async () => {
                setProbe((p) => ({ ...p, [i]: "connecting…" }));
                try {
                  const tools = await api.mcpProbe(s.mcpServers[i]);
                  setProbe((p) => ({ ...p, [i]: `${tools.length} tools: ${tools.slice(0, 6).map((t) => t.name).join(", ")}${tools.length > 6 ? "…" : ""}` }));
                } catch (e) {
                  setProbe((p) => ({ ...p, [i]: String(e) }));
                }
              }}
            >
              test
            </button>
            <button className="quiet danger" onClick={() => patch({ mcpServers: s.mcpServers.filter((_, j) => j !== i) })}>
              ✕
            </button>
            {probe[i] && <div className="mcp-probe">{probe[i]}</div>}
          </div>
        ))}
        <div className="row">
          <div className="row-action">
            <button
              className="quiet accent"
              onClick={() =>
                patch({ mcpServers: [...s.mcpServers, { name: "", command: "", args: [], env: {}, url: "", headers: {}, enabled: true }] })
              }
            >
              + add server
            </button>
          </div>
        </div>
        <details className="meet-options">
          <summary>import from a Claude / Cursor config</summary>
          <textarea
            className="meet-notes"
            value={mcpJson}
            onChange={(e) => setMcpJson(e.target.value)}
            placeholder='{"mcpServers": {"notion": {"command": "npx", "args": ["-y", "@notionhq/notion-mcp-server"]}}}'
          />
          <button
            className="quiet accent"
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
            import
          </button>
        </details>
      </section>

      <section className="sheet-section">
        <div className="sheet-section-head">
          <h3>claude code</h3>
        </div>
        <label className="check-row">
          <input
            type="checkbox"
            checked={s.claudeCode}
            disabled={!claudePath}
            onChange={(e) => patch({ claudeCode: e.target.checked })}
          />
          hand open-ended requests to Claude Code (uses your Claude account, MCP servers and browser tools)
        </label>
        <p className="sheet-hint block">
          {claudePath === undefined
            ? "looking for Claude Code…"
            : claudePath
              ? `found at ${claudePath}. Only the request and the relevant meeting context are sent, and only after you approve.`
              : "Claude Code isn't installed (claude.com/claude-code)."}
        </p>
        {s.claudeCode && (
          <div className="row">
            <div className="row-info">
              <div className="row-name">allowed tools</div>
              <div className="row-desc">comma-separated, e.g. mcp__atlassian, mcp__claude-in-chrome, WebFetch — tools not listed are refused</div>
            </div>
            <div className="row-action">
              <input
                className="text-input wide"
                defaultValue={s.claudeAllowedTools.join(", ")}
                onBlur={(e) =>
                  patch({ claudeAllowedTools: e.target.value.split(",").map((x) => x.trim()).filter(Boolean) })
                }
              />
            </div>
          </div>
        )}
      </section>
    </>
  );
}

/* ── meetings ────────────────────────────────────────────────────── */

export function MeetingsSection({ models, onError }: { models: ModelInfo[]; onError: (msg: string | null) => void }) {
  const [s, setS] = useState<MeetingSettings | null>(null);
  const [cal, setCal] = useState<CalAccess | null>(null);
  const [voices, setVoices] = useState<[string, number][]>([]);

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
  const embs = models.filter((m) => m.kind === "diarize" && m.id !== "diarize-segmentation" && m.downloaded);

  return (
    <>
      <section className="sheet-section">
        <div className="sheet-section-head">
          <h3>starting meetings</h3>
        </div>
        <label className="check-row">
          <input type="checkbox" checked={s.detectMeetings} onChange={(e) => save({ ...s, detectMeetings: e.target.checked })} />
          offer to record when Zoom, Meet, Teams, Slack or FaceTime starts using the mic
        </label>
        <label className="check-row">
          <input
            type="checkbox"
            checked={s.useCalendar && cal === "granted"}
            onChange={async (e) => {
              if (!e.target.checked) return save({ ...s, useCalendar: false });
              let access = cal;
              if (access !== "granted") {
                const ok = await api.calendarRequest().catch(() => false);
                access = ok ? "granted" : await api.calendarAccess();
                setCal(access);
              }
              save({ ...s, useCalendar: access === "granted" });
            }}
          />
          name meetings and attendees from my calendar
        </label>
        {cal === "denied" && (
          <p className="sheet-hint block warn">
            calendar access is off — allow unsound in System Settings → Privacy &amp; Security → Calendars.
          </p>
        )}
      </section>

      <section className="sheet-section">
        <div className="sheet-section-head">
          <h3>voices</h3>
        </div>
        <div className="row">
          <div className="row-info">
            <div className="row-name">voice model</div>
            <div className="row-desc">tells remote speakers apart as they talk</div>
          </div>
          <div className="row-action">
            <select className="chip-select" value={s.embeddingModel} onChange={(e) => save({ ...s, embeddingModel: e.target.value })}>
              <option value="">default (CAM++)</option>
              {embs.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </div>
        </div>
        <p className="sheet-hint block">
          people you name in a meeting are remembered by voice and labeled automatically next time.
        </p>
        {voices.length === 0 ? (
          <p className="dim">no remembered voices yet.</p>
        ) : (
          voices.map(([name, n]) => (
            <div className="row" key={name}>
              <div className="row-info">
                <div className="row-name">{name}</div>
                <div className="row-desc">from {n} lines</div>
              </div>
              <div className="row-action">
                <button
                  className="quiet danger"
                  onClick={() =>
                    api.deleteVoice(name).then(() => setVoices((v) => v.filter(([x]) => x !== name)))
                  }
                >
                  forget
                </button>
              </div>
            </div>
          ))
        )}
      </section>
    </>
  );
}
