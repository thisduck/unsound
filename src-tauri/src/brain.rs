//! The assistant's model — separate from the dictation cleanup model, since
//! understanding a request, drafting a Jira ticket or driving a browser wants
//! something stronger. Everything the assistant asks for is a JSON object
//! matching a schema (that's also how its tool use works, see agent.rs), so a
//! backend only has to do one thing well: return schema-valid JSON.
//!
//! Backends:
//!  • **local** — a downloaded GGUF in-process; a grammar enforces the schema;
//!  • **openai** / **compatible** — OpenAI's API or any OpenAI-compatible
//!    server (Fireworks, Groq, Together, OpenRouter, Ollama, LM Studio,
//!    llama-server), asked for `json_schema` output, with fallbacks for
//!    servers that don't support it;
//!  • **anthropic** — the Messages API with structured outputs;
//!  • **claude** / **codex** — the user's own Claude or ChatGPT subscription,
//!    through the Claude Code / Codex CLI they're already signed into (both
//!    take a JSON Schema for their final answer).

use crate::tools::{claude_code, codex, shell_env};
use crate::{llm, models, secrets, settings, AppState};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tauri::{AppHandle, Manager};

pub enum Brain {
    Local { path: PathBuf, name: String },
    /// OpenAI or any OpenAI-compatible server.
    OpenAi { base: String, key: Option<String>, model: String },
    Anthropic { key: String, model: String },
    ClaudeCli { model: String },
    CodexCli { model: String },
}

/// Local models known to follow instructions and fill schemas well, best
/// first — used when no assistant model is chosen.
const PREFERRED: &[&str] = &[
    "gemma-4-26b-a4b",
    "qwen3.6-35b-a3b",
    "gpt-oss-20b",
    "gemma-4-12b",
    "qwen3-8b",
    "gemma-4-e4b",
    "qwen3-4b",
];

const ANTHROPIC_DEFAULT_MODEL: &str = "claude-opus-5-5";

impl Brain {
    /// The configured assistant model, or the best downloaded local one.
    pub fn resolve(app: &AppHandle) -> Result<Brain, String> {
        let s = settings::load(app);
        let a = &s.assistant;
        let key = |k: &str| secrets::get(k).filter(|v| !v.trim().is_empty());
        match a.backend.as_str() {
            "openai" => {
                if a.cloud_model.trim().is_empty() {
                    return Err("choose an OpenAI model in Settings → assistant".into());
                }
                Ok(Brain::OpenAi {
                    base: "https://api.openai.com/v1".into(),
                    key: Some(key(settings::OPENAI_KEY).ok_or("add your OpenAI API key in Settings → assistant")?),
                    model: a.cloud_model.trim().into(),
                })
            }
            "anthropic" => Ok(Brain::Anthropic {
                key: key(settings::ANTHROPIC_KEY).ok_or("add your Anthropic API key in Settings → assistant")?,
                model: Some(a.cloud_model.trim())
                    .filter(|m| !m.is_empty())
                    .unwrap_or(ANTHROPIC_DEFAULT_MODEL)
                    .into(),
            }),
            "compatible" | "endpoint" => {
                if a.endpoint_url.trim().is_empty() || a.endpoint_model.trim().is_empty() {
                    return Err("set the server's URL and model in Settings → assistant".into());
                }
                Ok(Brain::OpenAi {
                    base: a.endpoint_url.trim().trim_end_matches('/').to_string(),
                    key: key(settings::ENDPOINT_KEY),
                    model: a.endpoint_model.trim().into(),
                })
            }
            "claude" => {
                claude_code::find_claude().ok_or("Claude Code isn't installed — see claude.com/claude-code")?;
                Ok(Brain::ClaudeCli { model: a.cli_model.trim().into() })
            }
            "codex" => {
                codex::find_codex().ok_or("the Codex CLI isn't installed — see developers.openai.com/codex")?;
                Ok(Brain::CodexCli { model: a.cli_model.trim().into() })
            }
            _ => Self::local(app, &s),
        }
    }

    fn local(app: &AppHandle, s: &settings::Settings) -> Result<Brain, String> {
        let a = &s.assistant;
        let downloaded: Vec<models::ModelInfo> = models::all_models(app)?
            .into_iter()
            .filter(|m| m.kind == models::ModelKind::Llm && m.downloaded)
            .collect();
        let pick = [a.local_model.as_str(), s.meetings.summary_model.as_str()]
            .into_iter()
            .filter(|id| !id.is_empty())
            .find_map(|id| downloaded.iter().find(|m| m.id == id))
            .or_else(|| {
                PREFERRED
                    .iter()
                    .find_map(|id| downloaded.iter().find(|m| m.id == *id))
            })
            .or_else(|| downloaded.first())
            .ok_or("download a language model first (Settings → models) — the assistant needs one")?;
        Ok(Brain::Local {
            path: models::downloaded_model_path(app, &pick.id)?,
            name: pick.name.clone(),
        })
    }

    pub fn describe(&self) -> String {
        match self {
            Brain::Local { name, .. } => name.clone(),
            Brain::OpenAi { model, base, .. } => {
                let host = base.split("://").nth(1).unwrap_or(base).split('/').next().unwrap_or(base);
                format!("{model} ({host})")
            }
            Brain::Anthropic { model, .. } => format!("{model} (Anthropic API)"),
            Brain::ClaudeCli { model } => format!(
                "Claude via your subscription{}",
                if model.is_empty() { String::new() } else { format!(" ({model})") }
            ),
            Brain::CodexCli { model } => format!(
                "ChatGPT via your subscription{}",
                if model.is_empty() { String::new() } else { format!(" ({model})") }
            ),
        }
    }

    /// Whether this backend sends prompts off the machine.
    pub fn is_cloud(&self) -> bool {
        match self {
            Brain::Local { .. } => false,
            Brain::OpenAi { base, .. } => !(base.contains("localhost") || base.contains("127.0.0.1")),
            _ => true,
        }
    }

    /// A JSON object matching `schema`.
    pub fn json(&self, app: &AppHandle, messages: &[(&str, String)], schema: &Value, max_tokens: usize) -> Result<Value, String> {
        if let Brain::Local { path, .. } = self {
            return llm::chat_json(&app.state::<AppState>().llm, path, messages, schema, max_tokens);
        }
        // OpenAI's strict mode and Anthropic need closed objects; free-form
        // objects travel as JSON strings and are decoded on the way back.
        // Compatible servers (llama-server, Ollama, Fireworks…) enforce
        // schemas with grammars that take free-form objects as they are.
        let openai_strict = matches!(self, Brain::OpenAi { base, .. } if base.contains("api.openai.com"));
        let needs_strict = openai_strict || !matches!(self, Brain::OpenAi { .. });
        let (strict, encoded) = if needs_strict { strictify(schema) } else { (schema.clone(), Vec::new()) };
        let raw = match self {
            Brain::OpenAi { base, key, model } => {
                openai_chat(base, key.as_deref(), model, messages, &strict, openai_strict, max_tokens)?
            }
            Brain::Anthropic { key, model } => anthropic_json(key, model, messages, &strict)?,
            Brain::ClaudeCli { model } => claude_json(model, messages, &strict)?,
            Brain::CodexCli { model } => codex_json(model, messages, &strict)?,
            Brain::Local { .. } => unreachable!(),
        };
        let mut v = match raw {
            Value::String(text) => llm::parse_json_reply(&text)?,
            v => v,
        };
        decode_encoded(&mut v, &encoded);
        Ok(v)
    }
}

// ---- schema adaptation -------------------------------------------------------

/// Make a schema acceptable to strict hosted structured outputs (OpenAI,
/// Anthropic, the CLIs): every object closed (`additionalProperties: false`)
/// with all its properties required, and free-form objects (e.g. a tool's
/// arguments) turned into JSON-encoded strings. Returns the paths encoded.
pub fn strictify(schema: &Value) -> (Value, Vec<Vec<String>>) {
    let mut encoded = Vec::new();
    let out = strict_node(schema, &mut Vec::new(), &mut encoded);
    (out, encoded)
}

fn strict_node(node: &Value, path: &mut Vec<String>, encoded: &mut Vec<Vec<String>>) -> Value {
    let Some(obj) = node.as_object() else {
        return node.clone();
    };
    let mut out = obj.clone();
    match obj.get("type").and_then(|t| t.as_str()) {
        Some("object") => match obj.get("properties").and_then(|p| p.as_object()) {
            Some(props) if !props.is_empty() => {
                let mut new_props = serde_json::Map::new();
                for (k, v) in props {
                    path.push(k.clone());
                    new_props.insert(k.clone(), strict_node(v, path, encoded));
                    path.pop();
                }
                out.insert("required".into(), json!(props.keys().collect::<Vec<_>>()));
                out.insert("properties".into(), Value::Object(new_props));
                out.insert("additionalProperties".into(), json!(false));
            }
            _ => {
                encoded.push(path.clone());
                return json!({"type": "string", "description": "a JSON object serialized as a string, e.g. {\"url\": \"https://example.com\"}"});
            }
        },
        Some("array") => {
            if let Some(items) = obj.get("items") {
                path.push("[]".into());
                out.insert("items".into(), strict_node(items, path, encoded));
                path.pop();
            }
            // Item-count limits aren't supported by strict structured outputs.
            out.remove("maxItems");
            out.remove("minItems");
        }
        _ => {}
    }
    Value::Object(out)
}

/// Turn the string-encoded objects back into objects.
fn decode_encoded(v: &mut Value, encoded: &[Vec<String>]) {
    for path in encoded {
        decode_at(v, path);
    }
}

fn decode_at(v: &mut Value, path: &[String]) {
    let Some((head, rest)) = path.split_first() else {
        if let Value::String(s) = v {
            *v = serde_json::from_str(s).unwrap_or_else(|_| json!({}));
        }
        return;
    };
    if head == "[]" {
        if let Some(items) = v.as_array_mut() {
            for item in items {
                decode_at(item, rest);
            }
        }
    } else if let Some(child) = v.get_mut(head) {
        decode_at(child, rest);
    }
}

// ---- hosted APIs ---------------------------------------------------------------

fn http() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .map_err(|e| e.to_string())
}

/// OpenAI chat completions (or a compatible server), trying the strongest
/// structured-output mode the server accepts.
fn openai_chat(
    base: &str,
    key: Option<&str>,
    model: &str,
    messages: &[(&str, String)],
    schema: &Value,
    strict: bool,
    max_tokens: usize,
) -> Result<Value, String> {
    let url = if base.ends_with("/chat/completions") {
        base.to_string()
    } else {
        format!("{base}/chat/completions")
    };
    let msgs: Vec<Value> = messages
        .iter()
        .map(|(r, c)| json!({"role": r, "content": c}))
        .collect();
    let modes = [
        Some(json!({"type": "json_schema", "json_schema": {"name": "reply", "schema": schema, "strict": strict}})),
        Some(json!({"type": "json_object"})),
        None,
    ];
    let mut last_err = String::new();
    for mode in modes {
        let mut body = json!({"model": model, "messages": msgs, "max_completion_tokens": max_tokens.max(2000)});
        match &mode {
            Some(m) => body["response_format"] = m.clone(),
            None => {
                // No structured mode at all: spell the schema out.
                let last = body["messages"].as_array().map(|a| a.len()).unwrap_or(1) - 1;
                let content = body["messages"][last]["content"].as_str().unwrap_or("").to_string();
                body["messages"][last]["content"] = json!(format!(
                    "{content}\n\nReply with only a JSON object matching this JSON Schema:\n{schema}"
                ));
            }
        }
        let key = key.map(str::to_string);
        let url = url.clone();
        let res: Result<Value, (u16, String)> = tauri::async_runtime::block_on(async move {
            let mut req = http().map_err(|e| (0, e))?.post(&url).json(&body);
            if let Some(k) = key {
                req = req.bearer_auth(k);
            }
            let resp = req.send().await.map_err(|e| (0, format!("couldn't reach {url}: {e}")))?;
            let status = resp.status().as_u16();
            let text = resp.text().await.map_err(|e| (status, e.to_string()))?;
            if !(200..300).contains(&status) {
                let short: String = text.chars().take(400).collect();
                return Err((status, format!("model server error {status}: {short}")));
            }
            serde_json::from_str(&text).map_err(|e| (status, format!("bad reply: {e}")))
        });
        match res {
            Ok(v) => {
                return v["choices"][0]["message"]["content"]
                    .as_str()
                    .map(|s| Value::String(s.to_string()))
                    .ok_or_else(|| "the model returned no message".to_string());
            }
            // Older servers reject newer parameters; retry with the classic one.
            Err((400, e)) if e.contains("max_completion_tokens") => {
                last_err = e;
                continue;
            }
            Err((status, e)) if (400..500).contains(&status) && status != 401 && status != 403 && mode.is_some() => {
                last_err = e;
                continue;
            }
            Err((_, e)) => return Err(e),
        }
    }
    Err(last_err)
}

/// Anthropic Messages API with structured outputs. Refusals fall back to
/// Anthropic's recommended model server-side (`fallbacks: "default"`).
fn anthropic_json(key: &str, model: &str, messages: &[(&str, String)], schema: &Value) -> Result<Value, String> {
    let system: Vec<&str> = messages.iter().filter(|(r, _)| *r == "system").map(|(_, c)| c.as_str()).collect();
    let msgs: Vec<Value> = messages
        .iter()
        .filter(|(r, _)| *r != "system")
        .map(|(r, c)| json!({"role": if *r == "assistant" { "assistant" } else { "user" }, "content": c}))
        .collect();
    let mut body = json!({
        "model": model,
        "max_tokens": 16000,
        "messages": msgs,
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    });
    // Models with safety classifiers that can decline: let the API re-run a
    // declined request on its recommended fallback instead of failing.
    let fallbacks = ["claude-opus-5", "claude-fable-5", "claude-sonnet-5-5"]
        .iter()
        .any(|p| model.starts_with(p));
    if fallbacks {
        body["fallbacks"] = json!("default");
    }
    if !system.is_empty() {
        body["system"] = json!(system.join("\n\n"));
    }
    let key = key.to_string();
    let v: Value = tauri::async_runtime::block_on(async move {
        let mut req = http()?
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01");
        if fallbacks {
            req = req.header("anthropic-beta", "server-side-fallback-2026-07-01");
        }
        let resp = req
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("couldn't reach Anthropic: {e}"))?;
        let status = resp.status().as_u16();
        let text = resp.text().await.map_err(|e| e.to_string())?;
        if !(200..300).contains(&status) {
            let short: String = text.chars().take(400).collect();
            return Err(format!("Anthropic error {status}: {short}"));
        }
        serde_json::from_str(&text).map_err(|e| format!("bad reply from Anthropic: {e}"))
    })?;
    match v["stop_reason"].as_str() {
        Some("refusal") => {
            let why = v["stop_details"]["explanation"].as_str().unwrap_or("no reason given");
            return Err(format!("the model declined this request ({why})"));
        }
        Some("max_tokens") => return Err("the model ran out of room before finishing".into()),
        _ => {}
    }
    let text: String = v["content"]
        .as_array()
        .map(|blocks| {
            blocks
                .iter()
                .filter(|b| b["type"] == "text")
                .filter_map(|b| b["text"].as_str())
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default();
    Ok(Value::String(text))
}

/// List a provider's models (for the settings picker).
pub fn list_models(provider: &str, base: &str) -> Result<Vec<String>, String> {
    let (url, key, anthropic) = match provider {
        "anthropic" => (
            "https://api.anthropic.com/v1/models?limit=100".to_string(),
            secrets::get(settings::ANTHROPIC_KEY),
            true,
        ),
        "openai" => ("https://api.openai.com/v1/models".to_string(), secrets::get(settings::OPENAI_KEY), false),
        _ => (
            format!("{}/models", base.trim().trim_end_matches('/')),
            secrets::get(settings::ENDPOINT_KEY),
            false,
        ),
    };
    let v: Value = tauri::async_runtime::block_on(async move {
        let mut req = http()?.get(&url);
        if let Some(k) = key.filter(|k| !k.is_empty()) {
            req = if anthropic {
                req.header("x-api-key", k).header("anthropic-version", "2023-06-01")
            } else {
                req.bearer_auth(k)
            };
        }
        let resp = req.send().await.map_err(|e| format!("couldn't reach {url}: {e}"))?;
        let status = resp.status().as_u16();
        let text = resp.text().await.map_err(|e| e.to_string())?;
        if !(200..300).contains(&status) {
            let short: String = text.chars().take(300).collect();
            return Err(format!("error {status}: {short}"));
        }
        serde_json::from_str(&text).map_err(|e| e.to_string())
    })?;
    let mut ids: Vec<String> = v["data"]
        .as_array()
        .map(|a| a.iter().filter_map(|m| m["id"].as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    ids.sort();
    Ok(ids)
}

// ---- subscriptions (CLIs) --------------------------------------------------------

/// Flatten a conversation for a CLI that takes one prompt: the system text
/// separately, the turns as a labeled transcript.
fn flatten(messages: &[(&str, String)]) -> (String, String) {
    let system = messages
        .iter()
        .filter(|(r, _)| *r == "system")
        .map(|(_, c)| c.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let turns: Vec<&(&str, String)> = messages.iter().filter(|(r, _)| *r != "system").collect();
    let prompt = if turns.len() == 1 {
        turns[0].1.clone()
    } else {
        turns
            .iter()
            .map(|(r, c)| format!("{}:\n{c}", if *r == "assistant" { "You (earlier)" } else { "User" }))
            .collect::<Vec<_>>()
            .join("\n\n")
    };
    (system, prompt)
}

/// `claude -p` on the user's Claude subscription: our system prompt, no
/// tools, no MCP servers, the answer constrained by `--json-schema`.
fn claude_json(model: &str, messages: &[(&str, String)], schema: &Value) -> Result<Value, String> {
    let bin = claude_code::find_claude().ok_or("Claude Code isn't installed")?;
    let (system, prompt) = flatten(messages);
    let mut cmd = tokio::process::Command::new(bin);
    cmd.arg("-p")
        .arg(&prompt)
        .args(["--output-format", "json", "--tools", "", "--strict-mcp-config", "--no-session-persistence"])
        .arg("--json-schema")
        .arg(schema.to_string())
        .arg("--system-prompt")
        .arg(if system.is_empty() { "Reply only with the required structured output.".to_string() } else { system });
    if !model.is_empty() {
        cmd.args(["--model", model]);
    }
    // The subscription, not an API key that happens to be in the environment.
    for k in ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"] {
        cmd.env_remove(k);
    }
    let out = run_cli(cmd)?;
    let v = out
        .lines()
        .rev()
        .chain(std::iter::once(out.as_str()))
        .find_map(|l| serde_json::from_str::<Value>(l.trim()).ok().filter(|v| v.get("type").is_some() || v.get("result").is_some()))
        .ok_or_else(|| format!("Claude Code gave no result: {}", out.chars().take(300).collect::<String>()))?;
    if v["is_error"].as_bool().unwrap_or(false) {
        return Err(format!("Claude Code: {}", v["result"].as_str().unwrap_or("error")));
    }
    match v.get("structured_output") {
        Some(so) if !so.is_null() => Ok(so.clone()),
        _ => Ok(Value::String(v["result"].as_str().unwrap_or("").to_string())),
    }
}

/// `codex exec` on the user's ChatGPT subscription, read-only sandbox, the
/// final message constrained by `--output-schema`.
fn codex_json(model: &str, messages: &[(&str, String)], schema: &Value) -> Result<Value, String> {
    let bin = codex::find_codex().ok_or("the Codex CLI isn't installed")?;
    let (system, prompt) = flatten(messages);
    let dir = std::env::temp_dir().join("unsound-codex");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let stamp = format!("{}-{:x}", std::process::id(), crate::util::now_ms());
    let schema_file = dir.join(format!("schema-{stamp}.json"));
    let out_file = dir.join(format!("out-{stamp}.txt"));
    std::fs::write(&schema_file, schema.to_string()).map_err(|e| e.to_string())?;
    let full = if system.is_empty() {
        format!("{prompt}\n\nDon't run commands; just answer.")
    } else {
        format!("{system}\n\n{prompt}\n\nDon't run commands; just answer.")
    };
    let mut cmd = tokio::process::Command::new(bin);
    cmd.args(["exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only"])
        .arg("--output-schema")
        .arg(&schema_file)
        .arg("-o")
        .arg(&out_file);
    if !model.is_empty() {
        cmd.args(["-m", model]);
    }
    cmd.arg(&full).current_dir(&dir).env_remove("OPENAI_API_KEY");
    let result = run_cli(cmd);
    let answer = std::fs::read_to_string(&out_file);
    let _ = std::fs::remove_file(&schema_file);
    let _ = std::fs::remove_file(&out_file);
    match answer {
        Ok(text) if !text.trim().is_empty() => Ok(Value::String(text)),
        _ => Err(format!("Codex gave no answer{}", result.err().map(|e| format!(": {e}")).unwrap_or_default())),
    }
}

/// Run a CLI with the login PATH, returning stdout (5-minute limit).
fn run_cli(mut cmd: tokio::process::Command) -> Result<String, String> {
    cmd.env("PATH", shell_env::login_path())
        .env("NODE_OPTIONS", "")
        .env("NODE_EXTRA_CA_CERTS", "")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    tauri::async_runtime::block_on(async move {
        let child = cmd.spawn().map_err(|e| format!("couldn't start: {e}"))?;
        let out = tokio::time::timeout(Duration::from_secs(300), child.wait_with_output())
            .await
            .map_err(|_| "didn't finish within 5 minutes".to_string())?
            .map_err(|e| e.to_string())?;
        let stdout = String::from_utf8_lossy(&out.stdout).to_string();
        if !out.status.success() && stdout.trim().is_empty() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let tail: String = stderr.chars().rev().take(400).collect::<Vec<_>>().into_iter().rev().collect();
            return Err(format!("exited ({}): {tail}", out.status));
        }
        Ok(stdout)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strictifies_and_round_trips_free_form_objects() {
        let schema = json!({
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": ["a.b"]},
                "arguments": {"type": "object"},
                "items": {"type": "array", "maxItems": 3, "items": {"type": "object", "properties": {"x": {"type": "string"}}}}
            },
            "required": ["tool"]
        });
        let (strict, encoded) = strictify(&schema);
        assert_eq!(strict["additionalProperties"], false);
        assert_eq!(strict["required"].as_array().unwrap().len(), 3);
        assert_eq!(strict["properties"]["arguments"]["type"], "string");
        assert_eq!(strict["properties"]["items"]["items"]["additionalProperties"], false);
        assert!(strict["properties"]["items"].get("maxItems").is_none());
        assert_eq!(encoded, vec![vec!["arguments".to_string()]]);
        let mut reply = json!({"tool": "a.b", "arguments": "{\"url\": \"https://x\"}", "items": []});
        decode_encoded(&mut reply, &encoded);
        assert_eq!(reply["arguments"]["url"], "https://x");
    }

    /// Both subscription CLIs return schema-valid JSON, free-form objects
    /// included. `cargo test --lib -- --ignored subscription_json --nocapture`
    #[test]
    #[ignore]
    fn subscription_json() {
        let schema = json!({
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "action": {"type": "string", "enum": ["call", "finish"]},
                "tool": {"type": "string", "enum": ["browser.navigate", "browser.click"]},
                "arguments": {"type": "object"},
                "answer": {"type": "string"}
            },
            "required": ["thought", "action", "tool", "arguments", "answer"]
        });
        let messages = [
            ("system", "You operate a browser with tools browser.navigate {url} and browser.click {ref}. Reply with one JSON step.".to_string()),
            ("user", "Task: open https://en.wikipedia.org".to_string()),
        ];
        let (strict, encoded) = strictify(&schema);
        for (name, raw) in [
            ("claude", claude_json("haiku", &messages, &strict)),
            ("codex", codex_json("", &messages, &strict)),
        ] {
            let t = std::time::Instant::now();
            let mut v = match raw.unwrap() {
                Value::String(text) => llm::parse_json_reply(&text).unwrap(),
                v => v,
            };
            decode_encoded(&mut v, &encoded);
            println!("{name} ({:.1}s): {v}", t.elapsed().as_secs_f32());
            assert_eq!(v["tool"], "browser.navigate");
            assert!(v["arguments"]["url"].as_str().unwrap_or("").contains("wikipedia"));
        }
    }

    /// The Anthropic API path with a real key from the environment.
    /// `ANTHROPIC_API_KEY=… cargo test --lib -- --ignored anthropic_api_json --nocapture`
    #[test]
    #[ignore]
    fn anthropic_api_json() {
        let key = std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY");
        let schema = json!({
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["call", "finish"]},
                "tool": {"type": "string", "enum": ["browser.navigate", "browser.click"]},
                "arguments": {"type": "object"}
            },
            "required": ["action", "tool", "arguments"]
        });
        let (strict, encoded) = strictify(&schema);
        let messages = [
            ("system", "You operate a browser with tools browser.navigate {url} and browser.click {ref}. Reply with one JSON step.".to_string()),
            ("user", "Task: open https://en.wikipedia.org".to_string()),
        ];
        let t = std::time::Instant::now();
        let mut v = match anthropic_json(&key, ANTHROPIC_DEFAULT_MODEL, &messages, &strict).unwrap() {
            Value::String(text) => llm::parse_json_reply(&text).unwrap(),
            v => v,
        };
        decode_encoded(&mut v, &encoded);
        println!("anthropic ({:.1}s): {v}", t.elapsed().as_secs_f32());
        assert!(v["arguments"]["url"].as_str().unwrap_or("").contains("wikipedia"));
    }

    /// The OpenAI-compatible path against a local llama-server.
    /// `COMPAT_URL=http://localhost:8089/v1 cargo test --lib -- --ignored compatible_json --nocapture`
    #[test]
    #[ignore]
    fn compatible_json() {
        let base = std::env::var("COMPAT_URL").expect("set COMPAT_URL");
        let schema = json!({
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["call", "finish"]},
                "tool": {"type": "string", "enum": ["browser.navigate", "browser.click"]},
                "arguments": {"type": "object"}
            },
            "required": ["action", "tool", "arguments"]
        });
        let (strict, encoded) = strictify(&schema);
        let messages = [
            ("system", "You operate a browser with tools browser.navigate {url} and browser.click {ref}. Reply with one JSON step.".to_string()),
            ("user", "Task: open https://en.wikipedia.org".to_string()),
        ];
        let models = list_models("compatible", &base).unwrap();
        println!("models: {models:?}");
        let _ = (&strict, &encoded);
        let v = match openai_chat(&base, None, &models[0], &messages, &schema, false, 500).unwrap() {
            Value::String(text) => llm::parse_json_reply(&text).unwrap(),
            v => v,
        };
        println!("compatible: {v}");
        // The format is what's under test; small models vary in judgment.
        assert_eq!(v["tool"], "browser.navigate");
        assert!(v["arguments"].is_object());
    }

    #[test]
    fn flattens_conversations_for_clis() {
        let (sys, prompt) = flatten(&[("system", "S".into()), ("user", "U".into())]);
        assert_eq!((sys.as_str(), prompt.as_str()), ("S", "U"));
        let (_, prompt) = flatten(&[("user", "A".into()), ("assistant", "B".into()), ("user", "C".into())]);
        assert!(prompt.starts_with("User:\nA") && prompt.contains("You (earlier):\nB"));
    }
}
