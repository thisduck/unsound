//! A Model Context Protocol client, so the assistant can use the same tool
//! servers people already run for Claude (Atlassian, Linear, GitHub, Slack…).
//! Stdio servers are spawned with the login-shell PATH (that's where `npx` /
//! `uvx` live); streamable-HTTP servers take static headers. OAuth-only remote
//! servers work through a stdio bridge such as `npx mcp-remote <url>`.

use super::shell_env;
use rmcp::model::{CallToolRequestParams, ClientCapabilities, ClientConfig, Implementation};
use rmcp::service::RunningService;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::process::Stdio;
use std::time::Duration;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
const CALL_TIMEOUT: Duration = Duration::from_secs(60);
/// Tool output fed back to a local model has to stay small.
const MAX_RESULT_CHARS: usize = 20_000;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct McpServer {
    pub name: String,
    /// Stdio server when non-empty.
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    /// Streamable-HTTP server when non-empty (and `command` is empty).
    pub url: String,
    pub headers: HashMap<String, String>,
    pub enabled: bool,
}

// Hand-written so a server added without the flag is on, not silently off.
impl Default for McpServer {
    fn default() -> Self {
        Self {
            name: String::new(),
            command: String::new(),
            args: Vec::new(),
            env: HashMap::new(),
            url: String::new(),
            headers: HashMap::new(),
            enabled: true,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolInfo {
    pub server: String,
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

type Client = RunningService<RoleClient, ClientConfig>;

/// Live connections to the configured servers, plus the tools they offer
/// (listed once at connect).
pub struct McpSession {
    clients: Vec<(String, Client)>,
    tools: Vec<ToolInfo>,
}

impl McpSession {
    /// Connect to every enabled server in parallel. A server that fails is
    /// reported in the second value and skipped — one broken config must not
    /// take the whole assistant down.
    pub async fn connect(servers: &[McpServer]) -> (Self, Vec<String>) {
        let enabled: Vec<&McpServer> = servers.iter().filter(|s| s.enabled).collect();
        let results = futures_util::future::join_all(enabled.iter().map(|s| async move {
            let r = connect_one(s).await;
            (s.name.clone(), r)
        }))
        .await;
        let mut clients = Vec::new();
        let mut tools = Vec::new();
        let mut errors = Vec::new();
        for (name, r) in results {
            match r {
                Ok((client, list)) => {
                    tools.extend(list);
                    clients.push((name, client));
                }
                Err(e) => errors.push(format!("{name}: {e}")),
            }
        }
        (Self { clients, tools }, errors)
    }

    pub async fn tools(&self) -> Vec<ToolInfo> {
        self.tools.clone()
    }

    /// Call `tool` on `server`. Text content is joined; images, audio and
    /// resources are summarized; a tool-reported error becomes `Err`.
    pub async fn call(&self, server: &str, tool: &str, args: Value) -> Result<String, String> {
        let (_, client) = self
            .clients
            .iter()
            .find(|(n, _)| n == server)
            .ok_or_else(|| format!("no connected MCP server named '{server}'"))?;
        let mut params = CallToolRequestParams::new(tool.to_string());
        match args {
            Value::Null => {}
            Value::Object(map) => params = params.with_arguments(map),
            other => return Err(format!("tool arguments must be a JSON object, got {other}")),
        }
        let result = tokio::time::timeout(CALL_TIMEOUT, client.call_tool(params))
            .await
            .map_err(|_| format!("{server}/{tool} timed out after {}s", CALL_TIMEOUT.as_secs()))?
            .map_err(|e| format!("{server}/{tool} failed: {e}"))?;
        let mut parts: Vec<String> = Vec::new();
        for block in &result.content {
            let v = serde_json::to_value(block).unwrap_or(Value::Null);
            parts.push(describe_block(&v));
        }
        if parts.iter().all(|p| p.is_empty()) {
            if let Some(sc) = &result.structured_content {
                parts = vec![sc.to_string()];
            }
        }
        let text = truncate(parts.join("\n").trim());
        if result.is_error.unwrap_or(false) {
            Err(if text.is_empty() {
                format!("{server}/{tool} reported an error")
            } else {
                text
            })
        } else {
            Ok(text)
        }
    }

    /// Shut every server down (stdio children are reaped).
    pub async fn close(self) {
        for (_, client) in self.clients {
            let _ = client.cancel().await;
        }
    }
}

/// Connect to one server, list its tools, disconnect — for a settings "test"
/// button.
pub async fn probe(server: &McpServer) -> Result<Vec<ToolInfo>, String> {
    let (client, tools) = connect_one(server).await?;
    let _ = client.cancel().await;
    Ok(tools)
}

async fn connect_one(s: &McpServer) -> Result<(Client, Vec<ToolInfo>), String> {
    let config = ClientConfig::new(
        ClientCapabilities::default(),
        Implementation::new("unsound", env!("CARGO_PKG_VERSION")),
    );
    let client = if !s.command.trim().is_empty() {
        let program = shell_env::which(s.command.trim())
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| s.command.trim().to_string());
        let mut cmd = tokio::process::Command::new(&program);
        cmd.args(&s.args)
            .env("PATH", shell_env::login_path())
            .envs(&s.env)
            .kill_on_drop(true);
        // Server logs go to our stderr (useful when a server misbehaves), and
        // a piped-but-unread stderr could otherwise fill and stall the child.
        let (transport, _stderr) = TokioChildProcess::builder(cmd)
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("could not start '{program}': {e}"))?;
        tokio::time::timeout(CONNECT_TIMEOUT, config.serve(transport))
            .await
            .map_err(|_| "timed out connecting".to_string())?
            .map_err(|e| format!("handshake failed: {e}"))?
    } else if !s.url.trim().is_empty() {
        let mut headers = HashMap::new();
        for (k, v) in &s.headers {
            let name = reqwest::header::HeaderName::from_bytes(k.trim().as_bytes())
                .map_err(|_| format!("invalid header name '{k}'"))?;
            let value = reqwest::header::HeaderValue::from_str(v.trim())
                .map_err(|_| format!("invalid value for header '{k}'"))?;
            headers.insert(name, value);
        }
        let cfg = StreamableHttpClientTransportConfig::with_uri(s.url.trim().to_string())
            .custom_headers(headers);
        let transport = StreamableHttpClientTransport::from_config(cfg);
        tokio::time::timeout(CONNECT_TIMEOUT, config.serve(transport))
            .await
            .map_err(|_| "timed out connecting".to_string())?
            .map_err(|e| format!("handshake failed: {e}"))?
    } else {
        return Err("needs either a command or a url".into());
    };
    let tools = tokio::time::timeout(CONNECT_TIMEOUT, client.list_all_tools())
        .await
        .map_err(|_| "timed out listing tools".to_string())?
        .map_err(|e| format!("could not list tools: {e}"))?;
    let tools = tools
        .into_iter()
        .map(|t| ToolInfo {
            server: s.name.clone(),
            name: t.name.to_string(),
            description: t.description.map(|d| d.to_string()).unwrap_or_default(),
            input_schema: Value::Object((*t.input_schema).clone()),
        })
        .collect();
    Ok((client, tools))
}

/// One content block as text. Works off the JSON form so new block kinds in
/// later protocol versions degrade to a label instead of failing to compile.
fn describe_block(v: &Value) -> String {
    match v["type"].as_str().unwrap_or("") {
        "text" => v["text"].as_str().unwrap_or("").to_string(),
        "image" => format!("[image: {}]", v["mimeType"].as_str().unwrap_or("unknown type")),
        "audio" => format!("[audio: {}]", v["mimeType"].as_str().unwrap_or("unknown type")),
        "resource" => {
            let r = &v["resource"];
            match r["text"].as_str() {
                Some(t) => t.to_string(),
                None => format!("[resource: {}]", r["uri"].as_str().unwrap_or("?")),
            }
        }
        "resource_link" | "resourceLink" => format!(
            "[link: {}{}]",
            v["name"].as_str().map(|n| format!("{n} — ")).unwrap_or_default(),
            v["uri"].as_str().unwrap_or("?")
        ),
        other if !other.is_empty() => format!("[{other} content]"),
        _ => String::new(),
    }
}

fn truncate(s: &str) -> String {
    if s.chars().count() <= MAX_RESULT_CHARS {
        return s.to_string();
    }
    let mut t: String = s.chars().take(MAX_RESULT_CHARS).collect();
    t.push_str("\n…[truncated]");
    t
}

/// Parse a Claude-style MCP config (`{"mcpServers": {name: {...}}}`, or the
/// bare server map). Entries may be stdio (`command`/`args`/`env`) or remote
/// (`url` + `headers`, with an optional `type` of http/sse/streamable-http).
/// `"disabled": true` is honored.
pub fn parse_mcp_json(json: &str) -> Result<Vec<McpServer>, String> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("invalid JSON: {e}"))?;
    let map = match v.get("mcpServers") {
        Some(m) => m,
        None => &v,
    }
    .as_object()
    .ok_or("expected an object of servers (e.g. {\"mcpServers\": {...}})")?;
    let strings = |x: &Value| -> HashMap<String, String> {
        x.as_object()
            .map(|o| {
                o.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())))
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut out = Vec::new();
    for (name, entry) in map {
        let obj = entry
            .as_object()
            .ok_or_else(|| format!("server '{name}' must be an object"))?;
        let command = obj.get("command").and_then(Value::as_str).unwrap_or("").to_string();
        let url = obj
            .get("url")
            .or_else(|| obj.get("serverUrl"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if command.is_empty() && url.is_empty() {
            return Err(format!("server '{name}' needs a \"command\" or a \"url\""));
        }
        let args = obj
            .get("args")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        out.push(McpServer {
            name: name.clone(),
            command,
            args,
            env: obj.get("env").map(strings).unwrap_or_default(),
            url,
            headers: obj.get("headers").map(strings).unwrap_or_default(),
            enabled: !obj.get("disabled").and_then(Value::as_bool).unwrap_or(false),
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_claude_style_config() {
        let servers = parse_mcp_json(
            r#"{"mcpServers": {
                "atlassian": {"command": "npx", "args": ["-y", "mcp-remote", "https://mcp.atlassian.com/v1/sse"]},
                "linear": {"type": "http", "url": "https://mcp.linear.app/mcp", "headers": {"Authorization": "Bearer x"}},
                "local": {"command": "uvx", "args": ["thing"], "env": {"TOKEN": "t", "N": 3}, "disabled": true}
            }}"#,
        )
        .unwrap();
        assert_eq!(servers.len(), 3);
        let a = &servers[0];
        assert_eq!((a.name.as_str(), a.command.as_str()), ("atlassian", "npx"));
        assert_eq!(a.args, vec!["-y", "mcp-remote", "https://mcp.atlassian.com/v1/sse"]);
        assert!(a.enabled);
        let l = &servers[1];
        assert_eq!(l.url, "https://mcp.linear.app/mcp");
        assert_eq!(l.headers["Authorization"], "Bearer x");
        let local = &servers[2];
        assert!(!local.enabled);
        assert_eq!(local.env["N"], "3");
    }

    #[test]
    fn parses_bare_map_and_rejects_bad_entries() {
        let s = parse_mcp_json(r#"{"x": {"url": "http://localhost:3000/mcp"}}"#).unwrap();
        assert_eq!(s[0].url, "http://localhost:3000/mcp");
        assert!(parse_mcp_json(r#"{"mcpServers": {"x": {}}}"#).is_err());
        assert!(parse_mcp_json("[1]").is_err());
        assert!(parse_mcp_json("nope").is_err());
    }

    #[test]
    fn stored_server_without_flag_is_enabled() {
        let s: McpServer = serde_json::from_str(r#"{"name":"a","command":"x"}"#).unwrap();
        assert!(s.enabled);
    }

    #[test]
    fn describes_blocks() {
        assert_eq!(describe_block(&serde_json::json!({"type":"text","text":"hi"})), "hi");
        assert_eq!(
            describe_block(&serde_json::json!({"type":"image","data":"..","mimeType":"image/png"})),
            "[image: image/png]"
        );
        assert_eq!(
            describe_block(&serde_json::json!({"type":"resource","resource":{"uri":"a://b","text":"body"}})),
            "body"
        );
    }

    /// Live check against the reference server (needs node/npx + network):
    /// `cargo test --lib tools::mcp -- --ignored --nocapture`
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn everything_server_lists_and_calls_echo() {
        let server = McpServer {
            name: "everything".into(),
            command: "npx".into(),
            args: vec!["-y".into(), "@modelcontextprotocol/server-everything".into()],
            ..Default::default()
        };
        let tools = probe(&server).await.expect("probe");
        eprintln!("probe: {} tools: {:?}", tools.len(), tools.iter().map(|t| &t.name).collect::<Vec<_>>());
        assert!(tools.iter().any(|t| t.name == "echo"));

        let (session, errors) = McpSession::connect(&[server]).await;
        assert!(errors.is_empty(), "{errors:?}");
        let out = session
            .call("everything", "echo", serde_json::json!({"message": "hello from unsound"}))
            .await
            .expect("echo");
        eprintln!("echo → {out}");
        assert!(out.contains("hello from unsound"));
        let image = session
            .call("everything", "get-tiny-image", serde_json::json!({}))
            .await
            .expect("image");
        eprintln!("get-tiny-image → {image}");
        assert!(image.contains("[image: image/png]"));
        let links = session
            .call("everything", "get-resource-links", serde_json::json!({"count": 2}))
            .await
            .expect("links");
        eprintln!("get-resource-links → {links}");
        assert!(links.contains("[link:"));
        let bad = session.call("everything", "echo", serde_json::json!(["not", "an", "object"])).await;
        assert!(bad.is_err());
        let missing = session.call("nope", "echo", Value::Null).await;
        assert!(missing.is_err());
        session.close().await;
    }
}
