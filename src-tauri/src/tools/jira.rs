//! Jira Cloud (REST v3) — just enough to file the ticket a meeting asked for:
//! check the credentials, list projects for the settings picker, and create an
//! issue with an assignee resolved from a spoken name. Basic auth with the
//! user's email + API token (id.atlassian.com → Security → API tokens).

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

/// Keychain key for the API token (see `secrets`).
pub const TOKEN_KEY: &str = "jira-api-token";

const TIMEOUT: Duration = Duration::from_secs(30);
/// Jira rejects summaries over 255 characters.
const MAX_SUMMARY: usize = 255;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct JiraConfig {
    /// e.g. `https://acme.atlassian.net`
    pub site: String,
    pub email: String,
    pub default_project: String,
    /// e.g. "Task"
    pub default_issue_type: String,
}

#[derive(Clone, Debug, Default)]
pub struct IssueInput {
    pub project: String,
    pub issue_type: String,
    pub summary: String,
    /// Markdown-ish: paragraphs, "- " bullets, "1." lists, **bold**, `code`.
    pub description: String,
    pub priority: Option<String>,
    pub labels: Vec<String>,
    /// A name as spoken ("Priya"); resolved to an assignable account.
    pub assignee_query: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreatedIssue {
    pub key: String,
    pub url: String,
}

/// `https://acme.atlassian.net` from whatever the user pasted (no scheme,
/// trailing slash, a full issue URL…).
fn base_url(cfg: &JiraConfig) -> Result<String, String> {
    let raw = cfg.site.trim();
    if raw.is_empty() {
        return Err("set your Jira site (e.g. https://acme.atlassian.net) in Settings".into());
    }
    let with_scheme = if raw.contains("://") {
        raw.to_string()
    } else {
        format!("https://{raw}")
    };
    let url = reqwest::Url::parse(&with_scheme).map_err(|e| format!("invalid Jira site: {e}"))?;
    let host = url.host_str().ok_or("invalid Jira site: no host")?;
    Ok(match url.port() {
        Some(p) => format!("{}://{host}:{p}", url.scheme()),
        None => format!("{}://{host}", url.scheme()),
    })
}

fn client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(TIMEOUT)
        .build()
        .map_err(|e| e.to_string())
}

fn check_auth(cfg: &JiraConfig, token: &str) -> Result<(), String> {
    if cfg.email.trim().is_empty() || token.trim().is_empty() {
        return Err("add your Jira email and API token in Settings".into());
    }
    Ok(())
}

async fn get_json(cfg: &JiraConfig, token: &str, path_and_query: &str) -> Result<Value, String> {
    check_auth(cfg, token)?;
    let url = format!("{}{path_and_query}", base_url(cfg)?);
    let resp = client()?
        .get(&url)
        .basic_auth(cfg.email.trim(), Some(token.trim()))
        .header("Accept", "application/json")
        .send()
        .await
        .map_err(|e| format!("Jira request failed: {e}"))?;
    read_json(resp).await
}

async fn read_json(resp: reqwest::Response) -> Result<Value, String> {
    let status = resp.status();
    let body = resp.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(match status.as_u16() {
            401 => "Jira rejected the credentials (check the email and API token)".into(),
            403 => format!("Jira denied access: {}", error_text(&body)),
            404 => format!("Jira returned not found: {}", error_text(&body)),
            _ => format!("Jira error ({status}): {}", error_text(&body)),
        });
    }
    if body.trim().is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_str(&body).map_err(|e| format!("Jira returned invalid JSON: {e}"))
}

/// Flatten Jira's `{errorMessages: [...], errors: {field: msg}}` into one line.
fn error_text(body: &str) -> String {
    let Ok(v) = serde_json::from_str::<Value>(body) else {
        return body.chars().take(300).collect();
    };
    let mut parts: Vec<String> = v["errorMessages"]
        .as_array()
        .map(|a| a.iter().filter_map(|m| m.as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    if let Some(errs) = v["errors"].as_object() {
        for (field, msg) in errs {
            parts.push(format!("{field}: {}", msg.as_str().unwrap_or(&msg.to_string())));
        }
    }
    if parts.is_empty() {
        body.chars().take(300).collect()
    } else {
        parts.join("; ")
    }
}

/// Verify the credentials; returns the authenticated user's display name.
pub async fn test(cfg: &JiraConfig, token: &str) -> Result<String, String> {
    let me = get_json(cfg, token, "/rest/api/3/myself").await?;
    Ok(me["displayName"]
        .as_str()
        .or_else(|| me["emailAddress"].as_str())
        .unwrap_or("(unknown user)")
        .to_string())
}

/// All projects visible to the user, as `(key, name)`, for the settings picker.
pub async fn list_projects(cfg: &JiraConfig, token: &str) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    let mut start = 0usize;
    // Bounded: a runaway `isLast: false` must not loop forever.
    for _ in 0..20 {
        let page = get_json(
            cfg,
            token,
            &format!("/rest/api/3/project/search?startAt={start}&maxResults=100&orderBy=name"),
        )
        .await?;
        let values = page["values"].as_array().cloned().unwrap_or_default();
        for p in &values {
            if let (Some(k), Some(n)) = (p["key"].as_str(), p["name"].as_str()) {
                out.push((k.to_string(), n.to_string()));
            }
        }
        if values.is_empty() || page["isLast"].as_bool().unwrap_or(true) {
            break;
        }
        start += values.len();
    }
    Ok(out)
}

/// The account id of the best assignable match for a spoken name: an exact
/// display-name match wins, then a prefix match, then Jira's first result.
async fn resolve_assignee(
    cfg: &JiraConfig,
    token: &str,
    project: &str,
    query: &str,
) -> Result<Option<(String, String)>, String> {
    let q = query.trim();
    if q.is_empty() {
        return Ok(None);
    }
    let path = format!(
        "/rest/api/3/user/assignable/search?project={}&query={}&maxResults=10",
        pct(project),
        pct(q)
    );
    let users = get_json(cfg, token, &path).await?;
    let users: Vec<(String, String)> = users
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|u| {
                    Some((
                        u["accountId"].as_str()?.to_string(),
                        u["displayName"].as_str().unwrap_or("").to_string(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default();
    let ql = q.to_lowercase();
    let pick = users
        .iter()
        .find(|(_, n)| n.to_lowercase() == ql)
        .or_else(|| users.iter().find(|(_, n)| n.to_lowercase().starts_with(&ql)))
        .or_else(|| users.first());
    Ok(pick.cloned())
}

/// Create the issue. Optional fields the project's create screen doesn't
/// allow (priority, labels, assignee) are dropped and retried rather than
/// failing the whole ticket.
pub async fn create_issue(
    cfg: &JiraConfig,
    token: &str,
    input: &IssueInput,
) -> Result<CreatedIssue, String> {
    check_auth(cfg, token)?;
    let base = base_url(cfg)?;
    let project = first_non_empty(&input.project, &cfg.default_project)
        .ok_or("no Jira project — say which project, or set a default in Settings")?;
    let issue_type = first_non_empty(&input.issue_type, &cfg.default_issue_type)
        .unwrap_or_else(|| "Task".to_string());
    let summary = clean_summary(&input.summary);
    if summary.is_empty() {
        return Err("the ticket needs a summary".into());
    }

    let assignee = match input.assignee_query.as_deref() {
        Some(q) => match resolve_assignee(cfg, token, &project, q).await {
            Ok(found) => {
                if found.is_none() {
                    eprintln!("[jira] no assignable user matches '{q}'; leaving unassigned");
                }
                found
            }
            Err(e) => {
                eprintln!("[jira] assignee lookup for '{q}' failed: {e}; leaving unassigned");
                None
            }
        },
        None => None,
    };

    let mut fields = serde_json::Map::new();
    fields.insert("project".into(), json!({ "key": project }));
    fields.insert("issuetype".into(), json!({ "name": issue_type }));
    fields.insert("summary".into(), json!(summary));
    if !input.description.trim().is_empty() {
        fields.insert("description".into(), to_adf(&input.description));
    }
    if let Some(p) = input.priority.as_deref().map(str::trim).filter(|p| !p.is_empty()) {
        fields.insert("priority".into(), json!({ "name": p }));
    }
    let labels: Vec<String> = input
        .labels
        .iter()
        .map(|l| l.trim().replace(char::is_whitespace, "-"))
        .filter(|l| !l.is_empty())
        .collect();
    if !labels.is_empty() {
        fields.insert("labels".into(), json!(labels));
    }
    if let Some((id, name)) = &assignee {
        eprintln!("[jira] assigning to {name}");
        fields.insert("assignee".into(), json!({ "accountId": id }));
    }

    let http = client()?;
    // At most one retry per optional field.
    for _ in 0..4 {
        let resp = http
            .post(format!("{base}/rest/api/3/issue"))
            .basic_auth(cfg.email.trim(), Some(token.trim()))
            .header("Accept", "application/json")
            .json(&json!({ "fields": fields }))
            .send()
            .await
            .map_err(|e| format!("Jira request failed: {e}"))?;
        if resp.status().as_u16() == 400 {
            let body = resp.text().await.unwrap_or_default();
            let rejected: Vec<String> = serde_json::from_str::<Value>(&body)
                .ok()
                .and_then(|v| v["errors"].as_object().map(|o| o.keys().cloned().collect()))
                .unwrap_or_default();
            let droppable: Vec<String> = rejected
                .into_iter()
                .filter(|f| ["priority", "labels", "assignee"].contains(&f.as_str()))
                .filter(|f| fields.contains_key(f))
                .collect();
            if droppable.is_empty() {
                return Err(format!("Jira rejected the ticket: {}", error_text(&body)));
            }
            for f in droppable {
                eprintln!("[jira] project screen rejected '{f}'; retrying without it");
                fields.remove(&f);
            }
            continue;
        }
        let created = read_json(resp).await?;
        let key = created["key"]
            .as_str()
            .ok_or("Jira created the issue but returned no key")?
            .to_string();
        return Ok(CreatedIssue {
            url: format!("{base}/browse/{key}"),
            key,
        });
    }
    Err("Jira kept rejecting the ticket".into())
}

/// Browser fallback when there's no token: Jira's create screen with the
/// summary and description filled in. The prefill needs numeric project and
/// issue-type ids, which we only include when that's what we were given — the
/// user picks the rest on the page.
pub fn create_page_url(cfg: &JiraConfig, input: &IssueInput) -> String {
    let base = base_url(cfg).unwrap_or_else(|_| "https://atlassian.net".into());
    let mut q = vec![
        format!("summary={}", pct(&clean_summary(&input.summary))),
        format!("description={}", pct(input.description.trim())),
    ];
    let project = first_non_empty(&input.project, &cfg.default_project).unwrap_or_default();
    if !project.is_empty() && project.chars().all(|c| c.is_ascii_digit()) {
        q.push(format!("pid={project}"));
    }
    let issue_type = first_non_empty(&input.issue_type, &cfg.default_issue_type).unwrap_or_default();
    if !issue_type.is_empty() && issue_type.chars().all(|c| c.is_ascii_digit()) {
        q.push(format!("issuetype={issue_type}"));
    }
    if let Some(p) = input.priority.as_deref().filter(|p| !p.trim().is_empty()) {
        q.push(format!("priority={}", pct(p.trim())));
    }
    format!("{base}/secure/CreateIssueDetails!init.jspa?{}", q.join("&"))
}

fn first_non_empty(a: &str, b: &str) -> Option<String> {
    [a, b]
        .iter()
        .map(|s| s.trim())
        .find(|s| !s.is_empty())
        .map(str::to_string)
}

/// One line, no markdown heading marks, within Jira's length limit.
fn clean_summary(s: &str) -> String {
    let one_line = s
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("")
        .trim_start_matches('#')
        .trim()
        .to_string();
    if one_line.chars().count() > MAX_SUMMARY {
        let mut t: String = one_line.chars().take(MAX_SUMMARY - 1).collect();
        t.push('…');
        t
    } else {
        one_line
    }
}

/// Percent-encode a query component (RFC 3986 unreserved kept).
fn pct(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

// ---- Markdown → Atlassian Document Format ----------------------------------

/// Convert the draft's light markdown into ADF, which v3 requires for rich
/// fields. Handles headings, paragraphs (single newlines become hard breaks),
/// bullet/numbered lists, task-list boxes, fenced code, **bold**, *italic*,
/// and `code`. Anything else stays literal text rather than being dropped.
pub fn to_adf(markdown: &str) -> Value {
    let mut blocks: Vec<Value> = Vec::new();
    let mut para: Vec<String> = Vec::new();
    let mut list: Option<(bool, Vec<Value>)> = None; // (ordered, items)
    let mut code: Option<(String, Vec<String>)> = None; // (language, lines)

    fn flush_para(para: &mut Vec<String>, blocks: &mut Vec<Value>) {
        if para.is_empty() {
            return;
        }
        let mut content = Vec::new();
        for (i, line) in para.iter().enumerate() {
            if i > 0 {
                content.push(json!({ "type": "hardBreak" }));
            }
            content.extend(inline(line));
        }
        blocks.push(json!({ "type": "paragraph", "content": content }));
        para.clear();
    }
    fn flush_list(list: &mut Option<(bool, Vec<Value>)>, blocks: &mut Vec<Value>) {
        if let Some((ordered, items)) = list.take() {
            if !items.is_empty() {
                let kind = if ordered { "orderedList" } else { "bulletList" };
                blocks.push(json!({ "type": kind, "content": items }));
            }
        }
    }

    for raw in markdown.lines() {
        if let Some((lang, lines)) = code.as_mut() {
            if raw.trim_start().starts_with("```") {
                let mut node = json!({
                    "type": "codeBlock",
                    "content": [{ "type": "text", "text": lines.join("\n") }],
                });
                if lines.is_empty() {
                    node["content"] = json!([]);
                }
                if !lang.is_empty() {
                    node["attrs"] = json!({ "language": lang });
                }
                blocks.push(node);
                code = None;
            } else {
                lines.push(raw.to_string());
            }
            continue;
        }
        let line = raw.trim();
        if let Some(rest) = line.strip_prefix("```") {
            flush_para(&mut para, &mut blocks);
            flush_list(&mut list, &mut blocks);
            code = Some((rest.trim().to_string(), Vec::new()));
            continue;
        }
        if line.is_empty() {
            flush_para(&mut para, &mut blocks);
            flush_list(&mut list, &mut blocks);
            continue;
        }
        if let Some((level, text)) = heading(line) {
            flush_para(&mut para, &mut blocks);
            flush_list(&mut list, &mut blocks);
            blocks.push(json!({
                "type": "heading",
                "attrs": { "level": level },
                "content": inline(text),
            }));
            continue;
        }
        if let Some((ordered, text)) = list_item(line) {
            flush_para(&mut para, &mut blocks);
            if list.as_ref().is_some_and(|(o, _)| *o != ordered) {
                flush_list(&mut list, &mut blocks);
            }
            let (_, items) = list.get_or_insert((ordered, Vec::new()));
            let content = inline(text);
            let paragraph = if content.is_empty() {
                json!({ "type": "paragraph" })
            } else {
                json!({ "type": "paragraph", "content": content })
            };
            items.push(json!({ "type": "listItem", "content": [paragraph] }));
            continue;
        }
        flush_list(&mut list, &mut blocks);
        para.push(line.to_string());
    }
    if let Some((lang, lines)) = code.take() {
        // Unterminated fence: keep what we have.
        let mut node = json!({
            "type": "codeBlock",
            "content": [{ "type": "text", "text": lines.join("\n") }],
        });
        if !lang.is_empty() {
            node["attrs"] = json!({ "language": lang });
        }
        blocks.push(node);
    }
    flush_para(&mut para, &mut blocks);
    flush_list(&mut list, &mut blocks);
    json!({ "type": "doc", "version": 1, "content": blocks })
}

fn heading(line: &str) -> Option<(u8, &str)> {
    let hashes = line.chars().take_while(|&c| c == '#').count();
    if (1..=6).contains(&hashes) && line[hashes..].starts_with(' ') {
        Some((hashes as u8, line[hashes..].trim()))
    } else {
        None
    }
}

/// `- x`, `* x`, `+ x`, `- [ ] x` (task boxes flattened), or `1. x` / `1) x`.
fn list_item(line: &str) -> Option<(bool, &str)> {
    for marker in ["- ", "* ", "+ "] {
        if let Some(rest) = line.strip_prefix(marker) {
            let rest = rest
                .strip_prefix("[ ] ")
                .or_else(|| rest.strip_prefix("[x] "))
                .or_else(|| rest.strip_prefix("[X] "))
                .unwrap_or(rest);
            return Some((false, rest.trim()));
        }
    }
    let digits = line.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0 && digits <= 3 {
        let rest = &line[digits..];
        if let Some(r) = rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") ")) {
            return Some((true, r.trim()));
        }
    }
    None
}

/// Inline marks: `code`, **strong**, *em* / _em_. Unmatched markers stay text.
fn inline(text: &str) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    let mut plain = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    let flush = |plain: &mut String, out: &mut Vec<Value>| {
        if !plain.is_empty() {
            out.push(json!({ "type": "text", "text": std::mem::take(plain) }));
        }
    };
    while i < chars.len() {
        let rest: String = chars[i..].iter().collect();
        let span = if rest.starts_with('`') {
            closing(&chars, i + 1, "`").map(|end| (i + 1, end, 1, "code"))
        } else if rest.starts_with("**") {
            closing(&chars, i + 2, "**").map(|end| (i + 2, end, 2, "strong"))
        } else if rest.starts_with("__") {
            closing(&chars, i + 2, "__").map(|end| (i + 2, end, 2, "strong"))
        } else if (rest.starts_with('*') || rest.starts_with('_'))
            && chars.get(i + 1).is_some_and(|c| !c.is_whitespace())
            // Don't read snake_case or 2*3*4 as emphasis.
            && (i == 0 || !chars[i - 1].is_alphanumeric())
        {
            let m = &rest[..1];
            closing(&chars, i + 1, m).map(|end| (i + 1, end, 1, "em"))
        } else {
            None
        };
        match span {
            Some((start, end, width, mark)) if end > start => {
                flush(&mut plain, &mut out);
                let inner: String = chars[start..end].iter().collect();
                out.push(json!({ "type": "text", "text": inner, "marks": [{ "type": mark }] }));
                i = end + width;
            }
            _ => {
                plain.push(chars[i]);
                i += 1;
            }
        }
    }
    flush(&mut plain, &mut out);
    out
}

/// Index of the closing `marker` at or after `from`, if any.
fn closing(chars: &[char], from: usize, marker: &str) -> Option<usize> {
    let m: Vec<char> = marker.chars().collect();
    (from..chars.len().saturating_sub(m.len() - 1)).find(|&j| chars[j..j + m.len()] == m[..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adf_paragraphs_lists_and_marks() {
        let doc = to_adf(
            "Login fails on **Safari**.\nSeen by two users.\n\n- step `one`\n- [ ] step two\n\n1. first\n2. second",
        );
        assert_eq!(doc["type"], "doc");
        assert_eq!(doc["version"], 1);
        let c = doc["content"].as_array().unwrap();
        assert_eq!(c.len(), 3);
        assert_eq!(c[0]["type"], "paragraph");
        let p = c[0]["content"].as_array().unwrap();
        assert_eq!(p[0]["text"], "Login fails on ");
        assert_eq!(p[1]["text"], "Safari");
        assert_eq!(p[1]["marks"][0]["type"], "strong");
        assert_eq!(p[3]["type"], "hardBreak");
        assert_eq!(c[1]["type"], "bulletList");
        assert_eq!(c[1]["content"].as_array().unwrap().len(), 2);
        let item0 = &c[1]["content"][0]["content"][0]["content"];
        assert_eq!(item0[1]["marks"][0]["type"], "code");
        assert_eq!(c[1]["content"][1]["content"][0]["content"][0]["text"], "step two");
        assert_eq!(c[2]["type"], "orderedList");
    }

    #[test]
    fn adf_headings_code_and_literals() {
        let doc = to_adf("## Steps\n```sh\nnpm test\n```\nsnake_case and 2*3*4 stay literal");
        let c = doc["content"].as_array().unwrap();
        assert_eq!(c[0]["type"], "heading");
        assert_eq!(c[0]["attrs"]["level"], 2);
        assert_eq!(c[1]["type"], "codeBlock");
        assert_eq!(c[1]["attrs"]["language"], "sh");
        assert_eq!(c[1]["content"][0]["text"], "npm test");
        assert_eq!(c[2]["content"][0]["text"], "snake_case and 2*3*4 stay literal");
        assert_eq!(c[2]["content"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn adf_empty_is_valid_doc() {
        let doc = to_adf("");
        assert_eq!(doc["content"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn site_normalization_and_fallback_url() {
        let cfg = JiraConfig {
            site: "acme.atlassian.net/jira/software/projects/X/boards/1".into(),
            ..Default::default()
        };
        assert_eq!(base_url(&cfg).unwrap(), "https://acme.atlassian.net");
        let input = IssueInput {
            summary: "Fix login & signup".into(),
            description: "It breaks".into(),
            project: "10001".into(),
            ..Default::default()
        };
        let url = create_page_url(&cfg, &input);
        assert!(url.starts_with("https://acme.atlassian.net/secure/CreateIssueDetails!init.jspa?"));
        assert!(url.contains("summary=Fix%20login%20%26%20signup"));
        assert!(url.contains("pid=10001"));
    }

    #[test]
    fn summary_is_one_line_and_bounded() {
        assert_eq!(clean_summary("\n# Title here\nmore"), "Title here");
        let long = "x".repeat(400);
        assert_eq!(clean_summary(&long).chars().count(), MAX_SUMMARY);
    }
}
