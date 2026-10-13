//! What the meeting assistant can actually *do* once it has drafted something:
//! local Mac actions (`apple`), Jira tickets (`jira`), any MCP tool server
//! (`mcp`), or a hand-off to Claude Code for open-ended work (`claude_code`).
//! Everything here runs only after the user approves the draft.

pub mod apple;
pub mod claude_code;
pub mod codex;
pub mod jira;
pub mod mcp;
pub mod shell_env;
