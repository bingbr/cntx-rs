use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::json;

use super::tools::{ToolExecution, check_timeout, finish_tool, parse_input, validate_relative_path};
use super::types::{AgentWorkspace, ToolCall, ToolDefinition, ToolName};

use std::time::Instant;

const MAX_GIT_LOG_ENTRIES: usize = 100;

#[derive(Debug, Deserialize)]
struct GitDiffInput {
    rev_a: String,
    #[serde(default)]
    rev_b: Option<String>,
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GitShowInput {
    rev: String,
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GitLogInput {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

pub fn git_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: ToolName::GitStatusReadonly,
            description: "Read the current git status for the prepared workspace.",
            safe_to_parallelize: false,
            input_schema: json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: ToolName::GitDiff,
            description: "Read a git diff between two revisions, optionally restricted to one path.",
            safe_to_parallelize: false,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "rev_a": { "type": "string" },
                    "rev_b": { "type": "string" },
                    "path": { "type": "string" }
                },
                "required": ["rev_a"]
            }),
        },
        ToolDefinition {
            name: ToolName::GitShow,
            description: "Read a git object or revision, optionally restricted to one path.",
            safe_to_parallelize: false,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "rev": { "type": "string" },
                    "path": { "type": "string" }
                },
                "required": ["rev"]
            }),
        },
        ToolDefinition {
            name: ToolName::GitLog,
            description: "Read recent git history for the workspace or one path.",
            safe_to_parallelize: false,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": MAX_GIT_LOG_ENTRIES }
                }
            }),
        },
    ]
}

pub fn execute_git_tool(workspace: &AgentWorkspace, call: &ToolCall, started: Instant) -> Result<ToolExecution> {
    match call.tool {
        ToolName::GitStatusReadonly => git_status_readonly(workspace, started),
        ToolName::GitDiff => git_diff(workspace, parse_input::<GitDiffInput>(call)?, started),
        ToolName::GitShow => git_show(workspace, parse_input::<GitShowInput>(call)?, started),
        ToolName::GitLog => git_log(workspace, parse_input::<GitLogInput>(call)?, started),
        _ => bail!("not a git tool: {}", call.tool),
    }
}

fn git_status_readonly(workspace: &AgentWorkspace, started: Instant) -> Result<ToolExecution> {
    check_timeout(started)?;
    let output = run_git_readonly(workspace, &["status", "--short", "--branch"])?;
    Ok(finish_tool(output, Vec::new(), false))
}

fn git_diff(workspace: &AgentWorkspace, input: GitDiffInput, started: Instant) -> Result<ToolExecution> {
    check_timeout(started)?;
    let mut args = vec!["diff", "--no-ext-diff", "--no-textconv"];
    args.push(input.rev_a.as_str());
    if let Some(rev_b) = input.rev_b.as_deref() {
        args.push(rev_b);
    }
    if let Some(path) = input.path.as_deref() {
        validate_git_relative_path(path)?;
        args.push("--");
        args.push(path);
    }
    let output = run_git_readonly(workspace, &args)?;
    Ok(finish_tool(output, Vec::new(), false))
}

fn git_show(workspace: &AgentWorkspace, input: GitShowInput, started: Instant) -> Result<ToolExecution> {
    check_timeout(started)?;
    let mut args = vec!["show", "--no-ext-diff", "--no-textconv", input.rev.as_str()];
    if let Some(path) = input.path.as_deref() {
        validate_git_relative_path(path)?;
        args.push("--");
        args.push(path);
    }
    let output = run_git_readonly(workspace, &args)?;
    Ok(finish_tool(output, Vec::new(), false))
}

fn git_log(workspace: &AgentWorkspace, input: GitLogInput, started: Instant) -> Result<ToolExecution> {
    check_timeout(started)?;
    let limit = input.limit.unwrap_or(20).clamp(1, MAX_GIT_LOG_ENTRIES);
    let limit_string = limit.to_string();
    let mut args = vec!["log", "--oneline", "--decorate", "-n", limit_string.as_str()];
    if let Some(path) = input.path.as_deref() {
        validate_git_relative_path(path)?;
        args.push("--");
        args.push(path);
    }
    let output = run_git_readonly(workspace, &args)?;
    Ok(finish_tool(output, Vec::new(), false))
}

fn run_git_readonly(workspace: &AgentWorkspace, args: &[&str]) -> Result<String> {
    let output = build_git_readonly_command(workspace, args)
        .output()
        .with_context(|| "Starting git for read-only inspection")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = if !stderr.is_empty() {
            stderr
        } else if !stdout.is_empty() {
            stdout
        } else {
            format!("exit status {}", output.status)
        };
        bail!("git inspection failed: {detail}");
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

pub fn build_git_readonly_command(workspace: &AgentWorkspace, args: &[&str]) -> Command {
    let mut command = Command::new("git");
    command.arg("--no-pager").arg("-C").arg(&workspace.root).args(args);
    command
}

fn validate_git_relative_path(raw: &str) -> Result<()> {
    let _ = validate_relative_path(raw)?;
    Ok(())
}
