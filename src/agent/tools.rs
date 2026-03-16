use std::{
    fmt::Write as _,
    fs,
    ops::ControlFlow,
    path::{Component, Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use globset::{Glob, GlobSetBuilder};
use ignore::WalkBuilder;
use serde::Deserialize;
use serde_json::json;

use super::types::{AgentWorkspace, ToolCall, ToolCitation, ToolDefinition, ToolName, ToolResult};

const TOOL_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_OUTPUT_BYTES: usize = 16 * 1024;
const MAX_FILE_BYTES: usize = 256 * 1024;
const MAX_READ_LINES: usize = 200;
const MAX_LIST_ENTRIES: usize = 200;
const MAX_GREP_MATCHES: usize = 50;
const MAX_GLOB_MATCHES: usize = 100;
const MAX_TREE_ENTRIES: usize = 400;
const MAX_TREE_DEPTH: usize = 8;
const MAX_READ_MANY_FILES: usize = 20;
const MAX_GIT_LOG_ENTRIES: usize = 100;

#[derive(Debug, Deserialize)]
struct ListInput {
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ReadInput {
    path: String,
    #[serde(default)]
    start_line: Option<usize>,
    #[serde(default)]
    end_line: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct GrepInput {
    query: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct GlobInput {
    pattern: String,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct TreeInput {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    depth: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ReadManyInput {
    files: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SearchInput {
    query: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    file_glob: Option<String>,
    #[serde(default)]
    regex: Option<bool>,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct StatInput {
    path: String,
}

#[derive(Debug, Deserialize)]
struct NameSearchInput {
    name: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

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

pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: ToolName::List,
            description: "List files and directories under a workspace path.",
            safe_to_parallelize: true,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path inside the workspace. Defaults to '.'."
                    }
                }
            }),
        },
        ToolDefinition {
            name: ToolName::Read,
            description: "Read a bounded line range from a text file inside the workspace.",
            safe_to_parallelize: true,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "start_line": { "type": "integer", "minimum": 1 },
                    "end_line": { "type": "integer", "minimum": 1 }
                },
                "required": ["path"]
            }),
        },
        ToolDefinition {
            name: ToolName::Grep,
            description: "Search for a plain-text query under a workspace path.",
            safe_to_parallelize: false,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "path": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": MAX_GREP_MATCHES }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: ToolName::Glob,
            description: "Find files using a glob pattern relative to the workspace root.",
            safe_to_parallelize: false,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": MAX_GLOB_MATCHES }
                },
                "required": ["pattern"]
            }),
        },
        ToolDefinition {
            name: ToolName::Tree,
            description: "List files and directories as an indented tree view.",
            safe_to_parallelize: true,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path inside the workspace to start from. Defaults to '.'."
                    },
                    "depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_TREE_DEPTH,
                        "description": "Maximum directory depth to recurse."
                    }
                }
            }),
        },
        ToolDefinition {
            name: ToolName::ReadMany,
            description: "Read multiple text files by name and emit a combined bounded output.",
            safe_to_parallelize: true,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_READ_MANY_FILES,
                        "items": { "type": "string" }
                    }
                },
                "required": ["files"]
            }),
        },
        ToolDefinition {
            name: ToolName::Search,
            description: "Search for text in files under a workspace path with an optional file glob filter.",
            safe_to_parallelize: false,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "path": { "type": "string" },
                    "file_glob": { "type": "string" },
                    "regex": { "type": "boolean" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": MAX_GREP_MATCHES }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: ToolName::Stat,
            description: "Inspect read-only metadata for a file or directory.",
            safe_to_parallelize: true,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        },
        ToolDefinition {
            name: ToolName::FindReferences,
            description: "Search for references to a symbol name.",
            safe_to_parallelize: false,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "path": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": MAX_GREP_MATCHES }
                },
                "required": ["name"]
            }),
        },
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

pub fn execute_tool(workspace: &AgentWorkspace, call: &ToolCall) -> Result<ToolResult> {
    let started = Instant::now();
    let result = match call.tool {
        ToolName::List => list(workspace, parse_input::<ListInput>(call)?, started),
        ToolName::Read => read(workspace, parse_input::<ReadInput>(call)?, started),
        ToolName::Grep => grep(workspace, parse_input::<GrepInput>(call)?, started),
        ToolName::Glob => glob(workspace, parse_input::<GlobInput>(call)?, started),
        ToolName::Tree => tree(workspace, parse_input::<TreeInput>(call)?, started),
        ToolName::ReadMany => read_many(workspace, parse_input::<ReadManyInput>(call)?, started),
        ToolName::Search => search(workspace, parse_input::<SearchInput>(call)?, started),
        ToolName::Stat => stat_path(workspace, parse_input::<StatInput>(call)?, started),
        ToolName::FindReferences => find_references(workspace, parse_input::<NameSearchInput>(call)?, started),
        ToolName::GitStatusReadonly => git_status_readonly(workspace, started),
        ToolName::GitDiff => git_diff(workspace, parse_input::<GitDiffInput>(call)?, started),
        ToolName::GitShow => git_show(workspace, parse_input::<GitShowInput>(call)?, started),
        ToolName::GitLog => git_log(workspace, parse_input::<GitLogInput>(call)?, started),
    }?;

    Ok(ToolResult {
        id: call.id.clone(),
        tool: call.tool,
        output: result.output,
        is_error: false,
        truncated: result.truncated,
        citations: result.citations,
    })
}

struct ToolExecution {
    output: String,
    truncated: bool,
    citations: Vec<ToolCitation>,
}

fn finish_tool(output: String, citations: Vec<ToolCitation>, truncated: bool) -> ToolExecution {
    let (output, output_truncated) = truncate_output(output);
    ToolExecution {
        output,
        truncated: truncated || output_truncated,
        citations,
    }
}

fn list(workspace: &AgentWorkspace, input: ListInput, started: Instant) -> Result<ToolExecution> {
    let path = resolve_existing_path(workspace, input.path.as_deref().unwrap_or("."))?;
    if !path.is_dir() {
        bail!("path '{}' is not a directory", display_path(workspace, &path));
    }

    let mut entries = fs::read_dir(&path)
        .with_context(|| format!("Listing '{}'", display_path(workspace, &path)))?
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("Listing '{}'", display_path(workspace, &path)))?;
    entries.sort_by_key(|entry| entry.file_name());

    let mut output = String::new();
    writeln!(output, "path: {}", display_path(workspace, &path)).expect("write to string");
    for (index, entry) in entries.iter().enumerate() {
        check_timeout(started)?;
        if index >= MAX_LIST_ENTRIES {
            output.push_str("... truncated\n");
            return Ok(finish_tool(output, Vec::new(), true));
        }

        let child_path = entry.path();
        let file_type = entry.file_type()?;
        let kind = if file_type.is_dir() {
            "dir"
        } else if file_type.is_file() {
            "file"
        } else if file_type.is_symlink() {
            "symlink"
        } else {
            "other"
        };
        writeln!(output, "{kind}\t{}", display_path(workspace, &child_path)).expect("write to string");
    }

    Ok(finish_tool(output, Vec::new(), false))
}

fn read(workspace: &AgentWorkspace, input: ReadInput, started: Instant) -> Result<ToolExecution> {
    let (path, text) = read_workspace_text_file(workspace, &input.path)?;
    let lines = text.lines().collect::<Vec<_>>();
    let start_line = input.start_line.unwrap_or(1).max(1);
    let mut end_line = input.end_line.unwrap_or_else(|| lines.len().max(start_line));
    if end_line < start_line {
        bail!("end_line must be greater than or equal to start_line");
    }
    if end_line - start_line + 1 > MAX_READ_LINES {
        end_line = start_line + MAX_READ_LINES - 1;
    }

    let mut output = String::new();
    writeln!(output, "path: {}", display_path(workspace, &path)).expect("write to string");
    let mut citations = Vec::new();
    for line_number in start_line..=end_line.min(lines.len()) {
        check_timeout(started)?;
        let line = lines[line_number - 1];
        writeln!(output, "{line_number}: {line}").expect("write to string");
        if citations.is_empty() {
            citations.push(citation_for_path(workspace, &path, line_number, 100));
        }
    }

    Ok(finish_tool(output, citations, false))
}

fn grep(workspace: &AgentWorkspace, input: GrepInput, started: Instant) -> Result<ToolExecution> {
    let query = input.query.trim();
    if query.is_empty() {
        bail!("query must not be empty");
    }
    let base = resolve_existing_path(workspace, input.path.as_deref().unwrap_or("."))?;
    let limit = input.limit.unwrap_or(20).min(MAX_GREP_MATCHES);
    let lower_query = query.to_ascii_lowercase();
    let mut output = String::new();
    let mut citations = Vec::new();
    let mut matches = 0usize;

    let truncated_result = for_each_workspace_file(&base, started, |path| {
        let Some(text) = read_small_text_file(&path) else {
            return Ok(ControlFlow::Continue(()));
        };

        for (index, line) in text.lines().enumerate() {
            check_timeout(started)?;
            if !line.to_ascii_lowercase().contains(&lower_query) {
                continue;
            }

            matches += 1;
            let line_number = index + 1;
            writeln!(output, "{}:{line_number}: {line}", display_path(workspace, &path)).expect("write to string");
            citations.push(citation_for_path(workspace, &path, line_number, 90));
            if matches >= limit {
                return Ok(ControlFlow::Break(finish_tool(
                    std::mem::take(&mut output),
                    std::mem::take(&mut citations),
                    true,
                )));
            }
        }
        Ok(ControlFlow::Continue(()))
    })?;
    if let Some(result) = truncated_result {
        return Ok(result);
    }

    if matches == 0 {
        output.push_str("no matches\n");
    }
    Ok(finish_tool(output, citations, false))
}

fn glob(workspace: &AgentWorkspace, input: GlobInput, started: Instant) -> Result<ToolExecution> {
    let pattern = input.pattern.trim();
    if pattern.is_empty() {
        bail!("pattern must not be empty");
    }
    let limit = input.limit.unwrap_or(20).min(MAX_GLOB_MATCHES);

    let mut builder = GlobSetBuilder::new();
    builder.add(Glob::new(pattern).with_context(|| format!("Invalid glob pattern '{pattern}'"))?);
    let matcher = builder.build()?;

    let mut output = String::new();
    let mut matches = 0usize;
    let mut walker = WalkBuilder::new(&workspace.root);
    walker.hidden(false).git_ignore(false).git_exclude(false);
    for entry in walker.build() {
        check_timeout(started)?;
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let path = entry.path();
        if path == workspace.root {
            continue;
        }

        let relative = relative_path(workspace, path)?;
        if !matcher.is_match(&relative) {
            continue;
        }

        matches += 1;
        writeln!(output, "{relative}").expect("write to string");
        if matches >= limit {
            return Ok(finish_tool(output, Vec::new(), true));
        }
    }

    if matches == 0 {
        output.push_str("no matches\n");
    }
    Ok(finish_tool(output, Vec::new(), false))
}

fn tree(workspace: &AgentWorkspace, input: TreeInput, started: Instant) -> Result<ToolExecution> {
    let path = resolve_existing_path(workspace, input.path.as_deref().unwrap_or("."))?;
    if !path.is_dir() {
        bail!("path '{}' is not a directory", display_path(workspace, &path));
    }

    let max_depth = input.depth.unwrap_or(4).clamp(1, MAX_TREE_DEPTH);
    let mut output = String::new();
    writeln!(output, "path: {}", display_path(workspace, &path)).expect("write to string");
    let mut tree = TreeRenderer {
        workspace,
        max_depth,
        started,
        count: 0,
        output: &mut output,
    };
    let truncated = tree.walk(&path, "", 0)?;

    Ok(finish_tool(output, Vec::new(), truncated))
}

fn read_many(workspace: &AgentWorkspace, input: ReadManyInput, started: Instant) -> Result<ToolExecution> {
    if input.files.is_empty() {
        bail!("files must not be empty");
    }
    if input.files.len() > MAX_READ_MANY_FILES {
        bail!("at most {MAX_READ_MANY_FILES} files can be read at once");
    }

    let mut output = String::new();
    writeln!(output, "count: {}", input.files.len()).expect("write to string");
    let mut citations = Vec::new();

    for raw_path in input.files {
        check_timeout(started)?;
        let (path, text) = read_workspace_text_file(workspace, &raw_path)?;

        writeln!(output, "---- {} ----", display_path(workspace, &path)).expect("write to string");
        writeln!(output, "{text}").expect("write to string");
        citations.push(citation_for_path(workspace, &path, 1, 80));
    }

    Ok(finish_tool(output, citations, false))
}

fn search(workspace: &AgentWorkspace, input: SearchInput, started: Instant) -> Result<ToolExecution> {
    if input.regex.unwrap_or(false) {
        bail!("regex search is not supported yet; use a plain-text query");
    }

    let matcher = build_optional_glob_matcher(input.file_glob.as_deref())?;
    search_lines(
        workspace,
        input.query.trim(),
        input.path.as_deref().unwrap_or("."),
        input.limit.unwrap_or(20).min(MAX_GREP_MATCHES),
        matcher.as_ref(),
        started,
        "no matches\n",
    )
}

fn stat_path(workspace: &AgentWorkspace, input: StatInput, started: Instant) -> Result<ToolExecution> {
    check_timeout(started)?;
    let path = resolve_existing_path(workspace, &input.path)?;
    let metadata = fs::metadata(&path).with_context(|| format!("Reading '{}'", display_path(workspace, &path)))?;
    let kind = if metadata.is_dir() {
        "dir"
    } else if metadata.is_file() {
        "file"
    } else {
        "other"
    };
    let mut output = String::new();
    writeln!(output, "path: {}", display_path(workspace, &path)).expect("write to string");
    writeln!(output, "kind: {kind}").expect("write to string");
    writeln!(output, "size_bytes: {}", metadata.len()).expect("write to string");
    writeln!(output, "readonly: {}", metadata.permissions().readonly()).expect("write to string");
    Ok(finish_tool(
        output,
        vec![citation_for_path(workspace, &path, 1, 70)],
        false,
    ))
}

fn find_references(workspace: &AgentWorkspace, input: NameSearchInput, started: Instant) -> Result<ToolExecution> {
    search_lines(
        workspace,
        input.name.trim(),
        input.path.as_deref().unwrap_or("."),
        input.limit.unwrap_or(20).min(MAX_GREP_MATCHES),
        None,
        started,
        "no references found\n",
    )
}

fn git_status_readonly(workspace: &AgentWorkspace, started: Instant) -> Result<ToolExecution> {
    check_timeout(started)?;
    let output = run_git_readonly(workspace, &["status", "--short", "--branch"])?;
    Ok(finish_tool(output, Vec::new(), false))
}

fn git_diff(workspace: &AgentWorkspace, input: GitDiffInput, started: Instant) -> Result<ToolExecution> {
    check_timeout(started)?;
    let mut args = vec!["diff", "--no-ext-diff"];
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
    let mut args = vec!["show", "--no-ext-diff", input.rev.as_str()];
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

fn build_optional_glob_matcher(pattern: Option<&str>) -> Result<Option<globset::GlobSet>> {
    let Some(pattern) = pattern.map(str::trim).filter(|pattern| !pattern.is_empty()) else {
        return Ok(None);
    };

    let mut builder = GlobSetBuilder::new();
    builder.add(Glob::new(pattern).with_context(|| format!("Invalid glob pattern '{pattern}'"))?);
    Ok(Some(builder.build()?))
}

fn search_lines(
    workspace: &AgentWorkspace,
    query: &str,
    raw_path: &str,
    limit: usize,
    file_glob: Option<&globset::GlobSet>,
    started: Instant,
    empty_message: &str,
) -> Result<ToolExecution> {
    let query = query.trim();
    if query.is_empty() {
        bail!("query must not be empty");
    }
    let base = resolve_existing_path(workspace, raw_path)?;
    let mut output = String::new();
    let mut citations = Vec::new();
    let mut matches = 0usize;
    let lower_query = query.to_ascii_lowercase();

    let truncated_result = for_each_workspace_file(&base, started, |path| {
        let relative = relative_path(workspace, &path)?;
        if let Some(file_glob) = file_glob
            && !file_glob.is_match(relative.as_str())
        {
            return Ok(ControlFlow::Continue(()));
        }
        let Some(text) = read_small_text_file(&path) else {
            return Ok(ControlFlow::Continue(()));
        };

        for (index, line) in text.lines().enumerate() {
            check_timeout(started)?;
            if !line.to_ascii_lowercase().contains(&lower_query) {
                continue;
            }

            let line_number = index + 1;
            writeln!(output, "{}:{line_number}: {line}", relative).expect("write to string");
            citations.push(citation_for_path(workspace, &path, line_number, 90));
            matches += 1;
            if matches >= limit {
                return Ok(ControlFlow::Break(finish_tool(
                    std::mem::take(&mut output),
                    std::mem::take(&mut citations),
                    true,
                )));
            }
        }
        Ok(ControlFlow::Continue(()))
    })?;
    if let Some(result) = truncated_result {
        return Ok(result);
    }

    if matches == 0 {
        output.push_str(empty_message);
    }
    Ok(finish_tool(output, citations, false))
}

fn run_git_readonly(workspace: &AgentWorkspace, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(&workspace.root)
        .args(args)
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

fn validate_git_relative_path(raw: &str) -> Result<()> {
    let _ = validate_relative_path(raw)?;
    Ok(())
}

struct TreeRenderer<'a> {
    workspace: &'a AgentWorkspace,
    max_depth: usize,
    started: Instant,
    count: usize,
    output: &'a mut String,
}

impl TreeRenderer<'_> {
    fn walk(&mut self, root: &Path, prefix: &str, depth: usize) -> Result<bool> {
        if depth >= self.max_depth {
            return Ok(false);
        }

        let mut entries = fs::read_dir(root)
            .with_context(|| format!("Reading directory '{}'", display_path(self.workspace, root)))?
            .collect::<std::io::Result<Vec<_>>>()
            .with_context(|| format!("Reading directory '{}'", display_path(self.workspace, root)))?;
        entries.sort_by_key(|entry| entry.file_name());

        for (local_index, entry) in entries.iter().enumerate() {
            check_timeout(self.started)?;
            if self.count >= MAX_TREE_ENTRIES {
                self.output.push_str("... truncated\n");
                return Ok(true);
            }

            let child_path = entry.path();
            let file_type = entry.file_type()?;
            let file_name = entry.file_name();
            let child_label = file_name.to_string_lossy();

            let is_last = local_index + 1 == entries.len();
            let branch = if is_last { "└── " } else { "├── " };
            let marker = if file_type.is_dir() { "/" } else { "" };
            writeln!(self.output, "{}{}{}{}", prefix, branch, child_label, marker).expect("write to string");
            self.count += 1;

            if file_type.is_dir() {
                let next_prefix = if is_last {
                    format!("{prefix}    ")
                } else {
                    format!("{prefix}│   ")
                };
                if self.walk(&child_path, &next_prefix, depth + 1)? {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}

fn parse_input<T: for<'de> Deserialize<'de>>(call: &ToolCall) -> Result<T> {
    serde_json::from_value(call.input.clone()).with_context(|| format!("Invalid '{}' tool input", call.tool.as_str()))
}

fn resolve_existing_path(workspace: &AgentWorkspace, raw: &str) -> Result<PathBuf> {
    let relative = validate_relative_path(raw)?;
    let trimmed = raw.trim();

    let candidate = workspace.root.join(relative);
    let canonical = candidate
        .canonicalize()
        .with_context(|| format!("Path '{}' does not exist", trimmed))?;
    ensure_under_workspace(&workspace.root, &canonical, trimmed)?;
    Ok(canonical)
}

fn validate_relative_path(raw: &str) -> Result<&Path> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("path must not be empty");
    }

    let path = Path::new(trimmed);
    if path.is_absolute() {
        bail!("absolute paths are not allowed");
    }

    for component in path.components() {
        match component {
            Component::ParentDir => {
                bail!("path '{}' escapes the workspace", trimmed)
            }
            Component::RootDir | Component::Prefix(_) => {
                bail!("absolute paths are not allowed")
            }
            Component::CurDir | Component::Normal(_) => {}
        }
    }

    Ok(path)
}

fn ensure_under_workspace(root: &Path, candidate: &Path, raw: &str) -> Result<()> {
    if !candidate.starts_with(root) {
        bail!("path '{}' escapes the workspace", raw);
    }
    Ok(())
}

fn resolve_existing_file(workspace: &AgentWorkspace, raw: &str) -> Result<PathBuf> {
    let path = resolve_existing_path(workspace, raw)?;
    if !path.is_file() {
        bail!("path '{}' is not a file", display_path(workspace, &path));
    }
    Ok(path)
}

fn read_workspace_text_file(workspace: &AgentWorkspace, raw: &str) -> Result<(PathBuf, String)> {
    let path = resolve_existing_file(workspace, raw)?;
    let text = read_text_file_with_limit(workspace, &path)?;
    Ok((path, text))
}

fn read_text_file_with_limit(workspace: &AgentWorkspace, path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("Reading '{}'", display_path(workspace, path)))?;
    if bytes.len() > MAX_FILE_BYTES {
        bail!(
            "file '{}' exceeds the {} byte limit",
            display_path(workspace, path),
            MAX_FILE_BYTES
        );
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn read_small_text_file(path: &Path) -> Option<String> {
    let bytes = match fs::read(path) {
        Ok(bytes) if bytes.len() <= MAX_FILE_BYTES => bytes,
        _ => return None,
    };
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

fn for_each_workspace_file<T>(
    base: &Path,
    started: Instant,
    mut visit: impl FnMut(PathBuf) -> Result<ControlFlow<T>>,
) -> Result<Option<T>> {
    let mut walker = WalkBuilder::new(base);
    walker.hidden(false).git_ignore(false).git_exclude(false);
    for entry in walker.build() {
        check_timeout(started)?;
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        if !entry.file_type().is_some_and(|kind| kind.is_file()) {
            continue;
        }

        match visit(entry.into_path())? {
            ControlFlow::Continue(()) => {}
            ControlFlow::Break(result) => return Ok(Some(result)),
        }
    }

    Ok(None)
}

fn truncate_output(mut output: String) -> (String, bool) {
    if output.len() <= MAX_OUTPUT_BYTES {
        return (output, false);
    }

    let mut boundary = MAX_OUTPUT_BYTES.min(output.len());
    while boundary > 0 && !output.is_char_boundary(boundary) {
        boundary -= 1;
    }
    output.truncate(boundary);
    output.push_str("\n... truncated");
    (output, true)
}

fn check_timeout(started: Instant) -> Result<()> {
    if started.elapsed() > TOOL_TIMEOUT {
        bail!("tool execution timed out after {}s", TOOL_TIMEOUT.as_secs());
    }
    Ok(())
}

fn display_path(workspace: &AgentWorkspace, path: &Path) -> String {
    relative_path(workspace, path).unwrap_or_else(|_| path.display().to_string())
}

fn relative_path(workspace: &AgentWorkspace, path: &Path) -> Result<String> {
    let relative = path
        .strip_prefix(&workspace.root)
        .map_err(|_| anyhow!("path '{}' is not under workspace", path.display()))?;
    if relative.as_os_str().is_empty() {
        return Ok(".".to_string());
    }
    Ok(relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/"))
}

fn citation_for_path(workspace: &AgentWorkspace, path: &Path, line: usize, score: usize) -> ToolCitation {
    let relative = display_path(workspace, path);
    let mut parts = relative.splitn(2, '/');
    let resource = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
    ToolCitation {
        resource,
        path,
        line,
        score,
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use serde_json::json;

    use super::execute_tool;
    use crate::{
        agent::types::{AgentWorkspace, ToolCall, ToolName, WorkspaceResourceSummary},
        temp_paths::create_test_dir,
    };

    fn workspace(root: PathBuf) -> AgentWorkspace {
        AgentWorkspace {
            root,
            resources: vec![WorkspaceResourceSummary {
                name: "repo".to_string(),
                kind: "local".to_string(),
                source: "local".to_string(),
                branch: None,
                search_paths: vec![".".to_string()],
                notes: None,
                mount_path: "repo".to_string(),
                ephemeral: false,
            }],
        }
    }

    #[test]
    fn read_rejects_parent_path_traversal() {
        let root = create_test_dir("agentic-tools-traversal");
        fs::create_dir_all(root.join("repo")).expect("create repo");
        let error = execute_tool(
            &workspace(root.clone()),
            &ToolCall {
                id: "1".to_string(),
                tool: ToolName::Read,
                input: json!({
                    "path": "../secret.txt"
                }),
                provider_data: None,
            },
        )
        .expect_err("path traversal should fail");

        assert!(error.to_string().contains("escapes the workspace"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn read_rejects_symlink_escape() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let root = create_test_dir("agentic-tools-symlink");
            let repo = root.join("repo");
            fs::create_dir_all(&repo).expect("create repo");
            let outside = create_test_dir("agentic-tools-outside");
            fs::write(outside.join("secret.txt"), "secret").expect("write outside file");
            symlink(outside.join("secret.txt"), repo.join("linked.txt")).expect("create symlink");

            let error = execute_tool(
                &workspace(root.clone()),
                &ToolCall {
                    id: "1".to_string(),
                    tool: ToolName::Read,
                    input: json!({
                        "path": "repo/linked.txt"
                    }),
                    provider_data: None,
                },
            )
            .expect_err("symlink escape should fail");

            assert!(error.to_string().contains("escapes the workspace"));

            let _ = fs::remove_dir_all(root);
            let _ = fs::remove_dir_all(outside);
        }
    }

    #[test]
    fn grep_observes_match_limit() {
        let root = create_test_dir("agentic-tools-grep-limit");
        let repo = root.join("repo");
        fs::create_dir_all(&repo).expect("create repo");
        fs::write(repo.join("file.txt"), "needle one\nneedle two\nneedle three\n").expect("write file");

        let result = execute_tool(
            &workspace(root.clone()),
            &ToolCall {
                id: "1".to_string(),
                tool: ToolName::Grep,
                input: json!({
                    "query": "needle",
                    "limit": 2
                }),
                provider_data: None,
            },
        )
        .expect("grep should succeed");

        assert!(result.truncated);
        assert!(result.output.contains("repo/file.txt:1"));
        assert!(result.output.contains("repo/file.txt:2"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn read_rejects_directory_paths() {
        let root = create_test_dir("agentic-tools-read-index");
        let tutorial = root.join("repo/docs/01-state");
        fs::create_dir_all(&tutorial).expect("create tutorial dir");
        fs::write(tutorial.join("index.md"), "# State\n\nReactive state lives here.\n").expect("write index");

        let error = execute_tool(
            &workspace(root.clone()),
            &ToolCall {
                id: "1".to_string(),
                tool: ToolName::Read,
                input: json!({
                    "path": "repo/docs/01-state"
                }),
                provider_data: None,
            },
        )
        .expect_err("read should reject directory paths");

        assert!(error.to_string().contains("is not a file"));

        let _ = fs::remove_dir_all(root);
    }
}
