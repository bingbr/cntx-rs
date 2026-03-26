use std::{any::Any, collections::HashSet, path::Path, thread};

use anyhow::{Context, Result, anyhow, bail};

use super::{
    sandbox::ToolExecutorImpl,
    tools,
    types::{
        AgentItem, AgentRunOutput, AgentRunStatus, AgentRuntimeEvent, AgentWorkspace, ToolCall, ToolDefinition,
        ToolExecutor, ToolName, ToolResult, profile_id_for_provider, resolve_profile,
    },
    workspace,
};
use crate::{
    ask::{AskOutput, Citation, ResourceSummary, RetrievalSummary},
    config::Config,
    providers, resources, validation,
};

type ChunkCallback<'a> = dyn FnMut(&str) -> Result<()> + 'a;
type ProgressCallback<'a> = dyn FnMut(&str) -> Result<()> + 'a;

pub fn ask_question(
    config: &Config,
    requested_resources: &[String],
    question: &str,
    on_chunk: Option<&mut ChunkCallback<'_>>,
    mut on_progress: Option<&mut ProgressCallback<'_>>,
) -> Result<AskOutput> {
    let mut config = config.clone();
    config.normalized()?;
    let agent_profile = profile_id_for_provider(&config.provider)?;
    let selected_profile = resolve_profile(agent_profile);
    let question = validation::validate_question(question)?;
    let resolved_resources = resources::resolve_resource_references(&config.resources, requested_resources, &question)?;
    let prepared_workspace =
        workspace::prepare_workspace_from_resources(&config.data_dir, &resolved_resources, on_progress.as_deref_mut())?;
    let workspace = prepared_workspace.workspace();
    let user_prompt = build_user_prompt(workspace, &question);
    let adapter = providers::build_adapter(&config.provider)?;
    let mut session = adapter.start_session(&config)?;
    session.profile = agent_profile;
    session.provider = config.provider.clone();
    session.system_prompt = r#"You are cntx, an expert research agent. Thoroughly answer user questions by searching available resources. Do not ask permission to research; execute immediately.

<agent_directives>
- **Persona & Style:** Direct, calm, and concise (internal channel). Use proper Markdown, leveraging lists and real codeblocks.
- **Tool Execution:** Prefer broad, high-yield discovery first, then 1-2 targeted reads. Parallelize only independent steps. Stop as soon as you have enough grounded evidence to answer.
- **Efficiency:** Avoid long exploratory chains. Aim to finish within a small number of tool rounds, and do not keep searching once additional tool calls are unlikely to change the answer.
- **Persistence & Depth:** Dig past the first plausible answer to identify edge cases and constraints. If tools return empty or partial results, retry with a better query or path before expanding scope.
- **Completeness:** Deliver fully grounded answers. If an item is blocked by missing data, explicitly mark it `[blocked]` and state exactly what is missing.
- **Paths:** All tool paths are relative to the workspace root, not to an individual repo. Resources are mounted under their listed mount paths, so a file inside resource `svelte` must be accessed as `svelte/...`, not `apps/...`.
- **Citations:** Always include sources. Format GitHub citations as markdown links: `- [repo/relative/path.ext](full_github_blob_url)`. Cite local file paths or npm package paths accurately.
</agent_directives>
"#.to_string();
    session.model = config.model.clone();
    session.conversation = vec![AgentItem::UserText(user_prompt)];
    let tool_definitions = tools::tool_definitions();
    let executor = ToolExecutorImpl::from_config(&config)?;
    let output = run_agent_loop(
        adapter.as_ref(),
        &executor,
        workspace,
        session,
        AgentLoopConfig {
            tools: &tool_definitions,
            max_steps: config.agentic_max_steps,
            allow_parallel_tools: selected_profile.parallel_tools,
            fail_closed_on_executor_error: config.agentic_require_sandbox,
        },
        on_progress.as_deref_mut(),
    )?;

    let citations = collect_citations(&output);
    let retrieval = build_retrieval_summary(workspace, &output);
    let answer = match output.status {
        AgentRunStatus::Completed => {
            let answer = output
                .answer
                .clone()
                .expect("completed agent run should contain a final answer");
            if let Some(on_chunk) = on_chunk {
                on_chunk(&answer)?;
            }
            answer
        }
        AgentRunStatus::MaxStepsReached => {
            if let Some(on_progress) = on_progress.as_mut() {
                (**on_progress)(
                    "Reached the agentic step limit; composing a best-effort answer from collected evidence...",
                )?;
            }
            synthesize_partial_answer(&config, workspace, &question, &output, on_chunk)?
        }
    };

    Ok(AskOutput {
        answer,
        resolved_resources: workspace
            .resources
            .iter()
            .map(|resource| ResourceSummary {
                name: resource.name.clone(),
                kind: resource.kind.clone(),
                source: resource.source.clone(),
                branch: resource.branch.clone(),
                search_paths: resource.search_paths.clone(),
                notes: resource.notes.clone(),
                ephemeral: resource.ephemeral,
            })
            .collect(),
        citations,
        retrieval,
    })
}

pub fn run_internal_tool(workspace_root: &Path, call: ToolCall) -> Result<ToolResult> {
    tools::execute_tool(
        &AgentWorkspace {
            root: workspace_root.to_path_buf(),
            resources: Vec::new(),
        },
        &call,
    )
}

struct AgentLoopConfig<'a> {
    tools: &'a [ToolDefinition],
    max_steps: usize,
    allow_parallel_tools: bool,
    fail_closed_on_executor_error: bool,
}

fn run_agent_loop(
    adapter: &dyn providers::ProviderAdapter,
    executor: &dyn ToolExecutor,
    workspace: &AgentWorkspace,
    mut session: crate::agent::types::AgentSession,
    config: AgentLoopConfig<'_>,
    mut on_progress: Option<&mut ProgressCallback<'_>>,
) -> Result<AgentRunOutput> {
    if config.max_steps == 0 {
        bail!("agentic_max_steps must be greater than zero");
    }

    let mut events = Vec::new();
    let mut tool_results = Vec::new();
    let mut steps = 0usize;

    for step in 0..config.max_steps {
        steps = step + 1;
        match adapter.complete_turn(crate::agent::types::ProviderTurnRequest {
            session: &session,
            tools: config.tools,
            stream: false,
        })? {
            crate::agent::types::ProviderTurn::Final {
                items,
                text,
                provider_state,
            } => {
                let final_text = apply_turn_items(&mut session, &mut events, items, provider_state);
                return Ok(AgentRunOutput {
                    answer: Some(final_text.unwrap_or(text)),
                    events,
                    tool_results,
                    steps,
                    status: AgentRunStatus::Completed,
                });
            }
            crate::agent::types::ProviderTurn::ToolCalls {
                items,
                calls,
                provider_state,
            } => {
                if calls.is_empty() {
                    bail!("provider returned a tool-call turn without any tool calls");
                }

                let thought = apply_turn_items(&mut session, &mut events, items, provider_state);
                if let Some(text) = thought.as_deref().filter(|text| !text.trim().is_empty())
                    && let Some(on_progress) = on_progress.as_mut()
                {
                    (**on_progress)(&format!("Thinking: {text}"))?;
                }

                for call in &calls {
                    if let Some(on_progress) = on_progress.as_mut() {
                        (**on_progress)(&describe_tool_call(call))?;
                    }
                }
                let results = if config.allow_parallel_tools
                    && adapter.capabilities().supports_parallel_tool_calls
                    && calls.len() > 1
                    && calls.iter().all(|call| is_parallel_safe_tool(config.tools, call))
                {
                    if config.fail_closed_on_executor_error {
                        execute_tool_calls_in_parallel_fail_closed(executor, workspace, &calls)?
                    } else {
                        execute_tool_calls_in_parallel(executor, workspace, &calls)
                    }
                } else if config.fail_closed_on_executor_error {
                    calls
                        .iter()
                        .map(|call| execute_tool_call_fail_closed(executor, workspace, call))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    calls
                        .iter()
                        .map(|call| execute_tool_call(executor, workspace, call))
                        .collect::<Vec<_>>()
                };

                for result in results {
                    session.conversation.push(AgentItem::ToolResult(result.clone()));
                    tool_results.push(result);
                }
            }
        }
    }

    Ok(AgentRunOutput {
        answer: None,
        events,
        tool_results,
        steps,
        status: AgentRunStatus::MaxStepsReached,
    })
}

fn apply_turn_items(
    session: &mut crate::agent::types::AgentSession,
    events: &mut Vec<AgentRuntimeEvent>,
    items: Vec<AgentItem>,
    provider_state: Option<serde_json::Value>,
) -> Option<String> {
    let mut last_text = None;
    for item in items {
        match &item {
            AgentItem::Reasoning(chunk) => {
                if let Some(text) = chunk.text.as_ref().filter(|text| !text.trim().is_empty()) {
                    events.push(AgentRuntimeEvent::Thinking(text.clone()));
                    last_text = Some(text.clone());
                }
            }
            AgentItem::AssistantText(text) => {
                events.push(AgentRuntimeEvent::Thinking(text.clone()));
                last_text = Some(text.clone());
            }
            AgentItem::FinalAnswer(text) => {
                last_text = Some(text.clone());
            }
            AgentItem::ToolCall(_) => {}
            AgentItem::ToolResult(_) | AgentItem::UserText(_) => {}
        }
        session.conversation.push(item);
    }
    session.set_provider_state(provider_state);
    last_text
}

fn execute_tool_call(executor: &dyn ToolExecutor, workspace: &AgentWorkspace, call: &ToolCall) -> ToolResult {
    executor
        .execute(workspace, call)
        .unwrap_or_else(|error| tool_error_result(call, error.to_string()))
}

fn execute_tool_call_fail_closed(
    executor: &dyn ToolExecutor,
    workspace: &AgentWorkspace,
    call: &ToolCall,
) -> Result<ToolResult> {
    executor
        .execute(workspace, call)
        .with_context(|| format!("tool execution aborted for `{}`", call.tool))
}

fn execute_tool_calls_in_parallel(
    executor: &dyn ToolExecutor,
    workspace: &AgentWorkspace,
    calls: &[ToolCall],
) -> Vec<ToolResult> {
    let mut results = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(calls.len());

        for (index, call) in calls.iter().enumerate() {
            let handle = scope.spawn(move || execute_tool_call_with_panic_handling(executor, workspace, call));
            handles.push((index, handle));
        }

        handles
            .into_iter()
            .map(|(index, handle)| {
                let result = match handle.join() {
                    Ok(result) => result,
                    Err(panic) => tool_error_result(
                        &calls[index],
                        format!("tool execution panicked: {}", panic_payload_to_string(&*panic)),
                    ),
                };

                (index, result)
            })
            .collect::<Vec<_>>()
    });

    results.sort_by_key(|(index, _)| *index);
    results.into_iter().map(|(_, result)| result).collect()
}

fn execute_tool_calls_in_parallel_fail_closed(
    executor: &dyn ToolExecutor,
    workspace: &AgentWorkspace,
    calls: &[ToolCall],
) -> Result<Vec<ToolResult>> {
    let mut results = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(calls.len());

        for (index, call) in calls.iter().enumerate() {
            let handle =
                scope.spawn(move || execute_tool_call_fail_closed_with_panic_handling(executor, workspace, call));
            handles.push((index, handle));
        }

        handles
            .into_iter()
            .map(|(index, handle)| {
                let result = match handle.join() {
                    Ok(result) => result,
                    Err(panic) => Err(anyhow!(
                        "tool execution aborted for `{}`: tool execution panicked: {}",
                        calls[index].tool,
                        panic_payload_to_string(&*panic)
                    )),
                };

                (index, result)
            })
            .collect::<Vec<_>>()
    });

    results.sort_by_key(|(index, _)| *index);
    results.into_iter().map(|(_, result)| result).collect()
}

fn execute_tool_call_with_panic_handling(
    executor: &dyn ToolExecutor,
    workspace: &AgentWorkspace,
    call: &ToolCall,
) -> ToolResult {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| executor.execute(workspace, call))) {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => tool_error_result(call, error.to_string()),
        Err(panic) => tool_error_result(
            call,
            format!("tool execution panicked: {}", panic_payload_to_string(&*panic)),
        ),
    }
}

fn execute_tool_call_fail_closed_with_panic_handling(
    executor: &dyn ToolExecutor,
    workspace: &AgentWorkspace,
    call: &ToolCall,
) -> Result<ToolResult> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| executor.execute(workspace, call))) {
        Ok(result) => result.with_context(|| format!("tool execution aborted for `{}`", call.tool)),
        Err(panic) => Err(anyhow!(
            "tool execution aborted for `{}`: tool execution panicked: {}",
            call.tool,
            panic_payload_to_string(&*panic)
        )),
    }
}

fn tool_error_result(call: &ToolCall, error: String) -> ToolResult {
    ToolResult::error(call, error)
}

fn panic_payload_to_string(payload: &(dyn Any + Send + 'static)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn is_parallel_safe_tool(tools: &[ToolDefinition], call: &ToolCall) -> bool {
    tools
        .iter()
        .find(|tool| tool.name == call.tool)
        .is_some_and(|tool| tool.safe_to_parallelize)
}

fn describe_tool_call(call: &ToolCall) -> String {
    match call.tool {
        ToolName::Read => {
            let path = input_string(&call.input, "path").unwrap_or_else(|| "<unknown path>".into());
            match (
                input_usize(&call.input, "start_line"),
                input_usize(&call.input, "end_line"),
            ) {
                (Some(start), Some(end)) => {
                    format!("Reading [{path}:{start}-{end}]")
                }
                _ => format!("Reading [{path}]"),
            }
        }
        ToolName::Grep | ToolName::Search | ToolName::FindReferences => {
            let query = input_string(&call.input, "query")
                .or_else(|| input_string(&call.input, "name"))
                .unwrap_or_else(|| "<query>".into());
            let path = input_string(&call.input, "path").unwrap_or_else(|| "<workspace>".into());
            format!("Searching {path} for `{query}`")
        }
        ToolName::Glob => {
            let pattern = input_string(&call.input, "pattern").unwrap_or_else(|| "<pattern>".into());
            let path = input_string(&call.input, "path").unwrap_or_else(|| "<workspace>".into());
            format!("Finding files in {path} matching `{pattern}`")
        }
        ToolName::Tree => {
            let path = input_string(&call.input, "path").unwrap_or_else(|| "<workspace>".into());
            let depth = input_usize(&call.input, "depth").unwrap_or(4);
            format!("Building tree for {path} (depth {depth})")
        }
        ToolName::ReadMany => {
            let files = call
                .input
                .get("files")
                .and_then(|value| value.as_array())
                .map(|files| files.len())
                .unwrap_or(0);
            format!("Reading {files} files")
        }
        ToolName::Stat => {
            let path = input_string(&call.input, "path").unwrap_or_else(|| "<workspace>".into());
            format!("Inspecting metadata for {path}")
        }
        ToolName::GitStatusReadonly => "Inspecting git status".to_string(),
        ToolName::GitDiff => "Inspecting git diff".to_string(),
        ToolName::GitShow => "Inspecting git object".to_string(),
        ToolName::GitLog => "Inspecting git history".to_string(),
        ToolName::List => {
            let path = input_string(&call.input, "path").unwrap_or_else(|| "<workspace>".into());
            format!("Listing [{path}]")
        }
    }
}

fn input_string(input: &serde_json::Value, key: &str) -> Option<String> {
    input
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string)
}

fn input_usize(input: &serde_json::Value, key: &str) -> Option<usize> {
    input
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn build_user_prompt(workspace: &AgentWorkspace, question: &str) -> String {
    let mut prompt = String::new();
    prompt.push_str("Tool path rules:\n");
    prompt.push_str("- All tool paths are relative to the workspace root.\n");
    prompt.push_str("- Each resource is mounted under its mount path.\n");
    prompt.push_str("- If a resource is mounted at `svelte`, files inside it must be accessed as `svelte/...`.\n\n");
    prompt.push_str("Workspace resources:\n");
    for resource in &workspace.resources {
        prompt.push_str(&format!(
            "- {} mounted at `{}` source={} branch={} search_paths={}\n",
            resource.name,
            resource.mount_path,
            resource.source,
            resource.branch.as_deref().unwrap_or("-"),
            resource.search_paths.join(",")
        ));
        if let Some(notes) = &resource.notes {
            prompt.push_str(&format!("  notes: {notes}\n"));
        }
    }
    prompt.push_str("\nTop-level resource directories:\n");
    for resource in &workspace.resources {
        prompt.push_str(&format!("- {}\n", resource.mount_path));
    }
    prompt.push_str("\nQuestion:\n");
    prompt.push_str(question);
    prompt
}

fn synthesize_partial_answer(
    config: &Config,
    workspace: &AgentWorkspace,
    question: &str,
    output: &AgentRunOutput,
    on_chunk: Option<&mut ChunkCallback<'_>>,
) -> Result<String> {
    let prompt = build_partial_answer_prompt(workspace, question, output);
    if let Some(on_chunk) = on_chunk {
        providers::ask_stream(config, &prompt, on_chunk)
    } else {
        providers::ask(config, &prompt)
    }
}

fn build_partial_answer_prompt(workspace: &AgentWorkspace, question: &str, output: &AgentRunOutput) -> String {
    const MAX_TOOL_RESULTS: usize = 12;
    const MAX_TOOL_OUTPUT_CHARS: usize = 1_500;

    let mut prompt = String::new();
    prompt.push_str("A research agent collected partial evidence but did not finish the tool loop.\n");
    prompt.push_str("Answer the user's question using only the grounded evidence below. Be direct and useful.\n");
    prompt.push_str("If the evidence is incomplete, say what is still uncertain instead of guessing.\n\n");
    prompt.push_str("Question:\n");
    prompt.push_str(question);
    prompt.push_str("\n\nWorkspace resources:\n");
    for resource in &workspace.resources {
        prompt.push_str(&format!(
            "- {} source={} search_paths={}\n",
            resource.name,
            resource.source,
            resource.search_paths.join(",")
        ));
    }

    let thoughts = output
        .events
        .iter()
        .map(|event| match event {
            AgentRuntimeEvent::Thinking(text) => text.as_str(),
        })
        .collect::<Vec<_>>();
    if !thoughts.is_empty() {
        prompt.push_str("\nAgent progress notes:\n");
        for thought in thoughts {
            prompt.push_str("- ");
            prompt.push_str(thought);
            prompt.push('\n');
        }
    }

    prompt.push_str("\nCollected tool evidence:\n");
    for result in output.tool_results.iter().take(MAX_TOOL_RESULTS) {
        prompt.push_str(&format!("\nTool `{}` output:\n", result.tool));
        if result.output.chars().count() > MAX_TOOL_OUTPUT_CHARS {
            let truncated = result.output.chars().take(MAX_TOOL_OUTPUT_CHARS).collect::<String>();
            prompt.push_str(&truncated);
            prompt.push_str("\n[tool output truncated for synthesis]\n");
        } else {
            prompt.push_str(&result.output);
            prompt.push('\n');
        }
    }

    prompt
}

fn collect_citations(output: &AgentRunOutput) -> Vec<Citation> {
    let mut seen = HashSet::new();
    output
        .tool_results
        .iter()
        .flat_map(|result| result.citations.iter())
        .filter(|citation| {
            !citation.resource.is_empty()
                && !citation.path.is_empty()
                && seen.insert(format!("{}:{}:{}", citation.resource, citation.path, citation.line))
        })
        .take(12)
        .map(|citation| Citation {
            resource: citation.resource.clone(),
            path: citation.path.clone(),
            line: citation.line,
            score: citation.score,
        })
        .collect()
}

fn build_retrieval_summary(workspace: &AgentWorkspace, output: &AgentRunOutput) -> RetrievalSummary {
    let touched_resources = output
        .tool_results
        .iter()
        .flat_map(|result| result.citations.iter())
        .map(|citation| citation.resource.clone())
        .collect::<HashSet<_>>();

    RetrievalSummary {
        resource_count: workspace.resources.len(),
        snippet_count: output.tool_results.iter().map(|result| result.citations.len()).sum(),
        search_path_count: workspace
            .resources
            .iter()
            .map(|resource| resource.search_paths.len())
            .sum(),
        retrieval_steps: output.steps,
        empty_resources: workspace
            .resources
            .iter()
            .filter(|resource| !touched_resources.contains(&resource.name))
            .map(|resource| resource.name.clone())
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, path::PathBuf, sync::Mutex};

    use anyhow::Result;
    use reqwest::blocking::Client;
    use serde_json::json;

    use super::{AgentLoopConfig, run_agent_loop};
    use crate::{
        agent::types::{
            AgentItem, AgentRunStatus, AgentSession, AgentWorkspace, ProviderCapabilities, ProviderTurn,
            ProviderTurnRequest, ToolCall, ToolExecutor, ToolName, ToolResult, WorkspaceResourceSummary,
        },
        config::Config,
        providers::{ProviderAdapter, ProviderError, ProviderResult},
    };

    struct ScriptedProviderAdapter {
        capabilities: ProviderCapabilities,
        turns: Mutex<VecDeque<ProviderTurn>>,
    }

    impl ScriptedProviderAdapter {
        fn new(capabilities: ProviderCapabilities, turns: Vec<ProviderTurn>) -> Self {
            Self {
                capabilities,
                turns: Mutex::new(turns.into()),
            }
        }
    }

    impl ProviderAdapter for ScriptedProviderAdapter {
        fn capabilities(&self) -> ProviderCapabilities {
            self.capabilities
        }

        fn ask(
            &self,
            _client: &Client,
            _config: &Config,
            _prompt: &str,
            _api_key: Option<&str>,
        ) -> ProviderResult<String> {
            Err(ProviderError::Protocol("unused in runtime tests".to_string()))
        }

        fn ask_stream(
            &self,
            _client: &Client,
            _config: &Config,
            _prompt: &str,
            _api_key: Option<&str>,
            _on_chunk: &mut dyn FnMut(&str) -> Result<()>,
        ) -> ProviderResult<String> {
            Err(ProviderError::Protocol("unused in runtime tests".to_string()))
        }

        fn complete_turn(&self, _request: ProviderTurnRequest<'_>) -> ProviderResult<ProviderTurn> {
            self.turns
                .lock()
                .expect("turn queue lock should not be poisoned")
                .pop_front()
                .ok_or_else(|| ProviderError::Protocol("no scripted turns remaining".to_string()))
        }
    }

    struct AlwaysFailingExecutor;

    impl ToolExecutor for AlwaysFailingExecutor {
        fn execute(&self, _workspace: &AgentWorkspace, _call: &ToolCall) -> Result<ToolResult> {
            Err(anyhow::anyhow!("sandbox startup failed"))
        }
    }

    fn test_session() -> AgentSession {
        AgentSession {
            profile: crate::agent::types::AgentProfileId::Anthropic,
            provider: "anthropic".to_string(),
            system_prompt: String::new(),
            model: "test-model".to_string(),
            conversation: Vec::new(),
            provider_state: None,
        }
    }

    fn test_workspace() -> AgentWorkspace {
        AgentWorkspace {
            root: PathBuf::from("/tmp/workspace"),
            resources: vec![WorkspaceResourceSummary {
                name: "repo".to_string(),
                kind: "git".to_string(),
                source: "https://github.com/example/repo".to_string(),
                branch: Some("main".to_string()),
                search_paths: vec!["docs".to_string()],
                notes: None,
                mount_path: "repo".to_string(),
                ephemeral: false,
            }],
        }
    }

    fn list_tool_call(id: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            tool: ToolName::List,
            input: json!({ "path": "." }),
            provider_data: None,
        }
    }

    #[test]
    fn run_agent_loop_aborts_when_sandboxed_tool_execution_fails() {
        let adapter = ScriptedProviderAdapter::new(
            ProviderCapabilities {
                supports_streaming: false,
                supports_parallel_tool_calls: false,
            },
            vec![ProviderTurn::ToolCalls {
                items: vec![AgentItem::AssistantText("Inspecting the workspace.".to_string())],
                calls: vec![list_tool_call("call-1")],
                provider_state: None,
            }],
        );
        let tools = crate::agent::tools::tool_definitions();

        let error = run_agent_loop(
            &adapter,
            &AlwaysFailingExecutor,
            &test_workspace(),
            test_session(),
            AgentLoopConfig {
                tools: &tools,
                max_steps: 2,
                allow_parallel_tools: false,
                fail_closed_on_executor_error: true,
            },
            None,
        )
        .expect_err("sandboxed tool failures should abort the agent loop");

        assert!(error.to_string().contains("tool execution aborted for `list`"));
        assert!(format!("{error:#}").contains("sandbox startup failed"));
    }

    #[test]
    fn run_agent_loop_aborts_when_parallel_sandboxed_tool_execution_fails() {
        let adapter = ScriptedProviderAdapter::new(
            ProviderCapabilities {
                supports_streaming: false,
                supports_parallel_tool_calls: true,
            },
            vec![ProviderTurn::ToolCalls {
                items: vec![AgentItem::AssistantText("Inspecting the workspace.".to_string())],
                calls: vec![list_tool_call("call-1"), list_tool_call("call-2")],
                provider_state: None,
            }],
        );
        let tools = crate::agent::tools::tool_definitions();

        let error = run_agent_loop(
            &adapter,
            &AlwaysFailingExecutor,
            &test_workspace(),
            test_session(),
            AgentLoopConfig {
                tools: &tools,
                max_steps: 2,
                allow_parallel_tools: true,
                fail_closed_on_executor_error: true,
            },
            None,
        )
        .expect_err("parallel sandboxed tool failures should abort the agent loop");

        assert!(error.to_string().contains("tool execution aborted for `list`"));
        assert!(format!("{error:#}").contains("sandbox startup failed"));
    }

    #[test]
    fn run_agent_loop_keeps_tool_errors_recoverable_when_fail_closed_is_disabled() {
        let adapter = ScriptedProviderAdapter::new(
            ProviderCapabilities {
                supports_streaming: false,
                supports_parallel_tool_calls: false,
            },
            vec![
                ProviderTurn::ToolCalls {
                    items: vec![AgentItem::AssistantText("Inspecting the workspace.".to_string())],
                    calls: vec![list_tool_call("call-1")],
                    provider_state: None,
                },
                ProviderTurn::Final {
                    text: "done".to_string(),
                    items: vec![AgentItem::FinalAnswer("done".to_string())],
                    provider_state: None,
                },
            ],
        );
        let tools = crate::agent::tools::tool_definitions();

        let output = run_agent_loop(
            &adapter,
            &AlwaysFailingExecutor,
            &test_workspace(),
            test_session(),
            AgentLoopConfig {
                tools: &tools,
                max_steps: 2,
                allow_parallel_tools: false,
                fail_closed_on_executor_error: false,
            },
            None,
        )
        .expect("non-fatal executor errors should be surfaced as tool results");

        assert_eq!(output.status, AgentRunStatus::Completed);
        assert_eq!(output.tool_results.len(), 1);
        assert!(output.tool_results[0].is_error);
        assert!(output.tool_results[0].output.contains("sandbox startup failed"));
    }
}
