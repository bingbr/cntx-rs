use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use reqwest::blocking::Client;
use serde_json::{Value, json};

use super::catalog::{ProviderDescriptor, default_model_for_provider, provider_descriptor};
use crate::{
    agent::types::{
        AgentItem, AgentSession, ProviderTurn, ProviderTurnRequest, ReasoningChunk, ReasoningPolicy, ToolCall,
        ToolDefinition, ToolResult, resolve_profile,
    },
    auth,
    config::Config,
    providers,
};

#[derive(Debug, Clone)]
pub(crate) enum ProviderMessage {
    AssistantToolCalls {
        text: Option<String>,
        reasoning: Option<ProviderReasoning>,
        calls: Vec<ToolCall>,
    },
    ToolResult(ToolResult),
}

#[derive(Debug, Clone)]
pub(crate) struct ProviderRequest {
    pub(crate) system_prompt: String,
    pub(crate) user_prompt: String,
    pub(crate) messages: Vec<ProviderMessage>,
}

#[derive(Debug, Clone)]
pub(crate) enum ProviderModelTurn {
    Final {
        text: String,
        reasoning: Option<ProviderReasoning>,
    },
    ToolCalls {
        text: Option<String>,
        reasoning: Option<ProviderReasoning>,
        calls: Vec<ToolCall>,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct ProviderReasoning {
    pub(crate) text: Option<String>,
    pub(crate) opaque_state: Option<Value>,
}

pub fn ask(config: &Config, prompt: &str) -> Result<String> {
    let (http_client, adapter, api_key) = provider_call_parts(config)?;
    adapter
        .ask(&http_client, config, prompt, api_key.as_deref())
        .map_err(|error| anyhow!(error))
}

pub fn ask_stream(config: &Config, prompt: &str, on_chunk: &mut dyn FnMut(&str) -> Result<()>) -> Result<String> {
    let (http_client, adapter, api_key) = provider_call_parts(config)?;
    adapter
        .ask_stream(&http_client, config, prompt, api_key.as_deref(), on_chunk)
        .map_err(|error| anyhow!(error))
}

pub(crate) fn client() -> Result<Client> {
    Ok(Client::builder().timeout(Duration::from_secs(600)).build()?)
}

pub(crate) fn require_api_key<'a>(api_key: Option<&'a str>, provider: &str) -> Result<&'a str> {
    api_key.ok_or_else(|| anyhow!("No API key found for '{provider}'"))
}

pub(crate) fn resolved_api_key<'a>(
    api_key: Option<&'a str>,
    descriptor: &ProviderDescriptor,
    provider: &str,
) -> Result<Option<&'a str>> {
    if descriptor.api_key_optional {
        Ok(api_key)
    } else {
        Ok(Some(require_api_key(api_key, provider)?))
    }
}

fn provider_call_parts(config: &Config) -> Result<(Client, providers::AdapterHandle, Option<String>)> {
    let descriptor = provider_descriptor(&config.provider)?;
    let api_key = auth::resolve_api_key(&config.provider, config)?;
    Ok((
        client()?,
        providers::build_adapter(descriptor.id).map_err(|error| anyhow!(error))?,
        resolved_api_key(api_key.as_deref(), descriptor, &config.provider)?.map(str::to_owned),
    ))
}

pub(crate) fn for_agent_tool_exchanges(
    state: &ProviderRequest,
    mut visit: impl FnMut(Option<&str>, Option<&ProviderReasoning>, &[ToolCall], &[&ToolResult]) -> Result<()>,
) -> Result<()> {
    let mut index = 0usize;
    while index < state.messages.len() {
        let ProviderMessage::AssistantToolCalls { text, calls, reasoning } = &state.messages[index] else {
            bail!("agent state contained a tool result without a preceding assistant tool call");
        };

        let mut tool_results = Vec::new();
        index += 1;
        while index < state.messages.len() {
            match &state.messages[index] {
                ProviderMessage::ToolResult(result) => {
                    tool_results.push(result);
                    index += 1;
                }
                ProviderMessage::AssistantToolCalls { .. } => break,
            }
        }

        visit(text.as_deref(), reasoning.as_ref(), calls, &tool_results)?;
    }

    Ok(())
}

pub(crate) fn complete_adapter_turn_with(
    request: ProviderTurnRequest<'_>,
    complete: impl FnOnce(&Client, &Config, &ProviderRequest, &[ToolDefinition], Option<&str>) -> Result<ProviderModelTurn>,
) -> Result<ProviderTurn> {
    let config = profile_config_for_session(request.session)?;
    let descriptor = provider_descriptor(&config.provider)?;
    let http_client = client()?;
    let api_key = auth::resolve_api_key(&config.provider, &config)?;
    let api_key = resolved_api_key(api_key.as_deref(), descriptor, &config.provider)?;
    let state = session_to_provider_request(request.session)?;
    let turn = complete(&http_client, &config, &state, request.tools, api_key)?;
    model_turn_to_runtime_turn(turn, request.session.provider_state.clone())
}

pub(crate) fn profile_config_for_session(session: &AgentSession) -> Result<Config> {
    let provider = session.provider.clone();
    let mut config = Config {
        provider: provider.clone(),
        ..Config::default()
    };
    if !session.model.trim().is_empty() {
        config.model = session.model.clone();
    } else if let Some(model) = default_model_for_provider(&provider) {
        config.model = model.to_string();
    }
    Ok(config)
}

pub(crate) fn session_to_provider_request(session: &AgentSession) -> Result<ProviderRequest> {
    let user_prompt = session
        .conversation
        .first()
        .and_then(|item| match item {
            AgentItem::UserText(text) => Some(text.clone()),
            _ => None,
        })
        .ok_or_else(|| anyhow!("agent session missing initial user prompt"))?;
    let keep_reasoning = !matches!(resolve_profile(session.profile).reasoning, ReasoningPolicy::Off);

    let mut messages = Vec::new();
    let mut pending_text = None::<String>;
    let mut pending_reasoning = None::<ProviderReasoning>;
    let mut pending_calls = Vec::new();

    for item in session.conversation.iter().skip(1) {
        match item {
            AgentItem::AssistantText(text) => {
                append_assistant_text(&mut pending_text, text);
            }
            AgentItem::Reasoning(chunk) if keep_reasoning => {
                pending_reasoning = Some(provider_reasoning_from_chunk(chunk));
            }
            AgentItem::ToolCall(call) => pending_calls.push(call.clone()),
            AgentItem::ToolResult(result) => {
                flush_pending_tool_calls(
                    &mut messages,
                    &mut pending_text,
                    &mut pending_reasoning,
                    &mut pending_calls,
                );
                messages.push(ProviderMessage::ToolResult(result.clone()));
            }
            AgentItem::FinalAnswer(_) | AgentItem::UserText(_) => {}
            AgentItem::Reasoning(_) => {}
        }
    }

    flush_pending_tool_calls(
        &mut messages,
        &mut pending_text,
        &mut pending_reasoning,
        &mut pending_calls,
    );

    Ok(ProviderRequest {
        system_prompt: session.system_prompt.clone(),
        user_prompt,
        messages,
    })
}

fn model_turn_to_runtime_turn(turn: ProviderModelTurn, provider_state: Option<Value>) -> Result<ProviderTurn> {
    match turn {
        ProviderModelTurn::Final { text, reasoning } => {
            let mut items = reasoning.into_iter().map(agent_reasoning_item).collect::<Vec<_>>();
            items.push(AgentItem::FinalAnswer(text.clone()));
            Ok(ProviderTurn::Final {
                text,
                items,
                provider_state,
            })
        }
        ProviderModelTurn::ToolCalls { text, reasoning, calls } => {
            let mut items = reasoning.iter().cloned().map(agent_reasoning_item).collect::<Vec<_>>();
            if let Some(text) = text {
                items.push(AgentItem::AssistantText(text));
            }
            let calls = calls
                .into_iter()
                .inspect(|call| {
                    items.push(AgentItem::ToolCall(call.clone()));
                })
                .collect::<Vec<_>>();
            Ok(ProviderTurn::ToolCalls {
                items,
                calls,
                provider_state,
            })
        }
    }
}

pub(crate) fn reasoning_from_chunks(blocks: &[Value], text_chunks: Vec<String>) -> Option<ProviderReasoning> {
    let text = text_chunks
        .into_iter()
        .filter_map(|chunk| {
            let trimmed = chunk.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    if blocks.is_empty() && text.trim().is_empty() {
        return None;
    }

    let opaque_state = if blocks.is_empty() {
        None
    } else if blocks.len() == 1 {
        Some(blocks.first().cloned().unwrap_or(Value::Null))
    } else {
        Some(Value::Array(blocks.to_vec()))
    };

    let reasoning_text = if text.trim().is_empty() { None } else { Some(text) };

    Some(ProviderReasoning {
        text: reasoning_text,
        opaque_state,
    })
}

pub(crate) fn provider_tool_metadata_payloads(tools: &[ToolDefinition], schema_field: &str) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            let mut payload = json!({
                "name": tool.name.as_str(),
                "description": tool.description,
            });
            payload[schema_field] = tool.input_schema.clone();
            payload
        })
        .collect()
}

fn append_assistant_text(slot: &mut Option<String>, text: &str) {
    let slot = slot.get_or_insert_with(String::new);
    if !slot.is_empty() {
        slot.push('\n');
    }
    slot.push_str(text);
}

fn flush_pending_tool_calls(
    messages: &mut Vec<ProviderMessage>,
    pending_text: &mut Option<String>,
    pending_reasoning: &mut Option<ProviderReasoning>,
    pending_calls: &mut Vec<ToolCall>,
) {
    if pending_calls.is_empty() {
        return;
    }

    messages.push(ProviderMessage::AssistantToolCalls {
        text: pending_text.take(),
        reasoning: pending_reasoning.take(),
        calls: std::mem::take(pending_calls),
    });
}

fn provider_reasoning_from_chunk(chunk: &ReasoningChunk) -> ProviderReasoning {
    ProviderReasoning {
        text: chunk.text.clone(),
        opaque_state: chunk.opaque_state.clone(),
    }
}

fn agent_reasoning_item(reasoning: ProviderReasoning) -> AgentItem {
    AgentItem::Reasoning(ReasoningChunk {
        text: reasoning.text,
        opaque_state: reasoning.opaque_state,
    })
}

fn normalize_optional_text(text: String) -> Option<String> {
    (!text.trim().is_empty()).then_some(text)
}

pub(crate) fn provider_model_turn_from_parts(
    text: String,
    calls: Vec<ToolCall>,
    reasoning: Option<ProviderReasoning>,
    empty_error: &str,
) -> Result<ProviderModelTurn> {
    if !calls.is_empty() {
        return Ok(ProviderModelTurn::ToolCalls {
            text: normalize_optional_text(text),
            reasoning,
            calls,
        });
    }

    if let Some(text) = normalize_optional_text(text) {
        return Ok(ProviderModelTurn::Final { text, reasoning });
    }

    bail!(empty_error.to_string())
}

pub(crate) fn parse_agent_turn_response(
    response: reqwest::blocking::Response,
    body_context: &str,
    parse: impl FnOnce(&Value) -> Result<ProviderModelTurn>,
) -> Result<ProviderModelTurn> {
    let status = response.status();
    let body: Value = response.json().with_context(|| body_context.to_string())?;
    if !status.is_success() {
        bail!("Agentic request failed: status {status} body {body}");
    }

    parse(&body)
}

pub(crate) fn extract_reasoning_text(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => {
            let value = value.trim();
            if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            }
        }
        Value::Array(values) => {
            let collected = values
                .iter()
                .filter_map(extract_reasoning_text)
                .collect::<Vec<_>>()
                .join("\n");
            if collected.trim().is_empty() {
                None
            } else {
                Some(collected)
            }
        }
        Value::Object(value) => value
            .get("text")
            .and_then(Value::as_str)
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty())
            .or_else(|| value.get("summary").and_then(extract_reasoning_text))
            .or_else(|| value.get("content").and_then(extract_reasoning_text))
            .or_else(|| {
                value
                    .get("reasoning")
                    .and_then(Value::as_str)
                    .map(|text| text.trim().to_string())
                    .filter(|text| !text.is_empty())
            }),
        _ => None,
    }
}

pub(crate) fn provider_base_url(descriptor: &'static ProviderDescriptor) -> Result<&'static str> {
    descriptor
        .default_api_url
        .ok_or_else(|| anyhow!("provider '{}' is missing a default API URL", descriptor.id))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::extract_reasoning_text;

    #[test]
    fn extracts_reasoning_summary_text() {
        let text = extract_reasoning_text(&json!({
            "type": "reasoning",
            "summary": [{
                "type": "summary_text",
                "text": "Checking repository context before calling tools."
            }]
        }));

        assert_eq!(
            text.as_deref(),
            Some("Checking repository context before calling tools.")
        );
    }
}
