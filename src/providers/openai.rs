use anyhow::{Context, Result, anyhow, bail};
use reqwest::{StatusCode, blocking::Client};
use serde_json::{Value, json};

use super::{catalog::provider_descriptor, transport};
use crate::{
    agent::types::{AgentItem, AgentSession, ProviderTurn, ProviderTurnRequest, ReasoningChunk, ToolCall},
    auth,
    config::Config,
};

pub(crate) fn complete_openai_adapter_turn(request: ProviderTurnRequest<'_>) -> Result<ProviderTurn> {
    let config = transport::profile_config_for_session(request.session)?;
    let descriptor = provider_descriptor("openai")?;
    let http_client = transport::client()?;
    let api_key = auth::resolve_api_key("openai", &config)?;
    let api_key = transport::resolved_api_key(api_key.as_deref(), descriptor, "openai")?;
    let previous_response_id = request
        .session
        .provider_state
        .as_ref()
        .and_then(|state| state.get("response_id"))
        .and_then(Value::as_str);

    let mut payload = responses_payload(
        &config.model,
        responses_input(request.session)?,
        request
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.name.as_str(),
                    "description": tool.description,
                    "parameters": tool.input_schema,
                })
            })
            .collect(),
        previous_response_id,
        request.stream,
    );
    payload["reasoning"] = json!({ "summary": "auto" });

    let endpoint = responses_endpoint(&config)?;
    let response = responses_request(&http_client, &endpoint, payload, api_key)
        .send()
        .with_context(|| format!("Sending OpenAI Responses turn to {endpoint}"))?;
    let turn = parse_responses_turn(response)?;
    Ok(with_responses_input_cursor(turn, request.session.conversation.len()))
}

pub(crate) fn ask_openai(client: &Client, config: &Config, prompt: &str, api_key: Option<&str>) -> Result<String> {
    let payload = responses_payload(
        &config.model,
        json!([
            {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": prompt }],
            }
        ]),
        Vec::new(),
        None,
        false,
    );
    let endpoint = responses_endpoint(config)?;
    let response = responses_request(client, &endpoint, payload, api_key)
        .send()
        .with_context(|| "Sending OpenAI Responses request")?;
    parse_responses_text(response)
}

pub(crate) fn ask_openai_stream(
    client: &Client,
    config: &Config,
    prompt: &str,
    _api_key: Option<&str>,
    _on_chunk: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<String> {
    let _ = (client, config, prompt);
    anyhow::bail!("provider 'openai' does not support true streaming yet")
}

fn responses_endpoint(_config: &Config) -> Result<String> {
    let descriptor = provider_descriptor("openai")?;
    Ok(format!(
        "{}/responses",
        transport::provider_base_url(descriptor)?.trim_end_matches('/')
    ))
}

fn responses_request(
    client: &Client,
    endpoint: &str,
    payload: Value,
    api_key: Option<&str>,
) -> reqwest::blocking::RequestBuilder {
    let mut request = client
        .post(endpoint)
        .header("Content-Type", "application/json")
        .json(&payload);
    if let Some(api_key) = api_key {
        request = request.bearer_auth(api_key);
    }
    request
}

fn responses_payload(
    model: &str,
    input: Value,
    tools: Vec<Value>,
    previous_response_id: Option<&str>,
    stream: bool,
) -> Value {
    let mut payload = json!({
        "model": model,
        "input": input,
        "tools": tools,
        "tool_choice": "auto",
        "stream": stream,
    });
    if let Some(previous_response_id) = previous_response_id.filter(|id| !id.trim().is_empty()) {
        payload["previous_response_id"] = json!(previous_response_id);
    }
    payload
}

fn responses_input(session: &AgentSession) -> Result<Value> {
    let mut items = Vec::new();
    for item in session.conversation.iter().skip(responses_input_start_index(session)) {
        match item {
            AgentItem::UserText(text) => items.push(json!({
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": text }],
            })),
            AgentItem::AssistantText(text) => items.push(json!({
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": text }],
            })),
            AgentItem::Reasoning(chunk) => {
                if let Some(opaque_state) = &chunk.opaque_state {
                    items.push(opaque_state.clone());
                } else if let Some(text) = chunk.text.as_ref().filter(|text| !text.trim().is_empty()) {
                    items.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": [{ "type": "output_text", "text": text }],
                    }));
                }
            }
            AgentItem::ToolCall(call) => {
                let mut payload = json!({
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.tool.as_str(),
                    "arguments": serde_json::to_string(&call.input)?,
                });
                if let Some(provider_state) = &call.provider_data
                    && let Some(id) = provider_state.get("id").and_then(Value::as_str)
                {
                    payload["id"] = json!(id);
                }
                items.push(payload);
            }
            AgentItem::ToolResult(result) => items.push(json!({
                "type": "function_call_output",
                "id": format!("fc_output_{}", result.id),
                "call_id": result.id,
                "output": result.output,
            })),
            AgentItem::FinalAnswer(text) => items.push(json!({
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": text }],
            })),
        }
    }
    Ok(Value::Array(items))
}

fn responses_input_start_index(session: &AgentSession) -> usize {
    let Some(state) = session.provider_state.as_ref() else {
        return 0;
    };
    if state.get("response_id").and_then(Value::as_str).is_none() {
        return 0;
    }
    if let Some(index) = state
        .get("input_cursor")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|index| *index <= session.conversation.len())
    {
        return index;
    }

    session
        .conversation
        .iter()
        .rposition(|item| !matches!(item, AgentItem::UserText(_) | AgentItem::ToolResult(_)))
        .map_or(session.conversation.len(), |index| index + 1)
}

fn with_responses_input_cursor(turn: ProviderTurn, conversation_len: usize) -> ProviderTurn {
    match turn {
        ProviderTurn::Final {
            text,
            items,
            provider_state,
        } => ProviderTurn::Final {
            text,
            provider_state: augment_responses_provider_state(provider_state, conversation_len + items.len()),
            items,
        },
        ProviderTurn::ToolCalls {
            items,
            calls,
            provider_state,
        } => ProviderTurn::ToolCalls {
            calls,
            provider_state: augment_responses_provider_state(provider_state, conversation_len + items.len()),
            items,
        },
    }
}

fn augment_responses_provider_state(provider_state: Option<Value>, next_input_index: usize) -> Option<Value> {
    let mut provider_state = provider_state?;
    if let Some(state) = provider_state.as_object_mut() {
        state.insert("input_cursor".to_string(), json!(next_input_index));
    }
    Some(provider_state)
}

fn parse_responses_turn(response: reqwest::blocking::Response) -> Result<ProviderTurn> {
    let status = response.status();
    let body: Value = response
        .json()
        .with_context(|| "Parsing OpenAI Responses turn JSON body")?;
    if !status.is_success() {
        bail!("{}", responses_error_message(status, &body, "turn"));
    }

    let provider_state = body
        .get("id")
        .and_then(Value::as_str)
        .map(|id| json!({ "response_id": id }));
    let outputs = body
        .get("output")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("OpenAI Responses output missing output array"))?;

    let mut text = String::new();
    let mut items = Vec::new();
    let mut calls = Vec::new();

    for output in outputs {
        match output.get("type").and_then(Value::as_str) {
            Some("message") => {
                if let Some(content) = output.get("content").and_then(Value::as_array) {
                    for part in content {
                        if let Some(part_text) = extract_responses_content_text(part) {
                            text.push_str(&part_text);
                        }
                    }
                }
            }
            Some("function_call") => {
                let name = output
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("OpenAI function_call missing name"))?;
                let call_id = output
                    .get("call_id")
                    .and_then(Value::as_str)
                    .or_else(|| output.get("id").and_then(Value::as_str))
                    .ok_or_else(|| anyhow!("OpenAI function_call missing call id"))?;
                let arguments = output.get("arguments").and_then(Value::as_str).unwrap_or("{}");
                let input = serde_json::from_str(arguments)
                    .with_context(|| format!("Invalid OpenAI function_call arguments for '{name}'"))?;
                let call = ToolCall {
                    id: call_id.to_string(),
                    tool: name.parse()?,
                    input,
                    provider_data: output.get("id").and_then(Value::as_str).map(|id| json!({ "id": id })),
                };
                items.push(AgentItem::ToolCall(call.clone()));
                calls.push(call);
            }
            Some("reasoning") => {
                items.push(AgentItem::Reasoning(ReasoningChunk {
                    text: transport::extract_reasoning_text(output),
                    opaque_state: Some(output.clone()),
                }));
            }
            _ => {}
        }
    }

    if !text.trim().is_empty() {
        items.insert(0, AgentItem::AssistantText(text.clone()));
    }

    if !calls.is_empty() {
        return Ok(ProviderTurn::ToolCalls {
            items,
            calls,
            provider_state,
        });
    }

    if !text.trim().is_empty() {
        if !matches!(items.last(), Some(AgentItem::FinalAnswer(_))) {
            items.push(AgentItem::FinalAnswer(text.clone()));
        }
        return Ok(ProviderTurn::Final {
            text,
            items,
            provider_state,
        });
    }

    bail!("OpenAI Responses turn did not contain text or tool calls")
}

fn parse_responses_text(response: reqwest::blocking::Response) -> Result<String> {
    let status = response.status();
    let body: Value = response
        .json()
        .with_context(|| "Parsing OpenAI Responses response JSON body")?;
    if !status.is_success() {
        return Err(anyhow!("{}", responses_error_message(status, &body, "request")));
    }
    extract_responses_output_text(&body).ok_or_else(|| anyhow!("OpenAI Responses payload missing output text"))
}

fn responses_error_message(status: StatusCode, body: &Value, phase: &str) -> String {
    format!("OpenAI Responses {phase} failed: status {status} body {body}")
}

fn extract_responses_output_text(body: &Value) -> Option<String> {
    let outputs = body.get("output").and_then(Value::as_array)?;
    let mut text = String::new();
    for item in outputs {
        if item.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        if let Some(content) = item.get("content").and_then(Value::as_array) {
            for part in content {
                if let Some(part_text) = extract_responses_content_text(part) {
                    text.push_str(&part_text);
                }
            }
        }
    }
    if text.is_empty() { None } else { Some(text) }
}

fn extract_responses_content_text(value: &Value) -> Option<String> {
    match value {
        Value::Object(map) => {
            for key in ["text", "content"] {
                if let Some(text) = map.get(key).and_then(extract_responses_content_text) {
                    return Some(text);
                }
            }
            map.get("value").and_then(Value::as_str).map(ToString::to_string)
        }
        _ => extract_openai_content_text(value),
    }
}

fn extract_openai_content_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Array(items) => {
            let mut text = String::new();
            for item in items {
                if let Some(part) = extract_openai_content_text(item) {
                    text.push_str(&part);
                }
            }
            if text.is_empty() { None } else { Some(text) }
        }
        Value::Object(map) => {
            for key in ["text", "content"] {
                if let Some(text) = map.get(key).and_then(extract_openai_content_text) {
                    return Some(text);
                }
            }
            None
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{responses_input, with_responses_input_cursor};
    use crate::agent::types::{
        AgentItem, AgentProfileId, AgentSession, ProviderTurn, ReasoningChunk, ToolCall, ToolName, ToolResult,
    };

    #[test]
    fn responses_history_includes_function_call_output_id() {
        let session = AgentSession {
            profile: AgentProfileId::OpenAi,
            provider: "openai".to_string(),
            system_prompt: "system".to_string(),
            model: "gpt-5-mini".to_string(),
            conversation: vec![
                AgentItem::UserText("question".to_string()),
                AgentItem::ToolResult(ToolResult {
                    id: "call-1".to_string(),
                    tool: ToolName::Read,
                    output: "README content".to_string(),
                    citations: Vec::new(),
                    truncated: false,
                    is_error: false,
                }),
            ],
            provider_state: None,
        };

        let payload = responses_input(&session).expect("payload should build");
        let items = payload.as_array().expect("input should be an array");

        assert_eq!(items[1]["type"], "function_call_output");
        assert_eq!(items[1]["call_id"], "call-1");
        assert_eq!(items[1]["id"], "fc_output_call-1");
    }

    #[test]
    fn responses_history_omits_null_function_call_id() {
        let session = AgentSession {
            profile: AgentProfileId::OpenAi,
            provider: "openai".to_string(),
            system_prompt: "system".to_string(),
            model: "gpt-5-mini".to_string(),
            conversation: vec![AgentItem::ToolCall(ToolCall {
                id: "call-1".to_string(),
                tool: ToolName::Read,
                input: json!({"path":"README.md"}),
                provider_data: Some(json!({"id": null})),
            })],
            provider_state: None,
        };

        let payload = responses_input(&session).expect("payload should build");
        let items = payload.as_array().expect("input should be an array");

        assert_eq!(items[0]["type"], "function_call");
        assert!(items[0].get("id").is_none());
    }

    #[test]
    fn responses_history_replays_only_new_items_after_previous_response() {
        let session = AgentSession {
            profile: AgentProfileId::OpenAi,
            provider: "openai".to_string(),
            system_prompt: "system".to_string(),
            model: "gpt-5-mini".to_string(),
            conversation: vec![
                AgentItem::UserText("question".to_string()),
                AgentItem::Reasoning(ReasoningChunk {
                    text: Some("thinking".to_string()),
                    opaque_state: Some(json!({"id":"rs_123","type":"reasoning"})),
                }),
                AgentItem::ToolCall(ToolCall {
                    id: "call-1".to_string(),
                    tool: ToolName::Read,
                    input: json!({"path":"README.md"}),
                    provider_data: Some(json!({"id":"fc_123"})),
                }),
                AgentItem::ToolResult(ToolResult {
                    id: "call-1".to_string(),
                    tool: ToolName::Read,
                    output: "README content".to_string(),
                    citations: Vec::new(),
                    truncated: false,
                    is_error: false,
                }),
            ],
            provider_state: Some(json!({"response_id":"resp_123","input_cursor":3})),
        };

        let payload = responses_input(&session).expect("payload should build");
        let items = payload.as_array().expect("input should be an array");

        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "function_call_output");
        assert_eq!(items[0]["call_id"], "call-1");
    }

    #[test]
    fn responses_history_without_cursor_keeps_only_trailing_user_items() {
        let session = AgentSession {
            profile: AgentProfileId::OpenAi,
            provider: "openai".to_string(),
            system_prompt: "system".to_string(),
            model: "gpt-5-mini".to_string(),
            conversation: vec![
                AgentItem::UserText("question".to_string()),
                AgentItem::Reasoning(ReasoningChunk {
                    text: Some("thinking".to_string()),
                    opaque_state: Some(json!({"id":"rs_123","type":"reasoning"})),
                }),
                AgentItem::ToolCall(ToolCall {
                    id: "call-1".to_string(),
                    tool: ToolName::Read,
                    input: json!({"path":"README.md"}),
                    provider_data: Some(json!({"id":"fc_123"})),
                }),
                AgentItem::ToolResult(ToolResult {
                    id: "call-1".to_string(),
                    tool: ToolName::Read,
                    output: "README content".to_string(),
                    citations: Vec::new(),
                    truncated: false,
                    is_error: false,
                }),
            ],
            provider_state: Some(json!({"response_id":"resp_123"})),
        };

        let payload = responses_input(&session).expect("payload should build");
        let items = payload.as_array().expect("input should be an array");

        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "function_call_output");
        assert_eq!(items[0]["call_id"], "call-1");
    }

    #[test]
    fn responses_turn_provider_state_tracks_next_input_cursor() {
        let turn = with_responses_input_cursor(
            ProviderTurn::ToolCalls {
                items: vec![
                    AgentItem::AssistantText("calling tools".to_string()),
                    AgentItem::ToolCall(ToolCall {
                        id: "call-1".to_string(),
                        tool: ToolName::Read,
                        input: json!({"path":"README.md"}),
                        provider_data: Some(json!({"id":"fc_123"})),
                    }),
                ],
                calls: vec![ToolCall {
                    id: "call-1".to_string(),
                    tool: ToolName::Read,
                    input: json!({"path":"README.md"}),
                    provider_data: Some(json!({"id":"fc_123"})),
                }],
                provider_state: Some(json!({"response_id":"resp_123"})),
            },
            1,
        );

        let ProviderTurn::ToolCalls { provider_state, .. } = turn else {
            panic!("expected tool-call turn");
        };
        assert_eq!(provider_state.expect("provider state should exist")["input_cursor"], 3);
    }
}
