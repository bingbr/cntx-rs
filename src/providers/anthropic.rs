use std::io::{BufRead, BufReader};

use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::{Value, json};

use super::{catalog::provider_descriptor, transport};
use crate::{
    agent::types::{ProviderTurn, ProviderTurnRequest, ToolCall, ToolDefinition},
    config::Config,
};

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Serialize)]
struct AnthropicPayload<'a> {
    model: &'a str,
    max_tokens: u16,
    messages: Vec<ChatMessage<'a>>,
}

pub(crate) fn complete_anthropic_adapter_turn(request: ProviderTurnRequest<'_>) -> Result<ProviderTurn> {
    transport::complete_adapter_turn_with(request, |client, config, state, tools, api_key| {
        complete_anthropic_agent_turn(
            client,
            &anthropic_messages_endpoint()?,
            &config.model,
            transport::require_api_key(api_key, &config.provider)?,
            state,
            tools,
        )
    })
}

pub(crate) fn ask_anthropic(client: &Client, config: &Config, prompt: &str, api_key: Option<&str>) -> Result<String> {
    let payload = AnthropicPayload {
        model: &config.model,
        max_tokens: 1024,
        messages: vec![ChatMessage {
            role: "user",
            content: prompt,
        }],
    };

    let response = client
        .post(anthropic_messages_endpoint()?)
        .header("x-api-key", transport::require_api_key(api_key, &config.provider)?)
        .header("anthropic-version", "2023-06-01")
        .json(&payload)
        .send()?;

    parse_anthropic_response(response)
}

pub(crate) fn ask_anthropic_stream(
    client: &Client,
    config: &Config,
    prompt: &str,
    api_key: Option<&str>,
    on_chunk: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<String> {
    let payload = json!({
        "model": config.model,
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": prompt,
        }],
        "stream": true,
    });

    let response = client
        .post(anthropic_messages_endpoint()?)
        .header("x-api-key", transport::require_api_key(api_key, &config.provider)?)
        .header("anthropic-version", "2023-06-01")
        .json(&payload)
        .send()?;

    parse_anthropic_stream(response, on_chunk)
}

pub(crate) fn build_anthropic_messages(state: &transport::ProviderRequest) -> Result<Vec<Value>> {
    let mut messages = vec![json!({
        "role": "user",
        "content": state.user_prompt,
    })];

    transport::for_agent_tool_exchanges(state, |text, reasoning, calls, tool_results| {
        let mut content = anthropic_content_blocks(reasoning, text);
        for call in calls {
            content.push(json!({
                "type": "tool_use",
                "id": call.id,
                "name": call.tool.as_str(),
                "input": call.input,
            }));
        }
        messages.push(json!({
            "role": "assistant",
            "content": content,
        }));

        if !tool_results.is_empty() {
            let results = tool_results
                .iter()
                .map(|result| {
                    json!({
                        "type": "tool_result",
                        "tool_use_id": result.id,
                        "content": result.output,
                    })
                })
                .collect::<Vec<_>>();
            messages.push(json!({
                "role": "user",
                "content": results,
            }));
        }

        Ok(())
    })?;

    Ok(messages)
}

pub(crate) fn parse_anthropic_agent_turn(body: &Value) -> Result<transport::ProviderModelTurn> {
    let content = body
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("Anthropic agentic response missing content blocks"))?;

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut reasoning_blocks = Vec::new();
    let mut reasoning_text = Vec::new();

    for block in content {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(chunk) = block.get("text").and_then(Value::as_str) {
                    text.push_str(chunk);
                }
            }
            Some("thinking") => {
                reasoning_blocks.push(block.clone());
                if let Some(chunk) = block.get("text").and_then(Value::as_str) {
                    reasoning_text.push(chunk.to_string());
                }
            }
            Some("tool_use") => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("Anthropic tool_use block missing id"))?
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("Anthropic tool_use block missing name"))?;
                let tool = name.parse()?;
                let input = block.get("input").cloned().unwrap_or(Value::Object(Default::default()));
                tool_calls.push(ToolCall {
                    id,
                    tool,
                    input,
                    provider_data: None,
                });
            }
            _ => {}
        }
    }

    transport::provider_model_turn_from_parts(
        text,
        tool_calls,
        transport::reasoning_from_chunks(&reasoning_blocks, reasoning_text),
        "Anthropic agentic response did not contain text or tool calls",
    )
}

pub(crate) fn parse_anthropic_response(response: reqwest::blocking::Response) -> Result<String> {
    let status = response.status();
    let body: Value = response
        .json()
        .with_context(|| "Parsing Anthropic response JSON body")?;
    if !status.is_success() {
        return Err(anyhow!("Anthropic request failed: status {status} body {body}"));
    }

    body.get("content")
        .and_then(Value::as_array)
        .and_then(|items| items.iter().find_map(|item| item.get("text").and_then(Value::as_str)))
        .map(ToString::to_string)
        .ok_or_else(|| anyhow!("Anthropic response missing text content"))
}

fn parse_anthropic_stream(
    response: reqwest::blocking::Response,
    on_chunk: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<String> {
    let status = response.status();
    if !status.is_success() {
        let body = response
            .text()
            .with_context(|| format!("Anthropic request failed with status {status}"))?;
        return Err(anyhow!("Anthropic request failed: status {status} body {body}"));
    }

    let reader = BufReader::new(response);
    parse_anthropic_stream_reader(reader, on_chunk)
}

pub(crate) fn parse_anthropic_stream_reader<R: BufRead>(
    mut reader: R,
    on_chunk: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<String> {
    let mut answer = String::new();
    let mut plain_body = String::new();
    let mut saw_stream_data = false;
    let mut raw_stream_preview = String::new();
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader
            .read_line(&mut line)
            .with_context(|| "Reading Anthropic stream response")?;
        if bytes_read == 0 {
            break;
        }

        let trimmed = line.trim();
        if !saw_stream_data {
            plain_body.push_str(&line);
        }
        if trimmed.is_empty() || !trimmed.starts_with("data:") {
            continue;
        }

        saw_stream_data = true;
        if raw_stream_preview.len() < 1024 {
            let remaining = 1024 - raw_stream_preview.len();
            raw_stream_preview.push_str(&trimmed.chars().take(remaining).collect::<String>());
            raw_stream_preview.push('\n');
        }

        let data = trimmed.trim_start_matches("data:").trim();
        if data.is_empty() {
            continue;
        }

        let chunk = match serde_json::from_str::<Value>(data) {
            Ok(chunk) => chunk,
            Err(_) => continue,
        };

        if chunk.get("type").and_then(Value::as_str) != Some("content_block_delta") {
            continue;
        }

        let Some(delta) = chunk.get("delta") else {
            continue;
        };

        let content = match delta.get("type").and_then(Value::as_str) {
            Some("text_delta") => delta.get("text").and_then(Value::as_str),
            Some("thinking_delta") => delta.get("thinking").and_then(Value::as_str),
            _ => None,
        };

        if let Some(content) = content.filter(|content| !content.is_empty()) {
            answer.push_str(content);
            on_chunk(content)?;
        }
    }

    if saw_stream_data {
        if !answer.is_empty() {
            return Ok(answer);
        }
        let preview = raw_stream_preview.trim();
        return Err(anyhow!(
            "Anthropic stream completed without any text content. Stream preview: {}",
            if preview.is_empty() { "<empty stream>" } else { preview }
        ));
    }

    Err(anyhow!(
        "Provider did not return a streaming response. Response preview: {}",
        plain_body.trim()
    ))
}

fn complete_anthropic_agent_turn(
    client: &Client,
    endpoint: &str,
    model: &str,
    api_key: &str,
    state: &transport::ProviderRequest,
    tools: &[ToolDefinition],
) -> Result<transport::ProviderModelTurn> {
    let messages = build_anthropic_messages(state)?;
    let tool_payloads = transport::provider_tool_metadata_payloads(tools, "input_schema");

    let response = client
        .post(endpoint)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&json!({
            "model": model,
            "max_tokens": 1024,
            "system": state.system_prompt,
            "messages": messages,
            "tools": tool_payloads,
        }))
        .send()
        .with_context(|| format!("Sending Anthropic agent turn to {endpoint}"))?;

    transport::parse_agent_turn_response(
        response,
        "Parsing Anthropic agentic response JSON body",
        parse_anthropic_agent_turn,
    )
}

fn anthropic_messages_endpoint() -> Result<String> {
    let descriptor = provider_descriptor("anthropic").expect("anthropic descriptor");
    Ok(format!(
        "{}/messages",
        transport::provider_base_url(descriptor)?.trim_end_matches('/')
    ))
}

fn anthropic_content_blocks(reasoning: Option<&transport::ProviderReasoning>, text: Option<&str>) -> Vec<Value> {
    let mut content = Vec::new();
    if let Some(reasoning) = reasoning {
        if let Some(blocks) = reasoning_blocks_as_array(reasoning) {
            content.extend(blocks);
        } else if let Some(text) = reasoning.text.as_deref().filter(|text| !text.trim().is_empty()) {
            content.push(json!({
                "type": "thinking",
                "text": text,
            }));
        }
    }
    if let Some(text) = text.filter(|text| !text.trim().is_empty()) {
        content.push(json!({
            "type": "text",
            "text": text,
        }));
    }
    content
}

fn reasoning_blocks_as_array(reasoning: &transport::ProviderReasoning) -> Option<Vec<Value>> {
    if let Some(opaque_state) = &reasoning.opaque_state {
        return match opaque_state {
            Value::Array(blocks) => Some(blocks.clone()),
            Value::Object(block) => {
                if block.contains_key("type") {
                    Some(vec![opaque_state.clone()])
                } else {
                    None
                }
            }
            _ => None,
        };
    }

    None
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use serde_json::json;

    use super::{build_anthropic_messages, parse_anthropic_agent_turn, parse_anthropic_stream_reader};
    use crate::{
        agent::types::{ToolCall, ToolName, ToolResult},
        providers::transport::{ProviderMessage, ProviderModelTurn, ProviderReasoning, ProviderRequest},
    };

    fn provider_request_with_tool_result() -> ProviderRequest {
        ProviderRequest {
            system_prompt: "system".to_string(),
            user_prompt: "user question".to_string(),
            messages: vec![
                ProviderMessage::AssistantToolCalls {
                    text: Some("calling tools".to_string()),
                    reasoning: None,
                    calls: vec![ToolCall {
                        id: "call-1".to_string(),
                        tool: ToolName::Read,
                        input: json!({"path":"repo/README.md"}),
                        provider_data: None,
                    }],
                },
                ProviderMessage::ToolResult(ToolResult {
                    id: "call-1".to_string(),
                    tool: ToolName::Read,
                    output: "path: repo/README.md\n1: hello".to_string(),
                    is_error: false,
                    truncated: false,
                    citations: Vec::new(),
                }),
            ],
        }
    }

    #[test]
    fn parses_anthropic_sse_stream_incrementally() {
        let body = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n",
        );
        let mut chunks = Vec::new();

        let answer = parse_anthropic_stream_reader(Cursor::new(body.as_bytes()), &mut |chunk| {
            chunks.push(chunk.to_string());
            Ok(())
        })
        .expect("anthropic stream parse should succeed");

        assert_eq!(answer, "Hello");
        assert_eq!(chunks, vec!["Hel".to_string(), "lo".to_string()]);
    }

    #[test]
    fn parses_anthropic_agent_tool_use_turn() {
        let body = json!({
            "content": [
                {"type": "text", "text": "Let me inspect that."},
                {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "repo/README.md"}}
            ]
        });

        let turn = parse_anthropic_agent_turn(&body).expect("anthropic tool use should parse");

        match turn {
            ProviderModelTurn::ToolCalls { text, calls, reasoning } => {
                assert_eq!(text.as_deref(), Some("Let me inspect that."));
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "toolu_1");
                assert_eq!(calls[0].tool, ToolName::Read);
                assert_eq!(calls[0].input, json!({"path":"repo/README.md"}));
                assert!(reasoning.is_none());
            }
            ProviderModelTurn::Final { .. } => panic!("expected tool calls"),
        }
    }

    #[test]
    fn parses_anthropic_agent_reasoning_block() {
        let body = json!({
            "content": [
                {"type": "thinking", "text": "Let's inspect the README first."},
                {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "repo/README.md"}}
            ]
        });

        let turn = parse_anthropic_agent_turn(&body).expect("anthropic thinking should parse");

        match turn {
            ProviderModelTurn::ToolCalls { text, calls, reasoning } => {
                assert!(text.is_none());
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "toolu_1");
                let reasoning = reasoning.expect("reasoning chunk should be preserved");
                assert_eq!(reasoning.text, Some("Let's inspect the README first.".to_string()));
                assert_eq!(
                    reasoning.opaque_state,
                    Some(json!({"type":"thinking","text":"Let's inspect the README first."}))
                );
            }
            ProviderModelTurn::Final { .. } => panic!("expected tool calls"),
        }
    }

    #[test]
    fn anthropic_history_groups_tool_results_into_user_message() {
        let messages =
            build_anthropic_messages(&provider_request_with_tool_result()).expect("anthropic messages should build");

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"][0]["type"], "tool_result");
        assert_eq!(messages[2]["content"][0]["tool_use_id"], "call-1");
    }

    #[test]
    fn anthropic_history_replays_reasoning_block() {
        let mut state = provider_request_with_tool_result();
        if let ProviderMessage::AssistantToolCalls { reasoning, .. } = &mut state.messages[0] {
            *reasoning = Some(ProviderReasoning {
                text: Some("Let's inspect that file.".to_string()),
                opaque_state: Some(json!({"type":"thinking","text":"Let's inspect that file."})),
            });
        }

        let messages = build_anthropic_messages(&state).expect("anthropic messages should build");

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"][0]["type"], "thinking");
        assert_eq!(messages[1]["content"][0]["text"], "Let's inspect that file.");
    }
}
