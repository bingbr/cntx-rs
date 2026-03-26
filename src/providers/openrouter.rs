use std::io::{BufRead, BufReader};

use anyhow::{Context, Result, anyhow, bail};
use reqwest::blocking::Client;
use serde_json::{Value, json};

use super::{catalog::provider_descriptor, transport};
use crate::{
    agent::types::{ProviderTurn, ProviderTurnRequest, ToolCall, ToolDefinition, ToolName},
    config::Config,
};

pub(crate) fn complete_openrouter_adapter_turn(request: ProviderTurnRequest<'_>) -> Result<ProviderTurn> {
    transport::complete_adapter_turn_with(request, |client, config, state, tools, api_key| {
        complete_openrouter_agent_turn(client, config, state, tools, api_key)
    })
}

pub(crate) fn ask_openrouter(client: &Client, config: &Config, prompt: &str, api_key: Option<&str>) -> Result<String> {
    let response = openrouter_request(
        client,
        &openrouter_chat_endpoint()?,
        openrouter_prompt_payload(config.model.as_str(), prompt, None),
        api_key,
    )
    .send()
    .with_context(|| "Sending OpenRouter request")?;

    parse_openrouter_response(response)
}

pub(crate) fn ask_openrouter_stream(
    client: &Client,
    config: &Config,
    prompt: &str,
    api_key: Option<&str>,
    on_chunk: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<String> {
    let response = openrouter_request(
        client,
        &openrouter_chat_endpoint()?,
        openrouter_prompt_payload(config.model.as_str(), prompt, Some(true)),
        api_key,
    )
    .send()
    .with_context(|| "Sending OpenRouter streaming request")?;

    parse_openrouter_stream(response, on_chunk)
}

fn openrouter_chat_endpoint() -> Result<String> {
    let descriptor = provider_descriptor("openrouter")?;
    Ok(format!(
        "{}/chat/completions",
        transport::provider_base_url(descriptor)?.trim_end_matches('/')
    ))
}

fn complete_openrouter_agent_turn(
    client: &Client,
    config: &Config,
    state: &transport::ProviderRequest,
    tools: &[ToolDefinition],
    api_key: Option<&str>,
) -> Result<transport::ProviderModelTurn> {
    let messages = build_agent_messages(state);
    let tool_payloads = tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name.as_str(),
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            })
        })
        .collect::<Vec<_>>();

    let response = openrouter_request(
        client,
        &openrouter_chat_endpoint()?,
        openrouter_agent_turn_payload(config.model.as_str(), messages, tool_payloads),
        api_key,
    )
    .send()
    .with_context(|| "Sending OpenRouter agent turn")?;

    transport::parse_agent_turn_response(
        response,
        "Parsing OpenRouter agent response JSON body",
        parse_openrouter_agent_turn,
    )
}

fn openrouter_request(
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

fn openrouter_prompt_payload(model: &str, prompt: &str, stream: Option<bool>) -> Value {
    openrouter_chat_payload(
        model,
        vec![json!({
            "role": "user",
            "content": prompt,
        })],
        stream,
    )
}

fn openrouter_chat_payload(model: &str, messages: Vec<Value>, stream: Option<bool>) -> Value {
    let mut payload = json!({
        "model": model,
        "messages": messages,
        "max_tokens": 1024,
        // OpenRouter reasoning models may spend the entire completion budget on hidden reasoning,
        // returning a successful response with null message.content. Disable reasoning for the
        // plain chat-completions path because this adapter does not preserve reasoning blocks.
        "reasoning": {
            "effort": "none",
            "exclude": true,
        },
    });
    if let Some(stream) = stream {
        payload["stream"] = json!(stream);
    }
    payload
}

fn openrouter_agent_turn_payload(model: &str, messages: Vec<Value>, tool_payloads: Vec<Value>) -> Value {
    let mut payload = openrouter_chat_payload(model, messages, None);
    payload["tools"] = json!(tool_payloads);
    payload["tool_choice"] = json!("auto");
    payload
}

fn build_agent_messages(state: &transport::ProviderRequest) -> Vec<Value> {
    let mut messages = vec![
        json!({
            "role": "system",
            "content": state.system_prompt,
        }),
        json!({
            "role": "user",
            "content": state.user_prompt,
        }),
    ];

    for message in &state.messages {
        match message {
            transport::ProviderMessage::AssistantToolCalls { text, calls, reasoning } => {
                let _ = reasoning;
                let tool_calls = calls
                    .iter()
                    .map(|call| {
                        json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.tool.as_str(),
                                "arguments": serde_json::to_string(&call.input)
                                    .unwrap_or_else(|_| "{}".to_string())
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                messages.push(json!({
                    "role": "assistant",
                    "content": text.clone().unwrap_or_default(),
                    "tool_calls": tool_calls
                }));
            }
            transport::ProviderMessage::ToolResult(result) => messages.push(json!({
                "role": "tool",
                "tool_call_id": result.id,
                "content": result.output,
            })),
        }
    }

    messages
}

fn parse_openrouter_response(response: reqwest::blocking::Response) -> Result<String> {
    let status = response.status();
    let body: Value = response
        .json()
        .with_context(|| "Parsing OpenRouter response JSON body")?;
    if !status.is_success() {
        return Err(anyhow!("Request failed: status {status} body {body}"));
    }

    extract_openrouter_message_content(&body).ok_or_else(|| anyhow!("Model response missing message content"))
}

fn parse_openrouter_stream(
    response: reqwest::blocking::Response,
    on_chunk: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<String> {
    let status = response.status();
    if !status.is_success() {
        let body = response
            .text()
            .with_context(|| format!("Request failed with status {status}"))?;
        return Err(anyhow!("Request failed: status {status} body {body}"));
    }

    let reader = BufReader::new(response);
    parse_openai_like_stream_reader(reader, on_chunk)
}

fn parse_openai_like_stream_reader<R: BufRead>(
    mut reader: R,
    on_chunk: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<String> {
    let mut answer = String::new();
    let mut fallback_message = None;
    let mut plain_body = String::new();
    let mut saw_stream_data = false;
    let mut raw_stream_preview = String::new();
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader
            .read_line(&mut line)
            .with_context(|| "Reading OpenAI-like stream response")?;
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
        if data == "[DONE]" {
            break;
        }
        if data.is_empty() {
            continue;
        }

        let chunk = match serde_json::from_str::<Value>(data) {
            Ok(chunk) => chunk,
            Err(_) => continue,
        };

        if let Some(content) = chunk
            .get("choices")
            .and_then(|choices| choices.as_array())
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("delta"))
            .and_then(|delta| delta.get("content"))
            .and_then(extract_openai_content_text)
        {
            answer.push_str(&content);
            on_chunk(&content)?;
            continue;
        }

        if let Some(content) = chunk
            .get("choices")
            .and_then(|choices| choices.as_array())
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(extract_openai_content_text)
        {
            fallback_message = Some(content);
        }
    }

    if saw_stream_data {
        if !answer.is_empty() {
            return Ok(answer);
        }
        if let Some(content) = fallback_message {
            on_chunk(&content)?;
            return Ok(content);
        }
        let preview = raw_stream_preview.trim();
        return Err(anyhow!(
            "Model stream completed without any text content. Stream preview: {}",
            if preview.is_empty() { "<empty stream>" } else { preview }
        ));
    }

    Err(anyhow!(
        "Provider did not return a streaming response. Response preview: {}",
        plain_body.trim()
    ))
}

fn parse_openrouter_agent_turn(body: &Value) -> Result<transport::ProviderModelTurn> {
    let choice = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| anyhow!("Agentic model response missing choice payload"))?;
    let message = body
        .get("message")
        .or_else(|| choice.get("message"))
        .ok_or_else(|| anyhow!("Agentic model response missing message payload"))?;

    let text = message
        .get("content")
        .and_then(extract_openai_content_text)
        .filter(|text| !text.trim().is_empty());
    let refusal = message
        .get("refusal")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToString::to_string);

    let tool_calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| calls.iter().map(parse_tool_call).collect::<Result<Vec<_>>>())
        .transpose()?
        .unwrap_or_default();

    let reasoning = message
        .get("reasoning")
        .and_then(parse_reasoning_chunk)
        .or_else(|| message.get("reasoning_content").and_then(parse_reasoning_chunk))
        .filter(provider_reasoning_is_present);

    if !tool_calls.is_empty() {
        return Ok(transport::ProviderModelTurn::ToolCalls {
            text,
            reasoning,
            calls: tool_calls,
        });
    }

    if let Some(text) = text {
        return Ok(transport::ProviderModelTurn::Final { text, reasoning });
    }

    if let Some(refusal) = refusal {
        return Ok(transport::ProviderModelTurn::Final {
            text: refusal,
            reasoning,
        });
    }

    let finish_reason = choice.get("finish_reason").and_then(Value::as_str).unwrap_or("unknown");
    bail!("Agentic model response did not contain text or tool calls (finish_reason={finish_reason}). Body: {body}")
}

fn parse_reasoning_chunk(value: &Value) -> Option<transport::ProviderReasoning> {
    let text = transport::extract_reasoning_text(value);
    if text.is_none() && value.is_null() {
        return None;
    }
    Some(transport::ProviderReasoning {
        text,
        opaque_state: if value.is_null() { None } else { Some(value.clone()) },
    })
}

fn provider_reasoning_is_present(reasoning: &transport::ProviderReasoning) -> bool {
    reasoning.text.as_ref().filter(|text| !text.trim().is_empty()).is_some() || reasoning.opaque_state.is_some()
}

fn parse_tool_call(value: &Value) -> Result<ToolCall> {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("Tool call missing id"))?
        .to_string();
    let function = value
        .get("function")
        .ok_or_else(|| anyhow!("Tool call missing function payload"))?;
    let tool: ToolName = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("Tool call missing function name"))?
        .parse()?;
    let raw_arguments = function.get("arguments").and_then(Value::as_str).unwrap_or("{}");
    let input = serde_json::from_str(raw_arguments)
        .with_context(|| format!("Invalid tool call arguments for '{}'", tool.as_str()))?;

    Ok(ToolCall {
        id,
        tool,
        input,
        provider_data: None,
    })
}

fn extract_openrouter_message_content(body: &Value) -> Option<String> {
    body.get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(extract_openai_content_text)
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
    use std::io::Cursor;

    use serde_json::json;

    use super::{parse_openai_like_stream_reader, parse_openrouter_agent_turn};
    use crate::providers::transport::ProviderModelTurn;

    #[test]
    fn parses_sse_stream_incrementally() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
            "data: [DONE]\n"
        );
        let mut chunks = Vec::new();

        let answer = parse_openai_like_stream_reader(Cursor::new(body.as_bytes()), &mut |chunk| {
            chunks.push(chunk.to_string());
            Ok(())
        })
        .expect("stream parse should succeed");

        assert_eq!(answer, "Hello");
        assert_eq!(chunks, vec!["Hel".to_string(), "lo".to_string()]);
    }

    #[test]
    fn parses_sse_stream_with_block_content() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":[{\"type\":\"output_text\",\"text\":\"Hel\"}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":[{\"type\":\"output_text\",\"text\":\"lo\"}]}}]}\n\n",
            "data: [DONE]\n"
        );
        let mut chunks = Vec::new();

        let answer = parse_openai_like_stream_reader(Cursor::new(body.as_bytes()), &mut |chunk| {
            chunks.push(chunk.to_string());
            Ok(())
        })
        .expect("block stream parse should succeed");

        assert_eq!(answer, "Hello");
        assert_eq!(chunks, vec!["Hel".to_string(), "lo".to_string()]);
    }

    #[test]
    fn falls_back_to_message_content_when_stream_has_no_delta_text() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            "data: {\"choices\":[{\"message\":{\"content\":[{\"type\":\"output_text\",\"text\":\"final answer\"}]}}]}\n\n",
            "data: [DONE]\n"
        );
        let mut chunks = Vec::new();

        let answer = parse_openai_like_stream_reader(Cursor::new(body.as_bytes()), &mut |chunk| {
            chunks.push(chunk.to_string());
            Ok(())
        })
        .expect("message fallback should succeed");

        assert_eq!(answer, "final answer");
        assert_eq!(chunks, vec!["final answer".to_string()]);
    }

    #[test]
    fn parses_openrouter_agent_refusal_as_final_text() {
        let body = json!({
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "refusal": "I can't help with that."
                }
            }]
        });

        let turn = parse_openrouter_agent_turn(&body).expect("refusal should parse");

        match turn {
            ProviderModelTurn::Final { text, reasoning } => {
                assert_eq!(text, "I can't help with that.");
                assert!(reasoning.is_none());
            }
            ProviderModelTurn::ToolCalls { .. } => {
                panic!("expected final text")
            }
        }
    }

    #[test]
    fn openrouter_agent_parse_error_includes_finish_reason_and_body() {
        let body = json!({
            "choices": [{
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": []
                }
            }]
        });

        let error = parse_openrouter_agent_turn(&body).expect_err("empty message should fail");
        let rendered = error.to_string();

        assert!(rendered.contains("finish_reason=length"));
        assert!(rendered.contains("\"choices\""));
    }
}
