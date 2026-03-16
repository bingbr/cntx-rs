use std::io::{BufRead, BufReader};

use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::{Value, json};

use super::{catalog::provider_descriptor, transport};
use crate::{
    agent::types::{ProviderTurn, ProviderTurnRequest, ToolCall, ToolDefinition, ToolName},
    config::Config,
};

#[derive(Serialize)]
struct GeminiPart<'a> {
    text: &'a str,
}

#[derive(Serialize)]
struct GeminiContent<'a> {
    parts: Vec<GeminiPart<'a>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    max_output_tokens: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GeminiThinkingConfig>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiThinkingConfig {
    include_thoughts: bool,
}

#[derive(Serialize)]
struct GeminiPayload<'a> {
    contents: Vec<GeminiContent<'a>>,
    #[serde(rename = "generationConfig")]
    generation_config: GeminiGenerationConfig,
}

pub(crate) fn complete_gemini_adapter_turn(request: ProviderTurnRequest<'_>) -> Result<ProviderTurn> {
    transport::complete_adapter_turn_with(request, |client, config, state, tools, api_key| {
        complete_gemini_agent_turn(
            client,
            &gemini_endpoint(config, false),
            transport::require_api_key(api_key, &config.provider)?,
            state,
            tools,
        )
    })
}

pub(crate) fn ask_gemini(client: &Client, config: &Config, prompt: &str, api_key: &str) -> Result<String> {
    let endpoint = gemini_endpoint(config, false);
    call_gemini(client, &endpoint, prompt, api_key)
}

pub(crate) fn ask_gemini_stream(
    client: &Client,
    config: &Config,
    prompt: &str,
    api_key: &str,
    on_chunk: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<String> {
    let endpoint = gemini_endpoint(config, true);
    call_gemini_stream(client, &endpoint, prompt, api_key, on_chunk)
}

pub(crate) fn build_gemini_contents(state: &transport::ProviderRequest) -> Result<Vec<Value>> {
    let mut contents = vec![json!({
        "role": "user",
        "parts": [{ "text": state.user_prompt }],
    })];

    transport::for_agent_tool_exchanges(state, |text, reasoning, calls, tool_results| {
        let mut parts = gemini_content_parts(reasoning, text);
        for call in calls {
            let mut part = json!({
                "functionCall": {
                    "name": call.tool.as_str(),
                    "args": call.input,
                }
            });
            if let Some(provider_data) = &call.provider_data
                && let Some(thought_signature) = provider_data.get("thoughtSignature")
            {
                part["thoughtSignature"] = thought_signature.clone();
            }
            parts.push(part);
        }
        contents.push(json!({
            "role": "model",
            "parts": parts,
        }));

        if !tool_results.is_empty() {
            let parts = tool_results
                .iter()
                .map(|result| {
                    json!({
                        "functionResponse": {
                            "name": result.tool.as_str(),
                            "response": {
                                "output": result.output,
                                "is_error": result.is_error,
                                "truncated": result.truncated,
                                "citations": result.citations,
                            }
                        }
                    })
                })
                .collect::<Vec<_>>();
            contents.push(json!({
                "role": "user",
                "parts": parts,
            }));
        }

        Ok(())
    })?;

    Ok(contents)
}

pub(crate) fn parse_gemini_agent_turn(body: &Value) -> Result<transport::ProviderModelTurn> {
    let parts = body
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|candidates| candidates.first())
        .and_then(|candidate| candidate.get("content"))
        .and_then(|content| content.get("parts"))
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("Gemini agentic response missing candidate content"))?;

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut reasoning_blocks = Vec::new();
    let mut reasoning_text = Vec::new();

    for part in parts {
        if part.get("thought").and_then(Value::as_bool).unwrap_or(false) {
            reasoning_blocks.push(part.clone());
            if let Some(chunk) = part.get("text").and_then(Value::as_str) {
                reasoning_text.push(chunk.to_string());
            }
            continue;
        }

        if let Some(chunk) = part.get("text").and_then(Value::as_str) {
            text.push_str(chunk);
            continue;
        }

        if let Some(function_call) = part.get("functionCall") {
            let name = function_call
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("Gemini functionCall missing name"))?;
            let tool: ToolName = name.parse()?;
            let input = function_call
                .get("args")
                .cloned()
                .unwrap_or(Value::Object(Default::default()));
            let provider_data = part.get("thoughtSignature").cloned().map(|thought_signature| {
                json!({
                    "thoughtSignature": thought_signature
                })
            });
            tool_calls.push(ToolCall {
                id: format!("call-{}-{}", tool.as_str(), tool_calls.len() + 1),
                tool,
                input,
                provider_data,
            });
            continue;
        }

        if let Some(thought_signature) = part.get("thoughtSignature") {
            reasoning_blocks.push(json!({
                "thoughtSignature": thought_signature
            }));
        }
    }

    transport::provider_model_turn_from_parts(
        text,
        tool_calls,
        transport::reasoning_from_chunks(&reasoning_blocks, reasoning_text),
        "Gemini agentic response did not contain text or tool calls",
    )
}

pub(crate) fn parse_gemini_response(response: reqwest::blocking::Response) -> Result<String> {
    let status = response.status();
    let body: Value = response.json().with_context(|| "Parsing Gemini response JSON body")?;
    if !status.is_success() {
        return Err(anyhow!("Gemini request failed: status {status} body {body}"));
    }

    extract_gemini_text(&body).ok_or_else(|| anyhow!("Gemini response missing text content"))
}

pub(crate) fn parse_gemini_stream_reader<R: BufRead>(
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
            .with_context(|| "Reading Gemini stream response")?;
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
        if data == "[DONE]" || data.is_empty() {
            continue;
        }

        let chunk = match serde_json::from_str::<Value>(data) {
            Ok(chunk) => chunk,
            Err(_) => continue,
        };

        if let Some(content) = extract_gemini_text(&chunk) {
            answer.push_str(&content);
            on_chunk(&content)?;
        }
    }

    if saw_stream_data {
        if !answer.is_empty() {
            return Ok(answer);
        }

        let preview = raw_stream_preview.trim();
        return Err(anyhow!(
            "Gemini stream completed without any text content. Stream preview: {}",
            if preview.is_empty() { "<empty stream>" } else { preview }
        ));
    }

    Err(anyhow!(
        "Provider did not return a streaming response. Response preview: {}",
        plain_body.trim()
    ))
}

fn complete_gemini_agent_turn(
    client: &Client,
    endpoint: &str,
    api_key: &str,
    state: &transport::ProviderRequest,
    tools: &[ToolDefinition],
) -> Result<transport::ProviderModelTurn> {
    let contents = build_gemini_contents(state)?;
    let function_declarations = transport::provider_tool_metadata_payloads(tools, "parameters");

    let response = client
        .post(endpoint)
        .header("x-goog-api-key", api_key)
        .header("Content-Type", "application/json")
        .json(&json!({
            "systemInstruction": {
                "parts": [{ "text": state.system_prompt }]
            },
            "contents": contents,
            "generationConfig": {
                "thinkingConfig": {
                    "includeThoughts": true
                }
            },
            "tools": [{
                "functionDeclarations": function_declarations
            }],
        }))
        .send()
        .with_context(|| format!("Sending Gemini agent turn to {endpoint}"))?;

    transport::parse_agent_turn_response(
        response,
        "Parsing Gemini agentic response JSON body",
        parse_gemini_agent_turn,
    )
}

fn gemini_endpoint(config: &Config, stream: bool) -> String {
    let base = transport::provider_base_url(provider_descriptor("gemini").expect("gemini descriptor"))
        .expect("gemini default URL");
    let suffix = if stream {
        ":streamGenerateContent?alt=sse"
    } else {
        ":generateContent"
    };
    format!("{}/models/{}{}", base.trim_end_matches('/'), config.model, suffix)
}

fn call_gemini(client: &Client, endpoint: &str, prompt: &str, api_key: &str) -> Result<String> {
    let response = gemini_request(client, endpoint, prompt, api_key)
        .send()
        .with_context(|| format!("Sending Gemini request to {endpoint}"))?;

    parse_gemini_response(response)
}

fn call_gemini_stream(
    client: &Client,
    endpoint: &str,
    prompt: &str,
    api_key: &str,
    on_chunk: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<String> {
    let response = gemini_request(client, endpoint, prompt, api_key)
        .send()
        .with_context(|| format!("Sending Gemini streaming request to {endpoint}"))?;

    parse_gemini_stream(response, on_chunk)
}

fn gemini_request<'a>(
    client: &'a Client,
    endpoint: &'a str,
    prompt: &'a str,
    api_key: &'a str,
) -> reqwest::blocking::RequestBuilder {
    client
        .post(endpoint)
        .header("x-goog-api-key", api_key)
        .header("Content-Type", "application/json")
        .json(&gemini_payload(prompt))
}

fn gemini_payload(prompt: &str) -> GeminiPayload<'_> {
    GeminiPayload {
        contents: vec![GeminiContent {
            parts: vec![GeminiPart { text: prompt }],
        }],
        generation_config: GeminiGenerationConfig {
            max_output_tokens: 1024,
            thinking_config: None,
        },
    }
}

fn parse_gemini_stream(
    response: reqwest::blocking::Response,
    on_chunk: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<String> {
    let status = response.status();
    if !status.is_success() {
        let body = response
            .text()
            .with_context(|| format!("Gemini request failed with status {status}"))?;
        return Err(anyhow!("Gemini request failed: status {status} body {body}"));
    }

    let reader = BufReader::new(response);
    parse_gemini_stream_reader(reader, on_chunk)
}

fn extract_gemini_text(value: &Value) -> Option<String> {
    let parts = value
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|candidates| candidates.first())
        .and_then(|candidate| candidate.get("content"))
        .and_then(|content| content.get("parts"))
        .and_then(Value::as_array)?;

    let mut text = String::new();
    for part in parts {
        if let Some(part_text) = part.get("text").and_then(Value::as_str) {
            text.push_str(part_text);
        }
    }

    if text.is_empty() { None } else { Some(text) }
}

fn gemini_content_parts(reasoning: Option<&transport::ProviderReasoning>, text: Option<&str>) -> Vec<Value> {
    let mut parts = Vec::new();
    if let Some(reasoning) = reasoning {
        if let Some(reasoning_parts) = gemini_reasoning_parts_as_array(reasoning) {
            parts.extend(reasoning_parts);
        } else if let Some(text) = reasoning.text.as_deref().filter(|text| !text.trim().is_empty()) {
            parts.push(json!({
                "text": text,
                "thought": true,
            }));
        }
    }
    if let Some(text) = text.filter(|text| !text.trim().is_empty()) {
        parts.push(json!({ "text": text }));
    }
    parts
}

fn gemini_reasoning_parts_as_array(reasoning: &transport::ProviderReasoning) -> Option<Vec<Value>> {
    match reasoning.opaque_state.as_ref()? {
        Value::Array(parts) => Some(parts.clone()),
        Value::Object(part) => {
            if part.contains_key("text") || part.contains_key("thought") || part.contains_key("thoughtSignature") {
                Some(vec![reasoning.opaque_state.clone()?])
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use serde_json::json;

    use super::{build_gemini_contents, parse_gemini_agent_turn, parse_gemini_stream_reader};
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
    fn parses_gemini_sse_stream_incrementally() {
        let body = concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo\"}]}}]}\n\n"
        );
        let mut chunks = Vec::new();

        let answer = parse_gemini_stream_reader(Cursor::new(body.as_bytes()), &mut |chunk| {
            chunks.push(chunk.to_string());
            Ok(())
        })
        .expect("gemini stream parse should succeed");

        assert_eq!(answer, "Hello");
        assert_eq!(chunks, vec!["Hel".to_string(), "lo".to_string()]);
    }

    #[test]
    fn errors_when_provider_does_not_stream_gemini() {
        let body = r#"{"candidates":[{"content":{"parts":[{"text":"final answer"}]}}]}"#;
        let error = parse_gemini_stream_reader(Cursor::new(body.as_bytes()), &mut |_| Ok(()))
            .expect_err("non-stream body should fail");

        assert!(error.to_string().contains("did not return a streaming response"));
    }

    #[test]
    fn parses_gemini_agent_function_call_turn() {
        let body = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "I need to inspect the file."},
                        {
                            "functionCall": {
                                "name": "read",
                                "args": {"path": "repo/README.md"}
                            },
                            "thoughtSignature": "abc123"
                        }
                    ]
                }
            }]
        });

        let turn = parse_gemini_agent_turn(&body).expect("gemini function call should parse");

        match turn {
            ProviderModelTurn::ToolCalls { text, calls, reasoning } => {
                assert_eq!(text.as_deref(), Some("I need to inspect the file."));
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].tool, ToolName::Read);
                assert_eq!(calls[0].input, json!({"path":"repo/README.md"}));
                assert_eq!(calls[0].provider_data, Some(json!({"thoughtSignature":"abc123"})));
                assert!(reasoning.is_none());
            }
            ProviderModelTurn::Final { .. } => panic!("expected tool calls"),
        }
    }

    #[test]
    fn parses_gemini_thought_parts_as_reasoning() {
        let body = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {
                            "text": "I should inspect the file first.",
                            "thought": true,
                            "thoughtSignature": "sig-1"
                        },
                        {
                            "functionCall": {
                                "name": "read",
                                "args": {"path": "repo/README.md"}
                            }
                        }
                    ]
                }
            }]
        });

        let turn = parse_gemini_agent_turn(&body).expect("gemini thought should parse");

        match turn {
            ProviderModelTurn::ToolCalls { text, calls, reasoning } => {
                assert!(text.is_none());
                assert_eq!(calls.len(), 1);
                let reasoning = reasoning.expect("reasoning should be preserved");
                assert_eq!(reasoning.text, Some("I should inspect the file first.".to_string()));
                assert_eq!(
                    reasoning.opaque_state,
                    Some(json!({
                        "text": "I should inspect the file first.",
                        "thought": true,
                        "thoughtSignature": "sig-1"
                    }))
                );
            }
            ProviderModelTurn::Final { .. } => panic!("expected tool calls"),
        }
    }

    #[test]
    fn gemini_history_groups_function_responses_into_user_content() {
        let mut state = provider_request_with_tool_result();
        if let ProviderMessage::AssistantToolCalls { calls, .. } = &mut state.messages[0] {
            calls[0].provider_data = Some(json!({"thoughtSignature":"abc123"}));
        }
        let contents = build_gemini_contents(&state).expect("gemini contents should build");

        assert_eq!(contents.len(), 3);
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(contents[2]["role"], "user");
        assert_eq!(contents[1]["parts"][1]["thoughtSignature"], "abc123");
        assert_eq!(contents[2]["parts"][0]["functionResponse"]["name"], "read");
        assert_eq!(
            contents[2]["parts"][0]["functionResponse"]["response"]["output"],
            "path: repo/README.md\n1: hello"
        );
        assert_eq!(
            contents[2]["parts"][0]["functionResponse"]["response"]["is_error"],
            false
        );
    }

    #[test]
    fn gemini_history_replays_reasoning_part_with_thought_signature() {
        let mut state = provider_request_with_tool_result();
        if let ProviderMessage::AssistantToolCalls { reasoning, .. } = &mut state.messages[0] {
            *reasoning = Some(ProviderReasoning {
                text: Some("I should inspect the file first.".to_string()),
                opaque_state: Some(json!({
                    "text": "I should inspect the file first.",
                    "thought": true,
                    "thoughtSignature": "sig-1"
                })),
            });
        }

        let contents = build_gemini_contents(&state).expect("gemini contents should build");

        assert_eq!(contents[1]["parts"][0]["thought"], true);
        assert_eq!(contents[1]["parts"][0]["thoughtSignature"], "sig-1");
        assert_eq!(contents[1]["parts"][0]["text"], "I should inspect the file first.");
    }
}
