use std::io::{self, BufRead, BufReader, BufWriter, Write};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    ask,
    config::Config,
    resources::{ResourceConfig, ResourceKind},
};

const CURRENT_PROTOCOL_VERSION: &str = "2025-11-25";
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

#[derive(Debug, Deserialize)]
struct McpRequest {
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct McpResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<McpError>,
}

#[derive(Debug, Serialize)]
struct McpError {
    code: i64,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
struct AskParams {
    question: String,
    #[serde(default)]
    resources: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ToolCallParams {
    name: String,
    arguments: Option<Value>,
    #[serde(default, rename = "_meta")]
    meta: Option<RequestMeta>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RequestMeta {
    #[serde(default)]
    progress_token: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InitializeParams {
    protocol_version: Option<String>,
}

pub fn run_server(config: Config) -> Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = BufReader::new(stdin.lock());
    let mut output = BufWriter::new(stdout.lock());

    while let Some(payload) = read_message(&mut input)? {
        let response = match serde_json::from_str::<McpRequest>(&payload) {
            Ok(request) => route_request(&config, request, &mut output)?,
            Err(error) => Some(error_response_with_details(
                Value::Null,
                -32700,
                "Invalid JSON".to_string(),
                Some(error.to_string()),
            )),
        };

        if let Some(response) = response {
            write_message(&mut output, &response)?;
        }
    }

    Ok(())
}

fn route_request<W: Write>(config: &Config, request: McpRequest, output: &mut W) -> Result<Option<McpResponse>> {
    let id = request.id;
    let response = match request.method.as_str() {
        "initialize" => Some(handle_initialize(id, request.params)),
        "notifications/initialized" => None,
        "ping" => Some(success_response(id, json!({}))),
        "resources/list" => {
            let resources: Vec<Value> = config.resources.iter().map(format_mcp_resource).collect();
            Some(success_response(id, json!({ "resources": resources })))
        }
        "tools/list" => Some(handle_tools_list(config, id)),
        "tools/call" => Some(handle_tools_call(config, id, request.params, output)?),
        _ => Some(error_response(
            response_id(id),
            -32601,
            format!("Method '{}' is not supported", request.method),
        )),
    };

    Ok(response)
}

fn handle_tools_list(config: &Config, id: Option<Value>) -> McpResponse {
    let configured_resource_names = config
        .resources
        .iter()
        .map(|resource| resource.name.clone())
        .collect::<Vec<_>>();
    let single_resource_example = configured_resource_names
        .first()
        .map(|name| vec![name.clone()])
        .unwrap_or_default();
    let multi_resource_example = configured_resource_names.iter().take(2).cloned().collect::<Vec<_>>();
    let tools = vec![json!({
        "name": "ask",
        "description": "Ask a grounded question against configured resources or one-off HTTPS git repos. Prefer setting `resources` when the user names a repo, package, or docs site. Call `resources/list` first if you need to discover valid configured resource names.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The grounded question to answer. Use `@resource` mentions in the question when helpful."
                },
                "resources": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {
                                "type": "string",
                                "enum": configured_resource_names,
                                "description": "One configured resource name returned by `resources/list`."
                            },
                            {
                                "type": "string",
                                "pattern": "^https://",
                                "description": "A one-off HTTPS git repository URL."
                            }
                        ]
                    },
                    "description": "Preferred when the user names specific repos or docs. Omit only when the question should search across all configured resources."
                },
            },
            "required": ["question"],
            "examples": [
                {
                    "question": "Summarize the routing docs",
                    "resources": single_resource_example
                },
                {
                    "question": "Compare the forms docs in Svelte and SvelteKit",
                    "resources": multi_resource_example
                },
                {
                    "question": "Which file explains load functions?",
                    "resources": ["https://github.com/sveltejs/kit/tree/main/documentation"]
                }
            ]
        }
    })];

    success_response(id, json!({ "tools": tools }))
}

fn handle_initialize(id: Option<Value>, params: Option<Value>) -> McpResponse {
    let protocol_version = match params {
        Some(params) => {
            let request_id = response_id(id.clone());
            let params: InitializeParams = match parse_params(params, request_id) {
                Ok(params) => params,
                Err(response) => return *response,
            };
            negotiate_protocol_version(params.protocol_version.as_deref())
        }
        None => CURRENT_PROTOCOL_VERSION,
    };

    success_response(
        id,
        json!({
            "protocolVersion": protocol_version,
            "capabilities": {
                "resources": {},
                "tools": {}
            },
            "serverInfo": {
                "name": env!("CARGO_PKG_NAME"),
                "version": env!("CARGO_PKG_VERSION")
            }
        }),
    )
}

fn negotiate_protocol_version(requested: Option<&str>) -> &'static str {
    let Some(requested) = requested else {
        return CURRENT_PROTOCOL_VERSION;
    };

    for version in SUPPORTED_PROTOCOL_VERSIONS {
        if *version == requested {
            return version;
        }
    }

    CURRENT_PROTOCOL_VERSION
}

fn handle_tools_call<W: Write>(
    config: &Config,
    id: Option<Value>,
    params: Option<Value>,
    output: &mut W,
) -> Result<McpResponse> {
    let request_id = response_id(id.clone());
    let Some(params) = params else {
        return Ok(error_response_with_details(
            request_id,
            -32602,
            "Invalid request body".to_string(),
            Some("Missing tools/call parameters".to_string()),
        ));
    };

    let call: ToolCallParams = match parse_params(params, request_id.clone()) {
        Ok(call) => call,
        Err(response) => return Ok(*response),
    };

    if call.name == "ask" {
        let args: AskParams = match parse_params(call.arguments.unwrap_or(Value::Null), request_id.clone()) {
            Ok(args) => args,
            Err(response) => return Ok(*response),
        };

        let question = args.question.clone();
        let requested_resources = requested_resources_from_params(args);
        let progress_token = match progress_token_from_meta(call.meta.as_ref(), request_id.clone()) {
            Ok(progress_token) => progress_token,
            Err(response) => return Ok(*response),
        };
        let mut progress = 0_u64;
        let mut on_progress = |message: &str| -> Result<()> {
            progress = progress.saturating_add(1);
            if let Some(progress_token) = progress_token.as_ref() {
                write_notification(
                    output,
                    &progress_notification(progress_token.clone(), progress, message),
                )?;
            }
            Ok(())
        };

        return Ok(
            match ask::ask_question(config, &requested_resources, &question, None, Some(&mut on_progress)) {
                Ok(answer) => success_response(
                    id,
                    json!({
                        "content": [{
                            "type": "text",
                            "text": answer.answer
                        }],
                        "resolvedResources": answer.resolved_resources,
                        "citations": answer.citations,
                        "retrieval": answer.retrieval,
                        "isError": false
                    }),
                ),
                Err(error) => tool_error_result(
                    id,
                    json!({
                        "error": error.to_string()
                    }),
                ),
            },
        );
    }

    Ok(error_response(
        request_id,
        -32601,
        format!("Tool '{}' is not supported", call.name),
    ))
}
fn requested_resources_from_params(params: AskParams) -> Vec<String> {
    params.resources
}

fn progress_token_from_meta(
    meta: Option<&RequestMeta>,
    id: Value,
) -> std::result::Result<Option<Value>, Box<McpResponse>> {
    let Some(progress_token) = meta.and_then(|meta| meta.progress_token.as_ref()) else {
        return Ok(None);
    };

    if progress_token.is_string() {
        return Ok(Some(progress_token.clone()));
    }

    if let Some(number) = progress_token.as_i64() {
        return Ok(Some(json!(number)));
    }
    if let Some(number) = progress_token.as_u64() {
        return Ok(Some(json!(number)));
    }

    Err(Box::new(error_response_with_details(
        id,
        -32602,
        "Invalid request body".to_string(),
        Some("progressToken must be a string or integer".to_string()),
    )))
}

fn progress_notification(progress_token: Value, progress: u64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {
            "progressToken": progress_token,
            "progress": progress,
            "message": message,
        }
    })
}

fn success_response(id: Option<Value>, result: Value) -> McpResponse {
    McpResponse {
        jsonrpc: "2.0",
        id: response_id(id),
        result: Some(result),
        error: None,
    }
}

fn error_response(id: Value, code: i64, message: String) -> McpResponse {
    error_response_with_data(id, code, message, None)
}

fn error_response_with_details(id: Value, code: i64, message: String, details: Option<String>) -> McpResponse {
    let data = error_data(&message, details);
    error_response_with_data(id, code, message, Some(data))
}

fn error_response_with_data(id: Value, code: i64, message: String, data: Option<Value>) -> McpResponse {
    McpResponse {
        jsonrpc: "2.0",
        id,
        result: None,
        error: Some(McpError { code, message, data }),
    }
}

fn response_id(id: Option<Value>) -> Value {
    id.unwrap_or(Value::Null)
}

fn parse_params<T: serde::de::DeserializeOwned>(params: Value, id: Value) -> Result<T, Box<McpResponse>> {
    serde_json::from_value(params).map_err(|error| {
        Box::new(error_response_with_details(
            id,
            -32602,
            "Invalid request body".to_string(),
            Some(error.to_string()),
        ))
    })
}

fn tool_error_result(id: Option<Value>, error: Value) -> McpResponse {
    success_response(
        id,
        json!({
            "content": [],
            "structuredContent": error,
            "isError": true
        }),
    )
}

fn error_data(message: &str, details: Option<String>) -> Value {
    let mut data = json!({ "error": message });

    if let Some(details) = details.filter(|details| details != message) {
        data["details"] = Value::String(details);
    }

    data
}

fn read_message<R: BufRead>(reader: &mut R) -> Result<Option<String>> {
    let Some(first_line) = read_next_non_empty_line(reader)? else {
        return Ok(None);
    };
    let first_header = trim_line_endings(&first_line);

    if first_header.starts_with('{') {
        return Ok(Some(first_header.to_string()));
    }

    let mut content_length = None;
    parse_header_line(first_header, &mut content_length)?;

    loop {
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line).with_context(|| "Reading MCP header line")?;
        if bytes_read == 0 {
            return Err(anyhow!("Unexpected EOF while reading MCP headers"));
        }

        let header = trim_line_endings(&line);
        if header.is_empty() {
            break;
        }
        parse_header_line(header, &mut content_length)?;
    }

    let content_length = content_length.ok_or_else(|| anyhow!("Missing Content-Length header"))?;
    let mut body = vec![0_u8; content_length];
    reader
        .read_exact(&mut body)
        .with_context(|| format!("Reading MCP body of {content_length} bytes"))?;

    String::from_utf8(body)
        .map(Some)
        .map_err(|error| anyhow!("MCP body is not valid UTF-8: {error}"))
}

fn read_next_non_empty_line<R: BufRead>(reader: &mut R) -> Result<Option<String>> {
    loop {
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line).with_context(|| "Reading MCP input")?;
        if bytes_read == 0 {
            return Ok(None);
        }
        if !trim_line_endings(&line).is_empty() {
            return Ok(Some(line));
        }
    }
}

fn parse_header_line(line: &str, content_length: &mut Option<usize>) -> Result<()> {
    let (name, value) = line
        .split_once(':')
        .ok_or_else(|| anyhow!("Invalid MCP header '{line}'"))?;

    if name.trim().eq_ignore_ascii_case("Content-Length") {
        let parsed_length = value
            .trim()
            .parse::<usize>()
            .with_context(|| format!("Invalid Content-Length header value '{}'", value.trim()))?;
        *content_length = Some(parsed_length);
    }

    Ok(())
}

fn trim_line_endings(line: &str) -> &str {
    line.trim_end_matches(['\r', '\n'])
}

fn write_message<W: Write>(writer: &mut W, response: &McpResponse) -> Result<()> {
    write_jsonrpc_message(writer, response)
}

fn write_notification<W: Write>(writer: &mut W, notification: &Value) -> Result<()> {
    write_jsonrpc_message(writer, notification)
}

fn write_jsonrpc_message<W: Write, T: Serialize>(writer: &mut W, payload: &T) -> Result<()> {
    let payload = serde_json::to_vec(payload).with_context(|| "Serializing MCP response payload")?;
    writer
        .write_all(&payload)
        .with_context(|| "Writing MCP response body")?;
    writer
        .write_all(b"\n")
        .with_context(|| "Writing MCP response delimiter")?;
    writer.flush().with_context(|| "Flushing MCP response")?;
    Ok(())
}

fn format_mcp_resource(resource: &ResourceConfig) -> Value {
    let uri = match resource.kind {
        ResourceKind::Git => resource.git_url.as_deref().unwrap_or_default().to_owned(),
        ResourceKind::Local => resource
            .local_path
            .as_ref()
            .map(|path| format!("file://{}", path.display()))
            .unwrap_or_default(),
    };
    let branch = resource.branch.as_deref().unwrap_or_default().to_owned();
    let search_paths = resource
        .search_paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();

    json!({
        "kind": resource.kind,
        "name": resource.name,
        "uri": uri,
        "branch": branch,
        "searchPaths": search_paths,
        "notes": resource.notes,
    })
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use serde_json::json;

    use super::{
        McpRequest, negotiate_protocol_version, progress_notification, read_message, route_request, tool_error_result,
        write_message,
    };
    use crate::config::Config;

    #[test]
    fn reads_content_length_framed_messages() {
        let body = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
        let framed = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        let mut input = Cursor::new(framed.into_bytes());

        let parsed = read_message(&mut input)
            .expect("message read should succeed")
            .expect("one message should be present");

        assert_eq!(parsed, body);
    }

    #[test]
    fn keeps_mcp_server_alive_by_returning_request_errors() {
        let config = Config {
            provider: "openrouter".to_string(),
            ..Config::default()
        };
        let mut notifications = Vec::new();

        let response = route_request(
            &config,
            McpRequest {
                id: Some(json!(1)),
                method: "tools/call".to_string(),
                params: Some(json!({
                    "name": "ask",
                    "arguments": {
                        "resources": ["missing"],
                        "question": "x"
                    }
                })),
            },
            &mut notifications,
        )
        .expect("route should succeed")
        .expect("request with id should yield a response");

        assert!(response.error.is_none());
        assert!(notifications.is_empty());
        let result = response.result.expect("result response expected");
        assert_eq!(result.get("isError").and_then(|value| value.as_bool()), Some(true));
        assert!(
            result
                .get("structuredContent")
                .and_then(|value| value.get("error"))
                .and_then(|value| value.as_str())
                .expect("structured tool error")
                .contains("failed to resolve requested resources")
        );
    }

    #[test]
    fn returns_structured_invalid_request_body_errors() {
        let mut notifications = Vec::new();
        let response = route_request(
            &Config::default(),
            McpRequest {
                id: Some(json!(1)),
                method: "tools/call".to_string(),
                params: Some(json!({
                    "name": "ask",
                    "arguments": "wrong-shape"
                })),
            },
            &mut notifications,
        )
        .expect("route should succeed")
        .expect("request with id should yield a response");

        assert!(response.result.is_none());
        assert!(notifications.is_empty());
        let error = response.error.expect("error response expected");
        assert_eq!(error.code, -32602);
        assert_eq!(error.message, "Invalid request body");
        assert_eq!(
            error.data,
            Some(json!({
                "error": "Invalid request body",
                "details": "invalid type: string \"wrong-shape\", expected struct AskParams"
            }))
        );
    }

    #[test]
    fn tool_errors_are_returned_in_call_results() {
        let response = tool_error_result(
            Some(json!(1)),
            json!({
                "error": "Provider \"anthropic\" is not authenticated."
            }),
        );

        assert!(response.error.is_none());
        let result = response.result.expect("result response expected");
        assert_eq!(
            result,
            json!({
                "content": [],
                "structuredContent": {
                    "error": "Provider \"anthropic\" is not authenticated."
                },
                "isError": true
            })
        );
    }

    #[test]
    fn writes_newline_delimited_stdio_messages() {
        let mut notifications = Vec::new();
        let response = route_request(
            &Config::default(),
            McpRequest {
                id: Some(json!(1)),
                method: "ping".to_string(),
                params: Some(json!({})),
            },
            &mut notifications,
        )
        .expect("route should succeed")
        .expect("request with id should yield a response");
        let mut output = Vec::new();

        assert!(notifications.is_empty());
        write_message(&mut output, &response).expect("response write should succeed");

        let rendered = String::from_utf8(output).expect("response should be valid utf-8");
        assert!(rendered.ends_with('\n'));
        assert!(!rendered.contains("Content-Length:"));
        let parsed: serde_json::Value =
            serde_json::from_str(rendered.trim_end()).expect("response should be valid json");
        assert_eq!(parsed, json!({"jsonrpc":"2.0","id":1,"result":{}}));
    }

    #[test]
    fn initialize_negotiates_protocol_versions() {
        assert_eq!(negotiate_protocol_version(Some("2025-11-25")), "2025-11-25");
        assert_eq!(negotiate_protocol_version(Some("2025-06-18")), "2025-06-18");
        assert_eq!(negotiate_protocol_version(Some("2099-01-01")), "2025-11-25");
    }

    #[test]
    fn progress_notifications_include_token_message_and_counter() {
        let notification = progress_notification(json!("token-1"), 2, "Searching resource");

        assert_eq!(
            notification,
            json!({
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {
                    "progressToken": "token-1",
                    "progress": 2,
                    "message": "Searching resource"
                }
            })
        );
    }

    #[test]
    fn tools_list_guides_resource_selection_and_does_not_expose_model_controls() {
        let mut notifications = Vec::new();
        let response = route_request(
            &Config::default(),
            McpRequest {
                id: Some(json!(1)),
                method: "tools/list".to_string(),
                params: None,
            },
            &mut notifications,
        )
        .expect("route should succeed")
        .expect("request with id should yield a response");

        assert!(notifications.is_empty());
        let result = response.result.expect("tools/list result expected");
        let tool = result["tools"]
            .as_array()
            .and_then(|tools| tools.first())
            .expect("one tool should be advertised");
        let properties = tool["inputSchema"]["properties"]
            .as_object()
            .expect("input schema properties should exist");

        assert!(properties.contains_key("question"));
        assert!(properties.contains_key("resources"));
        assert!(!properties.contains_key("provider"));
        assert!(!properties.contains_key("model"));
        assert!(
            tool["description"]
                .as_str()
                .expect("tool description should exist")
                .contains("resources/list")
        );
    }
}
