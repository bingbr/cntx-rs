//! Shared agent/runtime types.

use std::{fmt, path::PathBuf, str::FromStr};

use anyhow::{Result, anyhow};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ToolName {
    List,
    Read,
    Grep,
    Glob,
    Tree,
    ReadMany,
    Search,
    Stat,
    FindReferences,
    GitStatusReadonly,
    GitDiff,
    GitShow,
    GitLog,
}

impl ToolName {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::List => "list",
            Self::Read => "read",
            Self::Grep => "grep",
            Self::Glob => "glob",
            Self::Tree => "tree",
            Self::ReadMany => "read_many",
            Self::Search => "search",
            Self::Stat => "stat",
            Self::FindReferences => "find_references",
            Self::GitStatusReadonly => "git_status_readonly",
            Self::GitDiff => "git_diff",
            Self::GitShow => "git_show",
            Self::GitLog => "git_log",
        }
    }
}

impl fmt::Display for ToolName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ToolName {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "list" => Ok(Self::List),
            "read" => Ok(Self::Read),
            "grep" => Ok(Self::Grep),
            "glob" => Ok(Self::Glob),
            "tree" => Ok(Self::Tree),
            "read_many" => Ok(Self::ReadMany),
            "search" => Ok(Self::Search),
            "stat" => Ok(Self::Stat),
            "find_references" => Ok(Self::FindReferences),
            "git_status_readonly" => Ok(Self::GitStatusReadonly),
            "git_diff" => Ok(Self::GitDiff),
            "git_show" => Ok(Self::GitShow),
            "git_log" => Ok(Self::GitLog),
            _ => Err(anyhow!("unsupported tool '{value}'")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceResourceSummary {
    pub name: String,
    pub kind: String,
    pub source: String,
    pub branch: Option<String>,
    pub search_paths: Vec<String>,
    pub notes: Option<String>,
    pub mount_path: String,
    pub ephemeral: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentWorkspace {
    pub root: PathBuf,
    pub resources: Vec<WorkspaceResourceSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: ToolName,
    pub description: &'static str,
    pub input_schema: Value,
    pub safe_to_parallelize: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub tool: ToolName,
    pub input: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_data: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCitation {
    pub resource: String,
    pub path: String,
    pub line: usize,
    pub score: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub id: String,
    pub tool: ToolName,
    pub output: String,
    #[serde(default)]
    pub is_error: bool,
    pub truncated: bool,
    #[serde(default)]
    pub citations: Vec<ToolCitation>,
}

pub trait ToolExecutor: Send + Sync {
    fn execute(&self, workspace: &AgentWorkspace, call: &ToolCall) -> Result<ToolResult>;
}

#[derive(Debug, Clone, Copy)]
pub struct ProviderCapabilities {
    pub supports_streaming: bool,
    pub supports_parallel_tool_calls: bool,
}

#[derive(Debug, Clone)]
pub enum AgentRuntimeEvent {
    Thinking(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningPolicy {
    Off,
    ProviderDefault,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    pub reasoning: ReasoningPolicy,
    pub parallel_tools: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentItem {
    UserText(String),
    AssistantText(String),
    Reasoning(ReasoningChunk),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    FinalAnswer(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningChunk {
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opaque_state: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSession {
    pub profile: AgentProfileId,
    pub provider: String,
    pub system_prompt: String,
    pub model: String,
    pub conversation: Vec<AgentItem>,
    #[serde(default)]
    pub provider_state: Option<Value>,
}

impl AgentSession {
    pub fn set_provider_state(&mut self, provider_state: Option<Value>) {
        self.provider_state = provider_state;
    }
}

#[derive(Debug, Clone)]
pub struct ProviderTurnRequest<'a> {
    pub session: &'a AgentSession,
    pub tools: &'a [ToolDefinition],
    pub stream: bool,
}

#[derive(Debug, Clone)]
pub enum ProviderTurn {
    Final {
        text: String,
        items: Vec<AgentItem>,
        provider_state: Option<Value>,
    },
    ToolCalls {
        items: Vec<AgentItem>,
        calls: Vec<ToolCall>,
        provider_state: Option<Value>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRunStatus {
    Completed,
    MaxStepsReached,
}

#[derive(Debug, Clone)]
pub struct AgentRunOutput {
    pub answer: Option<String>,
    pub events: Vec<AgentRuntimeEvent>,
    pub tool_results: Vec<ToolResult>,
    pub steps: usize,
    pub status: AgentRunStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AgentProfileId {
    OpenAi,
    Anthropic,
    Gemini,
    OpenRouter,
}

pub fn resolve_profile(profile: AgentProfileId) -> AgentProfile {
    match profile {
        AgentProfileId::OpenAi => AgentProfile {
            reasoning: ReasoningPolicy::Off,
            parallel_tools: false,
        },
        AgentProfileId::Anthropic => AgentProfile {
            reasoning: ReasoningPolicy::ProviderDefault,
            parallel_tools: false,
        },
        AgentProfileId::Gemini => AgentProfile {
            reasoning: ReasoningPolicy::ProviderDefault,
            parallel_tools: true,
        },
        AgentProfileId::OpenRouter => AgentProfile {
            reasoning: ReasoningPolicy::ProviderDefault,
            parallel_tools: true,
        },
    }
}

pub fn profile_id_for_provider(provider: &str) -> Result<AgentProfileId> {
    match provider {
        "anthropic" => Ok(AgentProfileId::Anthropic),
        "gemini" => Ok(AgentProfileId::Gemini),
        "openrouter" => Ok(AgentProfileId::OpenRouter),
        "openai" => Ok(AgentProfileId::OpenAi),
        _ => Err(anyhow!("unsupported provider '{provider}'")),
    }
}
