//! Provider adapter trait and factory.

pub mod anthropic;
pub mod catalog;
pub mod gemini;
pub mod openai;
pub mod openrouter;
pub mod transport;

use std::sync::Arc;

use anyhow::Result as AnyhowResult;
pub use catalog::{
    ProviderStatus, default_model_for_provider, provider_descriptor, provider_statuses, supported_providers,
    validate_provider_config,
};
use reqwest::blocking::Client;
use thiserror::Error;
pub use transport::{ask, ask_stream};

use crate::{
    agent::types::{AgentSession, ProviderCapabilities, ProviderTurn, ProviderTurnRequest, profile_id_for_provider},
    config::Config,
};

pub type AdapterHandle = Arc<dyn ProviderAdapter>;

#[derive(Clone, Copy)]
enum ProviderBackend {
    Anthropic,
    Gemini,
    OpenRouter,
    OpenAi,
}

struct ProviderSpec {
    id: &'static str,
    backend: ProviderBackend,
    capabilities: ProviderCapabilities,
}

struct StaticProviderAdapter {
    spec: &'static ProviderSpec,
}

const PROVIDER_SPECS: &[ProviderSpec] = &[
    ProviderSpec {
        id: "anthropic",
        backend: ProviderBackend::Anthropic,
        capabilities: ProviderCapabilities {
            supports_streaming: true,
            supports_parallel_tool_calls: false,
        },
    },
    ProviderSpec {
        id: "gemini",
        backend: ProviderBackend::Gemini,
        capabilities: ProviderCapabilities {
            supports_streaming: true,
            supports_parallel_tool_calls: true,
        },
    },
    ProviderSpec {
        id: "openrouter",
        backend: ProviderBackend::OpenRouter,
        capabilities: ProviderCapabilities {
            supports_streaming: true,
            supports_parallel_tool_calls: true,
        },
    },
    ProviderSpec {
        id: "openai",
        backend: ProviderBackend::OpenAi,
        capabilities: ProviderCapabilities {
            supports_streaming: true,
            supports_parallel_tool_calls: true,
        },
    },
];

pub fn build_adapter(provider_id: &str) -> ProviderResult<AdapterHandle> {
    let spec = PROVIDER_SPECS
        .iter()
        .find(|spec| spec.id == provider_id)
        .ok_or_else(|| ProviderError::unsupported(format!("unsupported provider '{provider_id}'")))?;
    Ok(Arc::new(StaticProviderAdapter { spec }))
}

pub trait ProviderAdapter: Send + Sync {
    fn capabilities(&self) -> ProviderCapabilities;

    fn start_session(&self, config: &Config) -> ProviderResult<AgentSession> {
        let profile_id = profile_id_for_provider(&config.provider).map_err(ProviderError::from_anyhow)?;
        Ok(AgentSession {
            profile: profile_id,
            provider: config.provider.clone(),
            system_prompt: String::new(),
            model: config.model.clone(),
            conversation: Vec::new(),
            provider_state: None,
        })
    }
    fn ask(&self, client: &Client, config: &Config, prompt: &str, api_key: Option<&str>) -> ProviderResult<String>;
    fn ask_stream(
        &self,
        client: &Client,
        config: &Config,
        prompt: &str,
        api_key: Option<&str>,
        on_chunk: &mut dyn FnMut(&str) -> AnyhowResult<()>,
    ) -> ProviderResult<String>;
    fn complete_turn(&self, request: ProviderTurnRequest<'_>) -> ProviderResult<ProviderTurn>;
}

impl ProviderAdapter for StaticProviderAdapter {
    fn capabilities(&self) -> ProviderCapabilities {
        self.spec.capabilities
    }

    fn ask(&self, client: &Client, config: &Config, prompt: &str, api_key: Option<&str>) -> ProviderResult<String> {
        match self.spec.backend {
            ProviderBackend::Anthropic => anthropic::ask_anthropic(client, config, prompt, api_key),
            ProviderBackend::Gemini => gemini::ask_gemini(client, config, prompt, require_api_key(api_key, "gemini")?),
            ProviderBackend::OpenRouter => openrouter::ask_openrouter(client, config, prompt, api_key),
            ProviderBackend::OpenAi => openai::ask_openai(client, config, prompt, api_key),
        }
        .map_err(ProviderError::from_anyhow)
    }

    fn ask_stream(
        &self,
        client: &Client,
        config: &Config,
        prompt: &str,
        api_key: Option<&str>,
        on_chunk: &mut dyn FnMut(&str) -> AnyhowResult<()>,
    ) -> ProviderResult<String> {
        match self.spec.backend {
            ProviderBackend::Anthropic => anthropic::ask_anthropic_stream(client, config, prompt, api_key, on_chunk)
                .map_err(ProviderError::from_anyhow),
            ProviderBackend::Gemini => {
                gemini::ask_gemini_stream(client, config, prompt, require_api_key(api_key, "gemini")?, on_chunk)
                    .map_err(ProviderError::from_anyhow)
            }
            ProviderBackend::OpenRouter => openrouter::ask_openrouter_stream(client, config, prompt, api_key, on_chunk)
                .map_err(ProviderError::from_anyhow),
            ProviderBackend::OpenAi => {
                openai::ask_openai_stream(client, config, prompt, api_key, on_chunk).map_err(ProviderError::from_anyhow)
            }
        }
    }

    fn complete_turn(&self, request: ProviderTurnRequest<'_>) -> ProviderResult<ProviderTurn> {
        match self.spec.backend {
            ProviderBackend::Anthropic => anthropic::complete_anthropic_adapter_turn(request),
            ProviderBackend::Gemini => gemini::complete_gemini_adapter_turn(request),
            ProviderBackend::OpenRouter => openrouter::complete_openrouter_adapter_turn(request),
            ProviderBackend::OpenAi => openai::complete_openai_adapter_turn(request),
        }
        .map_err(ProviderError::from_anyhow)
    }
}

fn require_api_key<'a>(api_key: Option<&'a str>, provider_name: &str) -> ProviderResult<&'a str> {
    api_key.ok_or_else(|| ProviderError::protocol(format!("{provider_name} requires a non-empty API key")))
}

fn err(message: impl Into<String>) -> ProviderError {
    ProviderError::protocol(message)
}

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("provider protocol error: {0}")]
    Protocol(String),
    #[error("unsupported provider feature: {0}")]
    Unsupported(String),
}

pub type ProviderResult<T> = std::result::Result<T, ProviderError>;

impl ProviderError {
    fn protocol(message: impl Into<String>) -> Self {
        Self::Protocol(message.into())
    }

    fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported(message.into())
    }

    pub(crate) fn from_anyhow(error: anyhow::Error) -> Self {
        err(format_error_chain(&error))
    }
}

fn format_error_chain(error: &anyhow::Error) -> String {
    let mut parts = vec![error.to_string()];
    let mut current = error.source();
    while let Some(source) = current {
        parts.push(source.to_string());
        current = source.source();
    }
    parts.join(": ")
}
