use anyhow::{Result, anyhow, bail};

use crate::{auth, config::Config};

#[derive(Debug, Clone, Copy)]
pub struct ProviderDescriptor {
    pub id: &'static str,
    pub default_model: Option<&'static str>,
    pub default_api_url: Option<&'static str>,
    pub env_vars: &'static [&'static str],
    pub api_key_optional: bool,
}

#[derive(Debug, Clone)]
pub struct ProviderStatus {
    pub id: &'static str,
    pub default_model: Option<&'static str>,
    pub authenticated: bool,
    pub auth_source: Option<String>,
    pub current: bool,
    pub hint: String,
}

const PROVIDERS: &[ProviderDescriptor] = &[
    ProviderDescriptor {
        id: "anthropic",
        default_model: Some("claude-haiku-4-5"),
        default_api_url: Some("https://api.anthropic.com/v1"),
        env_vars: &["ANTHROPIC_API_KEY"],
        api_key_optional: false,
    },
    ProviderDescriptor {
        id: "gemini",
        default_model: Some("gemini-3.1-flash-lite-preview"),
        default_api_url: Some("https://generativelanguage.googleapis.com/v1beta"),
        env_vars: &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        api_key_optional: false,
    },
    ProviderDescriptor {
        id: "openrouter",
        default_model: Some("nvidia/nemotron-3-super-120b-a12b:free"),
        default_api_url: Some("https://openrouter.ai/api/v1"),
        env_vars: &["OPENROUTER_API_KEY", "API_KEY"],
        api_key_optional: false,
    },
    ProviderDescriptor {
        id: "openai",
        default_model: Some("gpt-5-mini"),
        default_api_url: Some("https://api.openai.com/v1"),
        env_vars: &["OPENAI_API_KEY", "API_KEY"],
        api_key_optional: false,
    },
];

pub fn supported_providers() -> &'static [ProviderDescriptor] {
    PROVIDERS
}

pub fn provider_descriptor(provider_id: &str) -> Result<&'static ProviderDescriptor> {
    PROVIDERS
        .iter()
        .find(|provider| provider.id == provider_id)
        .ok_or_else(|| {
            anyhow!(
                "unsupported provider '{provider_id}', supported providers are: {}",
                PROVIDERS
                    .iter()
                    .map(|provider| provider.id)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

pub fn default_model_for_provider(provider_id: &str) -> Option<&'static str> {
    provider_descriptor(provider_id)
        .ok()
        .and_then(|provider| provider.default_model)
}

pub fn validate_provider_config(config: &Config) -> Result<()> {
    let descriptor = provider_descriptor(&config.provider)?;
    if config.model.trim().is_empty() {
        bail!("provider '{}' requires a model name", descriptor.id);
    }
    Ok(())
}

pub fn provider_statuses(config: &Config) -> Result<Vec<ProviderStatus>> {
    supported_providers()
        .iter()
        .map(|descriptor| {
            let auth = auth::auth_status(descriptor.id, config)?;
            Ok(ProviderStatus {
                id: descriptor.id,
                default_model: descriptor.default_model,
                authenticated: auth.authenticated,
                auth_source: auth.source,
                current: descriptor.id == config.provider,
                hint: auth.hint,
            })
        })
        .collect()
}
