use std::{
    env,
    io::{self, IsTerminal},
};

use anyhow::{Context, Result, anyhow};
use keyring::{Entry, Error as KeyringError};

use crate::{config::Config, providers};

const KEYRING_SERVICE: &str = env!("CARGO_PKG_NAME");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderAuthStatus {
    pub authenticated: bool,
    pub source: Option<String>,
    pub hint: String,
}

pub fn resolve_api_key(provider_id: &str, _config: &Config) -> Result<Option<String>> {
    let descriptor = providers::provider_descriptor(provider_id)?;
    let env_key = descriptor.env_vars.iter().find_map(|env_var| non_empty_env(env_var));

    if let Some(key) = env_key {
        return Ok(Some(key));
    }

    let keyring_error = match read_keyring_key(provider_id) {
        Ok(Some(key)) => return Ok(Some(key)),
        Ok(None) => None,
        Err(error) => Some(error),
    };

    if descriptor.api_key_optional {
        return Ok(None);
    }

    Err(anyhow!(missing_auth_message(provider_id, keyring_error)))
}

pub fn auth_status(provider_id: &str, _config: &Config) -> Result<ProviderAuthStatus> {
    let descriptor = providers::provider_descriptor(provider_id)?;

    for env_var in descriptor.env_vars {
        if non_empty_env(env_var).is_some() {
            return Ok(ProviderAuthStatus {
                authenticated: true,
                source: Some(format!("env:{env_var}")),
                hint: format!("using {env_var}"),
            });
        }
    }

    let keyring_error = match read_keyring_key(provider_id) {
        Ok(Some(_)) => {
            return Ok(ProviderAuthStatus {
                authenticated: true,
                source: Some(format!("keyring:{provider_id}")),
                hint: "local keychain".to_string(),
            });
        }
        Ok(None) => None,
        Err(error) => Some(error),
    };

    Ok(ProviderAuthStatus {
        authenticated: descriptor.api_key_optional,
        source: None,
        hint: missing_auth_hint(provider_id, keyring_error),
    })
}

pub fn resolve_api_key_for_storage(provider_id: &str, explicit_key: Option<&str>) -> Result<String> {
    if let Some(key) = explicit_key {
        let trimmed = key.trim();
        if trimmed.is_empty() {
            return Err(anyhow!("API key cannot be empty"));
        }
        return Ok(trimmed.to_string());
    }

    if let Some(key) = providers::provider_descriptor(provider_id)?
        .env_vars
        .iter()
        .find_map(|env_var| non_empty_env(env_var))
    {
        return Ok(key);
    }

    if !io::stdin().is_terminal() {
        return Err(anyhow!(
            "No API key available for provider '{}'. Re-run in a terminal, set the provider environment variable, or pass --api-key.",
            provider_id
        ));
    }

    let prompt = format!("API key for provider '{provider_id}': ");
    let entered = rpassword::prompt_password(prompt).context("Reading API key from the terminal")?;
    let trimmed = entered.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("API key cannot be empty"));
    }
    Ok(trimmed.to_string())
}

pub fn store_api_key(provider_id: &str, api_key: &str) -> Result<()> {
    keyring_entry(provider_id)?
        .set_password(api_key)
        .with_context(|| format!("Writing credentials for provider '{provider_id}' to keychain"))
}

pub fn delete_api_key(provider_id: &str) -> Result<bool> {
    match keyring_entry(provider_id)?.delete_credential() {
        Ok(()) => Ok(true),
        Err(KeyringError::NoEntry) => Ok(false),
        Err(error) => Err(error).with_context(|| format!("Removing credentials for provider '{provider_id}'")),
    }
}

fn non_empty_env(env_var: &str) -> Option<String> {
    let value = env::var(env_var).ok()?;
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn keyring_entry(provider_id: &str) -> Result<Entry> {
    Entry::new(KEYRING_SERVICE, provider_id)
        .with_context(|| format!("Opening local keychain entry for provider '{provider_id}'"))
}

fn read_keyring_key(provider_id: &str) -> Result<Option<String>> {
    match keyring_entry(provider_id)?.get_password() {
        Ok(key) => Ok((!key.trim().is_empty()).then_some(key)),
        Err(KeyringError::NoEntry) => Ok(None),
        Err(error) => Err(error).with_context(|| format!("Reading credentials for provider '{provider_id}'")),
    }
}

fn missing_auth_hint(provider_id: &str, keyring_error: Option<anyhow::Error>) -> String {
    let descriptor = match providers::provider_descriptor(provider_id) {
        Ok(descriptor) => descriptor,
        Err(_) => {
            return format!("provider '{provider_id}' is not supported");
        }
    };

    let mut hint = if descriptor.api_key_optional {
        "optional".to_string()
    } else {
        "not set".to_string()
    };

    if let Some(error) = keyring_error {
        hint.push_str(&format!(" (keychain unavailable: {error})"));
    }

    hint
}

fn missing_auth_message(provider_id: &str, keyring_error: Option<anyhow::Error>) -> String {
    let mut message = format!("Provider \"{provider_id}\" is not authenticated.");

    if let Some(error) = keyring_error {
        message.push_str(&format!(" Local keychain lookup failed: {error}."));
    }

    message
}
