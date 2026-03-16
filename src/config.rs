use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use dirs::config_dir;
use serde::{Deserialize, Serialize};

use crate::{
    providers,
    resources::{ResourceConfig, default_resources},
};

const DEFAULT_AGENTIC_MAX_STEPS: usize = 20;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub provider: String,
    pub model: String,
    pub data_dir: PathBuf,
    pub resources: Vec<ResourceConfig>,
    pub agentic_max_steps: usize,
    pub agentic_require_sandbox: bool,
    pub bwrap_path: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        let data_dir = default_data_dir();
        Self {
            provider: "anthropic".to_string(),
            model: default_model_for_provider("anthropic")
                .unwrap_or("claude-haiku-4-5")
                .to_string(),
            data_dir,
            resources: default_resources(),
            agentic_max_steps: DEFAULT_AGENTIC_MAX_STEPS,
            agentic_require_sandbox: true,
            bwrap_path: None,
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Config::default());
        }

        let content = fs::read_to_string(path).with_context(|| format!("Reading config file {}", path.display()))?;
        let raw: toml::Value =
            toml::from_str(&content).with_context(|| format!("Parsing config file {}", path.display()))?;
        let has_model = raw.get("model").is_some();
        let mut config: Config =
            toml::from_str(&content).with_context(|| format!("Parsing config file {}", path.display()))?;
        config.normalized()?;
        if !has_model && let Some(model) = default_model_for_provider(&config.provider) {
            config.model = model.to_string();
        }
        Ok(config)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut normalized = self.clone();
        normalized.normalized()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("Creating config directory {}", parent.display()))?;
        }
        let content = toml::to_string_pretty(&normalized)?;
        fs::write(path, content).with_context(|| format!("Writing config file {}", path.display()))?;
        Ok(())
    }

    pub fn normalized(&mut self) -> Result<()> {
        validate_provider(&self.provider)?;
        if self.data_dir.as_os_str().is_empty() {
            self.data_dir = default_data_dir();
        }
        if self.model.is_empty()
            && let Some(model) = default_model_for_provider(&self.provider)
        {
            self.model = model.to_string();
        }
        if self.resources.is_empty() {
            self.resources = default_resources();
        }
        if self.agentic_max_steps == 0 {
            self.agentic_max_steps = DEFAULT_AGENTIC_MAX_STEPS;
        }
        if self.bwrap_path.as_ref().is_some_and(|path| path.as_os_str().is_empty()) {
            self.bwrap_path = None;
        }
        for resource in &mut self.resources {
            resource.normalize_in_place()?;
        }
        providers::validate_provider_config(self)?;
        Ok(())
    }
}

pub fn default_config_path() -> PathBuf {
    config_dir()
        .or_else(|| env::var("HOME").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("cntx-rs")
        .join("config.toml")
}

fn default_data_dir() -> PathBuf {
    config_dir()
        .or_else(|| env::var("HOME").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("cntx-rs")
        .join("data")
}

pub fn validate_provider(provider: &str) -> Result<()> {
    providers::provider_descriptor(provider).map(|_| ())
}

pub fn default_model_for_provider(provider: &str) -> Option<&'static str> {
    providers::default_model_for_provider(provider)
}

pub fn resolve_provider_and_model(config: &Config, provider: Option<&str>, model: Option<&str>) -> Result<Config> {
    let mut effective = config.clone();

    if let Some(provider) = provider {
        let trimmed = provider.trim();
        if trimmed.is_empty() {
            anyhow::bail!("--provider cannot be empty");
        }
        validate_provider(trimmed)?;
        effective.provider = trimmed.to_string();
    }

    if let Some(model) = model {
        let trimmed = model.trim();
        if trimmed.is_empty() {
            anyhow::bail!("--model cannot be empty");
        }
        effective.model = trimmed.to_string();
    } else if effective.model.trim().is_empty() || effective.provider != config.provider {
        effective.model = if effective.provider == config.provider && !config.model.trim().is_empty() {
            config.model.clone()
        } else {
            default_model_for_provider(&effective.provider)
                .ok_or_else(|| anyhow::anyhow!("--model is required for provider '{}'", effective.provider))?
                .to_string()
        };
    }

    effective.normalized()?;
    Ok(effective)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{Config, DEFAULT_AGENTIC_MAX_STEPS, default_model_for_provider};
    use crate::temp_paths::unique_test_file;

    #[test]
    fn load_applies_defaults_to_partial_configs() {
        let path = unique_test_file("config-partial", "toml");
        fs::write(
            &path,
            "provider = \"openrouter\"\nmodel = \"nvidia/nemotron-3-super-120b-a12b:free\"\n",
        )
        .expect("write config");

        let config = Config::load(&path).expect("partial config should load");

        assert_eq!(config.provider, "openrouter");
        assert_eq!(config.model, "nvidia/nemotron-3-super-120b-a12b:free");
        assert!(!config.data_dir.as_os_str().is_empty());
        assert!(!config.resources.is_empty());
        assert_eq!(config.agentic_max_steps, DEFAULT_AGENTIC_MAX_STEPS);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn load_applies_provider_default_model_when_missing() {
        let path = unique_test_file("config-missing-model", "toml");
        fs::write(&path, "provider = \"gemini\"\n").expect("write config");

        let config = Config::load(&path).expect("config with missing model should load");

        assert_eq!(config.provider, "gemini");
        assert_eq!(
            config.model,
            default_model_for_provider("gemini").expect("gemini default model")
        );

        let _ = fs::remove_file(path);
    }
}
