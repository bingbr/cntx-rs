use std::{
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, anyhow, bail};

use super::{
    tools,
    types::{AgentWorkspace, ToolCall, ToolExecutor, ToolResult},
};
use crate::config::Config;

pub enum ToolExecutorImpl {
    Local,
    Bubblewrap(BubblewrapToolExecutor),
}

impl ToolExecutorImpl {
    pub fn from_config(config: &Config) -> Result<Self> {
        if config.agentic_require_sandbox {
            Ok(Self::Bubblewrap(BubblewrapToolExecutor::from_config(config)?))
        } else {
            Ok(Self::Local)
        }
    }
}

impl ToolExecutor for ToolExecutorImpl {
    fn execute(&self, workspace: &AgentWorkspace, call: &ToolCall) -> Result<ToolResult> {
        match self {
            Self::Local => tools::execute_tool(workspace, call),
            Self::Bubblewrap(executor) => executor.execute(workspace, call),
        }
    }
}

pub struct BubblewrapToolExecutor {
    bwrap_path: PathBuf,
    binary_path: PathBuf,
}

impl BubblewrapToolExecutor {
    fn from_config(config: &Config) -> Result<Self> {
        Ok(Self {
            bwrap_path: config.bwrap_path.clone().unwrap_or_else(|| PathBuf::from("bwrap")),
            binary_path: std::env::current_exe().with_context(|| "Resolving current cntx-rs binary path")?,
        })
    }

    fn execute(&self, workspace: &AgentWorkspace, call: &ToolCall) -> Result<ToolResult> {
        let call_json = serde_json::to_string(call)?;
        let mut command = Command::new(&self.bwrap_path);
        command.arg("--unshare-all");
        command.arg("--die-with-parent");
        command.arg("--new-session");
        command.arg("--clearenv");

        if let Some(path) = std::env::var_os("PATH") {
            command.arg("--setenv").arg("PATH").arg(path);
        }

        for path in ["/usr", "/bin", "/lib", "/lib64", "/nix/store"] {
            if Path::new(path).exists() {
                command.arg("--ro-bind").arg(path).arg(path);
            }
        }

        for path in ["/etc/alternatives", "/etc/ssl", "/etc/pki"] {
            if Path::new(path).exists() {
                command.arg("--ro-bind").arg(path).arg(path);
            }
        }

        command
            .arg("--ro-bind")
            .arg(&workspace.root)
            .arg("/workspace")
            .arg("--ro-bind")
            .arg(&self.binary_path)
            .arg("/cntx-rs-bin")
            .arg("--chdir")
            .arg("/workspace")
            .arg("--proc")
            .arg("/proc")
            .arg("--dev")
            .arg("/dev")
            .arg("--tmpfs")
            .arg("/tmp")
            .arg("/cntx-rs-bin")
            .arg("internal-tool")
            .arg("--workspace-root")
            .arg("/workspace")
            .arg("--call-json")
            .arg(call_json);

        let output = command.output().map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                anyhow!(
                    "bubblewrap is required for agentic mode but '{}' was not found",
                    self.bwrap_path.display()
                )
            } else {
                anyhow!("failed to start bubblewrap '{}': {error}", self.bwrap_path.display())
            }
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let detail = if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("exit status {}", output.status)
            };
            bail!("bubblewrap tool execution failed: {detail}");
        }

        serde_json::from_slice(&output.stdout).with_context(|| "Parsing sandboxed tool result JSON")
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use serde_json::json;

    use super::BubblewrapToolExecutor;
    use crate::{
        agent::types::{AgentWorkspace, ToolCall, ToolName},
        config::Config,
        temp_paths::create_test_dir,
    };

    #[test]
    fn bubblewrap_executor_fails_closed_when_binary_is_missing() {
        let config = Config {
            bwrap_path: Some(PathBuf::from("/definitely-missing-bwrap")),
            ..Config::default()
        };
        let executor = BubblewrapToolExecutor::from_config(&config).expect("executor should build");
        let workspace_root = create_test_dir("agentic-sandbox-missing-bwrap");
        fs::create_dir_all(workspace_root.join("repo")).expect("create repo dir");

        let error = executor
            .execute(
                &AgentWorkspace {
                    root: workspace_root.clone(),
                    resources: Vec::new(),
                },
                &ToolCall {
                    id: "1".to_string(),
                    tool: ToolName::List,
                    input: json!({ "path": "." }),
                    provider_data: None,
                },
            )
            .expect_err("missing bwrap should fail");

        assert!(error.to_string().contains("bubblewrap is required"));
        let _ = fs::remove_dir_all(workspace_root);
    }
}
