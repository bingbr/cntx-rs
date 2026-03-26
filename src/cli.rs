use std::{
    fs,
    io::{self, IsTerminal, Write},
    path::{Path, PathBuf},
};

use anyhow::{Result, anyhow};
use clap::{Parser, Subcommand, ValueEnum};

use crate::{
    agent, ask, auth,
    config::{self, Config},
    install, mcp, providers,
    resources::ResourceConfig,
};

#[derive(Parser)]
#[command(name = "cntx-rs")]
#[command(about = "Small local MCP-first context tool written in Rust")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(long, global = true)]
    config: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Manage provider credentials for local MCP usage.
    Auth {
        #[command(subcommand)]
        command: AuthCommands,
    },
    /// Inspect or update local cntx-rs configuration.
    Config {
        #[command(subcommand)]
        command: ConfigCommands,
    },
    /// Ask a grounded question against one or more configured resources.
    Ask {
        #[arg(long = "resource")]
        resources: Vec<String>,

        #[arg(long, visible_alias = "q", short = 'q')]
        question: String,

        #[arg(long, visible_alias = "p", short = 'p')]
        provider: Option<String>,

        #[arg(long, visible_alias = "m", short = 'm')]
        model: Option<String>,

        #[arg(long, default_value_t = false)]
        stream: bool,
    },
    /// Install cntx-rs into a Linux bin directory.
    Install {
        /// Install directory. Defaults to ~/.local/bin on Linux.
        #[arg(long)]
        dir: Option<PathBuf>,

        /// Replace an existing installed binary.
        #[arg(long, default_value_t = false)]
        force: bool,
    },
    /// Remove an installed cntx-rs binary on Linux.
    Uninstall {
        /// Directory containing the installed binary. Defaults to ~/.local/bin on Linux.
        #[arg(long)]
        dir: Option<PathBuf>,

        /// Also remove the config file and cached data directory.
        #[arg(long, default_value_t = false)]
        purge: bool,
    },
    /// Manage local resource definitions.
    Resources {
        #[command(subcommand)]
        command: ResourceCommands,
    },
    /// Start a stdio MCP server or print editor configuration.
    Mcp {
        #[command(subcommand)]
        command: Option<McpCommands>,
    },
    #[command(hide = true)]
    InternalTool {
        #[arg(long)]
        workspace_root: PathBuf,

        #[arg(long)]
        call_json: String,
    },
}

#[derive(Subcommand)]
enum ResourceCommands {
    /// List configured resources.
    List,
    /// Add or replace a resource.
    Add {
        /// Resource name. Optional for git resources when it can be derived from the URL.
        #[arg(long)]
        name: Option<String>,
        /// Git URL (set when adding git resources). GitHub tree/blob URLs can infer branch and search path.
        #[arg(long)]
        git: Option<String>,
        /// Local directory path (set when adding local resources).
        #[arg(long)]
        local: Option<String>,
        /// Branch for git resources. Optional when --git already points at a GitHub tree/blob URL.
        #[arg(long)]
        branch: Option<String>,
        /// Relative search paths within the resource root. Optional when --git already scopes a GitHub tree/blob URL.
        #[arg(long = "search-path")]
        search_paths: Vec<String>,
        /// Resource-specific notes to bias retrieval.
        #[arg(long)]
        notes: Option<String>,
    },
    /// Remove a resource by name.
    Remove {
        #[arg(long)]
        name: String,
    },
}

#[derive(Subcommand)]
enum AuthCommands {
    /// Store an API key in the local system keychain.
    Login {
        #[arg(long)]
        provider: Option<String>,

        #[arg(long)]
        api_key: Option<String>,

        #[arg(long)]
        model: Option<String>,
    },
    /// Remove an API key from the local system keychain.
    Logout {
        #[arg(long)]
        provider: Option<String>,
    },
    /// Show provider defaults and auth state.
    Status,
}

#[derive(Subcommand)]
enum ConfigCommands {
    /// Print the current configuration values that affect runtime behavior.
    Show,
    /// Update runtime defaults in the local config file.
    Set {
        #[arg(long)]
        agentic_max_steps: Option<usize>,

        #[arg(long, value_enum)]
        sandbox: Option<SandboxMode>,
    },
}

#[derive(Subcommand)]
enum McpCommands {
    /// Start the stdio MCP server.
    Serve {
        #[arg(long, visible_alias = "p", short = 'p')]
        provider: Option<String>,

        #[arg(long, visible_alias = "m", short = 'm')]
        model: Option<String>,
    },
    /// Print a stdio MCP configuration snippet for editors that use the VS Code-style shape.
    Config {
        #[arg(long, visible_alias = "p", short = 'p')]
        path: Option<PathBuf>,

        #[arg(long)]
        provider: Option<String>,

        #[arg(long)]
        model: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum SandboxMode {
    On,
    Off,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let context = CommandContext::new(cli.config.unwrap_or_else(config::default_config_path));

    match cli.command {
        Commands::Auth { command } => {
            let config = context.load_config()?;
            match command {
                AuthCommands::Login {
                    provider,
                    api_key,
                    model,
                } => {
                    let mut config = config;
                    let provider = select_login_provider(provider, &config, context.config_exists())?;
                    let api_key = auth::resolve_api_key_for_storage(&provider, api_key.as_deref())?;
                    let model = select_login_model(model, &config, &provider)?;
                    auth::store_api_key(&provider, &api_key)?;
                    apply_provider_settings(&mut config, provider.clone(), Some(model))?;
                    context.save_config(&config)?;
                    println!(
                        "Stored credentials for provider '{}' in the local system keychain.",
                        provider
                    );
                    println!("Configuration written to {}", context.config_path().display());
                }
                AuthCommands::Logout { provider } => {
                    let provider = provider.unwrap_or_else(|| config.provider.clone());
                    config::validate_provider(&provider)?;
                    if auth::delete_api_key(&provider)? {
                        println!("Removed local system keychain credentials for provider '{}'.", provider);
                    } else {
                        println!("No local system keychain credentials found for '{}'.", provider);
                    }
                }
                AuthCommands::Status => {
                    let statuses = providers::provider_statuses(&config)?;
                    print!("{}", render_provider_statuses(&statuses));
                }
            }
        }
        Commands::Config { command } => {
            let mut config = context.load_config()?;
            match command {
                ConfigCommands::Show => {
                    print!("{}", render_runtime_config(&config, context.config_path()));
                }
                ConfigCommands::Set {
                    agentic_max_steps,
                    sandbox,
                } => {
                    update_runtime_config(&mut config, agentic_max_steps, sandbox)?;
                    context.save_config(&config)?;
                    print!("{}", render_runtime_config(&config, context.config_path()));
                }
            }
        }
        Commands::Ask {
            resources,
            question,
            provider,
            model,
            stream,
        } => {
            let config = context.resolve_config(provider.as_deref(), model.as_deref())?;
            if stream
                && !providers::build_adapter(&config.provider)?
                    .capabilities()
                    .supports_streaming
            {
                return Err(anyhow!(
                    "provider '{}' does not support true streaming",
                    config.provider
                ));
            }
            let mut progress = ProgressOutput::stderr();
            let show_progress = progress.enabled();
            let mut on_progress = |message: &str| progress.emit(message);
            if stream {
                let mut out = io::BufWriter::new(io::stdout());
                let mut wrote = false;
                let mut on_chunk = |chunk: &str| -> Result<()> {
                    wrote = true;
                    out.write_all(chunk.as_bytes())?;
                    out.flush()?;
                    Ok(())
                };
                let progress_callback = if show_progress {
                    Some(&mut on_progress as &mut dyn FnMut(&str) -> Result<()>)
                } else {
                    None
                };
                let output = ask::ask_question(&config, &resources, &question, Some(&mut on_chunk), progress_callback)?;
                if !wrote {
                    println!("{}", ask::format_cli_output(&output));
                } else {
                    print!("{}", ask::format_cli_footer(&output));
                    io::stdout().flush()?;
                }
            } else {
                let progress_callback = if show_progress {
                    Some(&mut on_progress as &mut dyn FnMut(&str) -> Result<()>)
                } else {
                    None
                };
                let output = ask::ask_question(&config, &resources, &question, None, progress_callback)?;
                println!("{}", ask::format_cli_output(&output));
            }
        }
        Commands::Install { dir, force } => {
            let outcome = install::install(dir, force)?;
            if outcome.already_current {
                println!("cntx-rs is already running from {}", outcome.target_path.display());
            } else if outcome.replaced_existing {
                println!("Updated cntx-rs at {}", outcome.target_path.display());
            } else {
                println!("Installed cntx-rs to {}", outcome.target_path.display());
            }
            if let Some(path_hint) = outcome.path_hint {
                println!("{path_hint}");
            }
        }
        Commands::Uninstall { dir, purge } => {
            let outcome = install::uninstall(dir, context.config_path(), purge)?;
            if outcome.removed_binary {
                println!("Removed cntx-rs from {}", outcome.target_path.display());
            } else {
                println!("No installed cntx-rs binary found at {}", outcome.target_path.display());
            }
            if purge {
                if outcome.removed_config || outcome.removed_data_dir {
                    println!("Removed local config/data state.");
                } else {
                    println!("No local config/data state found.");
                }
            }
        }
        Commands::Resources { command } => {
            let mut config = context.load_config()?;
            match command {
                ResourceCommands::List => {
                    for resource in &config.resources {
                        println!(
                            "{}\t{:?}\t{}\t{}\t{}",
                            resource.name,
                            resource.kind,
                            resource.search_paths_display(),
                            resource.source_display(),
                            resource.notes.as_deref().unwrap_or("")
                        );
                    }
                }
                ResourceCommands::Add {
                    name,
                    git,
                    local,
                    branch,
                    search_paths,
                    notes,
                } => {
                    let resource = ResourceConfig::new(name, git, local, branch, search_paths, notes)?;
                    if let Some(existing) = config.resources.iter().position(|entry| entry.name == resource.name) {
                        config.resources[existing] = resource;
                    } else {
                        config.resources.push(resource);
                    }
                    context.save_config(&config)?;
                    println!("Resource configuration written to {}", context.config_path().display());
                }
                ResourceCommands::Remove { name } => {
                    let before = config.resources.len();
                    config.resources.retain(|resource| resource.name != name);
                    if config.resources.len() == before {
                        println!("No resource named '{name}'");
                    } else {
                        context.save_config(&config)?;
                        println!("Resource '{name}' removed.");
                    }
                }
            }
        }
        Commands::Mcp { command } => {
            match command.unwrap_or(McpCommands::Serve {
                provider: None,
                model: None,
            }) {
                McpCommands::Serve { provider, model } => {
                    let config = context.resolve_config(provider.as_deref(), model.as_deref())?;
                    mcp::run_server(config)?;
                }
                McpCommands::Config { path, provider, model } => {
                    let snippet = mcp_config_snippet(context.config_path(), provider.as_deref(), model.as_deref());
                    if let Some(path) = path {
                        write_mcp_config(&path, &snippet)?;
                        println!("Wrote MCP config snippet to {}", path.display());
                    } else {
                        println!("{snippet}");
                    }
                }
            }
        }
        Commands::InternalTool {
            workspace_root,
            call_json,
        } => {
            let call: agent::types::ToolCall =
                serde_json::from_str(&call_json).map_err(|error| anyhow!("Invalid --call-json payload: {error}"))?;
            let result = match agent::runtime::run_internal_tool(&workspace_root, call.clone()) {
                Ok(result) => result,
                Err(error) => agent::types::ToolResult::error(&call, error.to_string()),
            };
            println!("{}", serde_json::to_string(&result)?);
        }
    }

    Ok(())
}

pub fn run() -> Result<()> {
    main()
}

struct CommandContext {
    config_path: PathBuf,
}

impl CommandContext {
    fn new(config_path: PathBuf) -> Self {
        Self { config_path }
    }

    fn config_exists(&self) -> bool {
        self.config_path.exists()
    }

    fn config_path(&self) -> &Path {
        &self.config_path
    }

    fn load_config(&self) -> Result<Config> {
        Config::load(&self.config_path)
    }

    fn resolve_config(&self, provider: Option<&str>, model: Option<&str>) -> Result<Config> {
        let config = self.load_config()?;
        config::resolve_provider_and_model(&config, provider, model)
    }

    fn save_config(&self, config: &Config) -> Result<()> {
        config.save(&self.config_path)
    }
}

struct ProgressOutput {
    enabled: bool,
    writer: io::BufWriter<io::Stderr>,
}

impl ProgressOutput {
    fn stderr() -> Self {
        Self {
            enabled: io::stderr().is_terminal(),
            writer: io::BufWriter::new(io::stderr()),
        }
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn emit(&mut self, message: &str) -> Result<()> {
        writeln!(
            self.writer,
            "{}",
            message.split_whitespace().collect::<Vec<_>>().join(" ")
        )?;
        self.writer.flush()?;
        Ok(())
    }
}

fn render_runtime_config(config: &Config, config_path: &std::path::Path) -> String {
    format!(
        "Config file: {}\nprovider = {}\nmodel = {}\nagentic_max_steps = {}\nagentic_require_sandbox = {}\n",
        config_path.display(),
        config.provider,
        config.model,
        config.agentic_max_steps,
        yes_no(config.agentic_require_sandbox),
    )
}

fn update_runtime_config(
    config: &mut Config,
    agentic_max_steps: Option<usize>,
    sandbox: Option<SandboxMode>,
) -> Result<()> {
    if agentic_max_steps.is_none() && sandbox.is_none() {
        return Err(anyhow!(
            "config set requires at least one of --agentic-max-steps or --sandbox"
        ));
    }

    if let Some(agentic_max_steps) = agentic_max_steps {
        if agentic_max_steps == 0 {
            return Err(anyhow!("--agentic-max-steps must be greater than zero"));
        }
        config.agentic_max_steps = agentic_max_steps;
    }
    if let Some(sandbox) = sandbox {
        config.agentic_require_sandbox = matches!(sandbox, SandboxMode::On);
    }

    config.normalized()?;
    Ok(())
}

fn apply_provider_settings(config: &mut Config, provider: String, model: Option<String>) -> Result<()> {
    config::validate_provider(&provider)?;
    let model = model
        .or_else(|| default_model_for_login(config, &provider))
        .ok_or_else(|| anyhow!("--model is required for provider '{}'", provider))?;
    config.provider = provider;
    config.model = model;
    providers::validate_provider_config(config)?;
    Ok(())
}

fn render_provider_statuses(statuses: &[providers::ProviderStatus]) -> String {
    let mut output = String::new();
    output.push_str(&format!(
        "{:<18} {:<7} {:<6} {:<38} {:<22} {}",
        "provider", "current", "auth", "default model", "source", "status"
    ));
    output.push('\n');
    for status in statuses {
        output.push_str(&format!(
            "{:<18} {:<7} {:<6} {:<38} {:<22} {}",
            status.id,
            yes_no(status.current),
            yes_no(status.authenticated),
            status.default_model.unwrap_or("-"),
            status.auth_source.as_deref().unwrap_or("-"),
            status.hint
        ));
        output.push('\n');
    }
    output
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn default_model_for_login(config: &Config, provider: &str) -> Option<String> {
    if config.provider == provider && !config.model.trim().is_empty() {
        Some(config.model.clone())
    } else {
        config::default_model_for_provider(provider).map(ToString::to_string)
    }
}

fn select_login_provider(provider: Option<String>, config: &Config, has_saved_config: bool) -> Result<String> {
    if let Some(provider) = provider {
        let provider = provider.trim().to_string();
        config::validate_provider(&provider)?;
        return Ok(provider);
    }

    prompt_provider_selection(config, has_saved_config)
}

fn select_login_model(model: Option<String>, config: &Config, provider: &str) -> Result<String> {
    if let Some(model) = model {
        let trimmed = model.trim();
        if trimmed.is_empty() {
            return Err(anyhow!("--model cannot be empty"));
        }
        return Ok(trimmed.to_string());
    }

    let default = default_model_for_login(config, provider);
    prompt_optional_value(
        &format!("Model for provider '{provider}'"),
        default.as_deref(),
        default.is_none(),
    )
}

fn prompt_provider_selection(config: &Config, has_saved_config: bool) -> Result<String> {
    ensure_interactive_terminal("--provider")?;

    let descriptors = providers::supported_providers();
    println!("Select a provider for local MCP usage:");
    for (index, descriptor) in descriptors.iter().enumerate() {
        let model_hint = descriptor.default_model.unwrap_or("model required");
        println!("  {}. {} ({})", index + 1, descriptor.id, model_hint);
    }

    let default = has_saved_config.then_some(config.provider.as_str());
    loop {
        let prompt = if let Some(default) = default {
            format!("Provider [{}]: ", default)
        } else {
            "Provider: ".to_string()
        };
        let input = prompt_line(&prompt)?;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            if let Some(default) = default {
                return Ok(default.to_string());
            }
            eprintln!("A provider is required.");
            continue;
        }

        if let Ok(index) = trimmed.parse::<usize>()
            && let Some(descriptor) = descriptors.get(index.saturating_sub(1))
        {
            return Ok(descriptor.id.to_string());
        }

        if descriptors.iter().any(|descriptor| descriptor.id == trimmed) {
            return Ok(trimmed.to_string());
        }

        eprintln!("Unknown provider '{}'. Enter a number or provider id.", trimmed);
    }
}

fn prompt_optional_value(prompt: &str, default: Option<&str>, required: bool) -> Result<String> {
    ensure_interactive_terminal(prompt)?;

    loop {
        let rendered = match default {
            Some(default) => format!("{prompt} [{default}]: "),
            None => format!("{prompt}: "),
        };
        let input = prompt_line(&rendered)?;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            if let Some(default) = default {
                return Ok(default.to_string());
            }
            if required {
                eprintln!("{prompt} is required.");
                continue;
            }
        }

        return Ok(trimmed.to_string());
    }
}

fn ensure_interactive_terminal(flag_hint: &str) -> Result<()> {
    if io::stdin().is_terminal() {
        Ok(())
    } else {
        Err(anyhow!(
            "Missing required interactive input. Re-run in a terminal or pass {} explicitly.",
            flag_hint
        ))
    }
}

fn prompt_line(prompt: &str) -> Result<String> {
    let mut stdout = io::stdout();
    stdout.write_all(prompt.as_bytes())?;
    stdout.flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input)
}

fn mcp_config_snippet(config_path: &std::path::Path, provider: Option<&str>, model: Option<&str>) -> String {
    let binary = std::env::current_exe().ok().unwrap_or_else(|| PathBuf::from("cntx-rs"));
    let mut args = vec![
        "--config".to_string(),
        config_path.display().to_string(),
        "mcp".to_string(),
        "serve".to_string(),
    ];
    if let Some(provider) = provider.filter(|provider| !provider.trim().is_empty()) {
        args.push("--provider".to_string());
        args.push(provider.to_string());
    }
    if let Some(model) = model.filter(|model| !model.trim().is_empty()) {
        args.push("--model".to_string());
        args.push(model.to_string());
    }
    serde_json::to_string_pretty(&serde_json::json!({
        "servers": {
            "cntx-rs": {
                "type": "stdio",
                "command": binary.display().to_string(),
                "args": args,
            }
        }
    }))
    .expect("MCP config should serialize")
}

fn write_mcp_config(path: &std::path::Path, snippet: &str) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, snippet)?;
    Ok(())
}
