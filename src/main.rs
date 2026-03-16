mod agent;
mod ask;
mod auth;
mod cli;
mod config;
mod install;
mod mcp;
mod providers;
mod resources;
mod temp_paths;
mod validation;

use anyhow::Result;

fn main() -> Result<()> {
    cli::run()
}
