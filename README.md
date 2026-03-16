# cntx-rs

> **Warning**: This is an MVP project. Depending on the model, token usage may be high.

Point it at code or docs (local directories, Git repos, GitHub URLs) and ask questions. It runs a sandboxed retrieval loop, reads the relevant files, and gives you answers. Linux-only.

Written in Rust. Works from the CLI or as a stdio MCP server you can plug into your editor.

## Menu

- [Install](#install)
- [Quick start](#quick-start)
- [Resources](#resources)
- [Providers](#providers)
- [MCP](#mcp)
- [Configuration](#configuration)
- [Commands](#commands)

## Install

### From GitHub with cargo

```bash
cargo install --git https://github.com/bingbr/cntx-rs.git cntx-rs
```

On Ubuntu/Debian you may need build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y bubblewrap libdbus-1-dev pkg-config
```

### From GitHub releases

Grab the binary and checksum from the [releases page](https://github.com/bingbr/cntx-rs/releases).

```bash
sha256sum -c cntx-rs.sha256
chmod +x cntx-rs
./cntx-rs --help
```

On Linux, drop it into `~/.local/bin`:

```bash
./cntx-rs install
```

Pick a different directory or overwrite an existing binary:

```bash
./cntx-rs install --dir /usr/local/bin --force
```

### From a local checkout

```bash
cargo build --release
./target/release/cntx-rs --help
```

Tool execution runs inside a bubblewrap sandbox by default on Linux. If `bwrap` isn't available, turn it off:

```bash
cntx-rs config set --sandbox off
```

## Quick start

### 1. See what you have

```bash
cntx-rs config show
cntx-rs resources list
cntx-rs auth status
```

If there's no config file yet, `cntx-rs` ships with built-in `svelte`, `sveltekit`, and `tauri` Git resources so you have something to try right away.

### 2. Set up a provider

Store an API key in your system keychain:

```bash
cntx-rs auth login --provider openai
```

Or use an environment variable:

```bash
export OPENAI_API_KEY=...
cntx-rs auth status
```

### 3. Ask something

Ask across everything:

```bash
cntx-rs ask -q "Where is form handling documented?"
```

Ask one resource:

```bash
cntx-rs ask \
  --resource sveltekit \
  -q "Which file explains load functions?"
```

Pass a GitHub URL without saving it to config:

```bash
cntx-rs ask \
  --resource https://github.com/sveltejs/kit/tree/main/documentation \
  -q "Which file explains form actions?"
```

Override provider and model for a single run:

```bash
cntx-rs ask \
  -p openai \
  -m gpt-5-mini \
  -q "Compare the routing docs in @svelte and @sveltekit"
```

Stream output as it arrives:

```bash
cntx-rs ask --stream -q "Summarize the main entry points in @sveltekit"
```

## Resources

Resources are Git repos or local directories listed in your config. Each one can have search paths and optional notes that steer retrieval.

```bash
cntx-rs resources list
```

Add a Git resource:

```bash
cntx-rs resources add \
  --git https://github.com/example/project/tree/main/docs \
  --notes "Primary product documentation"
```

When the GitHub URL already includes info, `cntx-rs` infers `--name`, `--branch`, and `--search-path` for you:

```bash
cntx-rs resources add --git https://github.com/tauri-apps/tauri/tree/dev/crates/tauri-runtime
```

Add a local directory:

```bash
cntx-rs resources add \
  --name local-docs \
  --local ~/Projects/docs \
  --search-path reference
```

Remove one:

```bash
cntx-rs resources remove --name local-docs
```

Worth knowing:

- Omitting `--resource` searches everything.
- `@resource` mentions in a question resolve against configured names.
- HTTPS GitHub URLs passed to `ask` work as one-off resources without touching your config.
- Search paths can't escape the resource root.

## Providers

| Provider | Environment variables | Default model |
| --- | --- | --- |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-4-5` |
| `gemini` | `GEMINI_API_KEY`, `GOOGLE_API_KEY` | `gemini-3.1-flash-lite-preview` |
| `openrouter` | `OPENROUTER_API_KEY`, `API_KEY` | `nvidia/nemotron-3-super-120b-a12b:free` |
| `openai` | `OPENAI_API_KEY`, `API_KEY` | `gpt-5-mini` |

```bash
cntx-rs auth status
cntx-rs auth login --provider anthropic
cntx-rs auth logout --provider openai
```

## MCP

Start the stdio server:

```bash
cntx-rs mcp serve
```

Override provider or model:

```bash
cntx-rs mcp serve --provider anthropic --model claude-haiku-4-5
```

Print a VS Code MCP config snippet:

```bash
cntx-rs mcp config
```

Write it to a file:

```bash
cntx-rs mcp config --path /path/to/mcp.json
```

The server exposes the `ask` tool.

## Configuration

Config lives under your platform config directory. On Linux:

```text
~/.config/cntx-rs/config.toml
```

It stores the default provider, model, data directory, resources, max agentic steps, sandbox settings, and `bwrap` path.

See what's active:

```bash
cntx-rs config show
```

Change defaults:

```bash
cntx-rs config set --agentic-max-steps 30
cntx-rs config set --sandbox off
```

Use a different config file for any command:

```bash
cntx-rs --config /path/to/config.toml ask -q "What changed in @docs?"
```

## Commands

```bash
cntx-rs auth login
cntx-rs auth logout
cntx-rs auth status
cntx-rs config show
cntx-rs config set
cntx-rs ask
cntx-rs resources list
cntx-rs resources add
cntx-rs resources remove
cntx-rs mcp serve
cntx-rs mcp config
cntx-rs install
cntx-rs uninstall
```
