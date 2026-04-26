# Repository Guidelines

## Project Structure & Module Organization

This is a Rust 2024 CLI crate. The binary entrypoint is `src/bin/semtools.rs`; reusable library modules are exposed from `src/lib.rs`. Subcommand handlers live in `src/cmds/`, with domain modules under `src/parse/`, `src/search/`, `src/ask/`, and `src/workspace/`. The local fork's Jina semantic grep implementation is split across `src/cmds/jgrep.rs`, `src/jina/`, and `src/jina_runtime/`. Documentation lives in `README.md`, `docs/`, and `examples/`. Repo-owned AIHT skill sources live in `skills/semtools-skills-collection/`. Build output and local agent state (`target/`, `.omx/`) are not source files.

## Build, Test, and Development Commands

- `cargo build --all-features`: build the full local fork, including `jina-native-runtime`.
- `cargo run -- jgrep --help`: inspect the enhanced Jina grep CLI.
- `cargo run -- jgrep --models-status --json`: verify local Jina model assets.
- `cargo fmt --check`: verify Rust formatting without rewriting files.
- `cargo test --all-features`: run the full test suite.
- `cargo test --no-default-features --features parse,search,workspace,ask`: verify the upstream-style feature set still builds without Jina.

Use `cargo install --path . --force` from the repository root when refreshing the workstation-local binary.

## Coding Style & Naming Conventions

Use standard `rustfmt` output and keep Rust identifiers idiomatic: `snake_case` for functions/modules, `PascalCase` for types, and `SCREAMING_SNAKE_CASE` for constants. Prefer small command-layer functions that delegate to domain modules. Keep CLI flags and JSON field names stable once documented. Add comments only for non-obvious runtime behavior or safety constraints.

## Testing Guidelines

Tests are Rust unit tests colocated in module-level `#[cfg(test)]` blocks. Add tests near the behavior being changed, especially for parsing, output formatting, chunking, feature gating, and JSON wire shape. For Jina changes, include fast deterministic tests with fake embedding backends where possible; reserve real model checks for smoke commands.

## Commit & Pull Request Guidelines

Follow the existing concise imperative style, for example `feat: add local Jina semantic grep fork surface`, `fix npm`, or `format`. Keep commits scoped to one logical change. PR descriptions should summarize behavior changes, list validation commands, call out feature or JSON interface changes, and mention any required local model/cache setup. Include the trailer `Co-authored-by: Codex <noreply@openai.com>` exactly once in commit messages created with Codex.

## Security & Configuration Tips

Do not commit API keys, local model blobs, runtime sockets, caches, or `.omx/` state. `parse` uses LlamaParse credentials, `ask` uses OpenAI-compatible credentials, and `jgrep` expects local Hugging Face model assets or an explicit `--model-dir`.
