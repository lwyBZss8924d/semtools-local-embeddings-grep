# Repository Guidelines

## Project Structure & Module Organization

This is a Rust 2024 CLI crate. The binary entrypoint is `src/bin/semtools.rs`; reusable modules are exported from `src/lib.rs`. Subcommand handlers live in `src/cmds/`, with domain code under `src/parse/`, `src/search/`, `src/ask/`, and `src/workspace/`. The local Jina grep fork is split across `src/cmds/jgrep.rs`, `src/jina/`, and `src/jina_runtime/`. Documentation lives in `README.md`, `docs/`, and `examples/`. Repo-owned AIHT skills live in `skills/semtools-skills-collection/`.

## Build, Test, and Development Commands

- `cargo build --all-features`: build the full fork, including `jina-native-runtime`.
- `cargo run -- jgrep --help`: inspect the enhanced Jina grep CLI.
- `cargo run -- jgrep --models-status --json`: verify local MLX checkpoint assets.
- `cargo fmt --check`: verify Rust formatting.
- `cargo test --all-features`: run the full test suite.
- `cargo test --no-default-features --features parse,search,workspace,ask`: verify upstream-style features without Jina.
- `cargo install --path . --force`: refresh the workstation-local binary from the repo root.

## Coding Style & Testing

Use standard `rustfmt`. Follow Rust naming conventions: `snake_case` functions/modules, `PascalCase` types, and `SCREAMING_SNAKE_CASE` constants. Keep CLI flags and JSON fields stable once documented. Tests are colocated in module-level `#[cfg(test)]` blocks. For Jina behavior, prefer deterministic tests with fake embedding backends; reserve real model checks for smoke commands.

## Documentation & Model Cards

When documenting supported Jina models, use `hf models info <repo> --format json` plus `hf download <repo> README.md --local-dir /tmp/...`; do not vendor downloaded model cards. Current runtime targets MLX checkpoint repos such as `jinaai/jina-embeddings-v5-text-small-mlx` and `jinaai/jina-code-embeddings-1.5b-mlx`, while executing them through this crate's Rust-native local runtime. Keep README examples publication-safe: use `cargo install --path . --force`, `~/.parse`, and `~/.semtools`, not personal absolute paths.

## Fork Workflow

`main` tracks `upstream/main` and should stay a clean upstream mirror. `local-enhanced-main` is the fork's default enhanced branch on `origin`. Feature and docs branches should target `local-enhanced-main` via PR, then be merged there. Keep `origin` pointed at the personal fork and `upstream` pointed at `run-llama/semtools`.

## Commit & Pull Request Guidelines

Use concise imperative commits, for example `docs: add HF MLX model card details` or `feat: add local Jina semantic grep fork surface`. PRs should summarize behavior changes, validation commands, feature or JSON interface changes, and model/cache requirements. Commit messages created with Codex must include `Co-authored-by: Codex <noreply@openai.com>` exactly once.

## Security & Configuration Tips

Do not commit API keys, local model blobs, runtime sockets, caches, `target/`, or `.omx/` state. `parse` uses LlamaParse credentials, `ask` uses OpenAI-compatible credentials, and `jgrep` expects Hugging Face model assets in cache or an explicit `--model-dir`.
