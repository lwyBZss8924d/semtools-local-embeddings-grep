---
name: semtools-skills-collection
description: Collection index for semtools CLI workflows. Use for progressive discovery of semtools parse, semantic search, document question-answering, workspace management, shell pipeline patterns, and the local-source semtools jgrep Jina semantic grep/classification surface when working with local document corpora or code/document retrieval.
metadata:
  collection-kind: custom
  source: local
---

# Semtools Skills Collection

Custom collection for semtools CLI workflows backed by the local source tree at `~/dev-space/semtools`.
The active workstation command is the local-source Cargo build installed into `~/.cargo/bin/semtools` with `cargo install --path /Users/arthur/dev-space/semtools --force`, not the crates.io default package.

## Skills

- `semtools-parse` — Parse PDFs and other non-text documents into cached markdown paths for downstream retrieval.
- `semtools-search` — Run semantic keyword search over text files or parsed markdown, with optional workspace acceleration.
- `semtools-ask` — Answer questions over document sets with semtools ask, especially when file mode and workspace mode matter.
- `semtools-workspace` — Create, inspect, activate, and prune semtools workspaces for repeated corpus search.
- `semtools-pipelines` — Compose parse, search, ask, workspace, and enhanced jgrep commands into efficient shell pipelines.

## Local Enhanced Surface

Use `semtools jgrep` for the locally enhanced Rust-native Jina semantic grep and code-search surface. It is the active local `jina-grep` / `jina-semsearch`-style capability, exposed as a semtools subcommand rather than separate PATH-default binaries. It supports recursive file search, include/exclude globs, grep-style context flags, classification labels with repeated `-e` or `--classify`, stdin reranking, `--json` output, `--models-status`, workspace/profile sync and search through `--workspace`, `--profile`, and `--sync`, plus optional daemon lifecycle flags `--daemon-start`, `--daemon-status`, `--daemon-stop`, and `--daemon`.

Prefer `semtools jgrep` when the task needs semantic grep over code or text with Jina embeddings, label classification, local workspace-backed Jina indexes, or machine-readable ranking output. Prefer classic `semtools search` when the task only needs the existing lightweight semantic keyword search surface over explicit text files or parsed markdown. To verify the enhanced surface, check `semtools jgrep --help` and `semtools jgrep --models-status --json`.

## Workflow

Use this collection when the user is working with the semtools CLI itself, wants to parse/search/ask over local corpora, needs the enhanced `semtools jgrep` Jina semantic grep/classification surface, or needs semtools workspace setup and maintenance guidance. Prefer these subskills over generic shell advice because semtools has important mode differences: binary documents must be parsed first, `search`/`ask` switch behavior between stdin and file mode, `jgrep` switches between grep/classify/workspace/pipe modes based on flags and inputs, and workspace activation requires either `SEMTOOLS_WORKSPACE` or `--workspace`.
