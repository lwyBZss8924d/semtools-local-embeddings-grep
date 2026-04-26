---
name: semtools-search
description: Run semtools semantic search and the local enhanced jgrep surface over local files, parsed markdown, code, or stdin. Use when the user wants semantic keyword search across a document set, Jina-backed semantic grep/classification, top-k or threshold-based retrieval, or faster repeated search with a semtools workspace.
---

# Semtools Search

Use `semtools search` for local semantic keyword retrieval over text files or parsed markdown. Use the local-source `semtools jgrep` surface when the task needs Jina-backed semantic grep over code/text, classification labels, workspace/profile indexing, or daemon-backed embedding.

## Core command

```bash
semtools search [OPTIONS] <QUERY> [FILES]...
```

Useful options:

- `-n, --n-lines <N>` — context lines around each hit
- `--top-k <K>` — bounded result count
- `-m, --max-distance <D>` — threshold mode; returns all results below the distance
- `-i, --ignore-case` — case-insensitive search
- `-j, --json` — machine-readable output
- `-w, --workspace <WORKSPACE>` — use a specific workspace

## Enhanced Jina jgrep surface

The workstation `semtools` binary is built from `~/dev-space/semtools` and includes:

```bash
semtools jgrep [OPTIONS] [PATTERN] [FILES]...
```

High-value options:

- `-r, --recursive` — recurse through directories
- `--include <GLOB>` / `--exclude <GLOB>` / `--exclude-dir <GLOB>` — constrain file selection
- `-A`, `-B`, `-C` — grep-style context output
- `--threshold <D>` / `--top-k <K>` — semantic score filtering or bounded ranking
- `-e, --regexp <LABEL>` / `--classify` / `-f, --file <LABEL_FILE>` — classify content against labels
- `--workspace <NAME>` / `--profile <ID>` / `--sync` — index or query a Jina workspace profile
- `--json` — machine-readable output
- `--models-status` — verify local Jina model availability
- `--daemon-start`, `--daemon-status`, `--daemon-stop`, `--daemon` — optional daemon-backed embedding path

Examples:

```bash
semtools jgrep "retry backoff timeout" src --recursive --include '*.rs' --top-k 8 --json
printf '%s\n' "token refresh logic" "database migration runner" | semtools jgrep "OAuth token refresh" --top-k 1 --json
semtools jgrep --classify -e bug -e feature -e docs ./issues/*.txt --json
semtools jgrep --workspace infra-ops --profile code --sync AGENTS.md SPEC.md ops/*.txt --json
semtools jgrep --workspace infra-ops --profile code "shell path authority" --top-k 8 --json
```

## Mode selection

### File mode
Use file mode when you have explicit files and want workspace acceleration.

```bash
semtools search "retry backoff, timeout, circuit breaker" docs/*.md --top-k 8 --n-lines 4
```

### Stdin mode
Use stdin mode for ad hoc piped content when you do not need per-file workspace behavior.

```bash
printf '%s\n' "token refresh logic" | semtools search "token refresh"
```

## Workspace-aware pattern

If repeated searches target the same corpus, activate a workspace and keep passing the file list:

```bash
semtools search "retry backoff, timeout, circuit breaker" ./.parse/*.md --workspace papers --top-k 8 --n-lines 4
```

## Retrieval guidance

- Use `semtools jgrep` for Jina-backed semantic grep/code search, classification, workspace/profile sync, and daemon-backed local embedding workflows.
- Use `semtools search` when the caller wants the classic lightweight semtools search behavior over explicit text files or parsed markdown.
- Use `--top-k` when the caller wants a bounded shortlist.
- Use `--max-distance` when the caller wants all sufficiently similar hits.
- If `--max-distance` is present, it overrides the usual top-k behavior.
- `search` only works on text-readable content; parse binary docs first.

## Common pitfalls

- No files and no stdin means the command errors.
- Workspace mode still requires explicit files; the workspace accelerates indexing and lookup for that file subset.
- Search result headers use 0-based range values internally even though displayed line content is human-readable.
- `semtools jgrep` changes behavior based on flags and inputs: label flags trigger classification, `--workspace` without `--sync` queries a profile, `--sync` indexes files, and piped stdin triggers reranking mode.
