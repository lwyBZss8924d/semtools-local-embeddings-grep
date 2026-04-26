---
name: semtools-pipelines
description: Compose semtools parse, search, ask, workspace, and local enhanced jgrep commands into efficient shell pipelines. Use when the user wants an end-to-end semtools workflow, such as parsing PDFs then searching them, asking questions over a corpus, running Jina-backed semantic grep/classification, or choosing between stdin pipelines and file-based workspace flows.
---

# Semtools Pipelines

Use this skill when the user needs an end-to-end semtools workflow rather than help on just one subcommand.

## Decision guide

- Binary docs like PDF first: `semtools parse`
- Text retrieval over files: `semtools search`
- Jina-backed semantic grep/code search or classification: `semtools jgrep`
- Synthesis or QA over files: `semtools ask`
- Repeated corpus work: `semtools workspace`

## Recommended pipeline patterns

### Parse then search by emitted file paths

```bash
semtools parse ./papers/*.pdf | xargs semtools search "authentication, OAuth, token refresh" --top-k 10 --n-lines 5
```

This preserves file-based search behavior and works well with later workspace usage.

### Parse then ask over emitted file paths

```bash
semtools parse ./papers/*.pdf | xargs semtools ask "What are the main findings?"
```

### Jina semantic grep over code/text

```bash
semtools jgrep "authentication middleware token refresh" src --recursive --include '*.rs' --top-k 10 --json
```

### Jina classification pipeline

```bash
semtools jgrep --classify -e bug -e feature -e docs ./.agents/issues/open/*.txt --json
```

### Jina workspace sync then search

```bash
semtools jgrep --workspace infra-ops --profile code --sync AGENTS.md SPEC.md ops/*.txt --json
semtools jgrep --workspace infra-ops --profile code "shell path authority" --top-k 8 --json
```

### Stdin reranking with jgrep

```bash
printf '%s\n' "token refresh logic" "database migration runner" | semtools jgrep "OAuth token refresh" --top-k 1 --json
```

### Parse then aggregate to stdin for quick ad hoc scanning

```bash
semtools parse ./papers/*.pdf | xargs cat | semtools search "large language model, evaluation, benchmark" --n-lines 5 --max-distance 0.4
```

Use this only when losing per-file structure is acceptable.

### Workspace-backed repeated retrieval

```bash
semtools workspace use papers
export SEMTOOLS_WORKSPACE=papers
semtools search "retry backoff, timeout, circuit breaker" ./.parse/*.md --top-k 8 --n-lines 4
semtools ask "What are the main findings?" ./.parse/*.md
```

## Pipeline guidance

- Prefer `semtools jgrep` for Jina-backed semantic grep/code search, label classification, workspace/profile sync or search, `--models-status`, and optional daemon-backed embedding.
- Prefer `semtools search` for classic lightweight semantic keyword search over explicit text files or parsed markdown.
- Prefer emitted file-path pipelines when later steps benefit from file mode or workspace mode.
- Prefer stdin pipelines for quick one-off scans of already assembled text.
- Parse first for binary docs; skip parse for normal text/code files.
- Use `--json` whenever another tool or automation step consumes the result.

## Caveat reminder

If multiple source files share the same basename, parse cache collisions in `~/.parse` can produce confusing results. Separate those runs or rename files first.
For `semtools jgrep`, be explicit about mode: `--sync` indexes, `--workspace` queries an existing profile, classification uses `-e`/`--classify`, and piped stdin triggers reranking rather than file grep.
