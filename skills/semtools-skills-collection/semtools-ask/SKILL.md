---
name: semtools-ask
description: Answer questions over local document sets with semtools ask. Use when the user wants synthesis or question-answering across files, parsed markdown, or workspace-backed corpora, especially when the difference between stdin mode and file mode matters.
---

# Semtools Ask

Use `semtools ask` for document question-answering over local corpora.

## Core command

```bash
semtools ask [OPTIONS] <QUERY> [FILES]...
```

Useful options:

- `-c, --config <CONFIG>` — config file path
- `--api-key <API_KEY>` — override API key
- `--base-url <BASE_URL>` — override OpenAI-compatible endpoint
- `-m, --model <MODEL>` — override model
- `--api-mode <chat|responses>` — choose API mode
- `-j, --json` — machine-readable output
- `-w, --workspace <WORKSPACE>` — use a specific workspace

## Critical mode difference

### File mode
If you pass files, semtools ask runs an agent workflow with internal retrieval tools (`grep`, `search`, `read`) over the corpus.

```bash
semtools ask "What are the main findings?" ./.parse/*.md --model gpt-4o-mini --workspace papers
```

### Stdin mode
If you pipe content and do not pass files, semtools ask answers directly from the piped content and does not use the file-tool loop.

```bash
cat summary.md | semtools ask "What changed?"
```

Use file mode for larger corpora and repeated investigative work.

## Recommended workflow

1. Parse binary docs first if needed.
2. Create or activate a workspace for repeated corpora.
3. Run `semtools ask` with explicit files plus `--workspace <name>`.
4. Use `--json` when another tool or script needs the answer programmatically.

## Configuration guidance

Defaults come from `~/.semtools_config.json`, but command-line flags override config values.

Typical areas to verify:

- model selection
- API base URL
- API key availability
- API mode (`responses` is the modern default in source)

## Common pitfalls

- No files and no stdin means the command errors.
- Without files, you lose the internal retrieval-tool workflow.
- Workspace support only helps when an active workspace resolves and files are provided.
