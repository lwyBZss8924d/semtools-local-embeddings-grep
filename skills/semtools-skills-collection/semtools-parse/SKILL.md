---
name: semtools-parse
description: Parse PDFs and other non-text documents into cached markdown with the semtools CLI. Use when the user needs to ingest PDFs, DOCX-style binary documents, or mixed document sets before semantic search or question-answering, especially when building a local corpus for later semtools search or ask workflows.
---

# Semtools Parse

Use `semtools parse` to convert non-text documents into markdown paths that downstream shell commands can consume directly.

## Core command

```bash
semtools parse [OPTIONS] <FILES>...
```

Useful options:

- `-c, --config <CONFIG>` — config file path, default `~/.semtools_config.json`
- `-b, --backend <BACKEND>` — parser backend, currently `llama-parse`
- `-v, --verbose` — show parsing progress

## Workflow

1. Use `parse` for binary/non-text formats before trying semantic retrieval.
2. Feed the emitted markdown paths into `semtools search` or `semtools ask`.
3. Keep the original files if you need provenance; parse writes cached markdown under `~/.parse`.

## Good patterns

Parse a batch of PDFs:

```bash
semtools parse ./papers/*.pdf
```

Parse then search the parsed files by path:

```bash
semtools parse ./papers/*.pdf | xargs semtools search "authentication, OAuth, token refresh" --top-k 10 --n-lines 5
```

Parse then ask questions over the parsed files:

```bash
semtools parse ./papers/*.pdf | xargs semtools ask "What are the main findings?"
```

## Important caveats

- Text/code files are skipped and returned as original paths rather than reparsed markdown.
- Missing files are skipped rather than hard-failing the entire batch.
- Cache output lives in `~/.parse`.
- Cache filenames are keyed by basename, so same-named files from different directories can collide. If that risk matters, separate parsing runs or rename files first.

## Choose parse vs direct search

- Use `parse` first for PDFs and other binary docs.
- Skip `parse` for normal text/code files and go directly to `semtools search` or `semtools ask`.
