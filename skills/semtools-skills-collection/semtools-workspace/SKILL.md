---
name: semtools-workspace
description: Manage semtools workspaces for repeated document search and question-answering. Use when the user needs to create, activate, inspect, or prune a workspace, or when semtools search and ask should reuse persisted embeddings across repeated runs.
---

# Semtools Workspace

Use `semtools workspace` to manage persistent semtools embedding stores.

## Core commands

```bash
semtools workspace use <NAME>
semtools workspace status [NAME]
semtools workspace prune [NAME]
```

All workspace subcommands support `-j, --json`.

## Activation model

`semtools workspace use <NAME>` configures the workspace but does not activate it by itself.

Activate a workspace with either:

```bash
export SEMTOOLS_WORKSPACE=<NAME>
```

or per-command:

```bash
semtools search "query" docs/*.md --workspace <NAME>
semtools ask "question" docs/*.md --workspace <NAME>
```

## What workspaces actually do

- Workspaces live under `~/.semtools/workspaces/<name>`.
- They persist document metadata and embeddings for faster repeated lookup.
- Indexing is lazy: search still needs explicit files, then semtools updates only new or changed files in that file set.

## Maintenance patterns

Create a workspace:

```bash
semtools workspace use papers
```

Inspect status:

```bash
semtools workspace --json status papers
```

Prune stale files after deletions or large corpus changes:

```bash
semtools workspace --json prune papers
```

## Common pitfalls

- Creating a workspace is not the same as activating it.
- Search and ask do not automatically search the whole workspace; they still require an explicit file list.
- If no active workspace resolves, semtools falls back to normal non-workspace behavior instead of using persisted embeddings.
