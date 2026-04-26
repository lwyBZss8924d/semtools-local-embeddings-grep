# Fork-Local Jina Semantic Grep

This fork adds `semtools jgrep`, a Rust-native Jina embedding search surface for local code and text corpora. It is the local `jina-grep` / `jina-semsearch` style capability exposed as a semtools subcommand.

The upstream `semtools search` command remains the lightweight model2vec search path. Use `semtools jgrep` when the task needs Jina embeddings, semantic code search, classification labels, stdin reranking, workspace-backed Jina profiles, JSON output for automation, or a daemon-backed local embedding path.

## Installation And Model Status

Install this fork from the local source tree:

```bash
cargo install --path . --force
```

Check that the active command is the local fork and that the default Jina model is available:

```bash
which semtools
semtools jgrep --help
semtools jgrep --models-status --json
```

`--models-status --json` reports the selected model, Hugging Face repo id, embedding dimension, resolved local path, availability, and any missing asset files. A healthy default text model status has `available: true` and an empty `missing_files` list.

If assets are missing, download the selected model into the Hugging Face cache or pass an explicit model directory:

```bash
huggingface-cli download jinaai/jina-embeddings-v5-text-small-mlx
semtools jgrep --model-dir /path/to/model-dir --models-status --json
```

The model directory must contain `config.json`, `tokenizer.json`, and `model.safetensors`.

## Models, Tasks, And Features

Supported model names:

- `jina-embeddings-v5-small`
- `jina-embeddings-v5-nano`
- `jina-code-embeddings-1.5b`
- `jina-code-embeddings-0.5b`

Text models default to the `retrieval` task. Code models default to `nl2code`. The code model task set includes `nl2code`, `qa`, `code2code`, `code2nl`, and `code2completion`. Text model tasks include `retrieval`, `text-matching`, `clustering`, and `classification`.

Use `--truncate-dim` for a specific Matryoshka dimension, or `--fast` to choose a lower default dimension for faster local scoring and indexing. The default feature set enables `jina-native-runtime`, which uses Candle and local safetensors assets rather than Python, MLX, or an external embedding server.

## Modes

`semtools jgrep` selects its mode from flags and inputs:

- Grep mode: `PATTERN` plus files or directories.
- Pipe mode: no files and piped stdin; candidates are reranked against `PATTERN`.
- Classification mode: `--classify`, repeated `-e`, or `-f`.
- Workspace sync mode: `--workspace ... --sync`.
- Workspace search mode: `--workspace ...` without `--sync`.

### Semantic Grep

```bash
semtools jgrep "retry backoff timeout" src --recursive --include '*.rs' --top-k 8 --json
```

Useful file-selection and grep-style flags:

- `-r, --recursive`
- `--include <GLOB>`
- `--exclude <GLOB>`
- `--exclude-dir <GLOB>`
- `-A`, `-B`, `-C` for context
- `-l`, `-L`, `-c`, `-q`, `-v`, and `-m`

### Stdin Reranking

```bash
printf '%s\n' "token refresh logic" "database migration runner" | semtools jgrep "OAuth token refresh" --top-k 1 --json
```

Pipe mode parses grep-style `path:line:text` candidates when present, strips ANSI color, embeds candidate text, and returns ranked matches. JSON output reports `mode: "pipe"`.

### Classification

```bash
semtools jgrep --classify -e bug -e feature -e docs ./issues/*.txt --json
semtools jgrep --classify -f labels.txt ./issues/*.txt --json
```

Classification embeds each label and candidate chunk with the classification task, returns the best label, and includes all label scores in JSON output.

### Workspace Profiles

Jina workspace profiles are stored under the selected semtools workspace root in `jina_profiles/`. A profile records the model, task, dimension, truncate dimension, and preprocessing version used for indexing.

```bash
semtools workspace use semtools
semtools jgrep --workspace semtools --profile code --sync README.md src --recursive --include '*.rs' --json
semtools jgrep --workspace semtools --profile code "Jina workspace profile search" --top-k 8 --json
```

Use explicit profiles when one workspace needs multiple Jina indexes, such as code and prose. If `--profile` is omitted, semtools derives a profile id from model, task, dimension, and preprocessing version.

### Daemon Path

The optional daemon keeps embedding calls behind a local Unix socket:

```bash
semtools jgrep --daemon-start
semtools jgrep --daemon-status --json
semtools jgrep --daemon "retry backoff timeout" src --recursive --include '*.rs' --top-k 8 --json
semtools jgrep --daemon-stop
```

Runtime files live under `~/.semtools/runtime/`. Use daemon mode for repeated searches where process startup and model setup cost matter.

## JSON Output

`--json` returns:

```json
{
  "type": "jina_grep_results",
  "backend": "rust_native_jina",
  "mode": "grep",
  "model": "jina-embeddings-v5-small",
  "task": "retrieval",
  "results": []
}
```

Each result contains:

- `path`
- `line_number`
- `score`
- `distance`
- `text`
- `original_line`
- `context_before`
- `context_after`
- `label`
- `label_scores`

Automation should use `type`, `backend`, and `mode` to distinguish jgrep output from classic `semtools search` JSON.

## Verification Commands

Run these checks before relying on a changed local fork:

```bash
cargo fmt --check
cargo test --all-features
cargo test --no-default-features --features parse,search,workspace,ask
cargo run --quiet -- jgrep --models-status --json
printf '%s\n' "token refresh logic" "database migration runner" | cargo run --quiet -- jgrep "OAuth token refresh" --top-k 1 --json
```

The no-default-features test ensures upstream-style parse/search/workspace/ask builds still work without the Jina feature set.

## AIHT Skills Source

The repository copy at `skills/semtools-skills-collection/` is the maintenance source for the custom semtools AIHT skills. The global runtime copy under `~/.agents/skills/semtools-skills-collection/` can be regenerated or synchronized from the repository source when this fork changes.
