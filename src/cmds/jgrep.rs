use crate::jina::format::{print_human_results, print_json_results};
use crate::jina::search::{JinaSearchRun, pipe_rerank, semantic_classify, semantic_grep};
use crate::jina::types::{
    ColorWhen, DEFAULT_TEXT_MODEL, Granularity, JinaModelInfo, JinaSearchOptions, JinaTask,
};
#[cfg(feature = "workspace")]
use crate::jina::workspace::{search_workspace, sync_workspace};
use crate::jina_runtime::NativeJinaEmbedder;
#[cfg(feature = "jina-native-runtime")]
use crate::jina_runtime::daemon::{self, DaemonJinaClient};
use anyhow::{Result, anyhow};
use clap::Args;
use std::io::{self, BufRead, IsTerminal};
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct JgrepArgs {
    /// Recursive directory search
    #[arg(short = 'r', short_alias = 'R', long)]
    pub recursive: bool,

    /// Print only filenames with matches
    #[arg(short = 'l', long)]
    pub files_with_matches: bool,

    /// Print only filenames without matches
    #[arg(short = 'L', long)]
    pub files_without_match: bool,

    /// Print match count per file
    #[arg(short = 'c', long)]
    pub count: bool,

    /// Print line numbers
    #[arg(short = 'n', long, default_value_t = true)]
    pub line_number: bool,

    /// Print filename with matches
    #[arg(short = 'H', long, default_value_t = false)]
    pub with_filename: bool,

    /// Suppress filename
    #[arg(long, default_value_t = false)]
    pub no_filename: bool,

    /// Lines after match
    #[arg(short = 'A', long, default_value_t = 0)]
    pub after_context: usize,

    /// Lines before match
    #[arg(short = 'B', long, default_value_t = 0)]
    pub before_context: usize,

    /// Lines before and after match
    #[arg(short = 'C', long, default_value_t = 0)]
    pub context: usize,

    /// Search only files matching GLOB
    #[arg(long = "include")]
    pub include_patterns: Vec<String>,

    /// Skip files matching GLOB
    #[arg(long = "exclude")]
    pub exclude_patterns: Vec<String>,

    /// Skip directories matching GLOB
    #[arg(long = "exclude-dir")]
    pub exclude_dir_patterns: Vec<String>,

    /// Color mode
    #[arg(long, value_enum, default_value_t = ColorWhen::Auto)]
    pub color: ColorWhen,

    /// Invert match: keep lowest similarity scores
    #[arg(short = 'v', long)]
    pub invert_match: bool,

    /// Max matches per file
    #[arg(short = 'm', long)]
    pub max_count: Option<usize>,

    /// Quiet mode
    #[arg(short = 'q', long)]
    pub quiet: bool,

    /// Print only the matching label in classification mode
    #[arg(short = 'o', long)]
    pub only_matching: bool,

    /// Classification label. Multiple -e flags enable classification.
    #[arg(short = 'e', long = "regexp")]
    pub labels: Vec<String>,

    /// Read classification labels from file, one per line
    #[arg(short = 'f', long = "file")]
    pub label_file: Option<String>,

    /// Force classification mode
    #[arg(long)]
    pub classify: bool,

    /// Similarity threshold
    #[arg(long)]
    pub threshold: Option<f32>,

    /// Max results
    #[arg(long)]
    pub top_k: Option<usize>,

    /// Jina model name
    #[arg(long, default_value = DEFAULT_TEXT_MODEL)]
    pub model: String,

    /// Embedding task
    #[arg(long, value_enum)]
    pub task: Option<JinaTask>,

    /// Matryoshka output dimension
    #[arg(long)]
    pub truncate_dim: Option<usize>,

    /// Use a lower Matryoshka dimension for faster scoring/indexing
    #[arg(long)]
    pub fast: bool,

    /// Number of same-length texts to embed together
    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,

    /// Local model directory containing config.json/tokenizer.json/model.safetensors
    #[arg(long)]
    pub model_dir: Option<String>,

    /// Chunk granularity
    #[arg(long, value_enum, default_value_t = Granularity::Token)]
    pub granularity: Granularity,

    /// Output JSON
    #[arg(long, short = 'j')]
    pub json: bool,

    /// Print local status for the selected Jina model and exit
    #[arg(long = "models-status")]
    pub models_status: bool,

    /// Use a semtools workspace for Jina profile search or sync
    #[arg(short, long)]
    pub workspace: Option<String>,

    /// Jina workspace profile id
    #[arg(long)]
    pub profile: Option<String>,

    /// Sync/index files into the selected Jina workspace profile
    #[arg(long)]
    pub sync: bool,

    /// Start the Rust-native jgrep daemon in the background
    #[arg(long)]
    pub daemon_start: bool,

    /// Run the Rust-native jgrep daemon in the foreground
    #[arg(long, hide = true)]
    pub daemon_serve_foreground: bool,

    /// Print Rust-native jgrep daemon status
    #[arg(long)]
    pub daemon_status: bool,

    /// Stop the Rust-native jgrep daemon
    #[arg(long)]
    pub daemon_stop: bool,

    /// Use the Rust-native jgrep daemon for embedding calls
    #[arg(long)]
    pub daemon: bool,

    /// Natural-language search query
    pub pattern: Option<String>,

    /// Files or directories to search
    pub files: Vec<String>,
}

pub async fn jgrep_cmd(args: JgrepArgs) -> Result<()> {
    #[cfg(feature = "jina-native-runtime")]
    {
        if args.daemon_serve_foreground {
            return daemon::serve_foreground();
        }
        if args.daemon_start {
            let status = daemon::start_background()?;
            print_daemon_status(status, args.json)?;
            return Ok(());
        }
        if args.daemon_status {
            print_daemon_status(daemon::status(), args.json)?;
            return Ok(());
        }
        if args.daemon_stop {
            print_daemon_status(daemon::stop()?, args.json)?;
            return Ok(());
        }
    }

    if args.models_status {
        print_model_status(&args)?;
        return Ok(());
    }

    let options = build_options(&args)?;
    let embedder = RuntimeBackend::new(
        args.daemon,
        args.model_dir.as_ref().map(PathBuf::from),
        options.batch_size,
    );

    #[cfg(feature = "workspace")]
    if args.sync {
        let paths = collect_paths(&args)?;
        let output = sync_workspace(
            args.workspace.as_deref(),
            args.profile.as_deref(),
            &paths,
            &embedder,
            &options,
        )?;
        if args.json {
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            println!(
                "Synced {} file(s), {} chunk(s) into workspace '{}' profile '{}'",
                output.files_indexed, output.chunks_indexed, output.workspace, output.profile
            );
            println!("Index: {}", output.index_path);
        }
        return Ok(());
    }

    let mode = if is_classification_mode(&args) {
        "classify"
    } else if args.workspace.is_some() {
        "workspace"
    } else if args.files.is_empty() && !io::stdin().is_terminal() {
        "pipe"
    } else {
        "grep"
    };

    let run = match mode {
        "classify" => {
            let labels = collect_labels(&args)?;
            let paths = collect_paths_for_classification(&args)?;
            semantic_classify(&labels, &paths, &embedder, &options)?
        }
        #[cfg(feature = "workspace")]
        "workspace" => {
            let pattern = required_pattern(&args)?;
            search_workspace(
                args.workspace.as_deref(),
                args.profile.as_deref(),
                &pattern,
                &embedder,
                &options,
            )?
        }
        "pipe" => {
            let pattern = required_pattern(&args)?;
            let stdin_lines = read_stdin_lines()?;
            pipe_rerank(&pattern, &stdin_lines, &embedder, &options)?
        }
        _ => {
            let pattern = required_pattern(&args)?;
            let paths = collect_paths(&args)?;
            semantic_grep(&pattern, &paths, &embedder, &options)?
        }
    };

    if args.json {
        print_json_results(&run, &options, mode)?;
    } else {
        print_human_results(&run, &options, args.only_matching);
    }

    exit_for_results(&run, &options);
    Ok(())
}

fn build_options(args: &JgrepArgs) -> Result<JinaSearchOptions> {
    let model_info = JinaModelInfo::for_model(&args.model)
        .ok_or_else(|| anyhow!("unsupported Jina model: {}", args.model))?;
    let task = args
        .task
        .clone()
        .unwrap_or_else(|| model_info.default_task());
    let truncate_dim = if args.fast && args.truncate_dim.is_none() {
        Some(if model_info.is_code_model() { 512 } else { 256 })
    } else {
        args.truncate_dim
    };
    model_info.validate_task(&task)?;
    model_info.validate_truncate_dim(truncate_dim)?;

    let threshold = match (args.threshold, args.top_k) {
        (Some(threshold), _) => threshold,
        (None, Some(_)) => 0.0,
        (None, None) => 0.5,
    };
    let top_k = args.top_k.unwrap_or(10);
    let context = args.context;
    let after_context = if context > 0 {
        context
    } else {
        args.after_context
    };
    let before_context = if context > 0 {
        context
    } else {
        args.before_context
    };

    Ok(JinaSearchOptions {
        recursive: args.recursive,
        files_with_matches: args.files_with_matches,
        files_without_match: args.files_without_match,
        count: args.count,
        line_number: args.line_number,
        with_filename: !args.no_filename
            && (args.with_filename || args.files.len() > 1 || args.recursive),
        after_context,
        before_context,
        include_patterns: args.include_patterns.clone(),
        exclude_patterns: args.exclude_patterns.clone(),
        exclude_dir_patterns: args.exclude_dir_patterns.clone(),
        color: match args.color {
            ColorWhen::Never => false,
            ColorWhen::Always => true,
            ColorWhen::Auto => io::stdout().is_terminal(),
        },
        invert_match: args.invert_match,
        max_count: args.max_count,
        quiet: args.quiet,
        threshold,
        top_k,
        model: args.model.clone(),
        task,
        truncate_dim,
        batch_size: args.batch_size.max(1),
        granularity: args.granularity,
    })
}

fn is_classification_mode(args: &JgrepArgs) -> bool {
    args.classify || !args.labels.is_empty() || args.label_file.is_some()
}

fn collect_labels(args: &JgrepArgs) -> Result<Vec<String>> {
    let mut labels = args.labels.clone();
    if let Some(label_file) = &args.label_file {
        let content = std::fs::read_to_string(label_file)?;
        labels.extend(
            content
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(ToString::to_string),
        );
    }
    if labels.is_empty() {
        return Err(anyhow!("classification mode requires labels via -e or -f"));
    }
    Ok(labels)
}

fn collect_paths(args: &JgrepArgs) -> Result<Vec<PathBuf>> {
    if args.files.is_empty() {
        return Ok(vec![PathBuf::from(".")]);
    }
    Ok(args.files.iter().map(PathBuf::from).collect())
}

fn collect_paths_for_classification(args: &JgrepArgs) -> Result<Vec<PathBuf>> {
    let mut files = args.files.clone();
    if files.is_empty()
        && let Some(pattern) = &args.pattern
    {
        files.push(pattern.clone());
    }
    if files.is_empty() {
        files.push(".".to_string());
    }
    Ok(files.into_iter().map(PathBuf::from).collect())
}

fn required_pattern(args: &JgrepArgs) -> Result<String> {
    args.pattern
        .clone()
        .ok_or_else(|| anyhow!("PATTERN is required unless classification labels are provided"))
}

fn read_stdin_lines() -> Result<Vec<String>> {
    let stdin = io::stdin();
    stdin
        .lock()
        .lines()
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn exit_for_results(run: &JinaSearchRun, options: &JinaSearchOptions) {
    let has_result = if options.files_without_match {
        true
    } else {
        !run.results.is_empty()
    };

    if !has_result {
        std::process::exit(1);
    }
}

fn print_model_status(args: &JgrepArgs) -> Result<()> {
    let embedder = NativeJinaEmbedder::new(args.model_dir.as_ref().map(PathBuf::from));
    let status = embedder.model_status(&args.model)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&status)?);
    } else {
        println!("Model: {}", status.model);
        println!("Repo: {}", status.repo_id);
        println!("Dimension: {}", status.dimension);
        println!("Available: {}", status.available);
        if let Some(path) = status.path {
            println!("Path: {path}");
        }
        if !status.missing_files.is_empty() {
            println!("Missing: {}", status.missing_files.join(", "));
        }
    }
    Ok(())
}

enum RuntimeBackend {
    Local(NativeJinaEmbedder),
    #[cfg(feature = "jina-native-runtime")]
    Daemon(DaemonJinaClient),
}

impl RuntimeBackend {
    fn new(use_daemon: bool, model_dir: Option<PathBuf>, batch_size: usize) -> Self {
        #[cfg(feature = "jina-native-runtime")]
        {
            if use_daemon {
                return Self::Daemon(DaemonJinaClient::new(model_dir, batch_size));
            }
        }
        Self::Local(NativeJinaEmbedder::new_with_batch_size(
            model_dir, batch_size,
        ))
    }
}

impl crate::jina::types::EmbeddingBackend for RuntimeBackend {
    fn embed(
        &self,
        texts: &[String],
        model: &str,
        task: &JinaTask,
        prompt_name: Option<crate::jina::types::PromptName>,
        truncate_dim: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Local(embedder) => embedder.embed(texts, model, task, prompt_name, truncate_dim),
            #[cfg(feature = "jina-native-runtime")]
            Self::Daemon(client) => client.embed(texts, model, task, prompt_name, truncate_dim),
        }
    }
}

#[cfg(feature = "jina-native-runtime")]
fn print_daemon_status(status: daemon::DaemonStatus, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(&status)?);
    } else {
        println!("Daemon running: {}", status.running);
        println!("Socket: {}", status.socket_path);
        println!("Pid file: {}", status.pid_path);
        if let Some(pid) = status.pid {
            println!("Pid: {pid}");
        }
    }
    Ok(())
}
