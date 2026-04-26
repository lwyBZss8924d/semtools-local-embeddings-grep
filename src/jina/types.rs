use anyhow::{Result, bail};
use clap::ValueEnum;
use serde::Serialize;
use std::fmt;
use std::path::PathBuf;

pub const DEFAULT_TEXT_MODEL: &str = "jina-embeddings-v5-small";
pub const DEFAULT_CODE_MODEL: &str = "jina-code-embeddings-1.5b";

const CODE_TASKS: &[JinaTask] = &[
    JinaTask::Nl2Code,
    JinaTask::Qa,
    JinaTask::Code2Code,
    JinaTask::Code2Nl,
    JinaTask::Code2Completion,
];

const TEXT_TASKS: &[JinaTask] = &[
    JinaTask::Retrieval,
    JinaTask::TextMatching,
    JinaTask::Clustering,
    JinaTask::Classification,
];

#[derive(Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum JinaTask {
    Retrieval,
    #[value(name = "text-matching")]
    TextMatching,
    Clustering,
    Classification,
    #[value(name = "nl2code")]
    Nl2Code,
    Qa,
    #[value(name = "code2code")]
    Code2Code,
    #[value(name = "code2nl")]
    Code2Nl,
    #[value(name = "code2completion")]
    Code2Completion,
}

impl JinaTask {
    pub fn as_wire(&self) -> &'static str {
        match self {
            Self::Retrieval => "retrieval",
            Self::TextMatching => "text-matching",
            Self::Clustering => "clustering",
            Self::Classification => "classification",
            Self::Nl2Code => "nl2code",
            Self::Qa => "qa",
            Self::Code2Code => "code2code",
            Self::Code2Nl => "code2nl",
            Self::Code2Completion => "code2completion",
        }
    }

    pub fn from_wire(value: &str) -> Option<Self> {
        match value {
            "retrieval" => Some(Self::Retrieval),
            "text-matching" => Some(Self::TextMatching),
            "clustering" => Some(Self::Clustering),
            "classification" => Some(Self::Classification),
            "nl2code" => Some(Self::Nl2Code),
            "qa" => Some(Self::Qa),
            "code2code" => Some(Self::Code2Code),
            "code2nl" => Some(Self::Code2Nl),
            "code2completion" => Some(Self::Code2Completion),
            _ => None,
        }
    }

    pub fn uses_prompt_pair(&self) -> bool {
        matches!(
            self,
            Self::Retrieval
                | Self::Nl2Code
                | Self::Qa
                | Self::Code2Code
                | Self::Code2Nl
                | Self::Code2Completion
        )
    }
}

impl fmt::Display for JinaTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_wire())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PromptName {
    Query,
    Document,
    Passage,
}

impl PromptName {
    pub fn as_wire(&self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Document => "document",
            Self::Passage => "passage",
        }
    }

    pub fn from_wire(value: &str) -> Option<Self> {
        match value {
            "query" => Some(Self::Query),
            "document" => Some(Self::Document),
            "passage" => Some(Self::Passage),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum Granularity {
    Line,
    Paragraph,
    Sentence,
    Token,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ColorWhen {
    Never,
    Always,
    Auto,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JinaModelKind {
    TextV5,
    Code,
}

#[derive(Clone, Debug)]
pub struct JinaModelInfo {
    pub model: String,
    pub repo_id: String,
    pub kind: JinaModelKind,
    pub dimension: usize,
    pub max_seq_len: usize,
    pub supported_tasks: &'static [JinaTask],
    pub matryoshka_dims: &'static [usize],
}

impl JinaModelInfo {
    pub fn for_model(model: &str) -> Option<Self> {
        match model {
            "jina-embeddings-v5-small" => Some(Self {
                model: model.to_string(),
                repo_id: "jinaai/jina-embeddings-v5-text-small-mlx".to_string(),
                kind: JinaModelKind::TextV5,
                dimension: 1024,
                max_seq_len: 32768,
                supported_tasks: TEXT_TASKS,
                matryoshka_dims: &[32, 64, 128, 256, 512, 768, 1024],
            }),
            "jina-code-embeddings-1.5b" => Some(Self {
                model: model.to_string(),
                repo_id: "jinaai/jina-code-embeddings-1.5b-mlx".to_string(),
                kind: JinaModelKind::Code,
                dimension: 1536,
                max_seq_len: 32768,
                supported_tasks: CODE_TASKS,
                matryoshka_dims: &[128, 256, 512, 1024, 1536],
            }),
            "jina-embeddings-v5-nano" => Some(Self {
                model: model.to_string(),
                repo_id: "jinaai/jina-embeddings-v5-text-nano-mlx".to_string(),
                kind: JinaModelKind::TextV5,
                dimension: 768,
                max_seq_len: 8192,
                supported_tasks: TEXT_TASKS,
                matryoshka_dims: &[32, 64, 128, 256, 512, 768],
            }),
            "jina-code-embeddings-0.5b" => Some(Self {
                model: model.to_string(),
                repo_id: "jinaai/jina-code-embeddings-0.5b-mlx".to_string(),
                kind: JinaModelKind::Code,
                dimension: 896,
                max_seq_len: 32768,
                supported_tasks: CODE_TASKS,
                matryoshka_dims: &[64, 128, 256, 512, 896],
            }),
            _ => None,
        }
    }

    pub fn is_code_model(&self) -> bool {
        self.kind == JinaModelKind::Code
    }

    pub fn default_task(&self) -> JinaTask {
        if self.is_code_model() {
            JinaTask::Nl2Code
        } else {
            JinaTask::Retrieval
        }
    }

    pub fn validate_task(&self, task: &JinaTask) -> Result<()> {
        if self.supported_tasks.contains(task) {
            Ok(())
        } else {
            bail!(
                "model '{}' does not support task '{}'; supported tasks: {}",
                self.model,
                task,
                self.supported_tasks
                    .iter()
                    .map(JinaTask::as_wire)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }

    pub fn validate_truncate_dim(&self, truncate_dim: Option<usize>) -> Result<()> {
        if let Some(dim) = truncate_dim
            && !self.matryoshka_dims.contains(&dim)
        {
            bail!(
                "model '{}' does not support truncate_dim {}; supported dimensions: {}",
                self.model,
                dim,
                self.matryoshka_dims
                    .iter()
                    .map(|dim| dim.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct JinaSearchOptions {
    pub recursive: bool,
    pub files_with_matches: bool,
    pub files_without_match: bool,
    pub count: bool,
    pub line_number: bool,
    pub with_filename: bool,
    pub after_context: usize,
    pub before_context: usize,
    pub include_patterns: Vec<String>,
    pub exclude_patterns: Vec<String>,
    pub exclude_dir_patterns: Vec<String>,
    pub color: bool,
    pub invert_match: bool,
    pub max_count: Option<usize>,
    pub quiet: bool,
    pub threshold: f32,
    pub top_k: usize,
    pub model: String,
    pub task: JinaTask,
    pub truncate_dim: Option<usize>,
    pub batch_size: usize,
    pub granularity: Granularity,
}

#[derive(Clone, Debug)]
pub struct Chunk {
    pub line_number: usize,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct JinaSearchResult {
    pub path: Option<String>,
    pub line_number: Option<usize>,
    pub text: String,
    pub score: f32,
    pub context_before: Vec<String>,
    pub context_after: Vec<String>,
    pub original_line: Option<String>,
    pub label: Option<String>,
    pub label_scores: Vec<LabelScore>,
}

impl JinaSearchResult {
    pub fn distance(&self) -> f32 {
        1.0 - self.score
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct LabelScore {
    pub label: String,
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct ModelAssets {
    pub model: JinaModelInfo,
    pub root: PathBuf,
    pub config_json: PathBuf,
    pub tokenizer_json: PathBuf,
    pub model_safetensors: PathBuf,
}

pub trait EmbeddingBackend {
    fn embed(
        &self,
        texts: &[String],
        model: &str,
        task: &JinaTask,
        prompt_name: Option<PromptName>,
        truncate_dim: Option<usize>,
    ) -> Result<Vec<Vec<f32>>>;
}
