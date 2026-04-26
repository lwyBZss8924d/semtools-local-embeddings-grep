use crate::jina::chunk::{get_files, read_file_safely, split_into_chunks};
use crate::jina::search::{JinaSearchRun, cosine_similarity};
use crate::jina::types::{EmbeddingBackend, JinaSearchOptions, JinaSearchResult, PromptName};
use crate::workspace::Workspace;
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const PREPROCESSING_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JinaProfile {
    pub id: String,
    pub backend: String,
    pub model: String,
    pub task: String,
    pub dimension: usize,
    pub truncate_dim: Option<usize>,
    pub preprocessing_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JinaIndexRecord {
    pub path: String,
    pub line_number: usize,
    pub text: String,
    pub embedding: Vec<f32>,
    pub size_bytes: u64,
    pub mtime: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JinaWorkspaceIndex {
    pub profile: JinaProfile,
    pub records: Vec<JinaIndexRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JinaSyncOutput {
    pub workspace: String,
    pub profile: String,
    pub files_indexed: usize,
    pub chunks_indexed: usize,
    pub index_path: String,
}

pub fn default_profile_id(options: &JinaSearchOptions) -> String {
    let dimension = options.truncate_dim.unwrap_or_else(|| {
        crate::jina::types::JinaModelInfo::for_model(&options.model)
            .map(|info| info.dimension)
            .unwrap_or(0)
    });
    sanitize_profile_id(&format!(
        "jina-local__{}__{}__{}__v{}",
        options.model, options.task, dimension, PREPROCESSING_VERSION
    ))
}

pub fn sync_workspace<B: EmbeddingBackend>(
    workspace_name: Option<&str>,
    profile_id: Option<&str>,
    paths: &[PathBuf],
    backend: &B,
    options: &JinaSearchOptions,
) -> Result<JinaSyncOutput> {
    let workspace = Workspace::open(workspace_name)?;
    let profile_id = profile_id
        .map(ToString::to_string)
        .unwrap_or_else(|| default_profile_id(options));
    let index_path = profile_index_path(&workspace.config.root_dir, &profile_id)?;
    let profile = profile_for(&profile_id, options);
    let mut index = load_index(&index_path)?.unwrap_or(JinaWorkspaceIndex {
        profile,
        records: Vec::new(),
    });

    let files = get_files(paths, options)?;
    let file_set: HashSet<String> = files
        .iter()
        .map(|path| path.display().to_string())
        .collect();
    index
        .records
        .retain(|record| !file_set.contains(&record.path) && Path::new(&record.path).exists());

    let mut files_indexed = 0usize;
    let mut chunks_indexed = 0usize;
    for file in &files {
        let Some(content) = read_file_safely(file) else {
            continue;
        };
        let metadata = std::fs::metadata(file)?;
        let size_bytes = metadata.len();
        let mtime = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_secs() as i64)
            .unwrap_or(0);
        let chunks = split_into_chunks(&content, options.granularity);
        if chunks.is_empty() {
            continue;
        }

        let texts: Vec<String> = chunks.iter().map(|chunk| chunk.text.clone()).collect();
        let embeddings = backend.embed(
            &texts,
            &options.model,
            &options.task,
            options
                .task
                .uses_prompt_pair()
                .then_some(PromptName::Document),
            options.truncate_dim,
        )?;
        for (chunk, embedding) in chunks.into_iter().zip(embeddings.into_iter()) {
            index.records.push(JinaIndexRecord {
                path: file.display().to_string(),
                line_number: chunk.line_number,
                text: chunk.text,
                embedding,
                size_bytes,
                mtime,
            });
            chunks_indexed += 1;
        }
        files_indexed += 1;
    }

    save_index(&index_path, &index)?;
    Ok(JinaSyncOutput {
        workspace: workspace.config.name,
        profile: profile_id,
        files_indexed,
        chunks_indexed,
        index_path: index_path.display().to_string(),
    })
}

pub fn search_workspace<B: EmbeddingBackend>(
    workspace_name: Option<&str>,
    profile_id: Option<&str>,
    query: &str,
    backend: &B,
    options: &JinaSearchOptions,
) -> Result<JinaSearchRun> {
    let workspace = Workspace::open(workspace_name)?;
    let profile_id = profile_id
        .map(ToString::to_string)
        .unwrap_or_else(|| default_profile_id(options));
    let index_path = profile_index_path(&workspace.config.root_dir, &profile_id)?;
    let index = load_index(&index_path)?
        .ok_or_else(|| anyhow!("Jina workspace profile '{profile_id}' is not indexed yet"))?;

    if index.profile.model != options.model || index.profile.task != options.task.to_string() {
        return Err(anyhow!(
            "profile '{}' was built with model '{}' task '{}', but query requested model '{}' task '{}'",
            profile_id,
            index.profile.model,
            index.profile.task,
            options.model,
            options.task
        ));
    }

    let query_embedding = backend
        .embed(
            &[query.to_string()],
            &options.model,
            &options.task,
            options.task.uses_prompt_pair().then_some(PromptName::Query),
            options.truncate_dim,
        )?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("embedding backend returned no query embedding"))?;

    let mut results = Vec::new();
    for record in &index.records {
        if !Path::new(&record.path).exists() {
            continue;
        }
        let score = cosine_similarity(&query_embedding, &record.embedding)?;
        let passes = if options.invert_match {
            score < options.threshold
        } else {
            score >= options.threshold
        };
        if !passes {
            continue;
        }
        results.push(JinaSearchResult {
            path: Some(record.path.clone()),
            line_number: Some(record.line_number),
            text: record.text.clone(),
            score,
            context_before: Vec::new(),
            context_after: Vec::new(),
            original_line: None,
            label: None,
            label_scores: Vec::new(),
        });
    }

    if options.invert_match {
        results.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    results.truncate(options.top_k);

    Ok(JinaSearchRun {
        files: Vec::new(),
        results,
    })
}

fn profile_for(profile_id: &str, options: &JinaSearchOptions) -> JinaProfile {
    let dimension = options.truncate_dim.unwrap_or_else(|| {
        crate::jina::types::JinaModelInfo::for_model(&options.model)
            .map(|info| info.dimension)
            .unwrap_or(0)
    });
    JinaProfile {
        id: profile_id.to_string(),
        backend: "rust-native-candle".to_string(),
        model: options.model.clone(),
        task: options.task.to_string(),
        dimension,
        truncate_dim: options.truncate_dim,
        preprocessing_version: PREPROCESSING_VERSION,
    }
}

fn profile_index_path(workspace_root: &str, profile_id: &str) -> Result<PathBuf> {
    let path = Path::new(workspace_root)
        .join("jina_profiles")
        .join(format!("{profile_id}.json"));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(path)
}

fn load_index(path: &Path) -> Result<Option<JinaWorkspaceIndex>> {
    if !path.exists() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(Some(serde_json::from_str(&content)?))
}

fn save_index(path: &Path, index: &JinaWorkspaceIndex) -> Result<()> {
    let content = serde_json::to_string_pretty(index)?;
    std::fs::write(path, content)?;
    Ok(())
}

fn sanitize_profile_id(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}
