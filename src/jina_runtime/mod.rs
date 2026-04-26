use crate::jina::types::{
    DEFAULT_CODE_MODEL, DEFAULT_TEXT_MODEL, EmbeddingBackend, JinaModelInfo, JinaTask, ModelAssets,
    PromptName,
};
use anyhow::{Result, anyhow, bail};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[cfg(feature = "jina-native-runtime")]
mod candle_embedder;

#[cfg(feature = "jina-native-runtime")]
pub mod daemon;

const HF_HUB_MODELS_PREFIX: &str = ".cache/huggingface/hub";

#[derive(Clone, Debug)]
pub struct NativeJinaEmbedder {
    model_dir: Option<PathBuf>,
    batch_size: usize,
}

#[derive(Debug, Serialize)]
pub struct ModelStatus {
    pub model: String,
    pub repo_id: String,
    pub dimension: usize,
    pub available: bool,
    pub path: Option<String>,
    pub missing_files: Vec<String>,
}

impl NativeJinaEmbedder {
    pub fn new(model_dir: Option<PathBuf>) -> Self {
        Self::new_with_batch_size(model_dir, 256)
    }

    pub fn new_with_batch_size(model_dir: Option<PathBuf>, batch_size: usize) -> Self {
        Self {
            model_dir,
            batch_size: batch_size.max(1),
        }
    }

    pub fn resolve_assets(&self, model: &str) -> Result<ModelAssets> {
        let info = JinaModelInfo::for_model(model).ok_or_else(|| {
            anyhow!(
                "unsupported Jina model '{}'; supported models: {}, {}, jina-embeddings-v5-nano, jina-code-embeddings-0.5b",
                model,
                DEFAULT_TEXT_MODEL,
                DEFAULT_CODE_MODEL
            )
        })?;

        let root = if let Some(root) = &self.model_dir {
            root.clone()
        } else {
            find_huggingface_snapshot(&info.repo_id).ok_or_else(|| missing_model_error(&info))?
        };

        let assets = ModelAssets {
            model: info,
            config_json: root.join("config.json"),
            tokenizer_json: root.join("tokenizer.json"),
            model_safetensors: root.join("model.safetensors"),
            root,
        };

        let missing = missing_asset_files(&assets);
        if !missing.is_empty() {
            bail!(
                "model assets for '{}' are incomplete at {}. Missing: {}",
                assets.model.model,
                assets.root.display(),
                missing.join(", ")
            );
        }

        Ok(assets)
    }

    pub fn model_status(&self, model: &str) -> Result<ModelStatus> {
        let info =
            JinaModelInfo::for_model(model).ok_or_else(|| anyhow!("unsupported model: {model}"))?;
        let root = self
            .model_dir
            .clone()
            .or_else(|| find_huggingface_snapshot(&info.repo_id));

        let missing_files = root
            .as_ref()
            .map(|root| {
                let assets = ModelAssets {
                    model: info.clone(),
                    config_json: root.join("config.json"),
                    tokenizer_json: root.join("tokenizer.json"),
                    model_safetensors: root.join("model.safetensors"),
                    root: root.clone(),
                };
                missing_asset_files(&assets)
            })
            .unwrap_or_else(|| {
                vec![
                    "config.json".to_string(),
                    "tokenizer.json".to_string(),
                    "model.safetensors".to_string(),
                ]
            });

        Ok(ModelStatus {
            model: info.model,
            repo_id: info.repo_id,
            dimension: info.dimension,
            available: root.is_some() && missing_files.is_empty(),
            path: root.map(|path| path.display().to_string()),
            missing_files,
        })
    }
}

impl EmbeddingBackend for NativeJinaEmbedder {
    fn embed(
        &self,
        texts: &[String],
        model: &str,
        task: &JinaTask,
        prompt_name: Option<PromptName>,
        truncate_dim: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let assets = self.resolve_assets(model)?;
        assets.model.validate_task(task)?;
        assets.model.validate_truncate_dim(truncate_dim)?;

        #[cfg(feature = "jina-native-runtime")]
        {
            candle_embedder::embed(
                &assets,
                texts,
                task,
                prompt_name,
                truncate_dim,
                self.batch_size,
            )
        }

        #[cfg(not(feature = "jina-native-runtime"))]
        {
            let prompt_name = prompt_name.map(|p| p.as_wire()).unwrap_or("none");
            bail!(
                "Rust-native Jina inference is not implemented yet for model '{}' task '{}' prompt '{}' using assets at {}. Product code intentionally does not fall back to Python, MLX, or a local embedding server.",
                assets.model.model,
                task,
                prompt_name,
                assets.root.display()
            )
        }
    }
}

fn missing_model_error(info: &JinaModelInfo) -> anyhow::Error {
    anyhow!(
        "model '{}' is not available in the local Hugging Face cache. Download it explicitly before searching: huggingface-cli download {} --local-dir <model-dir>, then run semtools jgrep --model-dir <model-dir> ...",
        info.model,
        info.repo_id
    )
}

fn missing_asset_files(assets: &ModelAssets) -> Vec<String> {
    let mut missing = Vec::new();
    for (name, path) in [
        ("config.json", &assets.config_json),
        ("tokenizer.json", &assets.tokenizer_json),
        ("model.safetensors", &assets.model_safetensors),
    ] {
        if !path.exists() {
            missing.push(name.to_string());
        }
    }
    missing
}

fn find_huggingface_snapshot(repo_id: &str) -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let cache_root = home.join(HF_HUB_MODELS_PREFIX);
    let repo_cache_name = format!("models--{}", repo_id.replace('/', "--"));
    let snapshots = cache_root.join(repo_cache_name).join("snapshots");
    newest_snapshot(&snapshots)
}

fn newest_snapshot(snapshots: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(snapshots).ok()?;
    let mut candidates = Vec::new();

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let modified = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .ok();
        candidates.push((modified, path));
    }

    candidates.sort_by(|a, b| a.0.cmp(&b.0));
    candidates.pop().map(|(_, path)| path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn status_reports_missing_model() {
        let embedder = NativeJinaEmbedder::new(Some(PathBuf::from("/definitely/missing")));
        let status = embedder
            .model_status("jina-embeddings-v5-small")
            .expect("status");
        assert!(!status.available);
        assert!(status.missing_files.contains(&"config.json".to_string()));
    }

    #[test]
    fn resolves_complete_asset_directory() {
        let temp = TempDir::new().expect("tempdir");
        for name in ["config.json", "tokenizer.json", "model.safetensors"] {
            std::fs::write(temp.path().join(name), "{}").expect("write");
        }

        let embedder = NativeJinaEmbedder::new(Some(temp.path().to_path_buf()));
        let assets = embedder
            .resolve_assets("jina-code-embeddings-1.5b")
            .expect("assets");
        assert_eq!(assets.model.dimension, 1536);
    }
}
