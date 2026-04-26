use crate::jina::types::{JinaModelKind, JinaTask, ModelAssets, PromptName};
use anyhow::{Context, Result, anyhow, bail};
use candle::{DType, Device, IndexOp, Shape, Tensor};
use candle_nn::{Activation, VarBuilder, var_builder::SimpleBackend};
use candle_transformers::models::{qwen2, qwen3};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
struct RawConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    num_key_value_heads: usize,
    rms_norm_eps: f64,
    vocab_size: usize,
    max_position_embeddings: usize,
    rope_theta: f64,
    tie_word_embeddings: bool,
}

#[derive(Debug, Deserialize)]
struct LoraConfig {
    lora_alpha: f64,
    r: f64,
}

pub fn embed(
    assets: &ModelAssets,
    texts: &[String],
    task: &JinaTask,
    prompt_name: Option<PromptName>,
    truncate_dim: Option<usize>,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
    match assets.model.kind {
        JinaModelKind::TextV5 => {
            embed_v5_text(assets, texts, task, prompt_name, truncate_dim, batch_size)
        }
        JinaModelKind::Code => {
            embed_code(assets, texts, task, prompt_name, truncate_dim, batch_size)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn embed_code(
    assets: &ModelAssets,
    texts: &[String],
    task: &JinaTask,
    prompt_name: Option<PromptName>,
    truncate_dim: Option<usize>,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
    let runtime = code_runtime(assets)?;
    let prompt_type = match prompt_name.unwrap_or(PromptName::Query) {
        PromptName::Query => "query",
        PromptName::Document | PromptName::Passage => "passage",
    };
    let prepared = texts
        .iter()
        .map(|text| code_prefixed_text(task, prompt_type, text))
        .collect::<Vec<_>>();

    let tokenized = tokenize_many(
        &runtime.tokenizer,
        &prepared,
        runtime.raw.max_position_embeddings,
    )?;
    let mut model = runtime
        .model
        .lock()
        .map_err(|_| anyhow!("code model cache mutex poisoned"))?;
    encode_qwen2_batches(
        &mut model,
        &tokenized,
        truncate_dim,
        batch_size,
        &runtime.device,
    )
}

#[allow(clippy::too_many_arguments)]
fn embed_v5_text(
    assets: &ModelAssets,
    texts: &[String],
    task: &JinaTask,
    prompt_name: Option<PromptName>,
    truncate_dim: Option<usize>,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
    let device = default_device();
    let tokenizer = Tokenizer::from_file(&assets.tokenizer_json)
        .map_err(|err| anyhow!("failed to load tokenizer: {err}"))?;
    let raw = load_raw_config(&assets.config_json)?;
    let dtype = DType::F32;
    let cfg = qwen3_config(&raw);
    let task_name = task.as_wire();
    let adapter = assets.root.join("adapters").join(task_name);
    let adapter_model = adapter.join("adapter_model.safetensors");
    let adapter_config = adapter.join("adapter_config.json");
    let lora = if adapter_model.exists() && adapter_config.exists() {
        Some((adapter_model, load_lora_config(&adapter_config)?))
    } else {
        None
    };

    let prompt = match task {
        JinaTask::Retrieval => match prompt_name.unwrap_or(PromptName::Query) {
            PromptName::Query => "Query: ",
            PromptName::Document | PromptName::Passage => "Document: ",
        },
        JinaTask::Classification | JinaTask::TextMatching | JinaTask::Clustering => "Document: ",
        _ => "",
    };

    let backend = unsafe {
        LoraBackend::new(
            assets.model_safetensors.clone(),
            lora.map(|(path, cfg)| (path, cfg.lora_alpha / cfg.r)),
        )
    };

    let prepared = texts
        .iter()
        .map(|text| format!("{prompt}{text}"))
        .collect::<Vec<_>>();
    let tokenized = tokenize_many(&tokenizer, &prepared, raw.max_position_embeddings)?;
    encode_qwen3_batches(
        || {
            let vb = VarBuilder::from_backend(Box::new(backend.clone()), dtype, device.clone());
            qwen3::Model::new(&cfg, vb)
        },
        &tokenized,
        truncate_dim,
        // Candle's Qwen3 causal-mask implementation is not batch-safe on CPU
        // in 0.10.2, so keep v5 text batches at one item until that backend
        // path is replaced or patched. The public batch-size still applies to
        // code/Qwen2 and future batch-safe runtimes.
        batch_size.min(1),
        &device,
    )
}

fn encode_qwen2_batches(
    model: &mut qwen2::Model,
    tokenized: &[Vec<u32>],
    truncate_dim: Option<usize>,
    batch_size: usize,
    device: &Device,
) -> Result<Vec<Vec<f32>>> {
    let mut out = vec![None; tokenized.len()];
    for group in same_len_groups(tokenized, batch_size) {
        let ids = flatten_group(tokenized, &group);
        let len = tokenized[group[0]].len();
        let input = Tensor::from_vec(ids, (group.len(), len), device)?;
        model.clear_kv_cache();
        let hidden = model.forward(&input, 0, None)?;
        let embeddings = pooled_embeddings(hidden, len, group.len(), truncate_dim)?;
        for (idx, embedding) in group.into_iter().zip(embeddings.into_iter()) {
            out[idx] = Some(embedding);
        }
    }
    collect_ordered_embeddings(out)
}

fn encode_qwen3_batches<F>(
    mut model_factory: F,
    tokenized: &[Vec<u32>],
    truncate_dim: Option<usize>,
    batch_size: usize,
    device: &Device,
) -> Result<Vec<Vec<f32>>>
where
    F: FnMut() -> candle::Result<qwen3::Model>,
{
    let mut out = vec![None; tokenized.len()];
    for group in same_len_groups(tokenized, batch_size) {
        let ids = flatten_group(tokenized, &group);
        let len = tokenized[group[0]].len();
        let input = Tensor::from_vec(ids, (group.len(), len), device)?;
        let mut model = model_factory()?;
        let hidden = model.forward(&input, 0)?;
        let embeddings = pooled_embeddings(hidden, len, group.len(), truncate_dim)?;
        for (idx, embedding) in group.into_iter().zip(embeddings.into_iter()) {
            out[idx] = Some(embedding);
        }
    }
    collect_ordered_embeddings(out)
}

fn pooled_embeddings(
    hidden: Tensor,
    seq_len: usize,
    batch_size: usize,
    truncate_dim: Option<usize>,
) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(batch_size);
    for batch_idx in 0..batch_size {
        let mut embedding = hidden
            .i((batch_idx, seq_len - 1, ..))?
            .to_dtype(DType::F32)?;
        if let Some(dim) = truncate_dim {
            embedding = embedding.narrow(0, 0, dim)?;
        }
        let mut values = embedding.to_vec1::<f32>()?;
        l2_normalize(&mut values);
        out.push(values);
    }
    Ok(out)
}

fn tokenize_many(tokenizer: &Tokenizer, texts: &[String], max_len: usize) -> Result<Vec<Vec<u32>>> {
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), false)
        .map_err(|err| anyhow!("tokenization failed: {err}"))?;
    let mut out = Vec::with_capacity(encodings.len());
    for encoding in encodings {
        let mut ids = encoding.get_ids().to_vec();
        if ids.len() > max_len {
            ids.truncate(max_len);
        }
        if ids.is_empty() {
            bail!("tokenizer produced no tokens");
        }
        out.push(ids);
    }
    Ok(out)
}

fn same_len_groups(tokenized: &[Vec<u32>], batch_size: usize) -> Vec<Vec<usize>> {
    let mut buckets: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, ids) in tokenized.iter().enumerate() {
        buckets.entry(ids.len()).or_default().push(idx);
    }
    let mut groups = Vec::new();
    for mut indices in buckets.into_values() {
        indices.sort_unstable();
        for chunk in indices.chunks(batch_size.max(1)) {
            groups.push(chunk.to_vec());
        }
    }
    groups.sort_by_key(|group| group[0]);
    groups
}

fn flatten_group(tokenized: &[Vec<u32>], group: &[usize]) -> Vec<u32> {
    let len = tokenized[group[0]].len();
    let mut ids = Vec::with_capacity(group.len() * len);
    for idx in group {
        ids.extend_from_slice(&tokenized[*idx]);
    }
    ids
}

fn collect_ordered_embeddings(out: Vec<Option<Vec<f32>>>) -> Result<Vec<Vec<f32>>> {
    out.into_iter()
        .enumerate()
        .map(|(idx, embedding)| {
            embedding.ok_or_else(|| anyhow!("missing embedding for input {idx}"))
        })
        .collect()
}

fn load_raw_config(path: &Path) -> Result<RawConfig> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(serde_json::from_str(&content)?)
}

fn load_lora_config(path: &Path) -> Result<LoraConfig> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(serde_json::from_str(&content)?)
}

fn qwen2_config(raw: &RawConfig) -> qwen2::Config {
    qwen2::Config {
        vocab_size: raw.vocab_size,
        hidden_size: raw.hidden_size,
        intermediate_size: raw.intermediate_size,
        num_hidden_layers: raw.num_hidden_layers,
        num_attention_heads: raw.num_attention_heads,
        num_key_value_heads: raw.num_key_value_heads,
        max_position_embeddings: raw.max_position_embeddings,
        sliding_window: raw.max_position_embeddings,
        max_window_layers: raw.num_hidden_layers,
        tie_word_embeddings: raw.tie_word_embeddings,
        rope_theta: raw.rope_theta,
        rms_norm_eps: raw.rms_norm_eps,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    }
}

fn qwen3_config(raw: &RawConfig) -> qwen3::Config {
    qwen3::Config {
        vocab_size: raw.vocab_size,
        hidden_size: raw.hidden_size,
        intermediate_size: raw.intermediate_size,
        num_hidden_layers: raw.num_hidden_layers,
        num_attention_heads: raw.num_attention_heads,
        head_dim: raw
            .head_dim
            .unwrap_or(raw.hidden_size / raw.num_attention_heads),
        attention_bias: false,
        num_key_value_heads: raw.num_key_value_heads,
        max_position_embeddings: raw.max_position_embeddings,
        sliding_window: None,
        max_window_layers: raw.num_hidden_layers,
        tie_word_embeddings: raw.tie_word_embeddings,
        rope_theta: raw.rope_theta,
        rms_norm_eps: raw.rms_norm_eps,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    }
}

fn code_prefixed_text(task: &JinaTask, prompt_type: &str, text: &str) -> String {
    let prefix = match (task, prompt_type) {
        (JinaTask::Nl2Code, "query") => {
            "Find the most relevant code snippet given the following query:\n"
        }
        (JinaTask::Nl2Code, _) => "Candidate code snippet:\n",
        (JinaTask::Qa, "query") => "Find the most relevant answer given the following question:\n",
        (JinaTask::Qa, _) => "Candidate answer:\n",
        (JinaTask::Code2Code, "query") => {
            "Find an equivalent code snippet given the following code snippet:\n"
        }
        (JinaTask::Code2Code, _) => "Candidate code snippet:\n",
        (JinaTask::Code2Nl, "query") => {
            "Find the most relevant comment given the following code snippet:\n"
        }
        (JinaTask::Code2Nl, _) => "Candidate comment:\n",
        (JinaTask::Code2Completion, "query") => {
            "Find the most relevant completion given the following start of code snippet:\n"
        }
        (JinaTask::Code2Completion, _) => "Candidate completion:\n",
        _ => "",
    };
    format!("{prefix}{text}")
}

fn l2_normalize(values: &mut [f32]) {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in values {
            *value /= norm;
        }
    }
}

fn default_device() -> Device {
    Device::Cpu
}

struct CodeRuntime {
    tokenizer: Tokenizer,
    raw: RawConfig,
    model: Mutex<qwen2::Model>,
    device: Device,
}

static CODE_CACHE: OnceLock<Mutex<HashMap<String, Arc<CodeRuntime>>>> = OnceLock::new();

fn code_runtime(assets: &ModelAssets) -> Result<Arc<CodeRuntime>> {
    let key = assets.root.display().to_string();
    let cache = CODE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(runtime) = cache
        .lock()
        .map_err(|_| anyhow!("code runtime cache mutex poisoned"))?
        .get(&key)
        .cloned()
    {
        return Ok(runtime);
    }

    let device = default_device();
    let dtype = DType::F32;
    let tokenizer = Tokenizer::from_file(&assets.tokenizer_json)
        .map_err(|err| anyhow!("failed to load tokenizer: {err}"))?;
    let raw = load_raw_config(&assets.config_json)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[assets.model_safetensors.as_path()], dtype, &device)?
    };
    let model = qwen2::Model::new(&qwen2_config(&raw), vb)?;
    let runtime = Arc::new(CodeRuntime {
        tokenizer,
        raw,
        model: Mutex::new(model),
        device,
    });

    cache
        .lock()
        .map_err(|_| anyhow!("code runtime cache mutex poisoned"))?
        .insert(key, runtime.clone());
    Ok(runtime)
}

#[derive(Clone)]
struct LoraBackend {
    base_path: PathBuf,
    adapter: Option<(PathBuf, f64)>,
}

impl LoraBackend {
    unsafe fn new(base_path: PathBuf, adapter: Option<(PathBuf, f64)>) -> Self {
        Self { base_path, adapter }
    }

    fn base(&self) -> candle::Result<candle::safetensors::MmapedSafetensors> {
        unsafe { candle::safetensors::MmapedSafetensors::new(&self.base_path) }
    }

    fn adapter(&self) -> candle::Result<Option<(candle::safetensors::MmapedSafetensors, f64)>> {
        self.adapter
            .as_ref()
            .map(|(path, scale)| unsafe {
                candle::safetensors::MmapedSafetensors::new(path).map(|st| (st, *scale))
            })
            .transpose()
    }

    fn lora_keys(name: &str) -> (String, String) {
        let stripped = name.strip_prefix("model.").unwrap_or(name);
        let prefix = format!("base_model.model.{stripped}");
        (
            prefix.replace(".weight", ".lora_A.weight"),
            prefix.replace(".weight", ".lora_B.weight"),
        )
    }
}

impl SimpleBackend for LoraBackend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _hint: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle::Result<Tensor> {
        let tensor = self.get_unchecked(name, dtype, dev)?;
        if tensor.shape() != &shape {
            candle::bail!(
                "shape mismatch for {name}, expected {:?}, got {:?}",
                shape,
                tensor.shape()
            );
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle::Result<Tensor> {
        let base = self.base()?.load(name, dev)?.to_dtype(dtype)?;
        let Some((adapter, scale)) = self.adapter()? else {
            return Ok(base);
        };

        if !name.ends_with(".weight") {
            return Ok(base);
        }

        let (a_key, b_key) = Self::lora_keys(name);
        if adapter.get(&a_key).is_err() || adapter.get(&b_key).is_err() {
            return Ok(base);
        }

        let a = adapter.load(&a_key, dev)?.to_dtype(dtype)?;
        let b = adapter.load(&b_key, dev)?.to_dtype(dtype)?;
        let delta = (b.matmul(&a)? * scale)?;
        base.broadcast_add(&delta)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        match self.base() {
            Ok(base) => base.get(name).is_ok(),
            Err(_) => false,
        }
    }
}
