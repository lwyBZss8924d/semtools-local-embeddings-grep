use crate::jina::chunk::{get_files, read_file_safely, split_into_chunks};
use crate::jina::types::{
    EmbeddingBackend, JinaSearchOptions, JinaSearchResult, JinaTask, LabelScore, PromptName,
};
use anyhow::{Result, bail};
use regex::Regex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

#[derive(Debug)]
pub struct JinaSearchRun {
    pub files: Vec<PathBuf>,
    pub results: Vec<JinaSearchResult>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParsedGrepLine {
    pub path: Option<String>,
    pub line_number: Option<usize>,
    pub content: String,
    pub raw_line: String,
}

pub fn parse_grep_line(line: &str) -> ParsedGrepLine {
    let mut parts = line.splitn(3, ':');
    let first = parts.next().unwrap_or_default();
    let second = parts.next();
    let third = parts.next();

    match (second, third) {
        (Some(line_number), Some(content)) if line_number.parse::<usize>().is_ok() => {
            ParsedGrepLine {
                path: (!first.is_empty()).then(|| first.to_string()),
                line_number: line_number.parse::<usize>().ok(),
                content: content.to_string(),
                raw_line: line.to_string(),
            }
        }
        (Some(content), None) => ParsedGrepLine {
            path: (!first.is_empty()).then(|| first.to_string()),
            line_number: None,
            content: content.to_string(),
            raw_line: line.to_string(),
        },
        _ => ParsedGrepLine {
            path: None,
            line_number: None,
            content: line.to_string(),
            raw_line: line.to_string(),
        },
    }
}

pub fn strip_ansi(text: &str) -> String {
    static ANSI_RE: OnceLock<Regex> = OnceLock::new();
    let re = ANSI_RE.get_or_init(|| Regex::new(r"\x1b\[[0-9;]*m").expect("valid ANSI regex"));
    re.replace_all(text, "").into_owned()
}

pub fn pipe_rerank<B: EmbeddingBackend>(
    pattern: &str,
    stdin_lines: &[String],
    backend: &B,
    options: &JinaSearchOptions,
) -> Result<JinaSearchRun> {
    let parsed: Vec<ParsedGrepLine> = stdin_lines
        .iter()
        .map(|line| parse_grep_line(line))
        .collect();
    let valid: Vec<(usize, String)> = parsed
        .iter()
        .enumerate()
        .filter_map(|(idx, line)| {
            let content = strip_ansi(&line.content);
            (!content.trim().is_empty()).then_some((idx, content))
        })
        .collect();

    if valid.is_empty() {
        return Ok(JinaSearchRun {
            files: Vec::new(),
            results: Vec::new(),
        });
    }

    let query_embedding = embed_query(pattern, backend, options)?;
    let candidate_texts: Vec<String> = valid.iter().map(|(_, text)| text.clone()).collect();
    let candidate_embeddings = embed_documents(&candidate_texts, backend, options)?;

    let mut results = Vec::new();
    for ((parsed_idx, text), embedding) in valid.iter().zip(candidate_embeddings.iter()) {
        let score = cosine_similarity(&query_embedding, embedding)?;
        if !score_passes(score, options) {
            continue;
        }
        let line = &parsed[*parsed_idx];
        results.push(JinaSearchResult {
            path: line.path.clone(),
            line_number: line.line_number,
            text: text.clone(),
            score,
            context_before: Vec::new(),
            context_after: Vec::new(),
            original_line: Some(line.raw_line.clone()),
            label: None,
            label_scores: Vec::new(),
        });
    }

    sort_results(&mut results, options.invert_match);
    results.truncate(options.top_k);

    Ok(JinaSearchRun {
        files: Vec::new(),
        results,
    })
}

pub fn semantic_grep<B: EmbeddingBackend>(
    pattern: &str,
    paths: &[PathBuf],
    backend: &B,
    options: &JinaSearchOptions,
) -> Result<JinaSearchRun> {
    let files = get_files(paths, options)?;
    if files.is_empty() {
        return Ok(JinaSearchRun {
            files,
            results: Vec::new(),
        });
    }

    let query_embedding = embed_query(pattern, backend, options)?;
    let mut all_results = Vec::new();

    for file in &files {
        let Some(content) = read_file_safely(file) else {
            continue;
        };
        let chunks = split_into_chunks(&content, options.granularity);
        if chunks.is_empty() {
            continue;
        }

        let chunk_texts: Vec<String> = chunks.iter().map(|chunk| chunk.text.clone()).collect();
        let embeddings = embed_documents(&chunk_texts, backend, options)?;
        let lines: Vec<&str> = content.lines().collect();
        let mut file_results = Vec::new();

        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            let score = cosine_similarity(&query_embedding, embedding)?;
            if !score_passes(score, options) {
                continue;
            }

            let (context_before, context_after) = context_lines(
                &lines,
                chunk.line_number,
                options.before_context,
                options.after_context,
            );
            file_results.push(JinaSearchResult {
                path: Some(file.display().to_string()),
                line_number: Some(chunk.line_number),
                text: chunk.text.clone(),
                score,
                context_before,
                context_after,
                original_line: None,
                label: None,
                label_scores: Vec::new(),
            });
        }

        sort_results(&mut file_results, options.invert_match);
        if let Some(max_count) = options.max_count {
            file_results.truncate(max_count);
        }
        all_results.extend(file_results);
    }

    sort_results(&mut all_results, options.invert_match);
    all_results.truncate(options.top_k);

    Ok(JinaSearchRun {
        files,
        results: all_results,
    })
}

pub fn semantic_classify<B: EmbeddingBackend>(
    labels: &[String],
    paths: &[PathBuf],
    backend: &B,
    options: &JinaSearchOptions,
) -> Result<JinaSearchRun> {
    if labels.is_empty() {
        bail!("classification requires at least one label");
    }

    let files = get_files(paths, options)?;
    if files.is_empty() {
        return Ok(JinaSearchRun {
            files,
            results: Vec::new(),
        });
    }

    let mut class_options = options.clone();
    class_options.task = JinaTask::Classification;

    let label_embeddings = backend.embed(
        labels,
        &class_options.model,
        &class_options.task,
        None,
        class_options.truncate_dim,
    )?;

    let mut all_results = Vec::new();
    for file in &files {
        let Some(content) = read_file_safely(file) else {
            continue;
        };
        let chunks = split_into_chunks(&content, options.granularity);
        if chunks.is_empty() {
            continue;
        }

        let chunk_texts: Vec<String> = chunks.iter().map(|chunk| chunk.text.clone()).collect();
        let chunk_embeddings = backend.embed(
            &chunk_texts,
            &class_options.model,
            &class_options.task,
            None,
            class_options.truncate_dim,
        )?;
        let lines: Vec<&str> = content.lines().collect();

        for (chunk, embedding) in chunks.iter().zip(chunk_embeddings.iter()) {
            let mut label_scores = Vec::new();
            for (label, label_embedding) in labels.iter().zip(label_embeddings.iter()) {
                label_scores.push(LabelScore {
                    label: label.clone(),
                    score: cosine_similarity(embedding, label_embedding)?,
                });
            }
            label_scores.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let Some(best) = label_scores.first() else {
                continue;
            };
            if best.score < options.threshold {
                continue;
            }

            let (context_before, context_after) = context_lines(
                &lines,
                chunk.line_number,
                options.before_context,
                options.after_context,
            );
            all_results.push(JinaSearchResult {
                path: Some(file.display().to_string()),
                line_number: Some(chunk.line_number),
                text: chunk.text.clone(),
                score: best.score,
                context_before,
                context_after,
                original_line: None,
                label: Some(best.label.clone()),
                label_scores,
            });
        }
    }

    sort_results(&mut all_results, false);
    all_results.truncate(options.top_k);

    Ok(JinaSearchRun {
        files,
        results: all_results,
    })
}

pub fn count_by_file(results: &[JinaSearchResult]) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for result in results {
        if let Some(path) = &result.path {
            *counts.entry(path.clone()).or_insert(0) += 1;
        }
    }
    counts
}

fn embed_query<B: EmbeddingBackend>(
    pattern: &str,
    backend: &B,
    options: &JinaSearchOptions,
) -> Result<Vec<f32>> {
    let prompt = options.task.uses_prompt_pair().then_some(PromptName::Query);
    let embeddings = backend.embed(
        &[pattern.to_string()],
        &options.model,
        &options.task,
        prompt,
        options.truncate_dim,
    )?;
    embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("embedding backend returned no query embedding"))
}

fn embed_documents<B: EmbeddingBackend>(
    texts: &[String],
    backend: &B,
    options: &JinaSearchOptions,
) -> Result<Vec<Vec<f32>>> {
    let prompt = options
        .task
        .uses_prompt_pair()
        .then_some(PromptName::Document);
    backend.embed(
        texts,
        &options.model,
        &options.task,
        prompt,
        options.truncate_dim,
    )
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        bail!(
            "embedding dimension mismatch: query has {}, candidate has {}",
            a.len(),
            b.len()
        );
    }

    let mut dot = 0.0f32;
    let mut a_norm = 0.0f32;
    let mut b_norm = 0.0f32;

    for (left, right) in a.iter().zip(b.iter()) {
        dot += left * right;
        a_norm += left * left;
        b_norm += right * right;
    }

    if a_norm == 0.0 || b_norm == 0.0 {
        return Ok(0.0);
    }

    Ok(dot / (a_norm.sqrt() * b_norm.sqrt()))
}

fn score_passes(score: f32, options: &JinaSearchOptions) -> bool {
    if options.invert_match {
        score < options.threshold
    } else {
        score >= options.threshold
    }
}

fn sort_results(results: &mut [JinaSearchResult], invert: bool) {
    if invert {
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
}

fn context_lines(
    lines: &[&str],
    line_number_1based: usize,
    before: usize,
    after: usize,
) -> (Vec<String>, Vec<String>) {
    let line_idx = line_number_1based.saturating_sub(1);
    let before_start = line_idx.saturating_sub(before);
    let context_before = lines[before_start..line_idx.min(lines.len())]
        .iter()
        .map(|line| line.to_string())
        .collect();
    let after_start = line_number_1based.min(lines.len());
    let after_end = (after_start + after).min(lines.len());
    let context_after = lines[after_start..after_end]
        .iter()
        .map(|line| line.to_string())
        .collect();
    (context_before, context_after)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jina::types::{Granularity, JinaTask};

    struct FixedBackend;

    impl EmbeddingBackend for FixedBackend {
        fn embed(
            &self,
            texts: &[String],
            _model: &str,
            _task: &JinaTask,
            _prompt_name: Option<PromptName>,
            _truncate_dim: Option<usize>,
        ) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|text| {
                    if text.contains("retry") || text.contains("backoff") {
                        vec![1.0, 0.0]
                    } else {
                        vec![0.0, 1.0]
                    }
                })
                .collect())
        }
    }

    fn options() -> JinaSearchOptions {
        JinaSearchOptions {
            recursive: false,
            files_with_matches: false,
            files_without_match: false,
            count: false,
            line_number: true,
            with_filename: true,
            after_context: 0,
            before_context: 0,
            include_patterns: Vec::new(),
            exclude_patterns: Vec::new(),
            exclude_dir_patterns: Vec::new(),
            color: false,
            invert_match: false,
            max_count: None,
            quiet: false,
            threshold: 0.5,
            top_k: 10,
            model: "jina-code-embeddings-1.5b".to_string(),
            task: JinaTask::Nl2Code,
            truncate_dim: None,
            batch_size: 256,
            granularity: Granularity::Line,
        }
    }

    #[test]
    fn parses_grep_lines() {
        assert_eq!(
            parse_grep_line("src/lib.rs:42:retry logic"),
            ParsedGrepLine {
                path: Some("src/lib.rs".to_string()),
                line_number: Some(42),
                content: "retry logic".to_string(),
                raw_line: "src/lib.rs:42:retry logic".to_string(),
            }
        );
        assert_eq!(parse_grep_line("plain").content, "plain");
    }

    #[test]
    fn strips_ansi() {
        assert_eq!(strip_ansi("\u{1b}[31mred\u{1b}[0m"), "red");
    }

    #[test]
    fn cosine_scores_vectors() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]).unwrap() - 1.0).abs() < 1e-6);
        assert_eq!(
            cosine_similarity(&[1.0], &[1.0, 2.0])
                .unwrap_err()
                .to_string(),
            "embedding dimension mismatch: query has 1, candidate has 2"
        );
    }

    #[test]
    fn pipe_rerank_scores_candidates() {
        let input = vec![
            "a.rs:1:retry timeout".to_string(),
            "b.rs:2:unrelated".to_string(),
        ];
        let run = pipe_rerank("retry backoff", &input, &FixedBackend, &options()).unwrap();
        assert_eq!(run.results.len(), 1);
        assert_eq!(run.results[0].path.as_deref(), Some("a.rs"));
    }
}
