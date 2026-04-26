use crate::jina::search::{JinaSearchRun, count_by_file};
use crate::jina::types::{JinaSearchOptions, JinaSearchResult};
use crate::json_mode::{JgrepLabelScoreJSON, JgrepOutput, JgrepResultJSON};
use anyhow::Result;
use std::collections::HashSet;
use std::io::{self, IsTerminal};

pub fn print_human_results(run: &JinaSearchRun, options: &JinaSearchOptions, only_matching: bool) {
    if options.quiet {
        return;
    }

    if options.files_with_matches {
        let mut seen = HashSet::new();
        for result in &run.results {
            if let Some(path) = &result.path
                && seen.insert(path.clone())
            {
                println!("{path}");
            }
        }
        return;
    }

    if options.files_without_match {
        let matched: HashSet<String> = run.results.iter().filter_map(|r| r.path.clone()).collect();
        for path in &run.files {
            let path = path.display().to_string();
            if !matched.contains(&path) {
                println!("{path}");
            }
        }
        return;
    }

    if options.count {
        let counts = count_by_file(&run.results);
        for path in &run.files {
            let path = path.display().to_string();
            let count = counts.get(&path).copied().unwrap_or(0);
            if options.with_filename {
                println!("{path}:{count}");
            } else {
                println!("{count}");
            }
        }
        return;
    }

    for result in &run.results {
        if only_matching {
            if let Some(label) = &result.label {
                println!("{label}");
            } else {
                println!("{}", result.text);
            }
            continue;
        }

        println!("{}", format_result(result, options));
    }
}

pub fn print_json_results(
    run: &JinaSearchRun,
    options: &JinaSearchOptions,
    mode: &str,
) -> Result<()> {
    let output = JgrepOutput {
        r#type: "jina_grep_results".to_string(),
        backend: "rust_native_jina".to_string(),
        mode: mode.to_string(),
        model: options.model.clone(),
        task: options.task.to_string(),
        results: run
            .results
            .iter()
            .map(|result| JgrepResultJSON {
                path: result.path.clone(),
                line_number: result.line_number,
                score: result.score,
                distance: result.distance(),
                text: result.text.clone(),
                original_line: result.original_line.clone(),
                context_before: result.context_before.clone(),
                context_after: result.context_after.clone(),
                label: result.label.clone(),
                label_scores: result
                    .label_scores
                    .iter()
                    .map(|label| JgrepLabelScoreJSON {
                        label: label.label.clone(),
                        score: label.score,
                    })
                    .collect(),
            })
            .collect(),
    };

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

pub fn format_result(result: &JinaSearchResult, options: &JinaSearchOptions) -> String {
    if let Some(original_line) = &result.original_line {
        return format!(
            "{original_line}  [{}]",
            format_score(result.score, options.color)
        );
    }

    let prefix = format_prefix(result, options, ':');
    let context_prefix = format_prefix(result, options, '-');
    let mut lines = Vec::new();

    for line in &result.context_before {
        lines.push(format!("{context_prefix}{line}"));
    }

    let text = if let Some(label) = &result.label {
        let scores = result
            .label_scores
            .iter()
            .map(|score| format!("{}:{:.3}", score.label, score.score))
            .collect::<Vec<_>>()
            .join(" ");
        format!("{}  [{}] [{}]", result.text, label, scores)
    } else {
        format!(
            "{}  [{}]",
            result.text,
            format_score(result.score, options.color)
        )
    };
    lines.push(format!("{prefix}{text}"));

    for line in &result.context_after {
        lines.push(format!("{context_prefix}{line}"));
    }

    lines.join("\n")
}

fn format_prefix(
    result: &JinaSearchResult,
    options: &JinaSearchOptions,
    separator: char,
) -> String {
    let mut parts = Vec::new();
    if options.with_filename
        && let Some(path) = &result.path
    {
        parts.push(path.clone());
    }
    if options.line_number
        && let Some(line_number) = result.line_number
    {
        parts.push(line_number.to_string());
    }

    if parts.is_empty() {
        String::new()
    } else {
        format!("{}{}", parts.join(&separator.to_string()), separator)
    }
}

fn format_score(score: f32, color: bool) -> String {
    if color && io::stdout().is_terminal() {
        format!("\x1b[2m{score:.3}\x1b[0m")
    } else {
        format!("{score:.3}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jina::types::{Granularity, JinaTask};

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
    fn formats_basic_result() {
        let result = JinaSearchResult {
            path: Some("src/lib.rs".to_string()),
            line_number: Some(7),
            text: "retry logic".to_string(),
            score: 0.875,
            context_before: Vec::new(),
            context_after: Vec::new(),
            original_line: None,
            label: None,
            label_scores: Vec::new(),
        };

        assert_eq!(
            format_result(&result, &options()),
            "src/lib.rs:7:retry logic  [0.875]"
        );
    }
}
