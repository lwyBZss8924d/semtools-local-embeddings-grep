use crate::jina::types::{Chunk, Granularity, JinaSearchOptions};
use anyhow::{Result, anyhow};
use globset::{Glob, GlobSet, GlobSetBuilder};
use std::path::{Path, PathBuf};
use walkdir::{DirEntry, WalkDir};

const DEFAULT_EXCLUDED_DIRS: &[&str] = &[
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
];

fn estimate_tokens(text: &str) -> usize {
    (text.len() / 3).max(1)
}

pub fn split_into_chunks(content: &str, granularity: Granularity) -> Vec<Chunk> {
    match granularity {
        Granularity::Line => content
            .lines()
            .enumerate()
            .filter_map(|(idx, line)| {
                let text = line.trim_end();
                (!text.trim().is_empty()).then(|| Chunk {
                    line_number: idx + 1,
                    text: text.to_string(),
                })
            })
            .collect(),
        Granularity::Paragraph => split_paragraphs(content),
        Granularity::Sentence => split_sentences(content),
        Granularity::Token => split_token_windows(content, 512),
    }
}

fn split_paragraphs(content: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut current = Vec::new();
    let mut start_line = 1;

    for (idx, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                chunks.push(Chunk {
                    line_number: start_line,
                    text: current.join("\n"),
                });
                current.clear();
            }
            continue;
        }

        if current.is_empty() {
            start_line = idx + 1;
        }
        current.push(line.to_string());
    }

    if !current.is_empty() {
        chunks.push(Chunk {
            line_number: start_line,
            text: current.join("\n"),
        });
    }

    chunks
}

fn split_sentences(content: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();

    for (idx, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        for sentence in sentence_segments(line) {
            let sentence = sentence.trim();
            if !sentence.is_empty() {
                chunks.push(Chunk {
                    line_number: idx + 1,
                    text: sentence.to_string(),
                });
            }
        }
    }

    chunks
}

fn sentence_segments(line: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    for ch in line.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?' | '。' | '！' | '？') {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                segments.push(trimmed.to_string());
            }
            current.clear();
        }
    }

    let trimmed = current.trim();
    if !trimmed.is_empty() {
        segments.push(trimmed.to_string());
    }

    segments
}

fn split_token_windows(content: &str, chunk_tokens: usize) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut current = Vec::new();
    let mut current_tokens = 0;
    let mut start_line = 1;

    for (idx, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        let line_tokens = estimate_tokens(line);
        if !current.is_empty() && current_tokens + line_tokens > chunk_tokens {
            chunks.push(Chunk {
                line_number: start_line,
                text: current.join("\n"),
            });
            current.clear();
            current_tokens = 0;
        }

        if current.is_empty() {
            start_line = idx + 1;
        }
        current.push(line.to_string());
        current_tokens += line_tokens;
    }

    if !current.is_empty() {
        chunks.push(Chunk {
            line_number: start_line,
            text: current.join("\n"),
        });
    }

    chunks
}

pub fn read_file_safely(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.contains(&0) {
        return None;
    }
    match String::from_utf8(bytes) {
        Ok(text) => Some(text),
        Err(err) => Some(err.into_bytes().into_iter().map(char::from).collect()),
    }
}

pub fn get_files(paths: &[PathBuf], options: &JinaSearchOptions) -> Result<Vec<PathBuf>> {
    let include = build_globset(&options.include_patterns)?;
    let exclude = build_globset(&options.exclude_patterns)?;
    let exclude_dirs = build_globset(&options.exclude_dir_patterns)?;
    let mut files = Vec::new();

    for path in paths {
        if path.is_file() {
            if should_include_file(path, include.as_ref(), exclude.as_ref()) {
                files.push(path.clone());
            }
            continue;
        }

        if !path.is_dir() {
            return Err(anyhow!("{}: No such file or directory", path.display()));
        }

        if options.recursive {
            for entry in WalkDir::new(path)
                .into_iter()
                .filter_entry(|entry| !should_exclude_dir(entry, exclude_dirs.as_ref()))
            {
                let entry = entry?;
                if entry.file_type().is_file()
                    && should_include_file(entry.path(), include.as_ref(), exclude.as_ref())
                {
                    files.push(entry.into_path());
                }
            }
        } else {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() && should_include_file(&path, include.as_ref(), exclude.as_ref())
                {
                    files.push(path);
                }
            }
        }
    }

    files.sort();
    Ok(files)
}

fn build_globset(patterns: &[String]) -> Result<Option<GlobSet>> {
    if patterns.is_empty() {
        return Ok(None);
    }

    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        builder.add(Glob::new(pattern)?);
    }
    Ok(Some(builder.build()?))
}

fn should_include_file(path: &Path, include: Option<&GlobSet>, exclude: Option<&GlobSet>) -> bool {
    if let Some(include) = include
        && !include.is_match(path)
        && !path
            .file_name()
            .is_some_and(|filename| include.is_match(Path::new(filename)))
    {
        return false;
    }

    if let Some(exclude) = exclude
        && (exclude.is_match(path)
            || path
                .file_name()
                .is_some_and(|filename| exclude.is_match(Path::new(filename))))
    {
        return false;
    }

    true
}

fn should_exclude_dir(entry: &DirEntry, exclude_dirs: Option<&GlobSet>) -> bool {
    if !entry.file_type().is_dir() {
        return true;
    }

    let name = entry.file_name().to_string_lossy();
    if DEFAULT_EXCLUDED_DIRS.contains(&name.as_ref()) {
        return true;
    }

    exclude_dirs.is_some_and(|patterns| {
        patterns.is_match(entry.path()) || patterns.is_match(Path::new(name.as_ref()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_lines() {
        let chunks = split_into_chunks("one\n\n two \n", Granularity::Line);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].line_number, 1);
        assert_eq!(chunks[1].line_number, 3);
    }

    #[test]
    fn chunks_paragraphs() {
        let chunks = split_into_chunks("a\nb\n\nc\n", Granularity::Paragraph);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "a\nb");
        assert_eq!(chunks[1].line_number, 4);
    }

    #[test]
    fn chunks_sentences() {
        let chunks = split_into_chunks("One. Two? 三！\n", Granularity::Sentence);
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn chunks_token_windows_keep_line_numbers() {
        let content = format!("{}\nshort", "x".repeat(1600));
        let chunks = split_token_windows(&content, 512);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[1].line_number, 2);
    }
}
