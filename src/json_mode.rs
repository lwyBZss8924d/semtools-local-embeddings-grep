use serde::Serialize;

// Parse
#[derive(Debug, Serialize)]
pub struct ParseResultJSON {
    pub input_path: String,
    pub output_path: String,
    pub was_cached: bool,
}

#[derive(Debug, Serialize)]
pub struct ParseOutput {
    pub results: Vec<ParseResultJSON>,
}

// Search
#[derive(Debug, Serialize)]
pub struct SearchResultJSON {
    pub filename: String,
    pub start_line_number: usize,
    pub end_line_number: usize,
    pub match_line_number: usize,
    pub distance: f64,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct SearchOutput {
    pub results: Vec<SearchResultJSON>,
}

// Jina grep
#[derive(Debug, Serialize)]
pub struct JgrepLabelScoreJSON {
    pub label: String,
    pub score: f32,
}

#[derive(Debug, Serialize)]
pub struct JgrepResultJSON {
    pub path: Option<String>,
    pub line_number: Option<usize>,
    pub score: f32,
    pub distance: f32,
    pub text: String,
    pub original_line: Option<String>,
    pub context_before: Vec<String>,
    pub context_after: Vec<String>,
    pub label: Option<String>,
    pub label_scores: Vec<JgrepLabelScoreJSON>,
}

#[derive(Debug, Serialize)]
pub struct JgrepOutput {
    pub r#type: String,
    pub backend: String,
    pub mode: String,
    pub model: String,
    pub task: String,
    pub results: Vec<JgrepResultJSON>,
}

// Ask
#[derive(Debug, Serialize)]
pub struct AskOutput {
    pub query: String,
    pub response: String,
    pub files_searched: Vec<String>,
}

// Workspace
#[derive(Debug, Serialize)]
pub struct WorkspaceOutput {
    pub name: String,
    pub root_dir: String,
    pub total_documents: usize,
}

#[derive(Debug, Serialize)]
pub struct PruneOutput {
    pub files_removed: usize,
    pub files_remaining: usize,
}

// Error output
#[derive(Debug, Serialize)]
pub struct ErrorOutput {
    pub error: String,
    pub error_type: String,
}
