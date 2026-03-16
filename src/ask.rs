use std::fmt::Write as _;

use anyhow::Result;
use serde::Serialize;

use crate::{agent::runtime, config::Config};

type Callback<'a> = dyn FnMut(&str) -> Result<()> + 'a;

#[derive(Debug, Clone, Serialize)]
pub struct AskOutput {
    pub answer: String,
    pub resolved_resources: Vec<ResourceSummary>,
    pub citations: Vec<Citation>,
    pub retrieval: RetrievalSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResourceSummary {
    pub name: String,
    pub kind: String,
    pub source: String,
    pub branch: Option<String>,
    pub search_paths: Vec<String>,
    pub notes: Option<String>,
    pub ephemeral: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Citation {
    pub resource: String,
    pub path: String,
    pub line: usize,
    pub score: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct RetrievalSummary {
    pub resource_count: usize,
    pub snippet_count: usize,
    pub search_path_count: usize,
    pub retrieval_steps: usize,
    pub empty_resources: Vec<String>,
}

pub fn ask_question(
    config: &Config,
    requested_resources: &[String],
    question: &str,
    on_chunk: Option<&mut Callback<'_>>,
    on_progress: Option<&mut Callback<'_>>,
) -> Result<AskOutput> {
    runtime::ask_question(config, requested_resources, question, on_chunk, on_progress)
}

pub fn format_cli_output(output: &AskOutput) -> String {
    let mut rendered = String::new();
    rendered.push_str(&output.answer);
    rendered.push_str(&format_cli_footer(output));
    rendered.trim_end().to_string()
}

pub fn format_cli_footer(output: &AskOutput) -> String {
    let mut rendered = String::new();
    let _ = writeln!(
        rendered,
        "\nRetrieved {} snippets across {} resources and {} search paths in {} agent turn(s).",
        output.retrieval.snippet_count,
        output.retrieval.resource_count,
        output.retrieval.search_path_count,
        output.retrieval.retrieval_steps,
    );
    if !output.retrieval.empty_resources.is_empty() {
        let _ = writeln!(
            rendered,
            "No matches in: {}.",
            output.retrieval.empty_resources.join(", ")
        );
    }
    rendered
}

#[cfg(test)]
mod tests {
    use super::{AskOutput, Citation, RetrievalSummary, format_cli_output};

    #[test]
    fn cli_output_uses_agent_footer() {
        let output = AskOutput {
            answer: "answer".to_string(),
            resolved_resources: vec![],
            citations: vec![Citation {
                resource: "repo".to_string(),
                path: "README.md".to_string(),
                line: 1,
                score: 9,
            }],
            retrieval: RetrievalSummary {
                resource_count: 1,
                snippet_count: 1,
                search_path_count: 1,
                retrieval_steps: 2,
                empty_resources: vec![],
            },
        };

        let rendered = format_cli_output(&output);

        assert!(rendered.contains("answer"));
        assert!(rendered.contains("2 agent turn(s)."));
        assert!(!rendered.contains("Sources:"));
    }

    #[test]
    fn cli_output_reports_empty_resources() {
        let output = AskOutput {
            answer: "answer".to_string(),
            resolved_resources: vec![],
            citations: vec![],
            retrieval: RetrievalSummary {
                resource_count: 2,
                snippet_count: 0,
                search_path_count: 2,
                retrieval_steps: 1,
                empty_resources: vec!["repo-a".to_string(), "repo-b".to_string()],
            },
        };

        let rendered = format_cli_output(&output);

        assert!(rendered.contains("No matches in: repo-a, repo-b."));
    }
}
