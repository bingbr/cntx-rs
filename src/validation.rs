use std::{
    net::IpAddr,
    path::{Component, Path, PathBuf},
};

use anyhow::{Result, anyhow, bail};
use url::{Host, Url};

pub const MAX_QUESTION_LENGTH: usize = 8_000;
pub const MAX_RESOURCE_COUNT: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedGitUrl {
    pub repo_url: String,
    pub branch: Option<String>,
    pub search_paths: Vec<PathBuf>,
}

pub fn validate_question(question: &str) -> Result<String> {
    let trimmed = question.trim();
    if trimmed.is_empty() {
        bail!("question must not be empty");
    }
    if trimmed.chars().count() > MAX_QUESTION_LENGTH {
        bail!("question exceeds the {MAX_QUESTION_LENGTH} character limit");
    }
    Ok(trimmed.to_string())
}

pub fn validate_requested_resource_count(count: usize) -> Result<()> {
    if count > MAX_RESOURCE_COUNT {
        bail!("too many resources requested; the limit is {MAX_RESOURCE_COUNT}");
    }
    Ok(())
}

pub fn validate_resource_name(name: &str) -> Result<String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        bail!("resource name must not be empty");
    }
    if trimmed.len() > 64 {
        bail!("resource name must be 64 characters or fewer");
    }
    let mut chars = trimmed.chars();
    let Some(first) = chars.next() else {
        bail!("resource name must not be empty");
    };
    if !first.is_ascii_alphanumeric() {
        bail!("resource name must start with an ASCII letter or digit");
    }
    if !trimmed
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
    {
        bail!("resource name may only contain ASCII letters, digits, '-', '_' or '.'");
    }
    Ok(trimmed.to_string())
}

pub fn validate_branch_name(branch: &str) -> Result<String> {
    let trimmed = branch.trim();
    if trimmed.is_empty() {
        bail!("branch must not be empty");
    }
    if trimmed.starts_with('-')
        || trimmed.contains("..")
        || trimmed.ends_with('/')
        || trimmed.ends_with('.')
        || trimmed.contains('@')
        || trimmed.contains('\\')
        || trimmed.contains("//")
        || trimmed.contains(" ")
        || trimmed.as_bytes().iter().any(|byte| byte.is_ascii_control())
    {
        bail!("branch '{trimmed}' is not a safe git ref name");
    }
    Ok(trimmed.to_string())
}

pub fn validate_local_path(path: &Path) -> Result<PathBuf> {
    if path.as_os_str().is_empty() {
        bail!("local path must not be empty");
    }
    Ok(path.to_path_buf())
}

pub fn validate_search_path(path: &Path) -> Result<PathBuf> {
    if path.as_os_str().is_empty() {
        return Ok(PathBuf::from("."));
    }
    if path.is_absolute() {
        bail!("search paths must be relative to the resource root");
    }

    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(segment) => {
                let segment = segment.to_string_lossy();
                if segment.chars().any(char::is_control) {
                    bail!("search path '{}' contains control characters", path.display());
                }
                normalized.push(segment.as_ref());
            }
            Component::ParentDir => {
                bail!("search path '{}' must not contain '..'", path.display());
            }
            Component::RootDir | Component::Prefix(_) => {
                bail!("search path '{}' must be relative", path.display());
            }
        }
    }

    if normalized.as_os_str().is_empty() {
        Ok(PathBuf::from("."))
    } else {
        Ok(normalized)
    }
}

pub fn normalize_git_url(input: &str) -> Result<NormalizedGitUrl> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        bail!("git URL must not be empty");
    }

    let url = Url::parse(trimmed).map_err(|error| anyhow!("invalid git URL '{trimmed}': {error}"))?;
    if url.scheme() != "https" {
        bail!("git URL '{trimmed}' must use https");
    }
    if !url.username().is_empty() || url.password().is_some() {
        bail!("git URL '{trimmed}' must not embed credentials");
    }

    reject_private_host(&url)?;

    if is_github_host(&url) {
        return normalize_github_url(url);
    }

    let host = url
        .host_str()
        .ok_or_else(|| anyhow!("git URL '{trimmed}' is missing a host"))?;
    let segments = url
        .path_segments()
        .map(|segments| segments.filter(|segment| !segment.is_empty()).collect::<Vec<_>>())
        .unwrap_or_default();
    if segments.len() < 2 {
        bail!("git URL '{trimmed}' must point to a repository root");
    }

    Ok(NormalizedGitUrl {
        repo_url: format!("https://{host}/{}", segments.join("/"))
            .trim_end_matches('/')
            .to_string(),
        branch: None,
        search_paths: vec![PathBuf::from(".")],
    })
}

fn normalize_github_url(url: Url) -> Result<NormalizedGitUrl> {
    let host = url.host_str().ok_or_else(|| anyhow!("GitHub URL is missing a host"))?;
    let segments = url
        .path_segments()
        .map(|segments| segments.filter(|segment| !segment.is_empty()).collect::<Vec<_>>())
        .unwrap_or_default();
    if segments.len() < 2 {
        bail!("GitHub URL '{}' must include owner and repo", url);
    }

    let owner = segments[0];
    let repo = segments[1].trim_end_matches(".git");
    let repo_url = format!("https://{host}/{owner}/{repo}");

    let mut branch = None;
    let mut search_paths = vec![PathBuf::from(".")];
    if segments.len() >= 4 && matches!(segments[2], "tree" | "blob") {
        branch = Some(validate_branch_name(segments[3])?);
        if segments.len() > 4 {
            let path = segments[4..].join("/");
            search_paths = vec![validate_search_path(Path::new(&path))?];
        }
    }

    Ok(NormalizedGitUrl {
        repo_url,
        branch,
        search_paths,
    })
}

fn reject_private_host(url: &Url) -> Result<()> {
    let Some(host) = url.host() else {
        bail!("git URL '{url}' is missing a host");
    };

    match host {
        Host::Domain(domain) => {
            let lower = domain.to_ascii_lowercase();
            if lower == "localhost" || lower.ends_with(".localhost") || lower.ends_with(".local") {
                bail!("git URL host '{domain}' is not allowed");
            }
        }
        Host::Ipv4(ip) => reject_private_ip(IpAddr::V4(ip))?,
        Host::Ipv6(ip) => reject_private_ip(IpAddr::V6(ip))?,
    }

    Ok(())
}

fn reject_private_ip(ip: IpAddr) -> Result<()> {
    let is_private = match ip {
        IpAddr::V4(ipv4) => {
            ipv4.is_private()
                || ipv4.is_loopback()
                || ipv4.is_link_local()
                || ipv4.is_broadcast()
                || ipv4.is_unspecified()
                || ipv4.octets()[0] == 169 && ipv4.octets()[1] == 254
        }
        IpAddr::V6(ipv6) => {
            ipv6.is_loopback() || ipv6.is_unspecified() || ipv6.is_unique_local() || ipv6.is_unicast_link_local()
        }
    };

    if is_private {
        bail!("git URL IP address '{ip}' is not allowed");
    }
    Ok(())
}

fn is_github_host(url: &Url) -> bool {
    matches!(
        url.host_str().map(|host| host.to_ascii_lowercase()),
        Some(host) if host == "github.com" || host == "www.github.com"
    )
}

pub fn extract_resource_mentions(question: &str) -> Vec<String> {
    let mut mentions = Vec::new();
    let chars: Vec<char> = question.chars().collect();
    let mut idx = 0usize;
    while idx < chars.len() {
        if chars[idx] == '@' {
            let start = idx + 1;
            let mut end = start;
            while end < chars.len() {
                let ch = chars[end];
                if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                    end += 1;
                } else {
                    break;
                }
            }
            if end > start {
                mentions.push(chars[start..end].iter().collect());
                idx = end;
                continue;
            }
        }
        idx += 1;
    }
    mentions
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{extract_resource_mentions, normalize_git_url, validate_resource_name, validate_search_path};

    #[test]
    fn normalizes_github_tree_urls() {
        let normalized = normalize_git_url("https://github.com/sveltejs/kit/tree/main/documentation")
            .expect("GitHub URL should normalize");

        assert_eq!(normalized.repo_url, "https://github.com/sveltejs/kit");
        assert_eq!(normalized.branch.as_deref(), Some("main"));
        assert_eq!(normalized.search_paths, vec![Path::new("documentation")]);
    }

    #[test]
    fn rejects_parent_search_paths() {
        let error = validate_search_path(Path::new("../secrets")).expect_err(".. should be rejected");
        assert!(error.to_string().contains("must not contain '..'"));
    }

    #[test]
    fn extracts_resource_mentions_from_question() {
        assert_eq!(
            extract_resource_mentions("compare @svelte and @kit.docs please"),
            vec!["svelte".to_string(), "kit.docs".to_string()]
        );
    }

    #[test]
    fn validates_resource_names() {
        assert!(validate_resource_name("svelte-docs").is_ok());
        assert!(validate_resource_name("../bad").is_err());
    }
}
