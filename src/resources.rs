use std::{
    collections::HashSet,
    fmt,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::validation;

const DEFAULT_RESOURCE_BRANCH: &str = "main";
type ProgressCallback<'a> = dyn FnMut(&str) -> Result<()> + 'a;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    pub name: String,
    #[serde(default)]
    pub kind: ResourceKind,
    pub git_url: Option<String>,
    pub local_path: Option<PathBuf>,
    pub branch: Option<String>,
    #[serde(default)]
    pub search_paths: Vec<PathBuf>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedResource {
    pub name: String,
    pub kind: ResourceKind,
    pub git_url: Option<String>,
    pub local_path: Option<PathBuf>,
    pub branch: Option<String>,
    pub search_paths: Vec<PathBuf>,
    pub notes: Option<String>,
    pub ephemeral: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ResourceKind {
    #[default]
    Git,
    Local,
}

impl fmt::Display for ResourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Git => write!(f, "git"),
            Self::Local => write!(f, "local"),
        }
    }
}

pub fn default_resources() -> Vec<ResourceConfig> {
    vec![
        ResourceConfig {
            name: "svelte".to_string(),
            kind: ResourceKind::Git,
            git_url: Some("https://github.com/sveltejs/svelte.dev".to_string()),
            local_path: None,
            branch: Some(DEFAULT_RESOURCE_BRANCH.to_string()),
            search_paths: vec![PathBuf::from("apps/svelte.dev")],
            notes: None,
        },
        ResourceConfig {
            name: "sveltekit".to_string(),
            kind: ResourceKind::Git,
            git_url: Some("https://github.com/sveltejs/kit".to_string()),
            local_path: None,
            branch: Some(DEFAULT_RESOURCE_BRANCH.to_string()),
            search_paths: vec![PathBuf::from(".")],
            notes: None,
        },
        ResourceConfig {
            name: "tauri".to_string(),
            kind: ResourceKind::Git,
            git_url: Some("https://github.com/tauri-apps/tauri".to_string()),
            local_path: None,
            branch: Some("dev".to_string()),
            search_paths: vec![PathBuf::from(".")],
            notes: None,
        },
    ]
}

pub fn resolve_resource_references(
    resources: &[ResourceConfig],
    requested: &[String],
    question: &str,
) -> Result<Vec<ResolvedResource>> {
    let mut raw_references = requested
        .iter()
        .map(|resource| resource.trim().to_string())
        .filter(|resource| !resource.is_empty())
        .collect::<Vec<_>>();

    for mention in validation::extract_resource_mentions(question) {
        raw_references.push(mention);
    }

    if raw_references.is_empty() {
        raw_references = resources.iter().map(|resource| resource.name.clone()).collect();
    }

    let mut seen = HashSet::new();
    let mut resolved = Vec::new();
    let mut invalid = Vec::new();

    for reference in raw_references {
        let dedupe_key = reference.to_ascii_lowercase();
        if !seen.insert(dedupe_key) {
            continue;
        }

        match resolve_reference(resources, &reference) {
            Ok(resource) => resolved.push(resource),
            Err(error) => invalid.push(format!("{reference}: {error}")),
        }
    }

    validation::validate_requested_resource_count(resolved.len())?;

    if !invalid.is_empty() {
        let configured = resources
            .iter()
            .map(|resource| resource.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        if resolved.is_empty() {
            bail!(
                "failed to resolve requested resources: {}. Configured resources: {}",
                invalid.join("; "),
                configured
            );
        }
        bail!(
            "resolved {} resources, but some references were invalid: {}. Configured resources: {}",
            resolved.len(),
            invalid.join("; "),
            configured
        );
    }

    if resolved.is_empty() {
        bail!("no resources are configured");
    }

    Ok(resolved)
}

fn resolve_reference(resources: &[ResourceConfig], reference: &str) -> Result<ResolvedResource> {
    if let Some(resource) = resources.iter().find(|resource| resource.name == reference) {
        return resource.to_resolved(false);
    }

    if reference.starts_with("https://") {
        return build_ephemeral_git_resource(reference);
    }

    Err(anyhow!("unknown resource"))
}

fn build_ephemeral_git_resource(reference: &str) -> Result<ResolvedResource> {
    let normalized = validation::normalize_git_url(reference)?;
    let mut hasher = Sha256::new();
    hasher.update(normalized.repo_url.as_bytes());
    if let Some(branch) = &normalized.branch {
        hasher.update(branch.as_bytes());
    }
    let hash = format!("{:x}", hasher.finalize());
    let repo_name = normalized
        .repo_url
        .rsplit('/')
        .next()
        .unwrap_or("repo")
        .trim_end_matches(".git");

    Ok(ResolvedResource {
        name: format!("anon-{repo_name}-{}", &hash[..12]),
        kind: ResourceKind::Git,
        git_url: Some(normalized.repo_url),
        local_path: None,
        branch: normalized.branch,
        search_paths: normalized.search_paths,
        notes: Some("Ephemeral git resource resolved from the ask request.".to_string()),
        ephemeral: true,
    })
}

pub fn ensure_local_resource_with_progress(
    resource: &ResolvedResource,
    data_dir: &Path,
    mut on_progress: Option<&mut ProgressCallback<'_>>,
) -> Result<PathBuf> {
    match resource.kind {
        ResourceKind::Git => ensure_git_resource(resource, data_dir, on_progress.as_mut()),
        ResourceKind::Local => ensure_local_directory(resource),
    }
}

#[cfg(test)]
fn resource_search_roots(resource: &ResolvedResource, root: &Path) -> Result<Vec<PathBuf>> {
    let root = root
        .canonicalize()
        .with_context(|| format!("Resolving resource root '{}'", root.display()))?;
    let mut resolved_paths = Vec::new();
    let search_paths = if resource.search_paths.is_empty() {
        vec![PathBuf::from(".")]
    } else {
        resource.search_paths.clone()
    };

    for search_path in search_paths {
        let scoped_path = root.join(&search_path);
        if !scoped_path.exists() {
            return Err(anyhow!(
                "Resource search path '{}' does not exist under '{}'",
                search_path.display(),
                root.display()
            ));
        }

        let scoped_path = scoped_path.canonicalize().with_context(|| {
            format!(
                "Resolving search path '{}' for resource '{}'",
                search_path.display(),
                resource.name
            )
        })?;
        if !scoped_path.starts_with(&root) {
            return Err(anyhow!(
                "Resource search path '{}' escapes resource root '{}'",
                search_path.display(),
                root.display()
            ));
        }
        resolved_paths.push(scoped_path);
    }

    Ok(resolved_paths)
}

impl ResourceConfig {
    pub fn new(
        name: Option<String>,
        git: Option<String>,
        local: Option<String>,
        branch: Option<String>,
        search_paths: Vec<String>,
        notes: Option<String>,
    ) -> Result<Self> {
        let mut normalized_search_paths = normalize_search_paths(search_paths)?;

        match (git, local) {
            (Some(git_url), None) => {
                let normalized_git = validation::normalize_git_url(&git_url)?;
                let name = resolve_git_resource_name(name, &normalized_git)?;
                if normalized_search_paths == vec![PathBuf::from(".")]
                    && normalized_git.search_paths != vec![PathBuf::from(".")]
                {
                    normalized_search_paths = normalized_git.search_paths.clone();
                }
                let branch = normalize_branch(branch, normalized_git.branch.as_deref())?;

                Ok(Self {
                    name,
                    kind: ResourceKind::Git,
                    git_url: Some(normalized_git.repo_url),
                    local_path: None,
                    branch,
                    search_paths: normalized_search_paths,
                    notes: normalize_notes(notes),
                })
            }
            (None, Some(local_path)) => {
                let name = name
                    .as_deref()
                    .ok_or_else(|| anyhow!("--name is required when adding a local resource"))
                    .and_then(validation::validate_resource_name)?;

                Ok(Self {
                    name,
                    kind: ResourceKind::Local,
                    git_url: None,
                    local_path: Some(validation::validate_local_path(Path::new(&local_path))?),
                    branch: None,
                    search_paths: normalized_search_paths,
                    notes: normalize_notes(notes),
                })
            }
            (Some(_), Some(_)) => Err(anyhow!("resource must be git OR local, not both")),
            (None, None) => Err(anyhow!(
                "must provide exactly one of --git or --local when adding a resource"
            )),
        }
    }

    pub fn normalize_in_place(&mut self) -> Result<()> {
        let had_search_paths = !self.search_paths.is_empty();
        self.name = validation::validate_resource_name(&self.name)?;
        if self.search_paths.is_empty() {
            self.search_paths = vec![PathBuf::from(".")];
        } else {
            self.search_paths = self
                .search_paths
                .iter()
                .map(|path| validation::validate_search_path(path))
                .collect::<Result<Vec<_>>>()?;
        }
        self.notes = normalize_notes(self.notes.take());

        match self.kind {
            ResourceKind::Git => {
                let git_url = self
                    .git_url
                    .as_deref()
                    .ok_or_else(|| anyhow!("missing git_url for git resource '{}'", self.name))?;
                let normalized_git = validation::normalize_git_url(git_url)?;
                self.git_url = Some(normalized_git.repo_url);
                self.local_path = None;
                if !had_search_paths && normalized_git.search_paths != vec![PathBuf::from(".")] {
                    self.search_paths = normalized_git.search_paths.clone();
                }
                self.branch = normalize_branch(self.branch.take(), normalized_git.branch.as_deref())?;
            }
            ResourceKind::Local => {
                let local_path = self
                    .local_path
                    .clone()
                    .ok_or_else(|| anyhow!("missing local_path for local resource '{}'", self.name))?;
                self.local_path = Some(validation::validate_local_path(&local_path)?);
                self.git_url = None;
                self.branch = None;
            }
        }

        Ok(())
    }

    pub fn to_resolved(&self, ephemeral: bool) -> Result<ResolvedResource> {
        let mut cloned = self.clone();
        cloned.normalize_in_place()?;
        Ok(ResolvedResource {
            name: cloned.name,
            kind: cloned.kind,
            git_url: cloned.git_url,
            local_path: cloned.local_path,
            branch: cloned.branch,
            search_paths: cloned.search_paths,
            notes: cloned.notes,
            ephemeral,
        })
    }

    pub fn search_paths_display(&self) -> String {
        self.search_paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    pub fn source_display(&self) -> String {
        match self.kind {
            ResourceKind::Git => self.git_url.clone().unwrap_or_else(|| "<missing git_url>".into()),
            ResourceKind::Local => self
                .local_path
                .as_ref()
                .map_or_else(|| "<missing local_path>".into(), |path| path.display().to_string()),
        }
    }
}

impl ResolvedResource {
    pub fn cache_key(&self) -> String {
        if self.ephemeral {
            let mut hasher = Sha256::new();
            hasher.update(self.name.as_bytes());
            if let Some(git_url) = &self.git_url {
                hasher.update(git_url.as_bytes());
            }
            if let Some(branch) = &self.branch {
                hasher.update(branch.as_bytes());
            }
            format!("ephemeral-{:x}", hasher.finalize())
        } else {
            self.name.clone()
        }
    }

    pub fn source_display(&self) -> String {
        match self.kind {
            ResourceKind::Git => self.git_url.clone().unwrap_or_else(|| "<missing git_url>".into()),
            ResourceKind::Local => self
                .local_path
                .as_ref()
                .map_or_else(|| "<missing local_path>".into(), |path| path.display().to_string()),
        }
    }
}

fn normalize_search_paths(search_paths: Vec<String>) -> Result<Vec<PathBuf>> {
    let raw_paths = if search_paths.is_empty() {
        vec![".".to_string()]
    } else {
        search_paths
    };

    raw_paths
        .into_iter()
        .map(|path| validation::validate_search_path(Path::new(&path)))
        .collect()
}

fn normalize_notes(notes: Option<String>) -> Option<String> {
    notes.and_then(|notes| {
        let trimmed = notes.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

fn normalize_branch(branch: Option<String>, fallback: Option<&str>) -> Result<Option<String>> {
    branch
        .as_deref()
        .map(str::trim)
        .filter(|branch| !branch.is_empty())
        .or(fallback)
        .map(validation::validate_branch_name)
        .transpose()
}

fn resolve_git_resource_name(name: Option<String>, normalized_git: &validation::NormalizedGitUrl) -> Result<String> {
    if let Some(name) = name.as_deref() {
        return validation::validate_resource_name(name);
    }

    let derived = normalized_git
        .search_paths
        .iter()
        .find_map(|path| {
            (path != Path::new(".")).then(|| {
                path.file_name()
                    .and_then(|segment| segment.to_str())
                    .map(str::to_string)
            })?
        })
        .or_else(|| {
            normalized_git
                .repo_url
                .rsplit('/')
                .next()
                .map(|segment| segment.trim_end_matches(".git").to_string())
        })
        .ok_or_else(|| anyhow!("could not derive a resource name from the git URL"))?;

    validation::validate_resource_name(&derived).map_err(|error| {
        anyhow!(
            "could not derive a valid resource name from the git URL (derived '{derived}'): {error}. Pass --name explicitly"
        )
    })
}

fn ensure_git_resource(
    resource: &ResolvedResource,
    data_dir: &Path,
    mut on_progress: Option<&mut &mut ProgressCallback<'_>>,
) -> Result<PathBuf> {
    let repo_dir = data_dir.join("resources").join(resource.cache_key());
    std::fs::create_dir_all(data_dir.join("resources"))?;

    if repo_dir.join(".git").exists() {
        if let Some(on_progress) = on_progress.as_mut() {
            (**on_progress)(&format!("Updating repository '{}'", resource.name))?;
        }
        git_fetch(&repo_dir, resource.branch.as_deref())?;
    } else {
        if repo_dir.exists() && repo_dir.read_dir()?.next().is_some() {
            return Err(anyhow!(
                "Refusing to clone into non-empty directory {}",
                repo_dir.display()
            ));
        }
        if let Some(on_progress) = on_progress.as_mut() {
            (**on_progress)(&format!("Cloning repository '{}'...", resource.name))?;
        }
        git_clone(
            resource
                .git_url
                .as_deref()
                .ok_or_else(|| anyhow!("Missing git URL for git resource '{}'", resource.name))?,
            resource.branch.as_deref(),
            &repo_dir,
        )?;
    }

    Ok(repo_dir)
}

fn ensure_local_directory(resource: &ResolvedResource) -> Result<PathBuf> {
    let path = resource
        .local_path
        .as_ref()
        .ok_or_else(|| anyhow!("Missing local_path for local resource '{}'", resource.name))?;

    if !path.exists() {
        return Err(anyhow!("Local resource path '{}' does not exist", path.display()));
    }
    if !path.is_dir() {
        return Err(anyhow!("Local resource path '{}' is not a directory", path.display()));
    }
    Ok(path.to_path_buf())
}

fn git_clone(url: &str, branch: Option<&str>, dest: &Path) -> Result<()> {
    let mut args = vec!["clone".to_string(), "--depth".to_string(), "1".to_string()];
    if let Some(branch) = branch {
        args.push("--branch".to_string());
        args.push(branch.to_string());
        args.push("--single-branch".to_string());
    }
    args.push(url.to_string());
    args.push(dest.to_string_lossy().to_string());
    run_git(args)?;
    Ok(())
}

fn git_fetch(repo_dir: &Path, branch: Option<&str>) -> Result<()> {
    let repo_dir = repo_dir.to_string_lossy().to_string();
    if let Some(branch) = branch {
        run_git([
            "-C".to_string(),
            repo_dir.clone(),
            "fetch".to_string(),
            "--depth".to_string(),
            "1".to_string(),
            "origin".to_string(),
            branch.to_string(),
        ])?;
        run_git([
            "-C".to_string(),
            repo_dir.clone(),
            "checkout".to_string(),
            branch.to_string(),
        ])?;
        run_git([
            "-C".to_string(),
            repo_dir,
            "reset".to_string(),
            "--hard".to_string(),
            format!("origin/{branch}"),
        ])?;
    } else {
        run_git([
            "-C".to_string(),
            repo_dir,
            "pull".to_string(),
            "--ff-only".to_string(),
            "--depth".to_string(),
            "1".to_string(),
        ])?;
    }
    Ok(())
}

fn run_git(args: impl IntoIterator<Item = String>) -> Result<()> {
    let args: Vec<String> = args.into_iter().collect();
    let output = Command::new("git")
        .args(&args)
        .output()
        .with_context(|| format!("Running git command: git {}", args.join(" ")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let mut message = format!("Git command failed: git {}", args.join(" "));
        if !stderr.is_empty() {
            message.push_str("\nstderr:\n");
            message.push_str(&stderr);
        }
        if !stdout.is_empty() {
            message.push_str("\nstdout:\n");
            message.push_str(&stdout);
        }
        return Err(anyhow!(message));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use super::{ResourceConfig, ResourceKind, resolve_resource_references, resource_search_roots};
    use crate::temp_paths::create_test_dir;

    #[test]
    fn search_roots_return_canonical_paths_within_root() {
        let root = create_test_dir("resources-valid-scope");
        let nested = root.join("docs").join("guide");
        fs::create_dir_all(&nested).expect("create nested scope");

        let resource = ResourceConfig {
            name: "docs".to_string(),
            kind: ResourceKind::Local,
            git_url: None,
            local_path: Some(root.clone()),
            branch: None,
            search_paths: vec![PathBuf::from("docs/guide")],
            notes: None,
        }
        .to_resolved(false)
        .expect("resource should normalize");

        let scoped_paths = resource_search_roots(&resource, &root).expect("scope should resolve");

        assert_eq!(
            scoped_paths,
            vec![nested.canonicalize().expect("canonical nested path")]
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn search_roots_reject_paths_that_escape_root() {
        let root = create_test_dir("resources-escaped-scope");
        let sibling = root
            .parent()
            .expect("temp dir has parent")
            .join("cntx-rs-resources-escaped-target");
        fs::create_dir_all(&sibling).expect("create sibling directory");

        let resource = ResourceConfig {
            name: "docs".to_string(),
            kind: ResourceKind::Local,
            git_url: None,
            local_path: Some(root.clone()),
            branch: None,
            search_paths: vec![PathBuf::from("../cntx-rs-resources-escaped-target")],
            notes: None,
        }
        .to_resolved(false)
        .expect_err("invalid search path should be rejected");

        assert!(resource.to_string().contains("must not contain '..'"));

        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(sibling);
    }

    #[test]
    fn resolves_mentions_and_ephemeral_git_resources() {
        let resources = vec![ResourceConfig {
            name: "svelte".to_string(),
            kind: ResourceKind::Git,
            git_url: Some("https://github.com/sveltejs/svelte.dev".to_string()),
            local_path: None,
            branch: Some("main".to_string()),
            search_paths: vec![PathBuf::from(".")],
            notes: None,
        }];

        let resolved = resolve_resource_references(
            &resources,
            &[String::from("https://github.com/sveltejs/kit")],
            "compare @svelte and kit docs",
        )
        .expect("resources should resolve");

        assert_eq!(resolved.len(), 2);
        assert!(resolved.iter().any(|resource| resource.name == "svelte"));
        assert!(resolved.iter().any(|resource| resource.ephemeral));
        assert!(resolved.iter().all(|resource| !resource.search_paths.is_empty()));
        assert_eq!(Path::new("."), resolved[0].search_paths[0]);
    }

    #[test]
    fn git_resources_can_infer_branch_and_search_path_from_github_tree_urls() {
        let resource = ResourceConfig::new(
            Some("tauri-runtime".to_string()),
            Some("https://github.com/tauri-apps/tauri/tree/dev/crates/tauri-runtime".to_string()),
            None,
            None,
            Vec::new(),
            None,
        )
        .expect("resource should normalize");

        assert_eq!(resource.git_url.as_deref(), Some("https://github.com/tauri-apps/tauri"));
        assert_eq!(resource.branch.as_deref(), Some("dev"));
        assert_eq!(resource.search_paths, vec![PathBuf::from("crates/tauri-runtime")]);
    }

    #[test]
    fn local_resources_still_require_name() {
        let error = ResourceConfig::new(None, None, Some("/tmp/docs".to_string()), None, Vec::new(), None)
            .expect_err("local resources should require an explicit name");

        assert!(error.to_string().contains("--name is required"));
    }
}
