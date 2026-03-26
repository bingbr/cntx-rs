use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};

use super::types::{AgentWorkspace, WorkspaceResourceSummary};
use crate::{
    resources::{self, ResolvedResource},
    temp_paths,
};

type ProgressCallback<'a> = dyn FnMut(&str) -> Result<()> + 'a;

pub struct PreparedWorkspace {
    workspace: AgentWorkspace,
    _root_guard: WorkspaceRootGuard,
}

impl PreparedWorkspace {
    pub fn workspace(&self) -> &AgentWorkspace {
        &self.workspace
    }

    #[cfg(test)]
    pub fn root(&self) -> &Path {
        &self._root_guard.path
    }
}

struct WorkspaceRootGuard {
    path: PathBuf,
}

impl Drop for WorkspaceRootGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

pub fn prepare_workspace_from_resources(
    data_dir: &Path,
    resources: &[ResolvedResource],
    mut on_progress: Option<&mut ProgressCallback<'_>>,
) -> Result<PreparedWorkspace> {
    let root = temp_paths::unique_temp_path("cntx-rs-agentic")?;
    fs::create_dir_all(&root).with_context(|| format!("Creating workspace root '{}'", root.display()))?;
    let root_guard = WorkspaceRootGuard { path: root.clone() };

    let mut summaries = Vec::new();
    for resource in resources {
        if let Some(on_progress) = on_progress.as_mut() {
            (**on_progress)(&format!("Preparing resource '{}'", resource.name))?;
        }
        let source_root =
            resources::ensure_local_resource_with_progress(resource, data_dir, on_progress.as_deref_mut())?;
        let resource_mount = root.join(&resource.name);
        if let Some(on_progress) = on_progress.as_mut() {
            (**on_progress)(&format!(
                "Copying resource '{}' into the agent workspace",
                resource.name
            ))?;
        }
        copy_tree(&source_root, &resource_mount).with_context(|| {
            format!(
                "Copying resource '{}' into workspace '{}'",
                resource.name,
                resource_mount.display()
            )
        })?;

        summaries.push(WorkspaceResourceSummary {
            name: resource.name.clone(),
            kind: resource.kind.to_string(),
            source: resource.source_display(),
            branch: resource.branch.clone(),
            search_paths: resource
                .search_paths
                .iter()
                .map(|path| path.display().to_string())
                .collect(),
            notes: resource.notes.clone(),
            mount_path: resource.name.clone(),
            ephemeral: resource.ephemeral,
        });
    }

    Ok(PreparedWorkspace {
        workspace: AgentWorkspace {
            root,
            resources: summaries,
        },
        _root_guard: root_guard,
    })
}

fn copy_tree(source: &Path, destination: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(source).with_context(|| format!("Inspecting '{}'", source.display()))?;
    if metadata.file_type().is_symlink() {
        return Ok(());
    }

    if metadata.is_dir() {
        fs::create_dir_all(destination).with_context(|| format!("Creating directory '{}'", destination.display()))?;
        for entry in fs::read_dir(source).with_context(|| format!("Reading directory '{}'", source.display()))? {
            let entry = entry?;
            let child_source = entry.path();
            let child_destination = destination.join(entry.file_name());
            copy_tree(&child_source, &child_destination)?;
        }
        return Ok(());
    }

    if metadata.is_file() {
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).with_context(|| format!("Creating directory '{}'", parent.display()))?;
        }
        fs::copy(source, destination)
            .with_context(|| format!("Copying file '{}' to '{}'", source.display(), destination.display()))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::prepare_workspace_from_resources;
    use crate::{
        resources::{ResolvedResource, ResourceKind},
        temp_paths::create_test_dir,
    };

    #[test]
    fn prepared_workspace_cleans_up_on_drop() {
        let data_dir = create_test_dir("agentic-workspace-data");
        let repo_root = data_dir.join("local-repo");
        fs::create_dir_all(&repo_root).expect("create repo");
        fs::write(repo_root.join("README.md"), "hello").expect("write repo file");

        let prepared = prepare_workspace_from_resources(
            &data_dir,
            &[ResolvedResource {
                name: "repo".to_string(),
                kind: ResourceKind::Local,
                git_url: None,
                local_path: Some(repo_root.clone()),
                branch: None,
                search_paths: vec![PathBuf::from(".")],
                notes: None,
                ephemeral: false,
            }],
            None,
        )
        .expect("workspace should build");

        let workspace_root = prepared.root().to_path_buf();
        assert!(workspace_root.exists());
        assert!(workspace_root.join("repo/README.md").exists());

        drop(prepared);

        assert!(!workspace_root.exists());
        let _ = fs::remove_dir_all(data_dir);
    }
}
