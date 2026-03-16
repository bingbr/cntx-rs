#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    env, fs,
    path::{Path, PathBuf},
    process,
};

use anyhow::{Context, Result, anyhow, bail};

use crate::config::{self, Config};

const BINARY_NAME: &str = env!("CARGO_PKG_NAME");

pub struct InstallOutcome {
    pub target_path: PathBuf,
    pub already_current: bool,
    pub replaced_existing: bool,
    pub path_hint: Option<String>,
}

pub struct UninstallOutcome {
    pub target_path: PathBuf,
    pub removed_binary: bool,
    pub removed_config: bool,
    pub removed_data_dir: bool,
}

pub fn install(dir: Option<PathBuf>, force: bool) -> Result<InstallOutcome> {
    ensure_linux()?;

    let source = env::current_exe().context("Resolving the current cntx-rs executable")?;
    let install_dir = resolve_install_dir(dir)?;
    let target_path = install_dir.join(BINARY_NAME);

    if paths_match(&source, &target_path) {
        return Ok(InstallOutcome {
            target_path,
            already_current: true,
            replaced_existing: false,
            path_hint: None,
        });
    }

    let replaced_existing = install_binary(&source, &target_path, force)?;
    let path_hint = (!path_contains_dir(&install_dir)).then(|| {
        format!(
            "Add '{}' to your PATH:\n\nexport PATH=\"$PATH:{}\"",
            install_dir.display(),
            install_dir.display()
        )
    });

    Ok(InstallOutcome {
        target_path,
        already_current: false,
        replaced_existing,
        path_hint,
    })
}

pub fn uninstall(dir: Option<PathBuf>, config_path: &Path, purge: bool) -> Result<UninstallOutcome> {
    ensure_linux()?;

    let install_dir = resolve_install_dir(dir)?;
    let target_path = install_dir.join(BINARY_NAME);
    let removed_binary = remove_binary(&target_path)?;
    let (removed_config, removed_data_dir) = if purge {
        purge_state(config_path)?
    } else {
        (false, false)
    };

    Ok(UninstallOutcome {
        target_path,
        removed_binary,
        removed_config,
        removed_data_dir,
    })
}

fn ensure_linux() -> Result<()> {
    if cfg!(target_os = "linux") {
        Ok(())
    } else {
        bail!("install and uninstall are currently supported on Linux only");
    }
}

fn resolve_install_dir(dir: Option<PathBuf>) -> Result<PathBuf> {
    match dir {
        Some(path) => Ok(path),
        None => {
            let home = dirs::home_dir().ok_or_else(|| anyhow!("Could not determine the home directory"))?;
            Ok(home.join(".local").join("bin"))
        }
    }
}

fn install_binary(source: &Path, target: &Path, force: bool) -> Result<bool> {
    let parent = target
        .parent()
        .ok_or_else(|| anyhow!("Install target '{}' does not have a parent directory", target.display()))?;
    fs::create_dir_all(parent).with_context(|| format!("Creating install directory '{}'", parent.display()))?;

    let replaced_existing = if target.exists() {
        if target.is_dir() {
            bail!("Install target '{}' is a directory, not a file", target.display());
        }
        if !force {
            bail!(
                "'{}' already exists. Re-run with --force to replace it.",
                target.display()
            );
        }
        true
    } else {
        false
    };

    let temp_path = temporary_target_path(target)?;
    if temp_path.exists() {
        let _ = fs::remove_file(&temp_path);
    }

    let install_result = (|| -> Result<()> {
        fs::copy(source, &temp_path).with_context(|| {
            format!(
                "Copying '{}' to temporary install path '{}'",
                source.display(),
                temp_path.display()
            )
        })?;
        set_executable_permissions(&temp_path)?;
        fs::rename(&temp_path, target)
            .with_context(|| format!("Moving '{}' into '{}'", temp_path.display(), target.display()))?;
        Ok(())
    })();

    if install_result.is_err() && temp_path.exists() {
        let _ = fs::remove_file(&temp_path);
    }

    install_result?;
    Ok(replaced_existing)
}

fn temporary_target_path(target: &Path) -> Result<PathBuf> {
    let parent = target
        .parent()
        .ok_or_else(|| anyhow!("Install target '{}' does not have a parent directory", target.display()))?;
    Ok(parent.join(format!(".{}-install-{}.tmp", BINARY_NAME, process::id())))
}

fn remove_binary(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    if path.is_dir() {
        bail!("Installed binary path '{}' is a directory", path.display());
    }

    fs::remove_file(path).with_context(|| format!("Removing installed binary '{}'", path.display()))?;
    if let Some(parent) = path.parent() {
        remove_dir_if_empty(parent)?;
    }

    Ok(true)
}

fn purge_state(config_path: &Path) -> Result<(bool, bool)> {
    let default_config_path = config::default_config_path();
    let data_dir = if config_path.exists() {
        Some(
            Config::load(config_path)
                .map(|config| config.data_dir)
                .unwrap_or_else(|_| Config::default().data_dir),
        )
    } else if config_path == default_config_path {
        Some(Config::default().data_dir)
    } else {
        None
    };

    let removed_config = if config_path.exists() {
        fs::remove_file(config_path).with_context(|| format!("Removing config file '{}'", config_path.display()))?;
        true
    } else {
        false
    };

    if let Some(parent) = config_path.parent() {
        remove_dir_if_empty(parent)?;
    }

    let mut removed_data_dir = false;
    if let Some(data_dir) = data_dir
        && data_dir.exists()
    {
        fs::remove_dir_all(&data_dir).with_context(|| format!("Removing data directory '{}'", data_dir.display()))?;
        if let Some(parent) = data_dir.parent() {
            remove_dir_if_empty(parent)?;
        }
        removed_data_dir = true;
    }

    Ok((removed_config, removed_data_dir))
}

fn remove_dir_if_empty(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    match fs::remove_dir(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::DirectoryNotEmpty => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("Removing empty directory '{}'", path.display())),
    }
}

fn path_contains_dir(dir: &Path) -> bool {
    env::var_os("PATH")
        .map(|paths| env::split_paths(&paths).any(|entry| paths_match(&entry, dir)))
        .unwrap_or(false)
}

fn paths_match(left: &Path, right: &Path) -> bool {
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

fn set_executable_permissions(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        fs::set_permissions(path, fs::Permissions::from_mode(0o755))
            .with_context(|| format!("Setting executable permissions on '{}'", path.display()))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::{BINARY_NAME, install_binary, purge_state, remove_binary};
    use crate::temp_paths::create_test_dir;

    fn write_file(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent directory");
        }
        fs::write(path, contents).expect("write test file");
    }

    #[test]
    fn install_binary_copies_the_source_file() {
        let root = create_test_dir("install-copy");
        let source = root.join("source").join(BINARY_NAME);
        let target = root.join("bin").join(BINARY_NAME);
        write_file(&source, "binary-content");

        let replaced = install_binary(&source, &target, false).expect("install binary");

        assert!(!replaced);
        assert_eq!(
            fs::read_to_string(&target).expect("read installed binary"),
            "binary-content"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn install_binary_requires_force_to_replace_existing_target() {
        let root = create_test_dir("install-force");
        let source = root.join("source").join(BINARY_NAME);
        let target = root.join("bin").join(BINARY_NAME);
        write_file(&source, "new-binary");
        write_file(&target, "existing-binary");

        let error = install_binary(&source, &target, false).expect_err("install should fail");

        assert!(error.to_string().contains("--force"));
        assert_eq!(fs::read_to_string(&target).expect("read target"), "existing-binary");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn remove_binary_deletes_the_installed_file() {
        let root = create_test_dir("install-remove-binary");
        let target = root.join("bin").join(BINARY_NAME);
        write_file(&target, "installed-binary");

        let removed = remove_binary(&target).expect("remove binary");

        assert!(removed);
        assert!(!target.exists());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn purge_state_removes_config_and_data_directory() {
        let root = create_test_dir("install-purge-state");
        let config_path = root.join("config").join("config.toml");
        let data_dir = root.join("data");

        fs::create_dir_all(&data_dir).expect("create data directory");
        write_file(
            &config_path,
            &format!(
                "provider = \"anthropic\"\nmodel = \"claude-haiku-4-5\"\ndata_dir = \"{}\"\n",
                data_dir.display()
            ),
        );

        let (removed_config, removed_data_dir) = purge_state(&config_path).expect("purge state should succeed");

        assert!(removed_config);
        assert!(removed_data_dir);
        assert!(!config_path.exists());
        assert!(!data_dir.exists());

        let _ = fs::remove_dir_all(root);
    }
}
