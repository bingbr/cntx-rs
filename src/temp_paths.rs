use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Result, anyhow};

pub fn unique_temp_path(prefix: &str) -> Result<PathBuf> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| anyhow!("System clock error: {error}"))?
        .as_nanos();
    Ok(std::env::temp_dir().join(format!("{prefix}-{nanos}-{}", std::process::id())))
}

#[cfg(test)]
pub fn create_test_dir(label: &str) -> PathBuf {
    let path = unique_temp_path(&format!("cntx-rs-{label}")).expect("build unique test path");
    std::fs::create_dir_all(&path).expect("create test directory");
    path
}

#[cfg(test)]
pub fn unique_test_file(label: &str, extension: &str) -> PathBuf {
    let mut path = unique_temp_path(&format!("cntx-rs-{label}")).expect("build unique test path");
    path.set_extension(extension);
    path
}
