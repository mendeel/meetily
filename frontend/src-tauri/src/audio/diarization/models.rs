//! Downloadable diarization model bundle under `models/diarization/`.
//!
//! Bundle files (pyannote-rs release assets):
//! - `wespeaker_en_voxceleb_CAM++.onnx` (required — speaker embeddings)
//! - `segmentation-3.0.onnx` (downloaded with the bundle; not used for clustering yet)

use anyhow::{anyhow, Result};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

const SEGMENTATION_FILE: &str = "segmentation-3.0.onnx";
const EMBEDDING_FILE: &str = "wespeaker_en_voxceleb_CAM++.onnx";

const SEGMENTATION_URL: &str =
    "https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/segmentation-3.0.onnx";
const EMBEDDING_URL: &str =
    "https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/wespeaker_en_voxceleb_CAM++.onnx";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DiarizationModelStatus {
    Available,
    Missing,
    Downloading { progress: u8 },
    Error { message: String },
}

/// Manages the on-disk diarization model bundle.
pub struct DiarizationModelManager {
    models_dir: PathBuf,
    status: Arc<RwLock<DiarizationModelStatus>>,
}

impl DiarizationModelManager {
    pub fn new_with_models_dir(models_dir: Option<PathBuf>) -> Result<Self> {
        let models_dir = if let Some(dir) = models_dir {
            dir.join("diarization")
        } else if cfg!(debug_assertions) {
            std::env::current_dir()?.join("models").join("diarization")
        } else {
            dirs::data_dir()
                .or_else(dirs::home_dir)
                .ok_or_else(|| anyhow!("Could not find system data directory"))?
                .join("Meetily")
                .join("models")
                .join("diarization")
        };

        if !models_dir.exists() {
            std::fs::create_dir_all(&models_dir)?;
        }

        info!(
            "DiarizationModelManager using directory: {}",
            models_dir.display()
        );

        let status = if bundle_present(&models_dir) {
            DiarizationModelStatus::Available
        } else {
            DiarizationModelStatus::Missing
        };

        Ok(Self {
            models_dir,
            status: Arc::new(RwLock::new(status)),
        })
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub fn embedding_model_path(&self) -> PathBuf {
        self.models_dir.join(EMBEDDING_FILE)
    }

    pub fn segmentation_model_path(&self) -> PathBuf {
        self.models_dir.join(SEGMENTATION_FILE)
    }

    pub fn is_available(&self) -> bool {
        bundle_present(&self.models_dir)
    }

    pub async fn status(&self) -> DiarizationModelStatus {
        if bundle_present(&self.models_dir) {
            DiarizationModelStatus::Available
        } else {
            self.status.read().await.clone()
        }
    }

    /// Download the model bundle. Safe to call when already complete (no-op).
    /// Downloads any missing files (WeSpeaker required; segmentation optional for now).
    pub async fn download_models(
        &self,
        progress: Option<Box<dyn Fn(u8) + Send + Sync>>,
    ) -> Result<()> {
        if embedding_present(&self.models_dir) && segmentation_present(&self.models_dir) {
            *self.status.write().await = DiarizationModelStatus::Available;
            return Ok(());
        }

        *self.status.write().await = DiarizationModelStatus::Downloading { progress: 0 };

        let files = [
            (SEGMENTATION_FILE, SEGMENTATION_URL),
            (EMBEDDING_FILE, EMBEDDING_URL),
        ];

        let client = reqwest::Client::new();
        let total = files.len() as f32;

        for (idx, (filename, url)) in files.iter().enumerate() {
            let dest = self.models_dir.join(filename);
            if dest.exists() {
                let pct = (((idx + 1) as f32 / total) * 100.0) as u8;
                if let Some(ref cb) = progress {
                    cb(pct);
                }
                continue;
            }

            info!("Downloading diarization model: {} → {}", url, dest.display());

            let response = client
                .get(*url)
                .send()
                .await
                .map_err(|e| anyhow!("Failed to download {}: {}", filename, e))?;

            if !response.status().is_success() {
                let msg = format!("HTTP {} for {}", response.status(), filename);
                *self.status.write().await =
                    DiarizationModelStatus::Error { message: msg.clone() };
                return Err(anyhow!(msg));
            }

            let bytes = response
                .bytes()
                .await
                .map_err(|e| anyhow!("Failed to read {}: {}", filename, e))?;

            let tmp = dest.with_extension("onnx.tmp");
            std::fs::write(&tmp, &bytes)
                .map_err(|e| anyhow!("Failed to write {}: {}", tmp.display(), e))?;
            std::fs::rename(&tmp, &dest)
                .map_err(|e| anyhow!("Failed to finalize {}: {}", dest.display(), e))?;

            let pct = (((idx + 1) as f32 / total) * 100.0) as u8;
            *self.status.write().await = DiarizationModelStatus::Downloading { progress: pct };
            if let Some(ref cb) = progress {
                cb(pct);
            }
        }

        if embedding_present(&self.models_dir) {
            *self.status.write().await = DiarizationModelStatus::Available;
            info!(
                "Diarization WeSpeaker model ready at {} (segmentation present={})",
                self.models_dir.display(),
                segmentation_present(&self.models_dir)
            );
            Ok(())
        } else {
            let msg = "Diarization download finished but WeSpeaker model missing".to_string();
            warn!("{}", msg);
            *self.status.write().await = DiarizationModelStatus::Error {
                message: msg.clone(),
            };
            Err(anyhow!(msg))
        }
    }
}

/// True when WeSpeaker is on disk — enough to run speaker embedding + clustering.
fn embedding_present(dir: &Path) -> bool {
    dir.join(EMBEDDING_FILE).exists()
}

fn segmentation_present(dir: &Path) -> bool {
    dir.join(SEGMENTATION_FILE).exists()
}

fn bundle_present(dir: &Path) -> bool {
    embedding_present(dir)
}
