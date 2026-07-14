use crate::qwen_engine::catalog;
use crate::qwen_engine::model::QwenModel;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::RwLock;
use tokio::time::timeout;

/// Model status for Qwen models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Available,
    Missing,
    Downloading { progress: u8 },
    Error(String),
    Corrupted { file_size: u64, expected_min_size: u64 },
}

/// Detailed download progress info (MB-based with speed)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub downloaded_mb: f64,
    pub total_mb: f64,
    pub speed_mbps: f64,
    pub percent: u8,
}

impl DownloadProgress {
    pub fn new(downloaded: u64, total: u64, speed_mbps: f64) -> Self {
        let percent = if total > 0 {
            ((downloaded as f64 / total as f64) * 100.0).min(100.0) as u8
        } else {
            0
        };
        Self {
            downloaded_bytes: downloaded,
            total_bytes: total,
            downloaded_mb: downloaded as f64 / (1024.0 * 1024.0),
            total_mb: total as f64 / (1024.0 * 1024.0),
            speed_mbps,
            percent,
        }
    }
}

/// Optional quantization tag for UI parity with other engines.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantizationType {
    Int4,
}

impl Default for QuantizationType {
    fn default() -> Self {
        QuantizationType::Int4
    }
}

/// Information about a Qwen model pack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub size_mb: u32,
    pub quantization: QuantizationType,
    pub speed: String,
    pub status: ModelStatus,
    pub description: String,
}

#[derive(Debug)]
pub enum QwenEngineError {
    ModelNotLoaded,
    ModelNotFound(String),
    TranscriptionFailed(String),
    DownloadFailed(String),
    IoError(std::io::Error),
    Other(String),
}

impl std::fmt::Display for QwenEngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QwenEngineError::ModelNotLoaded => write!(f, "No Qwen model loaded"),
            QwenEngineError::ModelNotFound(name) => write!(f, "Model '{}' not found", name),
            QwenEngineError::TranscriptionFailed(err) => {
                write!(f, "Transcription failed: {}", err)
            }
            QwenEngineError::DownloadFailed(err) => write!(f, "Download failed: {}", err),
            QwenEngineError::IoError(err) => write!(f, "IO error: {}", err),
            QwenEngineError::Other(err) => write!(f, "Error: {}", err),
        }
    }
}

impl std::error::Error for QwenEngineError {}

impl From<std::io::Error> for QwenEngineError {
    fn from(err: std::io::Error) -> Self {
        QwenEngineError::IoError(err)
    }
}

/// Approximate file sizes for weighted download progress (HF pack inspection).
fn approximate_file_sizes() -> HashMap<&'static str, u64> {
    [
        ("encoder.int4.onnx", 745_762_694u64),
        ("decoder_init.int4.onnx", 355_390u64),
        ("decoder_step.int4.onnx", 354_888u64),
        ("decoder_weights.int4.data", 962_460_672u64),
        ("embed_tokens.bin", 311_164_928u64),
        ("config.json", 1_250u64),
        ("tokenizer.json", 11_429_377u64),
    ]
    .into_iter()
    .collect()
}

fn description_for(id: &str) -> (&'static str, &'static str) {
    if id.contains("1.7") {
        ("Higher accuracy", "Higher accuracy multilingual ASR (1.7B INT4)")
    } else {
        ("Fast multilingual", "Fast multilingual ASR (0.6B INT4)")
    }
}

pub struct QwenEngine {
    models_dir: PathBuf,
    current_model: Arc<RwLock<Option<QwenModel>>>,
    current_model_name: Arc<RwLock<Option<String>>>,
    pub(crate) available_models: Arc<RwLock<HashMap<String, ModelInfo>>>,
    cancel_download_flag: Arc<RwLock<Option<String>>>,
    pub(crate) active_downloads: Arc<RwLock<HashSet<String>>>,
}

impl QwenEngine {
    /// Create a new Qwen engine. `models_dir` is used as-is (no `qwen` subdirectory);
    /// packs live at `{models_dir}/qwen3-asr-0.6b-int4/`, etc.
    pub fn new_with_models_dir(models_dir: Option<PathBuf>) -> Result<Self> {
        let models_dir = if let Some(dir) = models_dir {
            dir
        } else {
            let current_dir = std::env::current_dir()
                .map_err(|e| anyhow!("Failed to get current directory: {}", e))?;

            if cfg!(debug_assertions) {
                current_dir.join("models")
            } else {
                dirs::data_dir()
                    .or_else(|| dirs::home_dir())
                    .ok_or_else(|| anyhow!("Could not find system data directory"))?
                    .join("Meetily")
                    .join("models")
            }
        };

        log::info!("QwenEngine using models directory: {}", models_dir.display());

        if !models_dir.exists() {
            std::fs::create_dir_all(&models_dir)?;
        }

        Ok(Self {
            models_dir,
            current_model: Arc::new(RwLock::new(None)),
            current_model_name: Arc::new(RwLock::new(None)),
            available_models: Arc::new(RwLock::new(HashMap::new())),
            cancel_download_flag: Arc::new(RwLock::new(None)),
            active_downloads: Arc::new(RwLock::new(HashSet::new())),
        })
    }

    /// Discover catalogued Qwen model packs and their on-disk status.
    pub async fn discover_models(&self) -> Result<Vec<ModelInfo>> {
        let models_dir = &self.models_dir;
        let mut models = Vec::new();
        let active_downloads = self.active_downloads.read().await;

        for spec in catalog::all_specs() {
            let model_path = models_dir.join(spec.id);
            let (speed, description) = description_for(spec.id);

            let status = if active_downloads.contains(spec.id) {
                ModelStatus::Downloading { progress: 0 }
            } else if catalog::pack_is_complete(&model_path, spec.id) {
                ModelStatus::Available
            } else {
                ModelStatus::Missing
            };

            models.push(ModelInfo {
                name: spec.id.to_string(),
                path: model_path,
                size_mb: spec.size_mb,
                quantization: QuantizationType::Int4,
                speed: speed.to_string(),
                status,
                description: description.to_string(),
            });
        }

        let mut available_models = self.available_models.write().await;
        available_models.clear();
        for model in &models {
            available_models.insert(model.name.clone(), model.clone());
        }

        Ok(models)
    }

    async fn validate_model_directory(&self, model_dir: &PathBuf, model_id: &str) -> Result<()> {
        if !catalog::pack_is_complete(model_dir, model_id) {
            return Err(anyhow!("pack incomplete for {}", model_id));
        }

        let sizes = approximate_file_sizes();
        let Some(files) = catalog::required_files(model_id) else {
            return Err(anyhow!("unknown model id {}", model_id));
        };

        for filename in files {
            let file_path = model_dir.join(filename);
            let metadata = std::fs::metadata(&file_path)
                .map_err(|e| anyhow!("Failed to read {} metadata: {}", filename, e))?;
            let actual_size = metadata.len();
            let expected = sizes.get(*filename).copied().unwrap_or(0);
            let min_size = (expected as f64 * 0.90) as u64;
            if expected > 0 && actual_size < min_size {
                return Err(anyhow!(
                    "{} is incomplete: {} bytes (expected at least {} bytes)",
                    filename,
                    actual_size,
                    min_size
                ));
            }
        }
        Ok(())
    }

    async fn clean_incomplete_model_directory(
        &self,
        model_dir: &PathBuf,
        model_id: &str,
    ) -> Result<()> {
        if !model_dir.exists() {
            return Ok(());
        }

        match self.validate_model_directory(model_dir, model_id).await {
            Ok(_) => {
                log::info!("Qwen model directory is valid, no cleanup needed");
                Ok(())
            }
            Err(validation_error) => {
                log::warn!(
                    "Qwen model directory exists but is invalid: {}. Cleaning up...",
                    validation_error
                );

                let mut entries = fs::read_dir(model_dir)
                    .await
                    .map_err(|e| anyhow!("Failed to read model directory: {}", e))?;

                let mut removed_count = 0;
                while let Some(entry) = entries
                    .next_entry()
                    .await
                    .map_err(|e| anyhow!("Failed to read directory entry: {}", e))?
                {
                    let path = entry.path();
                    if path.is_file() {
                        match fs::remove_file(&path).await {
                            Ok(_) => {
                                log::info!("Removed incomplete file: {:?}", path.file_name());
                                removed_count += 1;
                            }
                            Err(e) => {
                                log::warn!("Failed to remove file {:?}: {}", path, e);
                            }
                        }
                    }
                }

                log::info!(
                    "Cleaned {} incomplete files from Qwen model directory",
                    removed_count
                );
                Ok(())
            }
        }
    }

    /// Load a Qwen model pack. Unloads any previously loaded model first.
    pub async fn load_model(&self, model_name: &str) -> Result<()> {
        if self.available_models.read().await.is_empty() {
            let _ = self.discover_models().await?;
        }

        let models = self.available_models.read().await;
        let model_info = models
            .get(model_name)
            .ok_or_else(|| anyhow!("Model {} not found", model_name))?
            .clone();
        drop(models);

        match model_info.status {
            ModelStatus::Available => {
                if let Some(current_model) = self.current_model_name.read().await.as_ref() {
                    if current_model == model_name {
                        log::info!(
                            "Qwen model {} is already loaded, skipping reload",
                            model_name
                        );
                        return Ok(());
                    }

                    log::info!(
                        "Unloading current Qwen model '{}' before loading '{}'",
                        current_model,
                        model_name
                    );
                    self.unload_model().await;
                }

                log::info!("Loading Qwen model: {}", model_name);

                let model = QwenModel::load(&model_info.path).map_err(|e| {
                    if model_name.contains("1.7") {
                        anyhow!(
                            "Failed to load Qwen model {}: {}. If this fails due to memory or ORT limits, try the 0.6B pack ({})",
                            model_name,
                            e,
                            catalog::DEFAULT_QWEN_MODEL
                        )
                    } else {
                        anyhow!("Failed to load Qwen model {}: {}", model_name, e)
                    }
                })?;

                *self.current_model.write().await = Some(model);
                *self.current_model_name.write().await = Some(model_name.to_string());

                log::info!("Successfully loaded Qwen model: {}", model_name);
                Ok(())
            }
            ModelStatus::Missing => Err(anyhow!("Qwen model {} is not downloaded", model_name)),
            ModelStatus::Downloading { .. } => {
                Err(anyhow!("Qwen model {} is currently downloading", model_name))
            }
            ModelStatus::Error(ref err) => {
                Err(anyhow!("Qwen model {} has error: {}", model_name, err))
            }
            ModelStatus::Corrupted { .. } => {
                Err(anyhow!(
                    "Qwen model {} is corrupted and cannot be loaded",
                    model_name
                ))
            }
        }
    }

    pub async fn unload_model(&self) -> bool {
        let mut model_guard = self.current_model.write().await;
        let unloaded = model_guard.take().is_some();
        if unloaded {
            log::info!("Qwen model unloaded");
        }

        let mut model_name_guard = self.current_model_name.write().await;
        model_name_guard.take();

        unloaded
    }

    pub async fn get_current_model(&self) -> Option<String> {
        self.current_model_name.read().await.clone()
    }

    pub async fn is_model_loaded(&self) -> bool {
        self.current_model.read().await.is_some()
    }

    pub async fn get_models_directory(&self) -> PathBuf {
        self.models_dir.clone()
    }

    pub async fn transcribe_audio(&self, audio_data: Vec<f32>) -> Result<String> {
        self.transcribe_audio_with_language(audio_data, None).await
    }

    pub async fn transcribe_audio_with_language(
        &self,
        audio_data: Vec<f32>,
        language: Option<String>,
    ) -> Result<String> {
        let mut model_guard = self.current_model.write().await;
        let model = model_guard
            .as_mut()
            .ok_or_else(|| anyhow!("No Qwen model loaded. Please load a model first."))?;

        let duration_seconds = audio_data.len() as f64 / 16000.0;
        log::debug!(
            "Qwen transcribing {} samples ({:.1}s duration, language={:?})",
            audio_data.len(),
            duration_seconds,
            language
        );

        let lang_ref = language.as_deref();
        let text = model
            .transcribe(&audio_data, lang_ref)
            .map_err(|e| anyhow!("Qwen transcription failed: {}", e))?;

        log::debug!("Qwen transcription result: '{}'", text);
        Ok(text)
    }

    pub async fn delete_model(&self, model_name: &str) -> Result<String> {
        log::info!("Attempting to delete Qwen model: {}", model_name);

        let model_info = {
            let models = self.available_models.read().await;
            models.get(model_name).cloned()
        };

        let model_info =
            model_info.ok_or_else(|| anyhow!("Qwen model '{}' not found", model_name))?;

        match &model_info.status {
            ModelStatus::Corrupted { .. } | ModelStatus::Available | ModelStatus::Missing => {
                if model_info.path.exists() {
                    fs::remove_dir_all(&model_info.path).await.map_err(|e| {
                        anyhow!(
                            "Failed to delete directory '{}': {}",
                            model_info.path.display(),
                            e
                        )
                    })?;
                    log::info!(
                        "Successfully deleted Qwen model directory: {}",
                        model_info.path.display()
                    );
                }

                {
                    let mut models = self.available_models.write().await;
                    if let Some(model) = models.get_mut(model_name) {
                        model.status = ModelStatus::Missing;
                    }
                }

                Ok(format!("Successfully deleted Qwen model '{}'", model_name))
            }
            _ => Err(anyhow!(
                "Cannot delete Qwen model '{}' with status: {:?}",
                model_name,
                model_info.status
            )),
        }
    }

    pub async fn download_model(
        &self,
        model_name: &str,
        progress_callback: Option<Box<dyn Fn(u8) + Send>>,
    ) -> Result<()> {
        let detailed_callback: Option<Box<dyn Fn(DownloadProgress) + Send>> =
            progress_callback.map(|cb| {
                Box::new(move |p: DownloadProgress| cb(p.percent))
                    as Box<dyn Fn(DownloadProgress) + Send>
            });
        self.download_model_detailed(model_name, detailed_callback)
            .await
    }

    /// Download a Qwen ONNX pack from Hugging Face with resume + weighted progress.
    pub async fn download_model_detailed(
        &self,
        model_name: &str,
        progress_callback: Option<Box<dyn Fn(DownloadProgress) + Send>>,
    ) -> Result<()> {
        log::info!("Starting download for Qwen model: {}", model_name);

        {
            let active = self.active_downloads.read().await;
            if active.contains(model_name) {
                log::warn!("Download already in progress for Qwen model: {}", model_name);
                return Err(anyhow!(
                    "Download already in progress for model: {}",
                    model_name
                ));
            }
        }

        {
            let mut active = self.active_downloads.write().await;
            active.insert(model_name.to_string());
        }

        {
            let mut cancel_flag = self.cancel_download_flag.write().await;
            *cancel_flag = None;
        }

        if self.available_models.read().await.get(model_name).is_none() {
            let _ = self.discover_models().await;
        }

        let model_info = {
            let models = self.available_models.read().await;
            match models.get(model_name).cloned() {
                Some(info) => info,
                None => {
                    let mut active = self.active_downloads.write().await;
                    active.remove(model_name);
                    return Err(anyhow!("Model {} not found", model_name));
                }
            }
        };

        let spec = match catalog::model_spec(model_name) {
            Some(s) => s,
            None => {
                let mut active = self.active_downloads.write().await;
                active.remove(model_name);
                return Err(anyhow!("No catalog spec for model {}", model_name));
            }
        };

        {
            let mut models = self.available_models.write().await;
            if let Some(model) = models.get_mut(model_name) {
                model.status = ModelStatus::Downloading { progress: 0 };
            }
        }

        let files_to_download: Vec<&str> = spec.files.to_vec();
        let model_dir = &model_info.path;
        if !model_dir.exists() {
            if let Err(e) = fs::create_dir_all(model_dir).await {
                let mut active = self.active_downloads.write().await;
                active.remove(model_name);
                return Err(anyhow!("Failed to create model directory: {}", e));
            }
        }

        log::info!("Checking for incomplete model files to clean up...");
        if let Err(e) = self
            .clean_incomplete_model_directory(model_dir, model_name)
            .await
        {
            log::warn!("Failed to clean incomplete model directory: {}", e);
        }

        let client = reqwest::Client::builder()
            .tcp_nodelay(true)
            .pool_max_idle_per_host(1)
            .timeout(Duration::from_secs(3600))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

        let total_files = files_to_download.len();
        let file_sizes = approximate_file_sizes();

        let total_size_bytes: u64 = files_to_download
            .iter()
            .filter_map(|f| file_sizes.get(*f))
            .copied()
            .sum();

        let mut already_downloaded: u64 = 0;
        for filename in &files_to_download {
            let file_path = model_dir.join(filename);
            if file_path.exists() {
                if let Ok(metadata) = fs::metadata(&file_path).await {
                    let file_size = metadata.len();
                    let expected_size = file_sizes.get(*filename).copied().unwrap_or(0);
                    already_downloaded += file_size.min(expected_size);
                }
            }
        }

        let mut total_downloaded: u64 = already_downloaded;
        let download_start_time = Instant::now();
        let mut last_report_time = Instant::now();
        let mut bytes_since_last_report: u64 = 0;
        let mut last_reported_progress: u8 = 0;

        log::info!(
            "Starting weighted download for {} files, total size: {:.2} MB (already downloaded: {:.2} MB)",
            total_files,
            total_size_bytes as f64 / 1_048_576.0,
            already_downloaded as f64 / 1_048_576.0
        );

        for (index, filename) in files_to_download.iter().enumerate() {
            let file_url = catalog::hf_file_url(spec.hf_repo, filename);
            let file_path = model_dir.join(filename);

            let existing_size: u64 = if file_path.exists() {
                fs::metadata(&file_path)
                    .await
                    .map(|m| m.len())
                    .unwrap_or(0)
            } else {
                0
            };

            let expected_size = file_sizes.get(*filename).copied().unwrap_or(0);
            let size_tolerance = (expected_size as f64 * 0.99) as u64;
            if existing_size >= size_tolerance && expected_size > 0 {
                log::info!(
                    "Skipping complete file: {} ({:.2} MB, expected: {:.2} MB)",
                    filename,
                    existing_size as f64 / 1_048_576.0,
                    expected_size as f64 / 1_048_576.0
                );
                continue;
            }

            log::info!(
                "Downloading file {}/{}: {} (resuming from {} bytes)",
                index + 1,
                total_files,
                filename,
                existing_size
            );

            let mut request = client.get(&file_url);
            if existing_size > 0 {
                request = request.header("Range", format!("bytes={}-", existing_size));
                log::info!("Resuming download from byte {}", existing_size);
            }

            let mut response = request
                .send()
                .await
                .map_err(|e| anyhow!("Failed to start download for {}: {}", filename, e))?;

            let (file_total_size, resuming) = if response.status()
                == reqwest::StatusCode::PARTIAL_CONTENT
            {
                let remaining = response.content_length().unwrap_or(0);
                log::info!("Server supports resume, remaining: {} bytes", remaining);
                (existing_size + remaining, true)
            } else if response.status().is_success() {
                if existing_size > 0 {
                    log::warn!(
                        "Server doesn't support resume for {}, starting fresh download",
                        filename
                    );
                }
                (response.content_length().unwrap_or(0), false)
            } else if response.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
                log::warn!("Server returned 416 Range Not Satisfiable for {}", filename);
                let size_tolerance = (expected_size as f64 * 0.99) as u64;
                if existing_size >= size_tolerance && expected_size > 0 {
                    log::info!(
                        "File {} complete ({} bytes). Skipping.",
                        filename,
                        existing_size
                    );
                    continue;
                } else {
                    log::warn!(
                        "File {} incomplete ({}/{} bytes). Deleting and retrying.",
                        filename,
                        existing_size,
                        expected_size
                    );

                    if let Err(e) = fs::remove_file(&file_path).await {
                        let mut active = self.active_downloads.write().await;
                        active.remove(model_name);
                        return Err(anyhow!(
                            "Failed to delete incomplete file {}: {}",
                            filename,
                            e
                        ));
                    }

                    log::info!("Retrying {} without resume", filename);
                    response = client
                        .get(&file_url)
                        .send()
                        .await
                        .map_err(|e| anyhow!("Retry failed for {}: {}", filename, e))?;

                    if !response.status().is_success() {
                        let mut active = self.active_downloads.write().await;
                        active.remove(model_name);
                        return Err(anyhow!(
                            "Retry failed for {} with status: {}",
                            filename,
                            response.status()
                        ));
                    }

                    (response.content_length().unwrap_or(0), false)
                }
            } else {
                let mut active = self.active_downloads.write().await;
                active.remove(model_name);
                return Err(anyhow!(
                    "Download failed for {} with status: {}",
                    filename,
                    response.status()
                ));
            };

            let file = if resuming {
                fs::OpenOptions::new()
                    .append(true)
                    .open(&file_path)
                    .await
                    .map_err(|e| anyhow!("Failed to open file for resume {}: {}", filename, e))?
            } else {
                fs::File::create(&file_path)
                    .await
                    .map_err(|e| anyhow!("Failed to create file {}: {}", filename, e))?
            };

            let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);

            use futures_util::StreamExt;
            let mut stream = response.bytes_stream();
            let mut file_downloaded = if resuming { existing_size } else { 0u64 };

            loop {
                {
                    let cancel_flag = self.cancel_download_flag.read().await;
                    if cancel_flag.as_ref() == Some(&model_name.to_string()) {
                        log::info!("Download cancelled for {}", model_name);
                        let _ = writer.flush().await;
                        drop(writer);
                        let mut active = self.active_downloads.write().await;
                        active.remove(model_name);
                        return Err(anyhow!("Download cancelled by user"));
                    }
                }

                let next_result = timeout(Duration::from_secs(30), stream.next()).await;

                let chunk = match next_result {
                    Err(_) => {
                        log::warn!(
                            "Download timeout for {}: no data received for 30 seconds",
                            model_name
                        );
                        let _ = writer.flush().await;

                        {
                            let mut active = self.active_downloads.write().await;
                            active.remove(model_name);
                        }
                        {
                            let mut models = self.available_models.write().await;
                            if let Some(model) = models.get_mut(model_name) {
                                model.status = ModelStatus::Missing;
                            }
                        }

                        return Err(anyhow!(
                            "Download timeout - No data received for 30 seconds"
                        ));
                    }
                    Ok(None) => break,
                    Ok(Some(chunk_result)) => match chunk_result {
                        Ok(c) => c,
                        Err(e) => {
                            log::error!("Download error for {}: {:?}", model_name, e);
                            let _ = writer.flush().await;

                            {
                                let mut active = self.active_downloads.write().await;
                                active.remove(model_name);
                            }
                            {
                                let mut models = self.available_models.write().await;
                                if let Some(model) = models.get_mut(model_name) {
                                    model.status = ModelStatus::Missing;
                                }
                            }

                            let error_msg = if e.is_timeout() {
                                "Connection timeout - Check your internet"
                            } else if e.is_connect() {
                                "Connection failed - Check your internet"
                            } else if e.is_body() {
                                "Stream interrupted - Network unstable"
                            } else {
                                "Download error"
                            };

                            return Err(anyhow!("{}: {}", error_msg, e));
                        }
                    },
                };

                if let Err(e) = writer.write_all(&chunk).await {
                    {
                        let mut active = self.active_downloads.write().await;
                        active.remove(model_name);
                    }
                    {
                        let mut models = self.available_models.write().await;
                        if let Some(model) = models.get_mut(model_name) {
                            model.status = ModelStatus::Missing;
                        }
                    }
                    return Err(anyhow!("Failed to write chunk to file: {}", e));
                }

                let chunk_len = chunk.len() as u64;
                file_downloaded += chunk_len;
                total_downloaded += chunk_len;
                bytes_since_last_report += chunk_len;

                let overall_progress = if total_size_bytes > 0 {
                    ((total_downloaded as f64 / total_size_bytes as f64) * 100.0).min(99.0) as u8
                } else {
                    ((index as f64
                        + (file_downloaded as f64 / file_total_size.max(1) as f64))
                        / total_files as f64
                        * 100.0) as u8
                };

                let elapsed_since_report = last_report_time.elapsed();
                let progress_changed = overall_progress > last_reported_progress;
                let time_threshold = elapsed_since_report >= Duration::from_millis(500);
                let is_complete = file_downloaded >= file_total_size;
                let should_report = progress_changed || time_threshold || is_complete;

                if should_report {
                    let speed_mbps = if elapsed_since_report.as_secs_f64() >= 0.1 {
                        (bytes_since_last_report as f64 / (1024.0 * 1024.0))
                            / elapsed_since_report.as_secs_f64()
                    } else {
                        let total_elapsed = download_start_time.elapsed().as_secs_f64();
                        if total_elapsed > 0.0 {
                            ((total_downloaded - already_downloaded) as f64 / (1024.0 * 1024.0))
                                / total_elapsed
                        } else {
                            0.0
                        }
                    };

                    last_reported_progress = overall_progress;
                    last_report_time = Instant::now();
                    bytes_since_last_report = 0;

                    let progress =
                        DownloadProgress::new(total_downloaded, total_size_bytes, speed_mbps);
                    if let Some(ref callback) = progress_callback {
                        callback(progress);
                    }

                    {
                        let mut models = self.available_models.write().await;
                        if let Some(model) = models.get_mut(model_name) {
                            model.status = ModelStatus::Downloading {
                                progress: overall_progress,
                            };
                        }
                    }
                }
            }

            if let Err(e) = writer.flush().await {
                {
                    let mut active = self.active_downloads.write().await;
                    active.remove(model_name);
                }
                {
                    let mut models = self.available_models.write().await;
                    if let Some(model) = models.get_mut(model_name) {
                        model.status = ModelStatus::Missing;
                    }
                }
                return Err(anyhow!("Failed to flush file {}: {}", filename, e));
            }

            log::info!(
                "Completed download: {} ({:.2} MB, overall progress: {:.1}%)",
                filename,
                file_downloaded as f64 / 1_048_576.0,
                (total_downloaded as f64 / total_size_bytes as f64) * 100.0
            );
        }

        let total_elapsed = download_start_time.elapsed().as_secs_f64();
        let final_speed = if total_elapsed > 0.0 {
            ((total_downloaded - already_downloaded) as f64 / (1024.0 * 1024.0)) / total_elapsed
        } else {
            0.0
        };
        let final_progress = DownloadProgress::new(total_size_bytes, total_size_bytes, final_speed);
        if let Some(ref callback) = progress_callback {
            callback(final_progress);
        }

        {
            let mut models = self.available_models.write().await;
            if let Some(model) = models.get_mut(model_name) {
                model.status = ModelStatus::Available;
                model.path = model_dir.clone();
            }
        }

        {
            let mut active = self.active_downloads.write().await;
            active.remove(model_name);
        }

        {
            let mut cancel_flag = self.cancel_download_flag.write().await;
            if cancel_flag.as_ref() == Some(&model_name.to_string()) {
                *cancel_flag = None;
            }
        }

        log::info!("Download completed for Qwen model: {}", model_name);
        Ok(())
    }

    pub async fn cancel_download(&self, model_name: &str) -> Result<()> {
        log::info!("Cancelling download for Qwen model: {}", model_name);

        {
            let mut cancel_flag = self.cancel_download_flag.write().await;
            *cancel_flag = Some(model_name.to_string());
        }

        {
            let mut active = self.active_downloads.write().await;
            active.remove(model_name);
        }

        {
            let mut models = self.available_models.write().await;
            if let Some(model) = models.get_mut(model_name) {
                model.status = ModelStatus::Missing;
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let model_path = self.models_dir.join(model_name);
        if model_path.exists() {
            if let Err(e) = fs::remove_dir_all(&model_path).await {
                log::warn!("Failed to clean up cancelled download directory: {}", e);
            } else {
                log::info!(
                    "Cleaned up cancelled download directory: {}",
                    model_path.display()
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn discover_marks_missing_when_dir_empty() {
        let dir = tempfile::tempdir().unwrap();
        let engine = QwenEngine::new_with_models_dir(Some(dir.path().to_path_buf())).unwrap();
        let models = engine.discover_models().await.unwrap();
        assert_eq!(models.len(), 2);
        assert!(models
            .iter()
            .all(|m| matches!(m.status, ModelStatus::Missing)));
    }
}
