use crate::config::DEFAULT_NEMOTRON_MODEL;
use crate::nemotron_engine::language::map_language_preference;
use anyhow::{anyhow, Result};
use parakeet_rs::Nemotron;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::RwLock;
use tokio::time::timeout;

/// 560ms of audio at 16kHz — Nemotron's required streaming chunk size.
pub const NEMOTRON_CHUNK_SAMPLES: usize = 8960;

/// Quantization type for Nemotron models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantizationType {
    FP32,
    Int8,
}

impl Default for QuantizationType {
    fn default() -> Self {
        QuantizationType::Int8
    }
}

/// Model status for Nemotron models
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

/// Information about a Nemotron model
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
pub enum NemotronEngineError {
    ModelNotLoaded,
    ModelNotFound(String),
    TranscriptionFailed(String),
    DownloadFailed(String),
    IoError(std::io::Error),
    Other(String),
}

impl std::fmt::Display for NemotronEngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NemotronEngineError::ModelNotLoaded => write!(f, "No Nemotron model loaded"),
            NemotronEngineError::ModelNotFound(name) => write!(f, "Model '{}' not found", name),
            NemotronEngineError::TranscriptionFailed(err) => {
                write!(f, "Transcription failed: {}", err)
            }
            NemotronEngineError::DownloadFailed(err) => write!(f, "Download failed: {}", err),
            NemotronEngineError::IoError(err) => write!(f, "IO error: {}", err),
            NemotronEngineError::Other(err) => write!(f, "Error: {}", err),
        }
    }
}

impl std::error::Error for NemotronEngineError {}

impl From<std::io::Error> for NemotronEngineError {
    fn from(err: std::io::Error) -> Self {
        NemotronEngineError::IoError(err)
    }
}

/// Required ONNX / tokenizer files for a Nemotron model directory.
const REQUIRED_FILES: &[&str] = &[
    "encoder.onnx",
    "encoder.onnx.data",
    "decoder_joint.onnx",
    "tokenizer.model",
];

/// Approximate file sizes for progress / validation (from HF repo inspection).
fn expected_file_sizes() -> HashMap<&'static str, u64> {
    [
        ("encoder.onnx", 42_963_073u64),
        ("encoder.onnx.data", 614_649_600u64),
        ("decoder_joint.onnx", 24_483_962u64),
        ("tokenizer.model", 406_554u64),
        ("config.json", 2_970u64),
    ]
    .into_iter()
    .collect()
}

pub struct NemotronEngine {
    models_dir: PathBuf,
    current_model: Arc<RwLock<Option<Nemotron>>>,
    current_model_name: Arc<RwLock<Option<String>>>,
    pub(crate) available_models: Arc<RwLock<HashMap<String, ModelInfo>>>,
    cancel_download_flag: Arc<RwLock<Option<String>>>,
    pub(crate) active_downloads: Arc<RwLock<HashSet<String>>>,
}

impl NemotronEngine {
    /// Create a new Nemotron engine with optional custom models directory.
    /// Models are stored under `{models_dir}/nemotron/{model-name}/`.
    pub fn new_with_models_dir(models_dir: Option<PathBuf>) -> Result<Self> {
        let models_dir = if let Some(dir) = models_dir {
            dir.join("nemotron")
        } else {
            let current_dir = std::env::current_dir()
                .map_err(|e| anyhow!("Failed to get current directory: {}", e))?;

            if cfg!(debug_assertions) {
                current_dir.join("models").join("nemotron")
            } else {
                dirs::data_dir()
                    .or_else(|| dirs::home_dir())
                    .ok_or_else(|| anyhow!("Could not find system data directory"))?
                    .join("Meetily")
                    .join("models")
                    .join("nemotron")
            }
        };

        log::info!(
            "NemotronEngine using models directory: {}",
            models_dir.display()
        );

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

    /// Discover available Nemotron models
    pub async fn discover_models(&self) -> Result<Vec<ModelInfo>> {
        let models_dir = &self.models_dir;
        let mut models = Vec::new();

        let model_configs = [(
            DEFAULT_NEMOTRON_MODEL,
            650u32,
            QuantizationType::Int8,
            "Multilingual Streaming",
            "NVIDIA Nemotron 3.5 ASR Streaming 0.6B INT8 — multilingual cache-aware streaming",
        )];

        let active_downloads = self.active_downloads.read().await;

        for (name, size_mb, quantization, speed, description) in model_configs {
            let model_path = models_dir.join(name);

            let status = if active_downloads.contains(name) {
                ModelStatus::Downloading { progress: 0 }
            } else if model_path.exists() {
                let all_files_exist = REQUIRED_FILES
                    .iter()
                    .all(|file| model_path.join(file).exists());

                if all_files_exist {
                    match self.validate_model_directory(&model_path).await {
                        Ok(_) => ModelStatus::Available,
                        Err(_) => {
                            log::warn!("Nemotron model directory {} appears corrupted", name);
                            let mut total_size = 0u64;
                            for file in REQUIRED_FILES {
                                if let Ok(metadata) = std::fs::metadata(model_path.join(file)) {
                                    total_size += metadata.len();
                                }
                            }
                            ModelStatus::Corrupted {
                                file_size: total_size,
                                expected_min_size: (size_mb as u64) * 1024 * 1024,
                            }
                        }
                    }
                } else {
                    ModelStatus::Missing
                }
            } else {
                ModelStatus::Missing
            };

            models.push(ModelInfo {
                name: name.to_string(),
                path: model_path,
                size_mb,
                quantization,
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

    async fn validate_model_directory(&self, model_dir: &PathBuf) -> Result<()> {
        let sizes = expected_file_sizes();
        for filename in REQUIRED_FILES {
            let file_path = model_dir.join(filename);
            if !file_path.exists() {
                return Err(anyhow!("{} not found", filename));
            }

            let metadata = std::fs::metadata(&file_path)
                .map_err(|e| anyhow!("Failed to read {} metadata: {}", filename, e))?;
            let actual_size = metadata.len();
            let expected = sizes.get(*filename).copied().unwrap_or(0);
            // Allow 5% variance for size checks
            let min_size = (expected as f64 * 0.95) as u64;
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

    async fn clean_incomplete_model_directory(&self, model_dir: &PathBuf) -> Result<()> {
        if !model_dir.exists() {
            return Ok(());
        }

        match self.validate_model_directory(model_dir).await {
            Ok(_) => {
                log::info!("Nemotron model directory is valid, no cleanup needed");
                Ok(())
            }
            Err(validation_error) => {
                log::warn!(
                    "Nemotron model directory exists but is invalid: {}. Cleaning up...",
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
                    "Cleaned {} incomplete files from Nemotron model directory",
                    removed_count
                );
                Ok(())
            }
        }
    }

    /// Load a Nemotron model via `Nemotron::from_pretrained`
    pub async fn load_model(&self, model_name: &str) -> Result<()> {
        // Ensure catalog is populated
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
                            "Nemotron model {} is already loaded, skipping reload",
                            model_name
                        );
                        return Ok(());
                    }
                    log::info!(
                        "Unloading current Nemotron model '{}' before loading '{}'",
                        current_model,
                        model_name
                    );
                    self.unload_model().await;
                }

                log::info!("Loading Nemotron model: {}", model_name);
                let model_path = model_info.path.clone();

                // Blocking ONNX load — run off the async runtime
                let model = tokio::task::spawn_blocking(move || {
                    Nemotron::from_pretrained(&model_path, None)
                        .map_err(|e| anyhow!("Failed to load Nemotron model: {}", e))
                })
                .await
                .map_err(|e| anyhow!("Nemotron load task panicked: {}", e))??;

                *self.current_model.write().await = Some(model);
                *self.current_model_name.write().await = Some(model_name.to_string());

                log::info!("Successfully loaded Nemotron model: {}", model_name);
                Ok(())
            }
            ModelStatus::Missing => Err(anyhow!("Nemotron model {} is not downloaded", model_name)),
            ModelStatus::Downloading { .. } => {
                Err(anyhow!("Nemotron model {} is currently downloading", model_name))
            }
            ModelStatus::Error(ref err) => {
                Err(anyhow!("Nemotron model {} has error: {}", model_name, err))
            }
            ModelStatus::Corrupted { .. } => Err(anyhow!(
                "Nemotron model {} is corrupted and cannot be loaded",
                model_name
            )),
        }
    }

    pub async fn unload_model(&self) -> bool {
        let mut model_guard = self.current_model.write().await;
        let unloaded = model_guard.take().is_some();
        if unloaded {
            log::info!("Nemotron model unloaded");
        }
        self.current_model_name.write().await.take();
        unloaded
    }

    pub async fn get_current_model(&self) -> Option<String> {
        self.current_model_name.read().await.clone()
    }

    pub async fn is_model_loaded(&self) -> bool {
        self.current_model.read().await.is_some()
    }

    /// Apply Meetily language preference to the loaded model.
    pub async fn set_language(&self, pref: Option<&str>) -> Result<()> {
        let locale = map_language_preference(pref);
        let mut model_guard = self.current_model.write().await;
        let model = model_guard
            .as_mut()
            .ok_or_else(|| anyhow!("No Nemotron model loaded"))?;

        model
            .set_target_lang(&locale)
            .map_err(|e| anyhow!("Failed to set Nemotron language '{}': {}", locale, e))?;
        log::info!("Nemotron target language set to '{}'", locale);
        Ok(())
    }

    /// Hybrid streaming transcription of a VAD utterance.
    ///
    /// Streams audio in 560ms chunks through Nemotron's cache-aware API,
    /// flushes with silence chunks, returns the full transcript, then resets.
    /// Optional `on_partial` is called with accumulated text as it grows.
    pub async fn transcribe_utterance<F>(
        &self,
        audio_data: Vec<f32>,
        language_pref: Option<&str>,
        mut on_partial: Option<F>,
    ) -> Result<String>
    where
        F: FnMut(&str) + Send,
    {
        let mut model_guard = self.current_model.write().await;
        let model = model_guard
            .as_mut()
            .ok_or_else(|| anyhow!("No Nemotron model loaded. Please load a model first."))?;

        let locale = map_language_preference(language_pref);
        if let Err(e) = model.set_target_lang(&locale) {
            log::warn!(
                "Failed to set Nemotron language '{}': {} — continuing with current setting",
                locale,
                e
            );
        }

        // Reset state for a fresh utterance
        model.reset();

        let duration_seconds = audio_data.len() as f64 / 16000.0;
        log::debug!(
            "Nemotron transcribing {} samples ({:.1}s) in {}-sample chunks (lang={})",
            audio_data.len(),
            duration_seconds,
            NEMOTRON_CHUNK_SAMPLES,
            locale
        );

        let mut last_emitted_len = 0usize;

        for chunk in audio_data.chunks(NEMOTRON_CHUNK_SAMPLES) {
            let mut buf = chunk.to_vec();
            buf.resize(NEMOTRON_CHUNK_SAMPLES, 0.0);
            model
                .transcribe_chunk(&buf)
                .map_err(|e| anyhow!("Nemotron transcribe_chunk failed: {}", e))?;

            let full = model.get_transcript();
            if let Some(ref mut cb) = on_partial {
                if !full.is_empty() && full.len() > last_emitted_len {
                    cb(&full);
                    last_emitted_len = full.len();
                }
            }
        }

        // Flush encoder pipeline with silence chunks
        for _ in 0..3 {
            model
                .transcribe_chunk(&vec![0.0f32; NEMOTRON_CHUNK_SAMPLES])
                .map_err(|e| anyhow!("Nemotron flush chunk failed: {}", e))?;

            let full = model.get_transcript();
            if let Some(ref mut cb) = on_partial {
                if !full.is_empty() && full.len() > last_emitted_len {
                    cb(&full);
                    last_emitted_len = full.len();
                }
            }
        }

        let result = model.get_transcript();
        model.reset();

        log::debug!("Nemotron transcription result: '{}'", result);
        Ok(result)
    }

    /// Convenience wrapper without partial callback (import / retranscription / commands).
    pub async fn transcribe_audio(
        &self,
        audio_data: Vec<f32>,
        language_pref: Option<&str>,
    ) -> Result<String> {
        self.transcribe_utterance(audio_data, language_pref, None::<fn(&str)>)
            .await
    }

    pub async fn get_models_directory(&self) -> PathBuf {
        self.models_dir.clone()
    }

    pub async fn delete_model(&self, model_name: &str) -> Result<String> {
        log::info!("Attempting to delete Nemotron model: {}", model_name);

        let model_info = {
            let models = self.available_models.read().await;
            models.get(model_name).cloned()
        };

        let model_info =
            model_info.ok_or_else(|| anyhow!("Nemotron model '{}' not found", model_name))?;

        match &model_info.status {
            ModelStatus::Corrupted { .. } | ModelStatus::Available => {
                if model_info.path.exists() {
                    fs::remove_dir_all(&model_info.path).await.map_err(|e| {
                        anyhow!(
                            "Failed to delete directory '{}': {}",
                            model_info.path.display(),
                            e
                        )
                    })?;
                    log::info!(
                        "Successfully deleted Nemotron model directory: {}",
                        model_info.path.display()
                    );
                }

                // Unload if this was the loaded model
                if self.current_model_name.read().await.as_deref() == Some(model_name) {
                    self.unload_model().await;
                }

                {
                    let mut models = self.available_models.write().await;
                    if let Some(model) = models.get_mut(model_name) {
                        model.status = ModelStatus::Missing;
                    }
                }

                Ok(format!(
                    "Successfully deleted Nemotron model '{}'",
                    model_name
                ))
            }
            _ => Err(anyhow!(
                "Can only delete corrupted or available Nemotron models. Model '{}' has status: {:?}",
                model_name,
                model_info.status
            )),
        }
    }

    pub async fn cancel_download(&self, model_name: &str) -> Result<()> {
        let mut cancel_flag = self.cancel_download_flag.write().await;
        *cancel_flag = Some(model_name.to_string());
        log::info!("Cancellation requested for Nemotron download: {}", model_name);
        Ok(())
    }

    /// Download a Nemotron model with detailed progress (MB/speed/resume support)
    pub async fn download_model_detailed(
        &self,
        model_name: &str,
        progress_callback: Option<Box<dyn Fn(DownloadProgress) + Send>>,
    ) -> Result<()> {
        log::info!("Starting download for Nemotron model: {}", model_name);

        {
            let active = self.active_downloads.read().await;
            if active.contains(model_name) {
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

        let model_info = {
            let models = self.available_models.read().await;
            match models.get(model_name).cloned() {
                Some(info) => info,
                None => {
                    self.active_downloads.write().await.remove(model_name);
                    return Err(anyhow!("Model {} not found", model_name));
                }
            }
        };

        {
            let mut models = self.available_models.write().await;
            if let Some(model) = models.get_mut(model_name) {
                model.status = ModelStatus::Downloading { progress: 0 };
            }
        }

        let base_url = format!(
            "https://huggingface.co/smcleod/{}/resolve/main",
            model_name
        );

        let mut files_to_download: Vec<&str> = REQUIRED_FILES.to_vec();
        // Optional but useful
        files_to_download.push("config.json");

        let model_dir = &model_info.path;
        if !model_dir.exists() {
            if let Err(e) = fs::create_dir_all(model_dir).await {
                self.active_downloads.write().await.remove(model_name);
                return Err(anyhow!("Failed to create model directory: {}", e));
            }
        }

        if let Err(e) = self.clean_incomplete_model_directory(model_dir).await {
            log::warn!("Failed to clean incomplete Nemotron model directory: {}", e);
        }

        let client = reqwest::Client::builder()
            .tcp_nodelay(true)
            .pool_max_idle_per_host(1)
            .timeout(Duration::from_secs(3600))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

        let file_sizes = expected_file_sizes();
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
                    let expected_size = file_sizes.get(*filename).copied().unwrap_or(0);
                    already_downloaded += metadata.len().min(expected_size);
                }
            }
        }

        let mut total_downloaded: u64 = already_downloaded;
        let download_start_time = Instant::now();
        let mut last_report_time = Instant::now();
        let mut bytes_since_last_report: u64 = 0;
        let mut last_reported_progress: u8 = 0;
        let total_files = files_to_download.len();

        log::info!(
            "Starting Nemotron download for {} files, total size: {:.2} MB (already: {:.2} MB)",
            total_files,
            total_size_bytes as f64 / 1_048_576.0,
            already_downloaded as f64 / 1_048_576.0
        );

        for (index, filename) in files_to_download.iter().enumerate() {
            let file_url = format!("{}/{}", base_url, filename);
            let file_path = model_dir.join(filename);

            let existing_size: u64 = if file_path.exists() {
                fs::metadata(&file_path).await.map(|m| m.len()).unwrap_or(0)
            } else {
                0
            };

            let expected_size = file_sizes.get(*filename).copied().unwrap_or(0);
            let size_tolerance = (expected_size as f64 * 0.99) as u64;
            if existing_size >= size_tolerance && expected_size > 0 {
                log::info!("Skipping complete file: {}", filename);
                continue;
            }

            log::info!(
                "Downloading file {}/{}: {} (resume from {} bytes)",
                index + 1,
                total_files,
                filename,
                existing_size
            );

            let mut request = client.get(&file_url);
            if existing_size > 0 {
                request = request.header("Range", format!("bytes={}-", existing_size));
            }

            let mut response = request
                .send()
                .await
                .map_err(|e| anyhow!("Failed to start download for {}: {}", filename, e))?;

            let (file_total_size, resuming) = if response.status()
                == reqwest::StatusCode::PARTIAL_CONTENT
            {
                let remaining = response.content_length().unwrap_or(0);
                (existing_size + remaining, true)
            } else if response.status().is_success() {
                (response.content_length().unwrap_or(0), false)
            } else if response.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
                let size_tolerance = (expected_size as f64 * 0.99) as u64;
                if existing_size >= size_tolerance && expected_size > 0 {
                    continue;
                }
                if let Err(e) = fs::remove_file(&file_path).await {
                    self.active_downloads.write().await.remove(model_name);
                    return Err(anyhow!("Failed to delete incomplete file {}: {}", filename, e));
                }
                response = client
                    .get(&file_url)
                    .send()
                    .await
                    .map_err(|e| anyhow!("Retry failed for {}: {}", filename, e))?;
                if !response.status().is_success() {
                    self.active_downloads.write().await.remove(model_name);
                    return Err(anyhow!(
                        "Retry failed for {} with status: {}",
                        filename,
                        response.status()
                    ));
                }
                (response.content_length().unwrap_or(0), false)
            } else {
                self.active_downloads.write().await.remove(model_name);
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
                        let _ = writer.flush().await;
                        drop(writer);
                        self.active_downloads.write().await.remove(model_name);
                        return Err(anyhow!("Download cancelled by user"));
                    }
                }

                let next_result = timeout(Duration::from_secs(30), stream.next()).await;
                let chunk = match next_result {
                    Err(_) => {
                        let _ = writer.flush().await;
                        self.active_downloads.write().await.remove(model_name);
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
                            let _ = writer.flush().await;
                            self.active_downloads.write().await.remove(model_name);
                            {
                                let mut models = self.available_models.write().await;
                                if let Some(model) = models.get_mut(model_name) {
                                    model.status = ModelStatus::Missing;
                                }
                            }
                            return Err(anyhow!("Download error: {}", e));
                        }
                    },
                };

                writer
                    .write_all(&chunk)
                    .await
                    .map_err(|e| anyhow!("Failed to write chunk: {}", e))?;

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
                let should_report = overall_progress > last_reported_progress
                    || elapsed_since_report >= Duration::from_millis(500)
                    || file_downloaded >= file_total_size;

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

            writer
                .flush()
                .await
                .map_err(|e| anyhow!("Failed to flush file {}: {}", filename, e))?;

            log::info!(
                "Completed download: {} ({:.2} MB)",
                filename,
                file_downloaded as f64 / 1_048_576.0
            );
        }

        let total_elapsed = download_start_time.elapsed().as_secs_f64();
        let final_speed = if total_elapsed > 0.0 {
            ((total_downloaded - already_downloaded) as f64 / (1024.0 * 1024.0)) / total_elapsed
        } else {
            0.0
        };
        if let Some(ref callback) = progress_callback {
            callback(DownloadProgress::new(
                total_size_bytes,
                total_size_bytes,
                final_speed,
            ));
        }

        {
            let mut models = self.available_models.write().await;
            if let Some(model) = models.get_mut(model_name) {
                model.status = ModelStatus::Available;
                model.path = model_dir.clone();
            }
        }

        self.active_downloads.write().await.remove(model_name);
        {
            let mut cancel_flag = self.cancel_download_flag.write().await;
            *cancel_flag = None;
        }

        // Final validation
        self.validate_model_directory(model_dir).await?;

        log::info!("Nemotron model '{}' download complete", model_name);
        Ok(())
    }
}
