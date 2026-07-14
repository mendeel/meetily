use crate::config::DEFAULT_QWEN_MODEL;
use crate::qwen_engine::{DownloadProgress, ModelInfo, ModelStatus, QwenEngine};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use tauri::{command, AppHandle, Emitter, Manager, Runtime};

/// Global Qwen engine
pub static QWEN_ENGINE: Mutex<Option<Arc<QwenEngine>>> = Mutex::new(None);

/// Global models directory path (set during app initialization)
static MODELS_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Initialize the models directory path using app_data_dir.
/// Packs live directly under `app_data/models/{pack-id}/`.
pub fn set_models_directory<R: Runtime>(app: &AppHandle<R>) {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .expect("Failed to get app data dir");

    let models_dir = app_data_dir.join("models");

    if !models_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(&models_dir) {
            log::error!("Failed to create models directory: {}", e);
            return;
        }
    }

    log::info!("Qwen models directory set to: {}", models_dir.display());

    let mut guard = MODELS_DIR.lock().unwrap();
    *guard = Some(models_dir);
}

fn get_models_directory() -> Option<PathBuf> {
    MODELS_DIR.lock().unwrap().clone()
}

#[command]
pub async fn qwen_init() -> Result<(), String> {
    let mut guard = QWEN_ENGINE.lock().unwrap();
    if guard.is_some() {
        return Ok(());
    }

    let models_dir = get_models_directory();
    let engine = QwenEngine::new_with_models_dir(models_dir)
        .map_err(|e| format!("Failed to initialize Qwen engine: {}", e))?;
    *guard = Some(Arc::new(engine));
    Ok(())
}

#[command]
pub async fn qwen_get_available_models() -> Result<Vec<ModelInfo>, String> {
    let engine = {
        let guard = QWEN_ENGINE.lock().unwrap();
        guard.as_ref().cloned()
    };

    if let Some(engine) = engine {
        engine
            .discover_models()
            .await
            .map_err(|e| format!("Failed to discover Qwen models: {}", e))
    } else {
        Err("Qwen engine not initialized".to_string())
    }
}

#[command]
pub async fn qwen_load_model<R: Runtime>(
    app_handle: AppHandle<R>,
    model_name: String,
) -> Result<(), String> {
    let engine = {
        let guard = QWEN_ENGINE.lock().unwrap();
        guard.as_ref().cloned()
    };

    if let Some(engine) = engine {
        if let Err(e) = app_handle.emit(
            "qwen-model-loading-started",
            serde_json::json!({
                "modelName": model_name
            }),
        ) {
            log::error!("Failed to emit qwen-model-loading-started event: {}", e);
        }

        let result = engine
            .load_model(&model_name)
            .await
            .map_err(|e| format!("Failed to load Qwen model: {}", e));

        if result.is_ok() {
            if let Err(e) = app_handle.emit(
                "qwen-model-loading-completed",
                serde_json::json!({
                    "modelName": model_name
                }),
            ) {
                log::error!("Failed to emit qwen-model-loading-completed event: {}", e);
            }
        } else if let Err(ref error) = result {
            if let Err(e) = app_handle.emit(
                "qwen-model-loading-failed",
                serde_json::json!({
                    "modelName": model_name,
                    "error": error
                }),
            ) {
                log::error!("Failed to emit qwen-model-loading-failed event: {}", e);
            }
        }

        result
    } else {
        Err("Qwen engine not initialized".to_string())
    }
}

#[command]
pub async fn qwen_get_current_model() -> Result<Option<String>, String> {
    let engine = {
        let guard = QWEN_ENGINE.lock().unwrap();
        guard.as_ref().cloned()
    };

    if let Some(engine) = engine {
        Ok(engine.get_current_model().await)
    } else {
        Err("Qwen engine not initialized".to_string())
    }
}

#[command]
pub async fn qwen_is_model_loaded() -> Result<bool, String> {
    let engine = {
        let guard = QWEN_ENGINE.lock().unwrap();
        guard.as_ref().cloned()
    };

    if let Some(engine) = engine {
        Ok(engine.is_model_loaded().await)
    } else {
        Err("Qwen engine not initialized".to_string())
    }
}

#[command]
pub async fn qwen_has_available_models() -> Result<bool, String> {
    let engine = {
        let guard = QWEN_ENGINE.lock().unwrap();
        guard.as_ref().cloned()
    };

    if let Some(engine) = engine {
        let models = engine
            .discover_models()
            .await
            .map_err(|e| format!("Failed to discover Qwen models: {}", e))?;

        let available_models: Vec<_> = models
            .iter()
            .filter(|model| matches!(model.status, ModelStatus::Available))
            .collect();

        Ok(!available_models.is_empty())
    } else {
        Ok(false)
    }
}

#[command]
pub async fn qwen_validate_model_ready() -> Result<String, String> {
    let engine = {
        let guard = QWEN_ENGINE.lock().unwrap();
        guard.as_ref().cloned()
    };

    if let Some(engine) = engine {
        if engine.is_model_loaded().await {
            if let Some(current_model) = engine.get_current_model().await {
                return Ok(current_model);
            }
        }

        let models = engine
            .discover_models()
            .await
            .map_err(|e| format!("Failed to discover Qwen models: {}", e))?;

        let available_models: Vec<_> = models
            .iter()
            .filter(|model| matches!(model.status, ModelStatus::Available))
            .collect();

        if available_models.is_empty() {
            return Err(
                "No Qwen models are available. Please download a model to enable transcription."
                    .to_string(),
            );
        }

        let first_model = available_models
            .iter()
            .find(|m| m.name == DEFAULT_QWEN_MODEL || m.name.contains("0.6"))
            .or_else(|| available_models.first())
            .unwrap();

        engine
            .load_model(&first_model.name)
            .await
            .map_err(|e| format!("Failed to load Qwen model {}: {}", first_model.name, e))?;

        Ok(first_model.name.clone())
    } else {
        Err("Qwen engine not initialized".to_string())
    }
}

/// Internal version that respects user's transcript config when provider is `qwen`.
pub async fn qwen_validate_model_ready_with_config<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
) -> Result<String, String> {
    let engine = {
        let guard = QWEN_ENGINE.lock().unwrap();
        guard.as_ref().cloned()
    };

    if let Some(engine) = engine {
        if engine.is_model_loaded().await {
            if let Some(current_model) = engine.get_current_model().await {
                log::info!("Qwen model already loaded: {}", current_model);
                return Ok(current_model);
            }
        }

        let model_to_load = match crate::api::api::api_get_transcript_config(
            app.clone(),
            app.state(),
            None,
        )
        .await
        {
            Ok(Some(config)) => {
                log::info!(
                    "Got transcript config from API - provider: {}, model: {}",
                    config.provider,
                    config.model
                );
                if config.provider == "qwen" {
                    if !config.model.is_empty() {
                        log::info!("Using user's configured Qwen model: {}", config.model);
                        Some(config.model)
                    } else {
                        log::info!(
                            "Qwen provider with empty model, using default {}",
                            DEFAULT_QWEN_MODEL
                        );
                        Some(DEFAULT_QWEN_MODEL.to_string())
                    }
                } else {
                    log::info!(
                        "API config uses non-Qwen provider ({}), will auto-select",
                        config.provider
                    );
                    None
                }
            }
            Ok(None) => {
                log::info!("No transcript config found in API, will auto-select Qwen model");
                None
            }
            Err(e) => {
                log::warn!(
                    "Failed to get transcript config from API: {}, will auto-select Qwen model",
                    e
                );
                None
            }
        };

        let models = engine
            .discover_models()
            .await
            .map_err(|e| format!("Failed to discover Qwen models: {}", e))?;

        let available_models: Vec<_> = models
            .iter()
            .filter(|model| matches!(model.status, ModelStatus::Available))
            .collect();

        if available_models.is_empty() {
            return Err(
                "No Qwen models are available. Please download a model to enable transcription."
                    .to_string(),
            );
        }

        let model_name = if let Some(configured_model) = model_to_load {
            if available_models.iter().any(|m| m.name == configured_model) {
                log::info!("Loading user's configured Qwen model: {}", configured_model);
                configured_model
            } else {
                log::warn!(
                    "Configured Qwen model '{}' not found, falling back to default/0.6B",
                    configured_model
                );
                available_models
                    .iter()
                    .find(|m| m.name == DEFAULT_QWEN_MODEL || m.name.contains("0.6"))
                    .or_else(|| available_models.first())
                    .unwrap()
                    .name
                    .clone()
            }
        } else {
            log::info!("No configured model, preferring DEFAULT_QWEN_MODEL / 0.6B");
            available_models
                .iter()
                .find(|m| m.name == DEFAULT_QWEN_MODEL || m.name.contains("0.6"))
                .or_else(|| available_models.first())
                .unwrap()
                .name
                .clone()
        };

        engine
            .load_model(&model_name)
            .await
            .map_err(|e| format!("Failed to load Qwen model {}: {}", model_name, e))?;

        Ok(model_name)
    } else {
        Err("Qwen engine not initialized".to_string())
    }
}

#[command]
pub async fn qwen_transcribe_audio(audio_data: Vec<f32>) -> Result<String, String> {
    let engine = {
        let guard = QWEN_ENGINE.lock().unwrap();
        guard.as_ref().cloned()
    };

    if let Some(engine) = engine {
        engine
            .transcribe_audio(audio_data)
            .await
            .map_err(|e| format!("Qwen transcription failed: {}", e))
    } else {
        Err("Qwen engine not initialized".to_string())
    }
}

#[command]
pub async fn qwen_transcribe_audio_with_language(
    audio_data: Vec<f32>,
    language: Option<String>,
) -> Result<String, String> {
    let engine = {
        let guard = QWEN_ENGINE.lock().unwrap();
        guard.as_ref().cloned()
    };

    if let Some(engine) = engine {
        engine
            .transcribe_audio_with_language(audio_data, language)
            .await
            .map_err(|e| format!("Qwen transcription failed: {}", e))
    } else {
        Err("Qwen engine not initialized".to_string())
    }
}

#[command]
pub async fn qwen_get_models_directory() -> Result<String, String> {
    let engine = {
        let guard = QWEN_ENGINE.lock().unwrap();
        guard.as_ref().cloned()
    };

    if let Some(engine) = engine {
        let path = engine.get_models_directory().await;
        Ok(path.to_string_lossy().to_string())
    } else {
        Err("Qwen engine not initialized".to_string())
    }
}

#[command]
pub async fn qwen_download_model<R: Runtime>(
    app_handle: AppHandle<R>,
    model_name: String,
) -> Result<(), String> {
    let engine = {
        let guard = QWEN_ENGINE.lock().unwrap();
        guard.as_ref().cloned()
    };

    if let Some(engine) = engine {
        let app_handle_clone = app_handle.clone();
        let model_name_clone = model_name.clone();

        let progress_callback = Box::new(move |progress: DownloadProgress| {
            log::info!(
                "Qwen download progress for {}: {:.1} MB / {:.1} MB ({:.1} MB/s) - {}%",
                model_name_clone,
                progress.downloaded_mb,
                progress.total_mb,
                progress.speed_mbps,
                progress.percent
            );

            if let Err(e) = app_handle_clone.emit(
                "qwen-model-download-progress",
                serde_json::json!({
                    "modelName": model_name_clone,
                    "progress": progress.percent,
                    "downloaded_bytes": progress.downloaded_bytes,
                    "total_bytes": progress.total_bytes,
                    "downloaded_mb": progress.downloaded_mb,
                    "total_mb": progress.total_mb,
                    "speed_mbps": progress.speed_mbps,
                    "status": if progress.percent == 100 { "completed" } else { "downloading" }
                }),
            ) {
                log::error!("Failed to emit qwen download progress event: {}", e);
            }
        });

        if let Err(e) = engine.discover_models().await {
            log::warn!("Failed to discover models before download: {}", e);
        }

        let result = engine
            .download_model_detailed(&model_name, Some(progress_callback))
            .await;

        match result {
            Ok(()) => {
                if let Err(e) = app_handle.emit(
                    "qwen-model-download-complete",
                    serde_json::json!({
                        "modelName": model_name
                    }),
                ) {
                    log::error!("Failed to emit qwen download complete event: {}", e);
                }

                log::info!("Qwen model download complete - updating tray menu");
                crate::tray::update_tray_menu(&app_handle);

                Ok(())
            }
            Err(e) => {
                if let Err(emit_e) = app_handle.emit(
                    "qwen-model-download-error",
                    serde_json::json!({
                        "modelName": model_name,
                        "error": e.to_string()
                    }),
                ) {
                    log::error!("Failed to emit qwen download error event: {}", emit_e);
                }
                Err(format!("Failed to download Qwen model: {}", e))
            }
        }
    } else {
        Err("Qwen engine not initialized".to_string())
    }
}

#[command]
pub async fn qwen_cancel_download<R: Runtime>(
    app_handle: AppHandle<R>,
    model_name: String,
) -> Result<(), String> {
    let engine = {
        let guard = QWEN_ENGINE.lock().unwrap();
        guard.as_ref().cloned()
    };

    if let Some(engine) = engine {
        engine
            .cancel_download(&model_name)
            .await
            .map_err(|e| format!("Failed to cancel Qwen download: {}", e))?;

        let _ = app_handle.emit(
            "qwen-model-download-progress",
            serde_json::json!({
                "modelName": model_name,
                "progress": 0,
                "status": "cancelled"
            }),
        );

        log::info!("Qwen download cancelled: {}", model_name);
        Ok(())
    } else {
        Err("Qwen engine not initialized".to_string())
    }
}

#[command]
pub async fn qwen_retry_download<R: Runtime>(
    app_handle: AppHandle<R>,
    model_name: String,
) -> Result<(), String> {
    log::info!("Retrying Qwen download for: {}", model_name);

    let engine = {
        let guard = QWEN_ENGINE.lock().unwrap();
        guard.as_ref().cloned()
    };

    if let Some(engine) = engine {
        {
            let mut active = engine.active_downloads.write().await;
            if active.contains(&model_name) {
                log::warn!(
                    "Retry: Model {} was still in active downloads, removing",
                    model_name
                );
                active.remove(&model_name);
            }
        }

        {
            let mut models = engine.available_models.write().await;
            if let Some(model) = models.get_mut(&model_name) {
                log::info!(
                    "Retry: Resetting model {} status from {:?} to Missing",
                    model_name,
                    model.status
                );
                model.status = ModelStatus::Missing;
            }
        }

        let _ = engine.discover_models().await;
        qwen_download_model(app_handle, model_name).await
    } else {
        Err("Qwen engine not initialized".to_string())
    }
}

#[command]
pub async fn qwen_delete_corrupted_model(model_name: String) -> Result<String, String> {
    let engine = {
        let guard = QWEN_ENGINE.lock().unwrap();
        guard.as_ref().cloned()
    };

    if let Some(engine) = engine {
        engine
            .delete_model(&model_name)
            .await
            .map_err(|e| format!("Failed to delete Qwen model: {}", e))
    } else {
        Err("Qwen engine not initialized".to_string())
    }
}

/// Open the Qwen models folder in the system file explorer
#[command]
pub async fn open_qwen_models_folder() -> Result<(), String> {
    let models_dir = get_models_directory()
        .ok_or_else(|| "Qwen models directory not initialized".to_string())?;

    if !models_dir.exists() {
        std::fs::create_dir_all(&models_dir)
            .map_err(|e| format!("Failed to create directory: {}", e))?;
    }

    let folder_path = models_dir.to_string_lossy().to_string();

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(&folder_path)
            .spawn()
            .map_err(|e| format!("Failed to open folder: {}", e))?;
    }

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(&folder_path)
            .spawn()
            .map_err(|e| format!("Failed to open folder: {}", e))?;
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(&folder_path)
            .spawn()
            .map_err(|e| format!("Failed to open folder: {}", e))?;
    }

    log::info!("Opened Qwen models folder: {}", folder_path);
    Ok(())
}
