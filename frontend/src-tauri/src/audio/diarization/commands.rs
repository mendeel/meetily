//! Tauri commands for diarization model status / download.

use super::models::{DiarizationModelManager, DiarizationModelStatus};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::{command, AppHandle, Emitter, Manager, Runtime};

pub static DIARIZATION_MANAGER: Mutex<Option<Arc<DiarizationModelManager>>> = Mutex::new(None);

static MODELS_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Initialize models directory from app data dir (call during setup).
pub fn set_models_directory<R: Runtime>(app: &AppHandle<R>) {
    let app_data_dir = match app.path().app_data_dir() {
        Ok(dir) => dir,
        Err(e) => {
            log::error!("Failed to get app data dir for diarization: {}", e);
            return;
        }
    };

    let models_dir = app_data_dir.join("models");
    if !models_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(&models_dir) {
            log::error!("Failed to create models directory: {}", e);
            return;
        }
    }

    log::info!("Diarization models parent directory: {}", models_dir.display());
    *MODELS_DIR.lock().unwrap() = Some(models_dir);
}

fn get_models_directory() -> Option<PathBuf> {
    MODELS_DIR.lock().unwrap().clone()
}

pub fn get_or_init_manager() -> Result<Arc<DiarizationModelManager>, String> {
    let mut guard = DIARIZATION_MANAGER.lock().unwrap();
    if let Some(ref mgr) = *guard {
        return Ok(mgr.clone());
    }

    let mgr = DiarizationModelManager::new_with_models_dir(get_models_directory())
        .map_err(|e| format!("Failed to init diarization manager: {}", e))?;
    let arc = Arc::new(mgr);
    *guard = Some(arc.clone());
    Ok(arc)
}

#[command]
pub async fn diarization_init() -> Result<(), String> {
    get_or_init_manager().map(|_| ())
}

#[command]
pub async fn diarization_status() -> Result<DiarizationModelStatus, String> {
    let mgr = get_or_init_manager()?;
    Ok(mgr.status().await)
}

#[command]
pub async fn diarization_is_available() -> Result<bool, String> {
    let mgr = get_or_init_manager()?;
    Ok(mgr.is_available())
}

/// Frontend alias: ConfigContext invokes `is_diarization_model_available`.
#[command]
pub async fn is_diarization_model_available() -> Result<bool, String> {
    diarization_is_available().await
}

/// Frontend alias: ConfigContext invokes `set_neural_diarization_enabled`.
/// Toggles online clustering on the active SharedDiarizer (and for future sessions).
#[command]
pub async fn set_neural_diarization_enabled(enabled: bool) -> Result<(), String> {
    super::online::set_neural_enabled_preference(enabled);
    Ok(())
}

#[command]
pub async fn diarization_download_models<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    let mgr = get_or_init_manager()?;
    let app_clone = app.clone();

    mgr.download_models(Some(Box::new(move |pct| {
        let _ = app_clone.emit(
            "diarization-model-download-progress",
            serde_json::json!({
                "percent": pct,
                "status": if pct >= 100 { "completed" } else { "downloading" }
            }),
        );
    })))
    .await
    .map_err(|e| e.to_string())?;

    let _ = app.emit(
        "diarization-model-download-complete",
        serde_json::json!({ "status": "completed" }),
    );
    Ok(())
}

#[command]
pub async fn diarization_get_models_directory() -> Result<String, String> {
    let mgr = get_or_init_manager()?;
    Ok(mgr.models_dir().display().to_string())
}
