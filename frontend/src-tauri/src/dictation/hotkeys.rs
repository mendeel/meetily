use std::str::FromStr;
use std::sync::Mutex;

use once_cell::sync::Lazy;
use tauri::{AppHandle, Runtime};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

use super::commands::{
    dictation_start_inner, dictation_stop_inner, emit_dictation_error, lock_runtime,
};
use super::config::{self, DictationConfig};
use super::session::TriggerMode;

/// Shortcuts currently registered with the OS (for clean re-registration).
static REGISTERED: Lazy<Mutex<Vec<String>>> = Lazy::new(|| Mutex::new(Vec::new()));

/// Install the global-shortcut plugin, load persisted config, and register hotkeys.
pub fn setup<R: Runtime>(app: &AppHandle<R>) -> Result<(), String> {
    let loaded = config::load_persisted(app);
    {
        let mut rt = lock_runtime();
        rt.session.mode = loaded.trigger_mode;
        rt.config = loaded;
    }

    app.plugin(
        tauri_plugin_global_shortcut::Builder::new()
            .with_handler(|app, shortcut, event| {
                handle_shortcut_event(app, shortcut, event.state);
            })
            .build(),
    )
    .map_err(|e| format!("Failed to init global-shortcut plugin: {e}"))?;

    reregister(app)
}

/// Unregister previous shortcuts and register based on current DictationConfig.
pub fn reregister<R: Runtime>(app: &AppHandle<R>) -> Result<(), String> {
    unregister_tracked(app);

    let config = lock_runtime().config.clone();
    if !config.enabled {
        log::info!("Dictation disabled — global hotkeys not registered");
        return Ok(());
    }

    let shortcut_str = active_shortcut_string(&config);
    app.global_shortcut()
        .register(shortcut_str.as_str())
        .map_err(|e| format!("Failed to register dictation shortcut '{shortcut_str}': {e}"))?;

    if let Ok(mut tracked) = REGISTERED.lock() {
        tracked.push(shortcut_str.clone());
    }
    log::info!(
        "Registered dictation hotkey '{}' (mode={:?})",
        shortcut_str,
        config.trigger_mode
    );
    Ok(())
}

fn active_shortcut_string(config: &DictationConfig) -> String {
    match config.trigger_mode {
        TriggerMode::PushToTalk => config.hold_shortcut.clone(),
        TriggerMode::Toggle => config.toggle_shortcut.clone(),
    }
}

fn unregister_tracked<R: Runtime>(app: &AppHandle<R>) {
    let shortcuts = REGISTERED
        .lock()
        .map(|mut g| std::mem::take(&mut *g))
        .unwrap_or_default();
    for s in shortcuts {
        if let Err(e) = app.global_shortcut().unregister(s.as_str()) {
            log::debug!("unregister '{s}': {e}");
        }
    }
}

fn handle_shortcut_event<R: Runtime>(
    app: &AppHandle<R>,
    shortcut: &Shortcut,
    state: ShortcutState,
) {
    let (mode, hold, toggle, enabled) = {
        let rt = lock_runtime();
        (
            rt.config.trigger_mode,
            rt.config.hold_shortcut.clone(),
            rt.config.toggle_shortcut.clone(),
            rt.config.enabled,
        )
    };

    if !enabled {
        return;
    }

    let hold_sc = Shortcut::from_str(&hold).ok();
    let toggle_sc = Shortcut::from_str(&toggle).ok();

    match mode {
        TriggerMode::PushToTalk => {
            if hold_sc.as_ref() != Some(shortcut) {
                return;
            }
            match state {
                ShortcutState::Pressed => spawn_start(app),
                ShortcutState::Released => spawn_stop(app),
            }
        }
        TriggerMode::Toggle => {
            if toggle_sc.as_ref() != Some(shortcut) {
                return;
            }
            if state == ShortcutState::Pressed {
                spawn_start(app);
            }
        }
    }
}

fn spawn_start<R: Runtime>(app: &AppHandle<R>) {
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        // Listening does not require AppState (DB); only finalize polish does.
        if let Err(e) = dictation_start_inner(&app).await {
            log::warn!("dictation_start from hotkey: {e}");
            emit_dictation_error(&app, &e);
        }
    });
}

fn spawn_stop<R: Runtime>(app: &AppHandle<R>) {
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        if let Err(e) = dictation_stop_inner(&app).await {
            log::warn!("dictation_stop from hotkey: {e}");
            emit_dictation_error(&app, &e);
        }
    });
}
