use tauri::{AppHandle, Manager, Runtime};

/// Show the always-on-top dictation pill HUD (non-activating window).
pub fn show_pill<R: Runtime>(app: &AppHandle<R>) -> Result<(), String> {
    if let Some(w) = app.get_webview_window("dictation-pill") {
        w.show().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Hide the dictation pill HUD.
pub fn hide_pill<R: Runtime>(app: &AppHandle<R>) -> Result<(), String> {
    if let Some(w) = app.get_webview_window("dictation-pill") {
        w.hide().map_err(|e| e.to_string())?;
    }
    Ok(())
}
