use once_cell::sync::Lazy;
use serde::Serialize;
use std::sync::Mutex;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, Runtime};
use tokio_util::sync::CancellationToken;

use super::capture::MicCapture;
use super::config::DictationConfig;
use super::context::frontmost_app;
use super::injector::{inject_or_clipboard, InjectResult};
use super::pill::{hide_pill, show_pill};
use super::polish::{rule_based_cleanup, system_prompt_for};
use super::profiles::PolishProfile;
use super::session::{
    decide_capture_start, decide_finalize, decide_stop, CaptureStartDecision, DictationPhase,
    DictationSession, FinalizeDecision, StopDecision, TriggerMode,
};
use crate::database::repositories::setting::SettingsRepository;
use crate::state::AppState;
use crate::summary::llm_client::{generate_summary, LLMProvider};

pub struct DictationRuntime {
    pub session: DictationSession,
    pub config: DictationConfig,
    pub capture: Option<MicCapture>,
    pub target_app_name: Option<String>,
    pub target_bundle_id: Option<String>,
    /// Set when PTT release arrives before start reaches Listening.
    pub pending_stop: bool,
    /// True while a finalize task owns the session (capture taken / STT in flight).
    pub finalizing: bool,
}

pub static DICTATION: Lazy<Mutex<DictationRuntime>> = Lazy::new(|| {
    Mutex::new(DictationRuntime {
        session: DictationSession::new(TriggerMode::PushToTalk),
        config: DictationConfig::default(),
        capture: None,
        target_app_name: None,
        target_bundle_id: None,
        pending_stop: false,
        finalizing: false,
    })
});

/// How long Done/Error stay visible on the pill before auto-hide.
const PILL_STATUS_VISIBLE_MS: u64 = 1000;

#[derive(Debug, Clone, Serialize)]
struct DictationPhasePayload {
    phase: String,
    app: Option<String>,
    message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct DictationResultPayload {
    text: String,
    delivery: String,
}

pub(crate) fn lock_runtime() -> std::sync::MutexGuard<'static, DictationRuntime> {
    DICTATION
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// True while dictation mic capture is active (Listening phase).
/// Used by meeting recording to soft-block concurrent capture.
pub fn dictation_is_listening() -> bool {
    lock_runtime().session.phase == DictationPhase::Listening
}

fn phase_name(phase: DictationPhase) -> &'static str {
    match phase {
        DictationPhase::Idle => "idle",
        DictationPhase::Listening => "listening",
        DictationPhase::Transcribing => "transcribing",
        DictationPhase::Polishing => "polishing",
        DictationPhase::Inserting => "inserting",
        DictationPhase::Done => "done",
        DictationPhase::Error => "error",
    }
}

pub(crate) fn emit_phase<R: Runtime>(
    app: &AppHandle<R>,
    phase: DictationPhase,
    app_name: Option<&str>,
    message: Option<&str>,
) {
    let _ = app.emit(
        "dictation-phase",
        DictationPhasePayload {
            phase: phase_name(phase).to_string(),
            app: app_name.map(|s| s.to_string()),
            message: message.map(|s| s.to_string()),
        },
    );
}

fn schedule_hide_pill<R: Runtime>(app: &AppHandle<R>) {
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(Duration::from_millis(PILL_STATUS_VISIBLE_MS)).await;
        let _ = hide_pill(&app);
    });
}

/// Emit a terminal phase while the pill is still visible, then hide after a short delay.
fn emit_terminal_phase_and_hide<R: Runtime>(
    app: &AppHandle<R>,
    phase: DictationPhase,
    app_name: Option<&str>,
    message: Option<&str>,
) {
    emit_phase(app, phase, app_name, message);
    schedule_hide_pill(app);
}

/// Surface a dictation error to the UI (pill + phase event). Used by hotkeys when
/// start/stop cannot proceed. Auto-hides after a short delay.
pub(crate) fn emit_dictation_error<R: Runtime>(app: &AppHandle<R>, message: &str) {
    let _ = show_pill(app);
    emit_terminal_phase_and_hide(app, DictationPhase::Error, None, Some(message));
}

fn is_silent(samples: &[f32]) -> bool {
    if samples.len() < 1600 {
        // < ~100ms at 16 kHz
        return true;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    let rms = (sum_sq / samples.len() as f32).sqrt();
    rms < 0.005
}

async fn transcribe_samples(engine: &str, samples: Vec<f32>) -> Result<String, String> {
    match engine {
        "nemotron" => {
            crate::nemotron_engine::commands::nemotron_transcribe_audio(samples).await
        }
        "parakeet" => {
            crate::parakeet_engine::commands::parakeet_transcribe_audio(samples).await
        }
        "whisper" => crate::whisper_engine::commands::whisper_transcribe_audio(samples).await,
        other => Err(format!("unsupported dictation STT engine: {other}")),
    }
}

async fn polish_with_llm<R: Runtime>(
    app: &AppHandle<R>,
    state: Option<&AppState>,
    profile: PolishProfile,
    cleaned: &str,
) -> String {
    let Some(state) = state else {
        // AppState (DB) may be absent on first launch — skip polish, keep cleaned text.
        return cleaned.to_string();
    };
    let pool = state.db_manager.pool();
    let setting = match SettingsRepository::get_model_config(pool).await {
        Ok(Some(s)) => s,
        _ => return cleaned.to_string(),
    };

    let provider = match LLMProvider::from_str(&setting.provider) {
        Ok(p) => p,
        Err(_) => return cleaned.to_string(),
    };

    let api_key = if matches!(
        provider,
        LLMProvider::Ollama | LLMProvider::BuiltInAI | LLMProvider::CustomOpenAI
    ) {
        String::new()
    } else {
        match SettingsRepository::get_api_key(pool, &setting.provider).await {
            Ok(Some(key)) if !key.is_empty() => key,
            _ => return cleaned.to_string(),
        }
    };

    let ollama_endpoint = if provider == LLMProvider::Ollama {
        setting.ollama_endpoint.clone()
    } else {
        None
    };

    let (custom_endpoint, custom_key, custom_max, custom_temp, custom_top_p) =
        if provider == LLMProvider::CustomOpenAI {
            match SettingsRepository::get_custom_openai_config(pool).await {
                Ok(Some(cfg)) => (
                    Some(cfg.endpoint),
                    cfg.api_key.unwrap_or_default(),
                    cfg.max_tokens.map(|t| t as u32).or(Some(512)),
                    Some(cfg.temperature.unwrap_or(0.2)),
                    cfg.top_p,
                ),
                _ => return cleaned.to_string(),
            }
        } else {
            (None, String::new(), Some(512), Some(0.2), None)
        };

    let effective_key = if provider == LLMProvider::CustomOpenAI {
        custom_key
    } else {
        api_key
    };

    let app_data_dir = app.path().app_data_dir().ok();
    let system_prompt = system_prompt_for(profile);
    let user_prompt = format!(
        "Polish the following dictated text. Return only the polished text with no quotes, labels, or commentary.\n\n{}",
        cleaned
    );

    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
    {
        Ok(c) => c,
        Err(_) => return cleaned.to_string(),
    };

    let token = CancellationToken::new();
    let fut = generate_summary(
        &client,
        &provider,
        &setting.model,
        &effective_key,
        system_prompt,
        &user_prompt,
        ollama_endpoint.as_deref(),
        custom_endpoint.as_deref(),
        custom_max,
        custom_temp,
        custom_top_p,
        app_data_dir.as_ref(),
        Some(&token),
    );

    match tokio::time::timeout(Duration::from_secs(18), fut).await {
        Ok(Ok(text)) => {
            let polished = text.trim().to_string();
            if polished.is_empty() {
                cleaned.to_string()
            } else {
                polished
            }
        }
        _ => cleaned.to_string(),
    }
}

fn reset_runtime(rt: &mut DictationRuntime) {
    rt.capture = None;
    rt.target_app_name = None;
    rt.target_bundle_id = None;
    rt.pending_stop = false;
    rt.finalizing = false;
    rt.session.reset();
}

fn fail_and_reset(rt: &mut DictationRuntime) {
    rt.capture = None;
    rt.target_app_name = None;
    rt.target_bundle_id = None;
    rt.pending_stop = false;
    rt.finalizing = false;
    rt.session.fail();
    rt.session.reset();
}

#[tauri::command]
pub async fn dictation_get_config() -> Result<DictationConfig, String> {
    Ok(lock_runtime().config.clone())
}

#[tauri::command]
pub async fn dictation_set_config<R: Runtime>(
    app: AppHandle<R>,
    config: DictationConfig,
) -> Result<(), String> {
    {
        let mut rt = lock_runtime();
        rt.session.mode = config.trigger_mode;
        rt.config = config.clone();
    }
    super::config::save_persisted(&app, &config)?;
    #[cfg(any(target_os = "macos", windows, target_os = "linux"))]
    super::hotkeys::reregister(&app)?;
    Ok(())
}

/// Start listening (or toggle-finalize). Does not require AppState — mic + pill only.
pub async fn dictation_start_inner<R: Runtime>(app: &AppHandle<R>) -> Result<(), String> {
    if crate::audio::recording_commands::is_recording().await {
        return Err("Stop meeting recording first".into());
    }

    let front = frontmost_app().ok();
    let app_name = front.as_ref().map(|f| f.name.clone());
    let bundle_id = front.as_ref().and_then(|f| f.bundle_id.clone());

    let finalize_now = {
        let mut rt = lock_runtime();
        if !rt.config.enabled {
            return Err("Dictation is disabled".into());
        }
        rt.session.meeting_recording_active = false;
        rt.session
            .on_hotkey_pressed()
            .map_err(|e| e.to_string())?;

        if rt.session.phase == DictationPhase::Transcribing {
            // Toggle second-press: finalize without starting a new capture.
            true
        } else if rt.session.phase == DictationPhase::Listening {
            match decide_capture_start(rt.session.phase, rt.capture.is_some()) {
                CaptureStartDecision::KeepExisting => {
                    // Re-entrant start must not restart/drop the live mic stream.
                }
                CaptureStartDecision::StartNew => {
                    match MicCapture::start_default_input() {
                        Ok(capture) => {
                            rt.capture = Some(capture);
                            rt.target_app_name = app_name.clone();
                            rt.target_bundle_id = bundle_id;
                        }
                        Err(e) => {
                            reset_runtime(&mut rt);
                            return Err(format!("Failed to start dictation mic: {e}"));
                        }
                    }
                }
                CaptureStartDecision::NotListening => return Ok(()),
            }

            // Fast PTT release raced ahead of start — finalize immediately.
            if rt.pending_stop {
                rt.pending_stop = false;
                rt.session.phase = DictationPhase::Transcribing;
                true
            } else {
                false
            }
        } else {
            return Ok(());
        }
    };

    if finalize_now {
        let state = app.try_state::<AppState>();
        return finalize_dictation(app, state.as_deref()).await;
    }

    let _ = show_pill(app);
    emit_phase(
        app,
        DictationPhase::Listening,
        app_name.as_deref(),
        None,
    );
    Ok(())
}

/// Stop / PTT release → finalize. AppState is optional (needed only for LLM polish).
pub async fn dictation_stop_inner<R: Runtime>(app: &AppHandle<R>) -> Result<(), String> {
    let should_finalize = {
        let mut rt = lock_runtime();
        let _ = rt.session.on_hotkey_released();
        match decide_stop(rt.session.phase, rt.finalizing) {
            StopDecision::Noop => false,
            StopDecision::PendingStop => {
                rt.pending_stop = true;
                false
            }
            StopDecision::FinalizeNow => {
                if rt.session.phase == DictationPhase::Listening {
                    rt.session.phase = DictationPhase::Transcribing;
                }
                true
            }
        }
    };
    if !should_finalize {
        return Ok(());
    }
    let state = app.try_state::<AppState>();
    finalize_dictation(app, state.as_deref()).await
}

#[tauri::command]
pub async fn dictation_start<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    dictation_start_inner(&app).await
}

#[tauri::command]
pub async fn dictation_stop<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    dictation_stop_inner(&app).await
}

async fn finalize_dictation<R: Runtime>(
    app: &AppHandle<R>,
    state: Option<&AppState>,
) -> Result<(), String> {
    // Stop mic capture synchronously before any `.await` — `MicCapture` / cpal::Stream is !Send.
    let (samples, config, target_app, target_bundle) = {
        let mut rt = lock_runtime();
        match decide_finalize(rt.session.phase, rt.finalizing) {
            FinalizeDecision::Noop => return Ok(()),
            FinalizeDecision::IdleCleanup => {
                reset_runtime(&mut rt);
                let _ = hide_pill(app);
                return Ok(());
            }
            FinalizeDecision::Proceed => {
                rt.finalizing = true;
            }
        }
        let config = rt.config.clone();
        let target_app = rt.target_app_name.clone();
        let target_bundle = rt.target_bundle_id.clone();
        let samples = match rt.capture.take() {
            Some(capture) => capture.stop(),
            None => {
                reset_runtime(&mut rt);
                emit_terminal_phase_and_hide(
                    app,
                    DictationPhase::Done,
                    target_app.as_deref(),
                    None,
                );
                return Ok(());
            }
        };
        (samples, config, target_app, target_bundle)
    };

    emit_phase(
        app,
        DictationPhase::Transcribing,
        target_app.as_deref(),
        None,
    );

    if is_silent(&samples) {
        let mut rt = lock_runtime();
        reset_runtime(&mut rt);
        emit_terminal_phase_and_hide(app, DictationPhase::Done, target_app.as_deref(), None);
        return Ok(());
    }

    let transcript = match transcribe_samples(&config.stt_engine, samples).await {
        Ok(t) => t,
        Err(e) => {
            let mut rt = lock_runtime();
            fail_and_reset(&mut rt);
            emit_terminal_phase_and_hide(
                app,
                DictationPhase::Error,
                target_app.as_deref(),
                Some(&e),
            );
            return Err(e);
        }
    };

    let cleaned = rule_based_cleanup(&transcript);
    if cleaned.is_empty() {
        let mut rt = lock_runtime();
        reset_runtime(&mut rt);
        emit_terminal_phase_and_hide(app, DictationPhase::Done, target_app.as_deref(), None);
        return Ok(());
    }

    {
        let mut rt = lock_runtime();
        rt.session.advance_after_transcript();
    }

    let profile = config.resolve_profile(
        target_app.as_deref().unwrap_or(""),
        target_bundle.as_deref(),
    );

    let final_text = if config.polish_enabled {
        emit_phase(app, DictationPhase::Polishing, target_app.as_deref(), None);
        {
            let mut rt = lock_runtime();
            if rt.session.phase == DictationPhase::Transcribing {
                rt.session.phase = DictationPhase::Polishing;
            }
        }
        polish_with_llm(app, state, profile, &cleaned).await
    } else {
        cleaned
    };

    {
        let mut rt = lock_runtime();
        rt.session.advance_after_polish();
        if matches!(
            rt.session.phase,
            DictationPhase::Polishing | DictationPhase::Transcribing
        ) {
            rt.session.phase = DictationPhase::Inserting;
        }
    }

    emit_phase(app, DictationPhase::Inserting, target_app.as_deref(), None);

    let delivery = match inject_or_clipboard(&final_text) {
        Ok(InjectResult::Inserted) => "inserted",
        Ok(InjectResult::CopiedToClipboard) => "clipboard",
        Err(e) => {
            let mut rt = lock_runtime();
            fail_and_reset(&mut rt);
            emit_terminal_phase_and_hide(
                app,
                DictationPhase::Error,
                target_app.as_deref(),
                Some(&e),
            );
            return Err(e);
        }
    };

    let _ = app.emit(
        "dictation-result",
        DictationResultPayload {
            text: final_text,
            delivery: delivery.to_string(),
        },
    );

    {
        let mut rt = lock_runtime();
        rt.session.finish_insert();
        reset_runtime(&mut rt);
    }

    emit_terminal_phase_and_hide(app, DictationPhase::Done, target_app.as_deref(), None);
    Ok(())
}

#[tauri::command]
pub async fn dictation_accessibility_status() -> bool {
    super::injector::accessibility_trusted()
}

#[tauri::command]
pub async fn dictation_open_accessibility_settings() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")
            .spawn()
            .map_err(|e| format!("Failed to open Accessibility settings: {e}"))?;
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("Accessibility settings are only available on macOS".into())
    }
}
