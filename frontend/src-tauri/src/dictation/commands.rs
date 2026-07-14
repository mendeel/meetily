use once_cell::sync::Lazy;
use serde::Serialize;
use std::sync::Mutex;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, Runtime};
use tokio_util::sync::CancellationToken;

use super::capture::MicCapture;
use super::config::DictationConfig;
use super::context::frontmost_app;
#[cfg(target_os = "macos")]
use super::context::{
    activate_app_by_bundle_id, may_paste_after_activation, MEETILY_BUNDLE_ID,
};
#[cfg(target_os = "macos")]
use super::injector::accessibility_trusted;
use super::injector::{inject_or_clipboard, InjectResult};
use super::pill::{hide_pill, show_pill};
use super::polish::{rule_based_cleanup, system_prompt_for};
use super::profiles::PolishProfile;
use super::session::{
    decide_capture_start, decide_finalize, decide_stop, should_hide_pill, CaptureStartDecision,
    DictationPhase, DictationSession, FinalizeDecision, StopDecision, TriggerMode,
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
    /// True while `dictation_start_inner` is in flight (before Listening or failure).
    pub starting: bool,
    /// Bumped whenever the pill is shown for a new Listening/terminal session.
    /// Auto-hide timers capture this and only hide if it still matches.
    pub pill_generation: u64,
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
        starting: false,
        pill_generation: 0,
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

fn schedule_hide_pill<R: Runtime>(app: &AppHandle<R>, generation: u64) {
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(Duration::from_millis(PILL_STATUS_VISIBLE_MS)).await;
        let current = lock_runtime().pill_generation;
        if should_hide_pill(generation, current) {
            let _ = hide_pill(&app);
        }
    });
}

/// Show the pill and bump generation so any prior auto-hide timer becomes a no-op.
fn show_pill_for_session<R: Runtime>(app: &AppHandle<R>) -> u64 {
    let generation = {
        let mut rt = lock_runtime();
        rt.pill_generation = rt.pill_generation.wrapping_add(1);
        rt.pill_generation
    };
    if let Err(e) = show_pill(app) {
        log::warn!("dictation pill show failed: {e}");
    }
    generation
}

/// Emit a terminal phase while the pill is still visible, then hide after a short delay.
/// `generation` must be the pill token from this session (captured before any reset that
/// could let a newer session start), so a stale timer cannot hide a newer pill.
fn emit_terminal_phase_and_hide<R: Runtime>(
    app: &AppHandle<R>,
    phase: DictationPhase,
    app_name: Option<&str>,
    message: Option<&str>,
    generation: u64,
) {
    emit_phase(app, phase, app_name, message);
    schedule_hide_pill(app, generation);
}

/// Surface a dictation error to the UI (pill + phase event). Used by hotkeys when
/// start/stop cannot proceed. Auto-hides after a short delay.
pub(crate) fn emit_dictation_error<R: Runtime>(app: &AppHandle<R>, message: &str) {
    let generation = show_pill_for_session(app);
    emit_terminal_phase_and_hide(app, DictationPhase::Error, None, Some(message), generation);
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

/// Which engine's init + validate_model_ready path dictation must run before STT.
/// Meeting recording already does this; dictation previously skipped it and failed
/// with "No Nemotron model loaded" on hotkey stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TranscriptionPreparation {
    Nemotron,
    Parakeet,
    Whisper,
}

fn transcription_preparation(engine: &str) -> TranscriptionPreparation {
    match engine {
        "nemotron" => TranscriptionPreparation::Nemotron,
        "parakeet" => TranscriptionPreparation::Parakeet,
        "whisper" => TranscriptionPreparation::Whisper,
        other => panic!("unsupported dictation STT engine: {other}"),
    }
}

/// Ensure the selected STT engine is initialized and has a model loaded.
/// Uses the same validate_model_ready helpers as meeting recording (auto-load first available).
async fn ensure_stt_ready(prep: TranscriptionPreparation) -> Result<(), String> {
    match prep {
        TranscriptionPreparation::Nemotron => {
            crate::nemotron_engine::commands::nemotron_init().await?;
            crate::nemotron_engine::commands::nemotron_validate_model_ready()
                .await
                .map(|_| ())
        }
        TranscriptionPreparation::Parakeet => {
            crate::parakeet_engine::commands::parakeet_init().await?;
            crate::parakeet_engine::commands::parakeet_validate_model_ready()
                .await
                .map(|_| ())
        }
        TranscriptionPreparation::Whisper => {
            crate::whisper_engine::commands::whisper_init().await?;
            crate::whisper_engine::commands::whisper_validate_model_ready()
                .await
                .map(|_| ())
        }
    }
}

async fn transcribe_samples(engine: &str, samples: Vec<f32>) -> Result<String, String> {
    match engine {
        "nemotron" | "parakeet" | "whisper" => {
            ensure_stt_ready(transcription_preparation(engine)).await?;
        }
        other => return Err(format!("unsupported dictation STT engine: {other}")),
    }

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
    rt.starting = false;
    rt.session.reset();
}

fn fail_and_reset(rt: &mut DictationRuntime) {
    rt.capture = None;
    rt.target_app_name = None;
    rt.target_bundle_id = None;
    rt.pending_stop = false;
    rt.finalizing = false;
    rt.starting = false;
    rt.session.fail();
    rt.session.reset();
}

/// Clear in-flight start flags after any early failure (before Listening).
fn clear_start_attempt(rt: &mut DictationRuntime) {
    rt.starting = false;
    rt.pending_stop = false;
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
    {
        let mut rt = lock_runtime();
        rt.starting = true;
    }

    if crate::audio::recording_commands::is_recording().await {
        clear_start_attempt(&mut lock_runtime());
        return Err("Stop meeting recording first".into());
    }

    let front = frontmost_app().ok();
    let app_name = front.as_ref().map(|f| f.name.clone());
    let bundle_id = front.as_ref().and_then(|f| f.bundle_id.clone());

    let finalize_now = {
        let mut rt = lock_runtime();
        if !rt.config.enabled {
            clear_start_attempt(&mut rt);
            return Err("Dictation is disabled".into());
        }
        rt.session.meeting_recording_active = false;
        if let Err(e) = rt.session.on_hotkey_pressed() {
            clear_start_attempt(&mut rt);
            return Err(e.to_string());
        }

        if rt.session.phase == DictationPhase::Transcribing {
            // Toggle second-press: finalize without starting a new capture.
            rt.starting = false;
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
                CaptureStartDecision::NotListening => {
                    clear_start_attempt(&mut rt);
                    return Ok(());
                }
            }

            // Fast PTT release raced ahead of start — finalize immediately.
            if rt.pending_stop {
                rt.pending_stop = false;
                rt.starting = false;
                rt.session.phase = DictationPhase::Transcribing;
                true
            } else {
                rt.starting = false;
                false
            }
        } else {
            clear_start_attempt(&mut rt);
            return Ok(());
        }
    };

    if finalize_now {
        // Pending-stop path never reached the Listening show_pill — surface HUD first.
        show_pill_for_session(app);
        let state = app.try_state::<AppState>();
        return finalize_dictation(app, state.as_deref()).await;
    }

    show_pill_for_session(app);
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
        match decide_stop(rt.session.phase, rt.finalizing, rt.starting) {
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
    // Capture pill_generation before reset so a later session cannot be hidden by this finalize's timer.
    let (samples, config, target_app, target_bundle, pill_generation) = {
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
        let pill_generation = rt.pill_generation;
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
                    pill_generation,
                );
                return Ok(());
            }
        };
        (samples, config, target_app, target_bundle, pill_generation)
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
        emit_terminal_phase_and_hide(
            app,
            DictationPhase::Done,
            target_app.as_deref(),
            None,
            pill_generation,
        );
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
                pill_generation,
            );
            return Err(e);
        }
    };

    let cleaned = rule_based_cleanup(&transcript);
    if cleaned.is_empty() {
        let mut rt = lock_runtime();
        reset_runtime(&mut rt);
        emit_terminal_phase_and_hide(
            app,
            DictationPhase::Done,
            target_app.as_deref(),
            None,
            pill_generation,
        );
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

    // Restore the originally frontmost target before Cmd+V. On failure / AX
    // unavailable / Meetily still focused, fall back to clipboard only.
    let allow_paste = prepare_target_paste(target_bundle.as_deref());
    let delivery = match inject_or_clipboard(&final_text, allow_paste) {
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
                pill_generation,
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

    emit_terminal_phase_and_hide(
        app,
        DictationPhase::Done,
        target_app.as_deref(),
        None,
        pill_generation,
    );
    Ok(())
}

/// Reactivate the stored target app (macOS), wait briefly, then decide if Cmd+V is safe.
/// Non-macOS / AX unavailable / activation failure → false (clipboard-only).
fn prepare_target_paste(target_bundle: Option<&str>) -> bool {
    #[cfg(target_os = "macos")]
    {
        if !accessibility_trusted() {
            return false;
        }
        let Some(bid) = target_bundle.filter(|s| !s.is_empty()) else {
            return false;
        };
        let activation_ok = activate_app_by_bundle_id(bid).is_ok();
        if activation_ok {
            // Brief settle so the target receives focus before synthesized Cmd+V.
            std::thread::sleep(Duration::from_millis(100));
        }
        let front = frontmost_app()
            .ok()
            .and_then(|f| f.bundle_id);
        may_paste_after_activation(bid, activation_ok, front.as_deref(), MEETILY_BUNDLE_ID)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = target_bundle;
        false
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nemotron_dictation_requires_model_preparation() {
        assert_eq!(
            transcription_preparation("nemotron"),
            TranscriptionPreparation::Nemotron
        );
    }

    #[test]
    fn parakeet_and_whisper_map_to_their_preparation_paths() {
        assert_eq!(
            transcription_preparation("parakeet"),
            TranscriptionPreparation::Parakeet
        );
        assert_eq!(
            transcription_preparation("whisper"),
            TranscriptionPreparation::Whisper
        );
    }
}
