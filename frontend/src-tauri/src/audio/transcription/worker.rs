// audio/transcription/worker.rs
//
// Parallel transcription worker pool and chunk processing logic.
// Dual-stream aware: preserves DeviceType → speaker/channel labels,
// provisional sequence IDs for Nemotron partials, and backpressure that
// drops system partials first so the recording mixer is never blocked.

use super::engine::TranscriptionEngine;
use super::provider::TranscriptionError;
use crate::audio::diarization::{
    channel_for_device, clear_active_diarizer, provisional_speaker_for_device,
    register_active_diarizer, shared_from_models_dir, OnlineDiarizer, SharedDiarizer,
};
use crate::audio::recording_state::DeviceType;
use crate::audio::AudioChunk;
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Runtime};

// Sequence counter for transcript updates
static SEQUENCE_COUNTER: AtomicU64 = AtomicU64::new(0);

// Speech detection flag - reset per recording session
static SPEECH_DETECTED_EMITTED: AtomicBool = AtomicBool::new(false);

/// Max in-flight ASR chunks before backpressure kicks in.
/// When exceeded, system-channel work is dropped/coalesced first.
const MAX_QUEUE_DEPTH: u64 = 6;

/// Dual-stream enables up to 2 workers (mic + system).
const NUM_WORKERS_DUAL: usize = 2;

/// Reset the speech detected flag for a new recording session
pub fn reset_speech_detected_flag() {
    SPEECH_DETECTED_EMITTED.store(false, Ordering::SeqCst);
    info!(
        "🔍 SPEECH_DETECTED_EMITTED reset to: {}",
        SPEECH_DETECTED_EMITTED.load(Ordering::SeqCst)
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TranscriptUpdate {
    pub text: String,
    pub timestamp: String, // Wall-clock time for reference (e.g., "14:30:05")
    pub source: String,
    pub sequence_id: u64,
    pub chunk_start_time: f64, // Legacy field, kept for compatibility
    pub is_partial: bool,
    pub confidence: f32,
    // Recording-relative timestamps for playback sync
    pub audio_start_time: f64,
    pub audio_end_time: f64,
    pub duration: f64,
    /// Speaker label: "you" | "others" | "speaker_1" | "speaker_2" | …
    pub speaker: String,
    /// Audio channel: "mic" | "system"
    pub channel: String,
}

/// Result of a provider transcription, including optional pre-allocated sequence_id
/// (used by Nemotron so partials and finals share a stable ID).
struct TranscriptionOutcome {
    text: String,
    confidence: Option<f32>,
    is_partial: bool,
    /// When Some, reuse this sequence_id instead of allocating a new one.
    sequence_id: Option<u64>,
}

/// Optimized parallel transcription task ensuring ZERO chunk loss (mic) with
/// dual-stream workers and system-first backpressure under load.
pub fn start_transcription_task<R: Runtime>(
    app: AppHandle<R>,
    transcription_receiver: tokio::sync::mpsc::UnboundedReceiver<AudioChunk>,
) -> tokio::task::JoinHandle<()> {
    start_transcription_task_with_options(app, transcription_receiver, None, true)
}

/// Same as [`start_transcription_task`] with explicit diarization models dir and
/// dual-stream worker count toggle.
pub fn start_transcription_task_with_options<R: Runtime>(
    app: AppHandle<R>,
    transcription_receiver: tokio::sync::mpsc::UnboundedReceiver<AudioChunk>,
    diarization_models_dir: Option<PathBuf>,
    dual_stream: bool,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        info!("🚀 Starting dual-stream transcription task (dual_stream={})", dual_stream);

        let transcription_engine = match super::engine::get_or_init_transcription_engine(&app).await
        {
            Ok(engine) => engine,
            Err(e) => {
                error!("Failed to initialize transcription engine: {}", e);
                let _ = app.emit(
                    "transcription-error",
                    serde_json::json!({
                        "error": e,
                        "userMessage": "Recording failed: Unable to initialize speech recognition. Please check your model settings.",
                        "actionable": true
                    }),
                );
                return;
            }
        };

        let num_workers = if dual_stream { NUM_WORKERS_DUAL } else { 1 };
        let diarizer: SharedDiarizer = shared_from_models_dir(diarization_models_dir.as_deref());
        register_active_diarizer(diarizer.clone());
        {
            let enabled = diarizer
                .lock()
                .map(|d| d.is_neural_enabled())
                .unwrap_or(false);
            info!(
                "🎙️ Diarizer neural={}, workers={}",
                enabled, num_workers
            );
        }

        let (work_sender, work_receiver) =
            tokio::sync::mpsc::unbounded_channel::<AudioChunk>();
        let work_receiver = Arc::new(tokio::sync::Mutex::new(work_receiver));

        let chunks_queued = Arc::new(AtomicU64::new(0));
        let chunks_completed = Arc::new(AtomicU64::new(0));
        let input_finished = Arc::new(AtomicBool::new(false));
        // Tracks latest dropped system partial text for coalescing
        let coalesced_system_partial: Arc<std::sync::Mutex<Option<AudioChunk>>> =
            Arc::new(std::sync::Mutex::new(None));

        info!(
            "📊 Starting {} transcription worker{}",
            num_workers,
            if num_workers == 1 { "" } else { "s" }
        );

        let mut worker_handles = Vec::new();
        for worker_id in 0..num_workers {
            let engine_clone = match &transcription_engine {
                TranscriptionEngine::Whisper(e) => TranscriptionEngine::Whisper(e.clone()),
                TranscriptionEngine::Parakeet(e) => TranscriptionEngine::Parakeet(e.clone()),
                TranscriptionEngine::Nemotron(e) => TranscriptionEngine::Nemotron(e.clone()),
                TranscriptionEngine::Provider(p) => TranscriptionEngine::Provider(p.clone()),
            };
            let app_clone = app.clone();
            let work_receiver_clone = work_receiver.clone();
            let chunks_completed_clone = chunks_completed.clone();
            let input_finished_clone = input_finished.clone();
            let chunks_queued_clone = chunks_queued.clone();
            let diarizer_clone = diarizer.clone();

            let worker_handle = tokio::spawn(async move {
                info!("👷 Worker {} started", worker_id);

                let initial_model_loaded = engine_clone.is_model_loaded().await;
                let current_model = engine_clone
                    .get_current_model()
                    .await
                    .unwrap_or_else(|| "unknown".to_string());
                let engine_name = engine_clone.provider_name();

                if initial_model_loaded {
                    info!(
                        "✅ Worker {} pre-validation: {} model '{}' is loaded and ready",
                        worker_id, engine_name, current_model
                    );
                } else {
                    warn!(
                        "⚠️ Worker {} pre-validation: {} model not loaded - chunks may be skipped",
                        worker_id, engine_name
                    );
                }

                loop {
                    let chunk = {
                        let mut receiver = work_receiver_clone.lock().await;
                        receiver.recv().await
                    };

                    match chunk {
                        Some(chunk) => {
                            let should_log_this_chunk = chunk.chunk_id % 10 == 0;
                            let device_type = chunk.device_type.clone();
                            let channel = channel_for_device(&device_type).to_string();

                            if should_log_this_chunk {
                                info!(
                                    "👷 Worker {} processing chunk {} ({}) with {} samples",
                                    worker_id,
                                    chunk.chunk_id,
                                    channel,
                                    chunk.data.len()
                                );
                            }

                            if !engine_clone.is_model_loaded().await {
                                warn!(
                                    "⚠️ Worker {}: Model unloaded, but continuing to preserve chunk {}",
                                    worker_id, chunk.chunk_id
                                );
                                chunks_completed_clone.fetch_add(1, Ordering::SeqCst);
                                continue;
                            }

                            let chunk_timestamp = chunk.timestamp;
                            let chunk_duration = chunk.data.len() as f64 / chunk.sample_rate as f64;
                            // Keep samples for system-channel diarization
                            let samples_for_diar = if matches!(device_type, DeviceType::System) {
                                Some(if chunk.sample_rate != 16000 {
                                    crate::audio::audio_processing::resample_audio(
                                        &chunk.data,
                                        chunk.sample_rate,
                                        16000,
                                    )
                                } else {
                                    chunk.data.clone()
                                })
                            } else {
                                None
                            };

                            match transcribe_chunk_with_provider(
                                &engine_clone,
                                chunk,
                                &app_clone,
                                &channel,
                                &device_type,
                            )
                            .await
                            {
                                Ok(outcome) => {
                                    let confidence_threshold = match &engine_clone {
                                        TranscriptionEngine::Whisper(_)
                                        | TranscriptionEngine::Provider(_) => 0.3,
                                        TranscriptionEngine::Parakeet(_)
                                        | TranscriptionEngine::Nemotron(_) => 0.0,
                                    };

                                    let confidence_str = match outcome.confidence {
                                        Some(c) => format!("{:.2}", c),
                                        None => "N/A".to_string(),
                                    };

                                    info!(
                                        "🔍 Worker {} transcription result: text='{}', confidence={}, partial={}, channel={}",
                                        worker_id,
                                        outcome.text,
                                        confidence_str,
                                        outcome.is_partial,
                                        channel
                                    );

                                    let meets_threshold = outcome
                                        .confidence
                                        .map_or(true, |c| c >= confidence_threshold);

                                    if !outcome.text.trim().is_empty() && meets_threshold {
                                        info!(
                                            "✅ Worker {} transcribed: {} (confidence: {}, partial: {}, channel: {})",
                                            worker_id,
                                            outcome.text,
                                            confidence_str,
                                            outcome.is_partial,
                                            channel
                                        );

                                        let current_flag =
                                            SPEECH_DETECTED_EMITTED.load(Ordering::SeqCst);
                                        if !current_flag {
                                            SPEECH_DETECTED_EMITTED.store(true, Ordering::SeqCst);
                                            let _ = app_clone.emit(
                                                "speech-detected",
                                                serde_json::json!({
                                                    "message": "Speech activity detected"
                                                }),
                                            );
                                        }

                                        let sequence_id = outcome
                                            .sequence_id
                                            .unwrap_or_else(|| {
                                                SEQUENCE_COUNTER.fetch_add(1, Ordering::SeqCst)
                                            });
                                        let audio_start_time = chunk_timestamp;
                                        let audio_end_time = chunk_timestamp + chunk_duration;

                                        // Channel labeling + optional neural refine on system
                                        let mut speaker =
                                            provisional_speaker_for_device(&device_type).to_string();
                                        if matches!(device_type, DeviceType::System) {
                                            if let Some(ref samples) = samples_for_diar {
                                                if let Ok(mut guard) = diarizer_clone.lock() {
                                                    speaker = guard
                                                        .assign_speaker(&device_type, samples);
                                                }
                                            }
                                        }

                                        let update = TranscriptUpdate {
                                            text: outcome.text,
                                            timestamp: format_current_timestamp(),
                                            source: "Audio".to_string(),
                                            sequence_id,
                                            chunk_start_time: chunk_timestamp,
                                            is_partial: outcome.is_partial,
                                            confidence: outcome.confidence.unwrap_or(0.85),
                                            audio_start_time,
                                            audio_end_time,
                                            duration: chunk_duration,
                                            speaker: speaker.clone(),
                                            channel: channel.clone(),
                                        };

                                        if let Err(e) =
                                            app_clone.emit("transcript-update", &update)
                                        {
                                            error!(
                                                "Worker {}: Failed to emit transcript update: {}",
                                                worker_id, e
                                            );
                                        }

                                        // Dedicated speaker-relabel event for UI that already
                                        // rendered "others" and needs speaker_N without text rewrite
                                        if matches!(device_type, DeviceType::System)
                                            && speaker != "others"
                                            && !outcome.is_partial
                                        {
                                            let _ = app_clone.emit(
                                                "transcript-speaker-update",
                                                serde_json::json!({
                                                    "sequence_id": sequence_id,
                                                    "speaker": speaker,
                                                    "channel": channel,
                                                }),
                                            );
                                        }
                                    }
                                }
                                Err(e) => match e {
                                    TranscriptionError::AudioTooShort { .. } => {
                                        info!("Worker {}: {}", worker_id, e);
                                        chunks_completed_clone.fetch_add(1, Ordering::SeqCst);
                                        continue;
                                    }
                                    TranscriptionError::ModelNotLoaded => {
                                        warn!(
                                            "Worker {}: Model unloaded during transcription",
                                            worker_id
                                        );
                                        chunks_completed_clone.fetch_add(1, Ordering::SeqCst);
                                        continue;
                                    }
                                    _ => {
                                        warn!(
                                            "Worker {}: Transcription failed: {}",
                                            worker_id, e
                                        );
                                        let _ =
                                            app_clone.emit("transcription-warning", e.to_string());
                                    }
                                },
                            }

                            let completed =
                                chunks_completed_clone.fetch_add(1, Ordering::SeqCst) + 1;
                            let queued = chunks_queued_clone.load(Ordering::SeqCst);

                            if completed % 5 == 0 || should_log_this_chunk {
                                info!(
                                    "Worker {}: Progress {}/{} chunks ({:.1}%)",
                                    worker_id,
                                    completed,
                                    queued,
                                    (completed as f64 / queued.max(1) as f64 * 100.0)
                                );
                            }

                            let progress_percentage = if queued > 0 {
                                (completed as f64 / queued as f64 * 100.0) as u32
                            } else {
                                100
                            };

                            let _ = app_clone.emit(
                                "transcription-progress",
                                serde_json::json!({
                                    "worker_id": worker_id,
                                    "chunks_completed": completed,
                                    "chunks_queued": queued,
                                    "progress_percentage": progress_percentage,
                                    "message": format!("Worker {} processing... ({}/{})", worker_id, completed, queued)
                                }),
                            );
                        }
                        None => {
                            if input_finished_clone.load(Ordering::SeqCst) {
                                let final_queued = chunks_queued_clone.load(Ordering::SeqCst);
                                let final_completed =
                                    chunks_completed_clone.load(Ordering::SeqCst);

                                if final_completed >= final_queued {
                                    info!(
                                        "👷 Worker {} finishing - all {}/{} chunks processed",
                                        worker_id, final_completed, final_queued
                                    );
                                    break;
                                } else {
                                    warn!(
                                        "👷 Worker {} detected potential chunk loss: {}/{} completed, waiting...",
                                        worker_id, final_completed, final_queued
                                    );
                                    tokio::time::sleep(tokio::time::Duration::from_millis(5))
                                        .await;
                                }
                            } else {
                                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                            }
                        }
                    }
                }

                info!("👷 Worker {} completed", worker_id);
            });

            worker_handles.push(worker_handle);
        }

        // Main dispatcher with backpressure: drop/coalesce system chunks first
        let mut receiver = transcription_receiver;
        while let Some(chunk) = receiver.recv().await {
            let queued = chunks_queued.load(Ordering::SeqCst);
            let completed = chunks_completed.load(Ordering::SeqCst);
            let depth = queued.saturating_sub(completed);

            if depth >= MAX_QUEUE_DEPTH && matches!(chunk.device_type, DeviceType::System) {
                // Backpressure: coalesce latest system chunk; never block recording mixer
                warn!(
                    "⚠️ ASR backpressure (depth={}): coalescing system chunk {}",
                    depth, chunk.chunk_id
                );
                if let Ok(mut slot) = coalesced_system_partial.lock() {
                    *slot = Some(chunk);
                }
                continue;
            }

            // Flush any previously coalesced system chunk when pressure eases
            if depth < MAX_QUEUE_DEPTH / 2 {
                if let Ok(mut slot) = coalesced_system_partial.lock() {
                    if let Some(coalesced) = slot.take() {
                        let q = chunks_queued.fetch_add(1, Ordering::SeqCst) + 1;
                        info!(
                            "📥 Flushing coalesced system chunk {} (total queued: {})",
                            coalesced.chunk_id, q
                        );
                        if work_sender.send(coalesced).is_err() {
                            error!("❌ Failed to send coalesced chunk to workers");
                            break;
                        }
                    }
                }
            }

            let queued = chunks_queued.fetch_add(1, Ordering::SeqCst) + 1;
            info!(
                "📥 Dispatching chunk {} ({:?}) to workers (total queued: {})",
                chunk.chunk_id, chunk.device_type, queued
            );

            if work_sender.send(chunk).is_err() {
                error!("❌ Failed to send chunk to workers - this should not happen!");
                break;
            }
        }

        // Drain coalesced system chunk on shutdown
        if let Ok(mut slot) = coalesced_system_partial.lock() {
            if let Some(coalesced) = slot.take() {
                let _ = chunks_queued.fetch_add(1, Ordering::SeqCst);
                let _ = work_sender.send(coalesced);
            }
        }

        input_finished.store(true, Ordering::SeqCst);
        drop(work_sender);

        let total_chunks_queued = chunks_queued.load(Ordering::SeqCst);
        info!(
            "📭 Input finished with {} total chunks queued. Waiting for all {} workers to complete...",
            total_chunks_queued, num_workers
        );

        let _ = app.emit(
            "transcription-queue-complete",
            serde_json::json!({
                "total_chunks": total_chunks_queued,
                "message": format!("{} chunks queued for processing - waiting for completion", total_chunks_queued)
            }),
        );

        for (worker_id, handle) in worker_handles.into_iter().enumerate() {
            if let Err(e) = handle.await {
                error!("❌ Worker {} panicked: {:?}", worker_id, e);
            } else {
                info!("✅ Worker {} completed successfully", worker_id);
            }
        }

        let mut verification_attempts = 0;
        const MAX_VERIFICATION_ATTEMPTS: u32 = 10;

        loop {
            let final_queued = chunks_queued.load(Ordering::SeqCst);
            let final_completed = chunks_completed.load(Ordering::SeqCst);

            if final_queued == final_completed {
                info!(
                    "🎉 ALL {} chunks processed successfully - ZERO chunks lost!",
                    final_completed
                );
                break;
            } else if verification_attempts < MAX_VERIFICATION_ATTEMPTS {
                verification_attempts += 1;
                warn!(
                    "⚠️ Chunk count mismatch (attempt {}): {} queued, {} completed - waiting for stragglers...",
                    verification_attempts, final_queued, final_completed
                );
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            } else {
                error!(
                    "❌ CRITICAL: After {} attempts, chunk loss detected: {} queued, {} completed",
                    MAX_VERIFICATION_ATTEMPTS, final_queued, final_completed
                );
                let _ = app.emit(
                    "transcript-chunk-loss-detected",
                    serde_json::json!({
                        "chunks_queued": final_queued,
                        "chunks_completed": final_completed,
                        "chunks_lost": final_queued - final_completed,
                        "message": "Some transcript chunks may have been lost during shutdown"
                    }),
                );
                break;
            }
        }

        info!("✅ Parallel transcription task completed - all workers finished, ready for model unload");
        clear_active_diarizer();
    })
}

/// Transcribe audio chunk using the appropriate provider (Whisper, Parakeet, Nemotron, or trait-based).
async fn transcribe_chunk_with_provider<R: Runtime>(
    engine: &TranscriptionEngine,
    chunk: AudioChunk,
    app: &AppHandle<R>,
    channel: &str,
    device_type: &DeviceType,
) -> std::result::Result<TranscriptionOutcome, TranscriptionError> {
    let transcription_data = if chunk.sample_rate != 16000 {
        crate::audio::audio_processing::resample_audio(&chunk.data, chunk.sample_rate, 16000)
    } else {
        chunk.data
    };

    let speech_samples = transcription_data;

    if speech_samples.is_empty() {
        warn!(
            "Audio chunk {} is empty, skipping transcription",
            chunk.chunk_id
        );
        return Err(TranscriptionError::AudioTooShort {
            samples: 0,
            minimum: 1600,
        });
    }

    let energy: f32 =
        speech_samples.iter().map(|&x| x * x).sum::<f32>() / speech_samples.len() as f32;
    info!(
        "Processing speech audio chunk {} ({:?}/{}) with {} samples (energy: {:.6})",
        chunk.chunk_id,
        device_type,
        channel,
        speech_samples.len(),
        energy
    );

    let speaker = provisional_speaker_for_device(device_type).to_string();

    match engine {
        TranscriptionEngine::Whisper(whisper_engine) => {
            let language = crate::get_language_preference_internal();

            match whisper_engine
                .transcribe_audio_with_confidence(speech_samples, language)
                .await
            {
                Ok((text, confidence, is_partial)) => {
                    let cleaned_text = text.trim().to_string();
                    if cleaned_text.is_empty() {
                        return Ok(TranscriptionOutcome {
                            text: String::new(),
                            confidence: Some(confidence),
                            is_partial,
                            sequence_id: None,
                        });
                    }

                    info!(
                        "Whisper transcription complete for chunk {}: '{}' (confidence: {:.2}, partial: {})",
                        chunk.chunk_id, cleaned_text, confidence, is_partial
                    );

                    Ok(TranscriptionOutcome {
                        text: cleaned_text,
                        confidence: Some(confidence),
                        is_partial,
                        sequence_id: None,
                    })
                }
                Err(e) => {
                    error!(
                        "Whisper transcription failed for chunk {}: {}",
                        chunk.chunk_id, e
                    );
                    let transcription_error = TranscriptionError::EngineFailed(e.to_string());
                    let _ = app.emit(
                        "transcription-error",
                        &serde_json::json!({
                            "error": transcription_error.to_string(),
                            "userMessage": format!("Transcription failed: {}", transcription_error),
                            "actionable": false
                        }),
                    );
                    Err(transcription_error)
                }
            }
        }
        TranscriptionEngine::Parakeet(parakeet_engine) => {
            match parakeet_engine.transcribe_audio(speech_samples).await {
                Ok(text) => {
                    let cleaned_text = text.trim().to_string();
                    if cleaned_text.is_empty() {
                        return Ok(TranscriptionOutcome {
                            text: String::new(),
                            confidence: None,
                            is_partial: false,
                            sequence_id: None,
                        });
                    }

                    info!(
                        "Parakeet transcription complete for chunk {}: '{}'",
                        chunk.chunk_id, cleaned_text
                    );

                    Ok(TranscriptionOutcome {
                        text: cleaned_text,
                        confidence: None,
                        is_partial: false,
                        sequence_id: None,
                    })
                }
                Err(e) => {
                    error!(
                        "Parakeet transcription failed for chunk {}: {}",
                        chunk.chunk_id, e
                    );
                    let transcription_error = TranscriptionError::EngineFailed(e.to_string());
                    let _ = app.emit(
                        "transcription-error",
                        &serde_json::json!({
                            "error": transcription_error.to_string(),
                            "userMessage": format!("Transcription failed: {}", transcription_error),
                            "actionable": false
                        }),
                    );
                    Err(transcription_error)
                }
            }
        }
        TranscriptionEngine::Nemotron(nemotron_engine) => {
            let language = crate::get_language_preference_internal();
            let chunk_id = chunk.chunk_id;
            let chunk_timestamp = chunk.timestamp;
            let chunk_duration = speech_samples.len() as f64 / 16000.0;
            let app_for_partial = app.clone();
            let channel_owned = channel.to_string();
            let speaker_owned = speaker.clone();

            // Allocate a stable provisional sequence ID so UI can replace partials in place.
            // Do NOT use sequence_id: 0.
            let provisional_seq = SEQUENCE_COUNTER.fetch_add(1, Ordering::SeqCst);

            let on_partial = move |partial: &str| {
                let cleaned = partial.trim();
                if cleaned.is_empty() {
                    return;
                }
                let update = TranscriptUpdate {
                    text: cleaned.to_string(),
                    timestamp: format_current_timestamp(),
                    source: "Audio".to_string(),
                    sequence_id: provisional_seq,
                    chunk_start_time: chunk_timestamp,
                    is_partial: true,
                    confidence: 0.85,
                    audio_start_time: chunk_timestamp,
                    audio_end_time: chunk_timestamp + chunk_duration,
                    duration: chunk_duration,
                    speaker: speaker_owned.clone(),
                    channel: channel_owned.clone(),
                };
                if let Err(e) = app_for_partial.emit("transcript-update", &update) {
                    log::warn!(
                        "Failed to emit Nemotron partial transcript for chunk {}: {}",
                        chunk_id,
                        e
                    );
                }
            };

            match nemotron_engine
                .transcribe_utterance(speech_samples, language.as_deref(), Some(on_partial))
                .await
            {
                Ok(text) => {
                    let cleaned_text = text.trim().to_string();
                    if cleaned_text.is_empty() {
                        return Ok(TranscriptionOutcome {
                            text: String::new(),
                            confidence: None,
                            is_partial: false,
                            sequence_id: Some(provisional_seq),
                        });
                    }

                    info!(
                        "Nemotron transcription complete for chunk {}: '{}' (seq={})",
                        chunk.chunk_id, cleaned_text, provisional_seq
                    );

                    Ok(TranscriptionOutcome {
                        text: cleaned_text,
                        confidence: None,
                        is_partial: false,
                        sequence_id: Some(provisional_seq),
                    })
                }
                Err(e) => {
                    error!(
                        "Nemotron transcription failed for chunk {}: {}",
                        chunk.chunk_id, e
                    );
                    let transcription_error = TranscriptionError::EngineFailed(e.to_string());
                    let _ = app.emit(
                        "transcription-error",
                        &serde_json::json!({
                            "error": transcription_error.to_string(),
                            "userMessage": format!("Transcription failed: {}", transcription_error),
                            "actionable": false
                        }),
                    );
                    Err(transcription_error)
                }
            }
        }
        TranscriptionEngine::Provider(provider) => {
            let language = crate::get_language_preference_internal();

            match provider.transcribe(speech_samples, language).await {
                Ok(result) => {
                    let cleaned_text = result.text.trim().to_string();
                    if cleaned_text.is_empty() {
                        return Ok(TranscriptionOutcome {
                            text: String::new(),
                            confidence: result.confidence,
                            is_partial: result.is_partial,
                            sequence_id: None,
                        });
                    }

                    let confidence_str = match result.confidence {
                        Some(c) => format!("confidence: {:.2}", c),
                        None => "no confidence".to_string(),
                    };

                    info!(
                        "{} transcription complete for chunk {}: '{}' ({}, partial: {})",
                        provider.provider_name(),
                        chunk.chunk_id,
                        cleaned_text,
                        confidence_str,
                        result.is_partial
                    );

                    Ok(TranscriptionOutcome {
                        text: cleaned_text,
                        confidence: result.confidence,
                        is_partial: result.is_partial,
                        sequence_id: None,
                    })
                }
                Err(e) => {
                    error!(
                        "{} transcription failed for chunk {}: {}",
                        provider.provider_name(),
                        chunk.chunk_id,
                        e
                    );
                    let _ = app.emit(
                        "transcription-error",
                        &serde_json::json!({
                            "error": e.to_string(),
                            "userMessage": format!("Transcription failed: {}", e),
                            "actionable": false
                        }),
                    );
                    Err(e)
                }
            }
        }
    }
}

/// Format current timestamp (wall-clock time)
fn format_current_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();

    let hours = (now.as_secs() / 3600) % 24;
    let minutes = (now.as_secs() / 60) % 60;
    let seconds = now.as_secs() % 60;

    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

/// Format recording-relative time as [MM:SS]
#[allow(dead_code)]
fn format_recording_time(seconds: f64) -> String {
    let total_seconds = seconds.floor() as u64;
    let minutes = total_seconds / 60;
    let secs = total_seconds % 60;

    format!("[{:02}:{:02}]", minutes, secs)
}

/// Resolve diarization models directory (app data / models / diarization).
#[allow(dead_code)]
pub fn resolve_diarization_models_dir() -> Option<PathBuf> {
    if cfg!(debug_assertions) {
        std::env::current_dir()
            .ok()
            .map(|d| d.join("models").join("diarization"))
    } else {
        dirs::data_dir()
            .or_else(dirs::home_dir)
            .map(|d| d.join("Meetily").join("models").join("diarization"))
    }
}

/// Helper kept for callers that want a fresh channel-only diarizer.
#[allow(dead_code)]
pub fn new_channel_only_diarizer() -> OnlineDiarizer {
    OnlineDiarizer::channel_only()
}
