//! Realtime neural diarization for system-audio channels.
//!
//! Mic stays channel-labeled as `"you"`. System audio starts as `"others"` and is
//! refined to `"speaker_1"`, `"speaker_2"`, … when ONNX models are present under
//! `models/diarization/`. Missing models fall back to channel-only labels.

pub mod models;
pub mod online;
pub mod commands;

pub use models::{DiarizationModelManager, DiarizationModelStatus};
pub use online::{
    channel_for_device, clear_active_diarizer, provisional_speaker_for_device,
    register_active_diarizer, shared_from_models_dir, OnlineDiarizer, SharedDiarizer,
};
pub use commands::set_models_directory as set_diarization_models_directory;
