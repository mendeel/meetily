//! Nemotron 3.5 ASR streaming speech recognition engine module.
//!
//! Wraps `parakeet-rs::Nemotron` for multilingual cache-aware streaming ASR.
//! Hybrid inference: Silero VAD segments utterances; within each utterance,
//! audio is streamed in 560ms chunks through Nemotron's cache-aware API.
//!
//! # Module Structure
//!
//! - `nemotron_engine`: Discovery, download, load/unload, hybrid streaming
//! - `language`: Meetily language-preference → Nemotron locale mapping
//! - `commands`: Tauri command interface for frontend integration

pub mod nemotron_engine;
pub mod language;
pub mod commands;

pub use nemotron_engine::{
    NemotronEngine, NemotronEngineError, QuantizationType, ModelInfo, ModelStatus, DownloadProgress,
};
pub use language::map_language_preference;
pub use commands::*;
