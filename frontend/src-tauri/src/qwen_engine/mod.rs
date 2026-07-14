//! Qwen3-ASR ONNX speech recognition engine.
pub mod catalog;
pub mod config;
pub mod mel;
pub mod model;
pub mod prompt;
pub mod text;
pub mod qwen_engine;
pub mod commands;

pub use catalog::{DEFAULT_QWEN_MODEL, QWEN_0_6B_INT4, QWEN_1_7B_INT4};
pub use model::QwenModel;
pub use qwen_engine::{QwenEngine, QwenEngineError, ModelInfo, ModelStatus, DownloadProgress};
pub use commands::*;
