//! Qwen3-ASR ONNX speech recognition engine.
pub mod catalog;
pub mod mel;
pub mod text;
// model, qwen_engine, commands added in later tasks

pub use catalog::{DEFAULT_QWEN_MODEL, QWEN_0_6B_INT4, QWEN_1_7B_INT4};
