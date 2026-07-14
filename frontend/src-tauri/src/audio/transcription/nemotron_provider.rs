// audio/transcription/nemotron_provider.rs
//
// Nemotron transcription provider implementation.

use super::provider::{TranscriptionError, TranscriptionProvider, TranscriptResult};
use async_trait::async_trait;
use std::sync::Arc;

/// Nemotron transcription provider (wraps NemotronEngine)
pub struct NemotronProvider {
    engine: Arc<crate::nemotron_engine::NemotronEngine>,
}

impl NemotronProvider {
    pub fn new(engine: Arc<crate::nemotron_engine::NemotronEngine>) -> Self {
        Self { engine }
    }
}

#[async_trait]
impl TranscriptionProvider for NemotronProvider {
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        language: Option<String>,
    ) -> std::result::Result<TranscriptResult, TranscriptionError> {
        match self
            .engine
            .transcribe_audio(audio, language.as_deref())
            .await
        {
            Ok(text) => Ok(TranscriptResult {
                text: text.trim().to_string(),
                confidence: None, // Nemotron streaming path does not expose confidence
                is_partial: false,
            }),
            Err(e) => Err(TranscriptionError::EngineFailed(e.to_string())),
        }
    }

    async fn is_model_loaded(&self) -> bool {
        self.engine.is_model_loaded().await
    }

    async fn get_current_model(&self) -> Option<String> {
        self.engine.get_current_model().await
    }

    fn provider_name(&self) -> &'static str {
        "Nemotron"
    }
}
