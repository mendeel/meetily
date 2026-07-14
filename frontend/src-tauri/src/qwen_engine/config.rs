//! Pack `config.json` for Qwen3-ASR ONNX (andrewleech / transcribe-rs layout).

use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Top-level model configuration loaded from `config.json`.
#[derive(Debug, Deserialize)]
pub struct QwenAsrConfig {
    pub encoder: EncoderConfig,
    pub decoder: DecoderConfig,
    pub mel: MelParams,
    pub special_tokens: SpecialTokens,
    /// Storage dtype of `embed_tokens.bin` (`float16` in published packs).
    #[serde(default)]
    pub embed_tokens_dtype: EmbedDtype,
    /// Optional override; defaults to 256 when absent.
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbedDtype {
    Float32,
    Float16,
}

impl Default for EmbedDtype {
    fn default() -> Self {
        Self::Float32
    }
}

impl EmbedDtype {
    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 => 2,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EncoderConfig {
    pub num_mel_bins: usize,
    pub output_dim: usize,
}

#[derive(Debug, Deserialize)]
pub struct DecoderConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
}

#[derive(Debug, Deserialize)]
pub struct MelParams {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    #[serde(default)]
    pub fmin: f64,
    #[serde(default = "default_fmax")]
    pub fmax: f64,
}

fn default_fmax() -> f64 {
    8000.0
}

#[derive(Debug, Deserialize)]
pub struct SpecialTokens {
    pub eos_token_ids: Vec<i64>,
    pub pad_token_id: i64,
    pub im_start_token_id: i64,
    pub im_end_token_id: i64,
    pub audio_start_token_id: i64,
    pub audio_end_token_id: i64,
    pub audio_pad_token_id: i64,
    #[serde(default = "default_asr_text_token_id")]
    pub asr_text_token_id: i64,
}

fn default_asr_text_token_id() -> i64 {
    151704
}

impl QwenAsrConfig {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let path = model_dir.join("config.json");
        let data =
            fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&data).map_err(|e| format!("parse {}: {e}", path.display()))
    }

    pub fn max_new_tokens_or_default(&self) -> usize {
        self.max_new_tokens.unwrap_or(256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "model_type": "qwen3_asr",
        "encoder": {
            "num_layers": 18, "hidden_size": 896, "num_heads": 14,
            "ffn_dim": 3584, "conv_channels": 480, "output_dim": 1024,
            "downsample_factor": 8, "num_mel_bins": 128
        },
        "decoder": {
            "num_layers": 28, "hidden_size": 1024, "num_attention_heads": 16,
            "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 3072,
            "vocab_size": 151936, "rope_theta": 1000000, "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
            "rope_scaling": { "mrope_section": [24, 20, 20], "interleaved": true }
        },
        "mel": {
            "sample_rate": 16000, "n_fft": 400, "hop_length": 160,
            "n_mels": 128, "fmin": 0, "fmax": 8000
        },
        "special_tokens": {
            "eos_token_ids": [151643, 151645],
            "pad_token_id": 151643,
            "im_start_token_id": 151644,
            "im_end_token_id": 151645,
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "audio_pad_token_id": 151676,
            "asr_text_token_id": 151704
        },
        "embed_tokens_dtype": "float16"
    }"#;

    #[test]
    fn deserializes_published_pack_shape() {
        let cfg: QwenAsrConfig = serde_json::from_str(SAMPLE).unwrap();
        assert_eq!(cfg.decoder.vocab_size, 151936);
        assert_eq!(cfg.mel.n_mels, 128);
        assert_eq!(cfg.embed_tokens_dtype, EmbedDtype::Float16);
        assert_eq!(cfg.max_new_tokens_or_default(), 256);
        assert_eq!(cfg.special_tokens.asr_text_token_id, 151704);
    }
}
