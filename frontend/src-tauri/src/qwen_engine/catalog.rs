//! Qwen3-ASR ONNX model catalog and pack completeness checks.

pub const DEFAULT_QWEN_MODEL: &str = "qwen3-asr-0.6b-int4";
pub const QWEN_0_6B_INT4: &str = "qwen3-asr-0.6b-int4";
pub const QWEN_1_7B_INT4: &str = "qwen3-asr-1.7b-int4";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenModelSpec {
    pub id: &'static str,
    pub hf_repo: &'static str,
    pub size_mb: u32,
    pub files: &'static [&'static str],
}

/// Required files for int4 ONNX packs from andrewleech/qwen3-asr-*-onnx.
/// Note: some HF packs also ship `decoder_init.int4.onnx.data` /
/// `decoder_step.int4.onnx.data`. During Task 4/5 spike, if ORT requires
/// those external data files, add them here before shipping.
const INT4_FILES: &[&str] = &[
    "encoder.int4.onnx",
    "decoder_init.int4.onnx",
    "decoder_step.int4.onnx",
    "decoder_weights.int4.data",
    "embed_tokens.bin",
    "config.json",
    "tokenizer.json",
];

static SPECS: &[QwenModelSpec] = &[
    QwenModelSpec {
        id: QWEN_0_6B_INT4,
        hf_repo: "andrewleech/qwen3-asr-0.6b-onnx",
        size_mb: 1950, // ~encoder+decoder+embed; refine after first real download
        files: INT4_FILES,
    },
    QwenModelSpec {
        id: QWEN_1_7B_INT4,
        hf_repo: "andrewleech/qwen3-asr-1.7b-onnx",
        size_mb: 4000, // refine after first real download
        files: INT4_FILES,
    },
];

pub fn model_spec(id: &str) -> Option<&'static QwenModelSpec> {
    SPECS.iter().find(|s| s.id == id)
}

pub fn all_specs() -> &'static [QwenModelSpec] {
    SPECS
}

pub fn required_files(id: &str) -> Option<&'static [&'static str]> {
    model_spec(id).map(|s| s.files)
}

pub fn pack_is_complete(dir: &std::path::Path, id: &str) -> bool {
    let Some(files) = required_files(id) else {
        return false;
    };
    files.iter().all(|f| dir.join(f).is_file())
}

pub fn hf_file_url(repo: &str, filename: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{filename}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn known_models_have_specs() {
        assert!(model_spec(QWEN_0_6B_INT4).is_some());
        assert!(model_spec(QWEN_1_7B_INT4).is_some());
        assert!(model_spec("nope").is_none());
    }

    #[test]
    fn pack_complete_requires_all_int4_files() {
        let dir = tempdir().unwrap();
        let id = QWEN_0_6B_INT4;
        let files = required_files(id).unwrap();
        assert!(!pack_is_complete(dir.path(), id));
        for f in files {
            fs::write(dir.path().join(f), b"x").unwrap();
        }
        assert!(pack_is_complete(dir.path(), id));
    }

    #[test]
    fn default_model_is_0_6b() {
        assert_eq!(DEFAULT_QWEN_MODEL, QWEN_0_6B_INT4);
    }
}
