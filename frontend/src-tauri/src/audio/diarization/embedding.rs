//! WeSpeaker CAM++ ONNX speaker embeddings via `ort` 2.0.0-rc.12.
//!
//! Model I/O (inspected from `wespeaker_en_voxceleb_CAM++.onnx`):
//! - Input `feats`: `[B, T, 80]` f32 (kaldi-native-fbank, mean-normalized)
//! - Output `embs`: `[B, 512]` f32 (not L2-normalized; we normalize here)
//!
//! Feature extraction matches pyannote-rs (`knf_rs::compute_fbank`).

use anyhow::{anyhow, Context, Result};
use log::{debug, info, warn};
use ndarray::Array3;
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

const INPUT_NAME: &str = "feats";
const OUTPUT_NAME: &str = "embs";
const FBANK_BINS: usize = 80;
const EMBEDDING_DIM: usize = 512;

/// ONNX WeSpeaker embedding extractor (16 kHz mono PCM).
pub struct SpeakerEmbeddingExtractor {
    session: Session,
}

impl SpeakerEmbeddingExtractor {
    pub fn try_load(model_path: &Path) -> Result<Self> {
        if !model_path.exists() {
            return Err(anyhow!(
                "WeSpeaker model not found at {}",
                model_path.display()
            ));
        }

        let providers = vec![CPUExecutionProvider::default().build()];
        let session = Session::builder()
            .map_err(|e| anyhow!("{}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("{}", e))?
            .with_execution_providers(providers)
            .map_err(|e| anyhow!("{}", e))?
            .with_intra_threads(2)
            .map_err(|e| anyhow!("{}", e))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("{}", e))?;

        for input in session.inputs() {
            info!(
                "WeSpeaker input: name={}, type={:?}",
                input.name(),
                input.dtype()
            );
        }
        for output in session.outputs() {
            info!(
                "WeSpeaker output: name={}, type={:?}",
                output.name(),
                output.dtype()
            );
        }

        info!(
            "WeSpeaker embedding session loaded from {}",
            model_path.display()
        );
        Ok(Self { session })
    }

    /// Compute an L2-normalized 512-D speaker embedding from 16 kHz f32 mono samples.
    pub fn embed(&mut self, samples_f32: &[f32]) -> Result<Vec<f32>> {
        if samples_f32.is_empty() {
            return Err(anyhow!("empty audio for WeSpeaker embedding"));
        }

        let features = knf_rs::compute_fbank(samples_f32)
            .map_err(|e| anyhow!("fbank failed: {}", e))?;
        let n_frames = features.nrows();
        let n_bins = features.ncols();
        if n_bins != FBANK_BINS {
            warn!(
                "WeSpeaker fbank bins={} (expected {}); continuing",
                n_bins, FBANK_BINS
            );
        }
        if n_frames == 0 {
            return Err(anyhow!("fbank produced zero frames"));
        }

        let data: Vec<f32> = features.iter().copied().collect();
        let feats = Array3::from_shape_vec((1, n_frames, n_bins), data)
            .context("reshape fbank to [1, T, bins]")?;

        let inputs = inputs![
            INPUT_NAME => TensorRef::from_array_view(feats.view())
                .map_err(|e| anyhow!("{}", e))?
        ];
        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| anyhow!("WeSpeaker inference failed: {}", e))?;

        let emb_tensor = outputs
            .get(OUTPUT_NAME)
            .ok_or_else(|| anyhow!("WeSpeaker output '{}' missing", OUTPUT_NAME))?;
        let emb_view = emb_tensor
            .try_extract_array::<f32>()
            .map_err(|e| anyhow!("extract embs: {}", e))?;

        let mut embedding: Vec<f32> = emb_view.iter().copied().collect();
        if embedding.len() < EMBEDDING_DIM {
            return Err(anyhow!(
                "WeSpeaker embedding dim {} < expected {}",
                embedding.len(),
                EMBEDDING_DIM
            ));
        }
        if embedding.len() > EMBEDDING_DIM {
            embedding.truncate(EMBEDDING_DIM);
        }

        l2_normalize(&mut embedding);
        debug!(
            "WeSpeaker embedding ok: frames={}, dim={}",
            n_frames,
            embedding.len()
        );
        Ok(embedding)
    }
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in v.iter_mut() {
        *x /= norm;
    }
}
