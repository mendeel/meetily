//! Online speaker assignment for system-audio ASR segments.
//!
//! Pyannote-rs-style pipeline (segmentation duty-cycled via upstream VAD +
//! embedding clustering). Full WeSpeaker ONNX via `pyannote-rs` is blocked by
//! an ort 2.0.0-rc.12 API mismatch, so when the downloadable model bundle is
//! present we run a local spectral embedding + cosine clustering backend that
//! mirrors the same `speaker_N` labeling contract. Missing models → channel-only.

use super::models::DiarizationModelManager;
use crate::audio::recording_state::DeviceType;
use log::{debug, info, warn};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const SIMILARITY_THRESHOLD: f32 = 0.72;
const MAX_SPEAKERS: usize = 10;
const EMBED_DIM: usize = 64;

/// User preference for neural-style clustering (default on; effective only when models exist).
static NEURAL_PREFERENCE: AtomicBool = AtomicBool::new(true);

/// Live session diarizer so settings toggles apply mid-recording.
static ACTIVE_DIARIZER: Mutex<Option<SharedDiarizer>> = Mutex::new(None);

/// Persist the UI toggle and apply it to any active recording session.
pub fn set_neural_enabled_preference(enabled: bool) {
    NEURAL_PREFERENCE.store(enabled, Ordering::SeqCst);
    if let Ok(guard) = ACTIVE_DIARIZER.lock() {
        if let Some(ref shared) = *guard {
            if let Ok(mut diarizer) = shared.lock() {
                diarizer.set_user_enabled(enabled);
            }
        }
    }
    info!("Neural diarization preference set to {}", enabled);
}

pub fn neural_enabled_preference() -> bool {
    NEURAL_PREFERENCE.load(Ordering::SeqCst)
}

pub fn register_active_diarizer(diarizer: SharedDiarizer) {
    if let Ok(mut guard) = ACTIVE_DIARIZER.lock() {
        *guard = Some(diarizer);
    }
}

pub fn clear_active_diarizer() {
    if let Ok(mut guard) = ACTIVE_DIARIZER.lock() {
        *guard = None;
    }
}

/// Map capture device → stable channel string for TranscriptUpdate.
pub fn channel_for_device(device: &DeviceType) -> &'static str {
    match device {
        DeviceType::Microphone => "mic",
        DeviceType::System => "system",
    }
}

/// Provisional speaker before neural diarization refines system segments.
pub fn provisional_speaker_for_device(device: &DeviceType) -> &'static str {
    match device {
        DeviceType::Microphone => "you",
        DeviceType::System => "others",
    }
}

/// Live online diarizer. Safe to share across workers via `Arc<Mutex<_>>`.
pub struct OnlineDiarizer {
    /// When None, always return channel-only provisional labels.
    backend: Option<SpectralBackend>,
    /// User toggle ("Enable neural-style clustering"). Models may be present while this is false.
    user_enabled: bool,
}

struct SpectralBackend {
    centroids: Vec<Vec<f32>>,
}

impl OnlineDiarizer {
    /// Create a channel-only diarizer (always returns provisional labels).
    pub fn channel_only() -> Self {
        info!("OnlineDiarizer: channel-only mode (neural models not loaded)");
        Self {
            backend: None,
            user_enabled: neural_enabled_preference(),
        }
    }

    /// Enable neural-style clustering when the diarization model bundle is present.
    pub fn try_load(models_dir: &Path) -> Self {
        let user_enabled = neural_enabled_preference();
        let embedding = models_dir.join("wespeaker_en_voxceleb_CAM++.onnx");
        let segmentation = models_dir.join("segmentation-3.0.onnx");
        if !(embedding.exists() && segmentation.exists()) {
            warn!(
                "Diarization model bundle incomplete under {} — falling back to channel-only",
                models_dir.display()
            );
            return Self {
                backend: None,
                user_enabled,
            };
        }

        // Bundle present: enable online spectral clustering (pyannote-rs-style contract).
        // Full WeSpeaker ORT path lands when pyannote-rs supports ort 2.0.0-rc.12.
        info!(
            "OnlineDiarizer: model bundle found at {} — neural preference={}",
            models_dir.display(),
            user_enabled
        );
        Self {
            backend: Some(SpectralBackend {
                centroids: Vec::new(),
            }),
            user_enabled,
        }
    }

    pub fn try_from_manager(manager: &DiarizationModelManager) -> Self {
        if manager.is_available() {
            Self::try_load(manager.models_dir())
        } else {
            Self::channel_only()
        }
    }

    /// Toggle the "Enable neural-style clustering" flag without unloading models.
    pub fn set_user_enabled(&mut self, enabled: bool) {
        self.user_enabled = enabled;
        info!(
            "OnlineDiarizer: user_enabled={}, models_loaded={}",
            enabled,
            self.backend.is_some()
        );
    }

    pub fn is_neural_enabled(&self) -> bool {
        self.user_enabled && self.backend.is_some()
    }

    /// Assign a speaker label for a speech window.
    /// Mic always returns `"you"`. System returns `"speaker_N"` when neural is
    /// enabled, otherwise `"others"`.
    pub fn assign_speaker(&mut self, device: &DeviceType, samples_f32: &[f32]) -> String {
        match device {
            DeviceType::Microphone => "you".to_string(),
            DeviceType::System => self.assign_system_speaker(samples_f32),
        }
    }

    fn assign_system_speaker(&mut self, samples_f32: &[f32]) -> String {
        if !self.user_enabled {
            return "others".to_string();
        }
        let Some(backend) = self.backend.as_mut() else {
            return "others".to_string();
        };

        if samples_f32.len() < 1600 {
            return "others".to_string();
        }

        let embedding = compute_spectral_embedding(samples_f32);
        match backend.assign(&embedding) {
            Some(idx) => {
                let label = format!("speaker_{}", idx + 1);
                debug!("Diarizer assigned {}", label);
                label
            }
            None => {
                debug!("Diarizer could not assign speaker — using others");
                "others".to_string()
            }
        }
    }
}

impl SpectralBackend {
    fn assign(&mut self, embedding: &[f32]) -> Option<usize> {
        let mut best_idx = None;
        let mut best_sim = SIMILARITY_THRESHOLD;

        for (i, centroid) in self.centroids.iter().enumerate() {
            let sim = cosine_similarity(embedding, centroid);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            // EMA update of matched centroid
            let c = &mut self.centroids[idx];
            for (i, v) in embedding.iter().enumerate() {
                if i < c.len() {
                    c[i] = 0.85 * c[i] + 0.15 * v;
                }
            }
            return Some(idx);
        }

        if self.centroids.len() < MAX_SPEAKERS {
            self.centroids.push(embedding.to_vec());
            return Some(self.centroids.len() - 1);
        }

        // At capacity — force best match even below threshold
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, cosine_similarity(embedding, c)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }
}

/// Compact log-mel style spectral fingerprint used as a speaker embedding.
fn compute_spectral_embedding(samples: &[f32]) -> Vec<f32> {
    let mut bands = vec![0.0f32; EMBED_DIM];
    let mut counts = vec![0u32; EMBED_DIM];

    let frame = 512.min(samples.len());
    if frame < 64 {
        return bands;
    }

    let hop = frame / 2;
    let mut offset = 0;
    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(frame);

    while offset + frame <= samples.len() {
        let window = &samples[offset..offset + frame];
        let mut input = window.to_vec();
        // Hann window
        for (i, s) in input.iter_mut().enumerate() {
            let w = 0.5
                - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (frame as f32 - 1.0)).cos();
            *s *= w;
        }
        let mut spectrum = fft.make_output_vec();
        if fft.process(&mut input, &mut spectrum).is_ok() {
            let n = spectrum.len().max(1);
            for (k, bin) in spectrum.iter().enumerate() {
                let mag = (bin.re * bin.re + bin.im * bin.im).sqrt();
                let band = (k * EMBED_DIM) / n;
                if band < EMBED_DIM {
                    bands[band] += mag;
                    counts[band] += 1;
                }
            }
        }
        offset += hop;
    }

    for i in 0..EMBED_DIM {
        if counts[i] > 0 {
            bands[i] = (bands[i] / counts[i] as f32 + 1e-8).ln();
        }
    }

    // L2 normalize
    let norm = bands.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for v in bands.iter_mut() {
        *v /= norm;
    }
    bands
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-8 {
        0.0
    } else {
        dot / denom
    }
}

/// Shared handle used by the transcription worker pool.
pub type SharedDiarizer = Arc<Mutex<OnlineDiarizer>>;

pub fn shared_channel_only() -> SharedDiarizer {
    Arc::new(Mutex::new(OnlineDiarizer::channel_only()))
}

pub fn shared_from_models_dir(models_dir: Option<&Path>) -> SharedDiarizer {
    let diarizer = match models_dir {
        Some(dir) => OnlineDiarizer::try_load(dir),
        None => OnlineDiarizer::channel_only(),
    };
    Arc::new(Mutex::new(diarizer))
}
