//! Online speaker assignment for system-audio ASR segments.
//!
//! Pipeline: VAD segment → WeSpeaker ONNX embedding (spectral fallback) →
//! cosine clustering with frozen centroids → `speaker_N` labels.
//! Missing models → channel-only provisional labels.

use super::embedding::SpeakerEmbeddingExtractor;
use super::models::DiarizationModelManager;
use crate::audio::recording_state::DeviceType;
use log::{debug, info, warn};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Cosine similarity threshold for WeSpeaker space.
/// pyannote-rs examples use 0.5; plan range is ~0.75–0.85. Start at 0.75
/// (stricter match → less collapse across similar voices).
const SIMILARITY_THRESHOLD: f32 = 0.75;
const MAX_SPEAKERS: usize = 10;
const SPECTRAL_EMBED_DIM: usize = 64;
/// Minimum segment length for a stable embedding (~1s at 16 kHz).
const MIN_SAMPLES: usize = 16_000;

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
    backend: Option<ClusteringBackend>,
    /// User toggle ("Enable neural-style clustering"). Models may be present while this is false.
    user_enabled: bool,
}

struct ClusteringBackend {
    centroids: Vec<Vec<f32>>,
    /// WeSpeaker ONNX session when load succeeded; otherwise spectral fallback.
    extractor: Option<SpeakerEmbeddingExtractor>,
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

    /// Enable neural clustering when the WeSpeaker embedding model is present.
    /// Prefers WeSpeaker ONNX embeddings; falls back to spectral fingerprints if
    /// the session fails to load. Segmentation ONNX is not required for clustering
    /// (speaker-change detection via segmentation remains out of scope).
    pub fn try_load(models_dir: &Path) -> Self {
        let user_enabled = neural_enabled_preference();
        let embedding_path = models_dir.join("wespeaker_en_voxceleb_CAM++.onnx");
        if !embedding_path.exists() {
            warn!(
                "WeSpeaker model missing under {} — falling back to channel-only",
                models_dir.display()
            );
            return Self {
                backend: None,
                user_enabled,
            };
        }

        let extractor = match SpeakerEmbeddingExtractor::try_load(&embedding_path) {
            Ok(ext) => {
                info!(
                    "OnlineDiarizer: WeSpeaker embeddings enabled (neural preference={})",
                    user_enabled
                );
                Some(ext)
            }
            Err(e) => {
                warn!(
                    "OnlineDiarizer: WeSpeaker load failed ({}); using spectral embedding fallback",
                    e
                );
                None
            }
        };

        Self {
            backend: Some(ClusteringBackend {
                centroids: Vec::new(),
                extractor,
            }),
            user_enabled,
        }
    }

    pub fn try_from_manager(manager: &DiarizationModelManager) -> Self {
        Self::try_load(manager.models_dir())
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

        if samples_f32.len() < MIN_SAMPLES {
            debug!(
                "Diarizer: segment too short ({} samples < {}); labeling others",
                samples_f32.len(),
                MIN_SAMPLES
            );
            return "others".to_string();
        }

        let embedding = match backend.compute_embedding(samples_f32) {
            Some(e) => e,
            None => {
                debug!("Diarizer: embedding failed — using others");
                return "others".to_string();
            }
        };

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

impl ClusteringBackend {
    fn compute_embedding(&mut self, samples_f32: &[f32]) -> Option<Vec<f32>> {
        if let Some(ref mut extractor) = self.extractor {
            match extractor.embed(samples_f32) {
                Ok(emb) => return Some(emb),
                Err(e) => {
                    warn!(
                        "WeSpeaker embed failed ({}); falling back to spectral for this segment",
                        e
                    );
                }
            }
        }
        Some(compute_spectral_embedding(samples_f32))
    }

    fn assign(&mut self, embedding: &[f32]) -> Option<usize> {
        let mut best_idx = None;
        let mut best_sim = f32::NEG_INFINITY;
        let mut second_best = f32::NEG_INFINITY;

        for (i, centroid) in self.centroids.iter().enumerate() {
            let sim = cosine_similarity(embedding, centroid);
            if sim > best_sim {
                second_best = best_sim;
                best_sim = sim;
                best_idx = Some(i);
            } else if sim > second_best {
                second_best = sim;
            }
        }

        debug!(
            "Diarizer cluster: best_sim={:.4}, second_best={:.4}, centroids={}, threshold={:.2}",
            if best_sim.is_finite() { best_sim } else { 0.0 },
            if second_best.is_finite() {
                second_best
            } else {
                0.0
            },
            self.centroids.len(),
            SIMILARITY_THRESHOLD
        );

        if let Some(idx) = best_idx {
            if best_sim > SIMILARITY_THRESHOLD {
                // Freeze centroid on create — no EMA drift.
                debug!(
                    "Diarizer matched speaker_{} (sim={:.4})",
                    idx + 1,
                    best_sim
                );
                return Some(idx);
            }
        }

        if self.centroids.len() < MAX_SPEAKERS {
            self.centroids.push(embedding.to_vec());
            let idx = self.centroids.len() - 1;
            debug!(
                "Diarizer new speaker_{} (best_sim={:.4} below threshold)",
                idx + 1,
                if best_sim.is_finite() { best_sim } else { 0.0 }
            );
            return Some(idx);
        }

        // At capacity — force best match even below threshold
        debug!(
            "Diarizer at capacity; forcing speaker_{} (sim={:.4})",
            best_idx.map(|i| i + 1).unwrap_or(1),
            if best_sim.is_finite() { best_sim } else { 0.0 }
        );
        best_idx.or_else(|| {
            self.centroids
                .iter()
                .enumerate()
                .map(|(i, c)| (i, cosine_similarity(embedding, c)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
        })
    }
}

/// Compact log-mel style spectral fingerprint used as a last-resort embedding.
fn compute_spectral_embedding(samples: &[f32]) -> Vec<f32> {
    let mut bands = vec![0.0f32; SPECTRAL_EMBED_DIM];
    let mut counts = vec![0u32; SPECTRAL_EMBED_DIM];

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
                let band = (k * SPECTRAL_EMBED_DIM) / n;
                if band < SPECTRAL_EMBED_DIM {
                    bands[band] += mag;
                    counts[band] += 1;
                }
            }
        }
        offset += hop;
    }

    for i in 0..SPECTRAL_EMBED_DIM {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::recording_state::DeviceType;

    fn l2(v: &mut [f32]) {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for x in v.iter_mut() {
            *x /= n;
        }
    }

    #[test]
    fn clustering_assigns_multiple_speakers_for_dissimilar_embeddings() {
        let mut backend = ClusteringBackend {
            centroids: Vec::new(),
            extractor: None,
        };

        let mut e1 = vec![0.0f32; SPECTRAL_EMBED_DIM];
        e1[0] = 1.0;
        l2(&mut e1);

        let mut e2 = vec![0.0f32; SPECTRAL_EMBED_DIM];
        e2[1] = 1.0;
        l2(&mut e2);

        assert_eq!(backend.assign(&e1), Some(0));
        assert_eq!(backend.assign(&e2), Some(1));
        // Near-identical to first speaker should rematch speaker_1 (idx 0)
        assert_eq!(backend.assign(&e1), Some(0));
        assert_eq!(backend.centroids.len(), 2);
    }

    #[test]
    fn assign_speaker_mic_is_you_and_short_system_is_others() {
        let mut diarizer = OnlineDiarizer {
            backend: Some(ClusteringBackend {
                centroids: Vec::new(),
                extractor: None,
            }),
            user_enabled: true,
        };

        assert_eq!(
            diarizer.assign_speaker(&DeviceType::Microphone, &[0.1; 32_000]),
            "you"
        );
        // Below MIN_SAMPLES → others
        assert_eq!(
            diarizer.assign_speaker(&DeviceType::System, &[0.1; 8_000]),
            "others"
        );
    }

    #[test]
    fn centroids_are_frozen_on_match() {
        let mut backend = ClusteringBackend {
            centroids: Vec::new(),
            extractor: None,
        };
        let mut e1 = vec![0.0f32; SPECTRAL_EMBED_DIM];
        e1[0] = 1.0;
        l2(&mut e1);
        backend.assign(&e1);
        let before = backend.centroids[0].clone();

        // Slightly perturbed but still above threshold vs frozen centroid
        let mut e1b = e1.clone();
        e1b[0] = 0.99;
        e1b[1] = 0.1;
        l2(&mut e1b);
        assert_eq!(backend.assign(&e1b), Some(0));
        assert_eq!(
            backend.centroids[0], before,
            "matched centroid must not EMA-update"
        );
    }
}
