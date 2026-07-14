//! Whisper-compatible log-mel spectrogram for Qwen3-ASR.
//!
//! Parameters: 128 mel bins, Hann window, n_fft=400, hop=160, Slaney mel scale.
//! Adapted from the community ONNX Qwen3-ASR frontend (transcribe-rs).

#![allow(clippy::needless_range_loop)]

use ndarray::Array3;
use once_cell::sync::Lazy;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

pub const SAMPLE_RATE: u32 = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
const FMIN: f64 = 0.0;
const FMAX: f64 = 8000.0;

/// Mel frontend configuration (from pack `config.json` / defaults).
#[derive(Debug, Clone)]
pub struct MelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
}

impl MelConfig {
    pub fn from_qwen_defaults() -> Self {
        Self {
            sample_rate: SAMPLE_RATE,
            n_fft: N_FFT,
            hop_length: HOP_LENGTH,
            n_mels: N_MELS,
        }
    }
}

static HANN_WINDOW: Lazy<Vec<f64>> = Lazy::new(|| hann_window(N_FFT));
static MEL_FILTERS: Lazy<Vec<Vec<f64>>> =
    Lazy::new(|| slaney_mel_filterbank(SAMPLE_RATE as f64, N_FFT, N_MELS, FMIN, FMAX));

/// Returns mel features shaped `[1, n_mels, time]` as f32.
pub fn log_mel_spectrogram(samples: &[f32], cfg: &MelConfig) -> Result<Array3<f32>, String> {
    if cfg.n_fft != N_FFT || cfg.hop_length != HOP_LENGTH || cfg.n_mels != N_MELS {
        return Err(format!(
            "unsupported mel config: n_fft={}, hop={}, n_mels={} (expected {}/{}/{})",
            cfg.n_fft, cfg.hop_length, cfg.n_mels, N_FFT, HOP_LENGTH, N_MELS
        ));
    }
    Ok(log_mel_spectrogram_impl(samples))
}

fn log_mel_spectrogram_impl(audio: &[f32]) -> Array3<f32> {
    if audio.len() <= N_FFT / 2 {
        return Array3::zeros((1, N_MELS, 0));
    }

    let mel_filters = &*MEL_FILTERS;
    let window = &*HANN_WINDOW;
    let num_fft_bins = N_FFT / 2 + 1;
    let num_frames = audio.len() / HOP_LENGTH + 1;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(N_FFT);

    let mut magnitudes = vec![vec![0.0f64; num_frames]; num_fft_bins];

    for frame_idx in 0..num_frames {
        let center = frame_idx * HOP_LENGTH;
        let mut fft_input: Vec<Complex<f64>> = Vec::with_capacity(N_FFT);

        for i in 0..N_FFT {
            let sample_idx = center as isize + i as isize - (N_FFT / 2) as isize;
            let sample = if sample_idx < 0 {
                let reflect_idx = (-sample_idx) as usize;
                if reflect_idx < audio.len() {
                    audio[reflect_idx] as f64
                } else {
                    0.0
                }
            } else if (sample_idx as usize) >= audio.len() {
                let overshoot = sample_idx as usize - audio.len();
                if audio.len() > 1 && overshoot < audio.len() {
                    audio[audio.len().saturating_sub(2).saturating_sub(overshoot)] as f64
                } else {
                    0.0
                }
            } else {
                audio[sample_idx as usize] as f64
            };
            fft_input.push(Complex::new(sample * window[i], 0.0));
        }

        fft.process(&mut fft_input);

        for k in 0..num_fft_bins {
            magnitudes[k][frame_idx] = fft_input[k].norm_sqr();
        }
    }

    // Drop last STFT frame (WhisperFeatureExtractor quirk)
    let time_steps = num_frames - 1;

    let mut mel_spec = vec![vec![0.0f64; time_steps]; N_MELS];
    for m in 0..N_MELS {
        for k in 0..num_fft_bins {
            if mel_filters[m][k] != 0.0 {
                for t in 0..time_steps {
                    mel_spec[m][t] += mel_filters[m][k] * magnitudes[k][t];
                }
            }
        }
    }

    let mut log_spec = vec![vec![0.0f32; time_steps]; N_MELS];
    let mut global_max: f32 = f32::NEG_INFINITY;

    for m in 0..N_MELS {
        for t in 0..time_steps {
            let val = (mel_spec[m][t].max(1e-10).log10()) as f32;
            log_spec[m][t] = val;
            if val > global_max {
                global_max = val;
            }
        }
    }

    let floor = global_max - 8.0;
    for m in 0..N_MELS {
        for t in 0..time_steps {
            log_spec[m][t] = (log_spec[m][t].max(floor) + 4.0) / 4.0;
        }
    }

    let mut result = Array3::<f32>::zeros((1, N_MELS, time_steps));
    for m in 0..N_MELS {
        for t in 0..time_steps {
            result[[0, m, t]] = log_spec[m][t];
        }
    }
    result
}

fn hann_window(length: usize) -> Vec<f64> {
    (0..length)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / length as f64).cos()))
        .collect()
}

fn slaney_mel_filterbank(
    sr: f64,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
) -> Vec<Vec<f64>> {
    let num_fft_bins = n_fft / 2 + 1;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let log_step = 6.4_f64.ln() / 27.0;

    let hz_to_mel = |hz: f64| -> f64 {
        if hz < min_log_hz {
            hz / f_sp
        } else {
            min_log_mel + (hz / min_log_hz).ln() / log_step
        }
    };
    let mel_to_hz = |mel: f64| -> f64 {
        if mel < min_log_mel {
            mel * f_sp
        } else {
            min_log_hz * ((mel - min_log_mel) * log_step).exp()
        }
    };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let n_points = n_mels + 2;
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_points - 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let fft_freqs: Vec<f64> = (0..num_fft_bins)
        .map(|k| k as f64 * sr / n_fft as f64)
        .collect();

    let mut filters = vec![vec![0.0f64; num_fft_bins]; n_mels];
    for m in 0..n_mels {
        let left = hz_points[m];
        let center = hz_points[m + 1];
        let right = hz_points[m + 2];
        let enorm = 2.0 / (right - left);
        for k in 0..num_fft_bins {
            let freq = fft_freqs[k];
            if freq > left && freq < center {
                filters[m][k] = enorm * (freq - left) / (center - left);
            } else if freq >= center && freq < right {
                filters[m][k] = enorm * (right - freq) / (right - center);
            }
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_has_expected_channel_and_mel_dims() {
        let cfg = MelConfig::from_qwen_defaults();
        let samples = vec![0.0f32; cfg.sample_rate as usize];
        let mel = log_mel_spectrogram(&samples, &cfg).unwrap();
        assert_eq!(mel.shape()[0], 1);
        assert_eq!(mel.shape()[1], cfg.n_mels);
        assert!(mel.shape()[2] > 0);
        assert_eq!(mel.shape()[2], 100); // 16000/160 + 1 - 1
    }
}
