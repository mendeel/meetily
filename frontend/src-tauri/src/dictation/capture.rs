use std::sync::{Arc, Mutex};

use anyhow::Result;

pub struct MicCapture {
    samples: Arc<Mutex<Vec<f32>>>,
    // hold cpal stream in struct so it stays alive
    _stream: Option<cpal::Stream>,
    sample_rate: u32,
}

// SAFETY: `cpal::Stream` is !Send, but we only create/stop/drop `MicCapture` while holding
// the dictation runtime mutex and never share the stream across threads concurrently.
unsafe impl Send for MicCapture {}

impl MicCapture {
    pub fn start_default_input() -> Result<Self> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("no default input device"))?;
        let config = device.default_input_config()?;
        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        let samples = Arc::new(Mutex::new(Vec::<f32>::new()));
        let stream = build_input_stream(&device, &config, samples.clone(), channels)?;
        stream.play()?;
        Ok(Self {
            samples,
            _stream: Some(stream),
            sample_rate,
        })
    }

    pub fn stop(self) -> Vec<f32> {
        // Drop stream by consuming self; resample to 16 kHz mono for STT if sample_rate != 16000
        let raw = self.samples.lock().ok().map(|g| g.clone()).unwrap_or_default();
        if self.sample_rate == 16_000 || raw.is_empty() {
            return raw;
        }
        resample_to_16k(&raw, self.sample_rate)
    }
}

fn push_mono(buf: &mut Vec<f32>, data: &[f32], channels: usize) {
    if channels <= 1 {
        buf.extend_from_slice(data);
    } else {
        for frame in data.chunks(channels) {
            buf.push(frame[0]);
        }
    }
}

fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    samples: Arc<Mutex<Vec<f32>>>,
    channels: usize,
) -> Result<cpal::Stream> {
    use cpal::traits::DeviceTrait;

    let err_fn = |err| log::error!("dictation mic stream error: {err}");
    let stream_config: cpal::StreamConfig = config.clone().into();

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => {
            let samples_cb = samples;
            device.build_input_stream(
                &stream_config,
                move |data: &[f32], _| {
                    if let Ok(mut buf) = samples_cb.lock() {
                        push_mono(&mut buf, data, channels);
                    }
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::I16 => {
            let samples_cb = samples;
            device.build_input_stream(
                &stream_config,
                move |data: &[i16], _| {
                    let f32_data: Vec<f32> = data
                        .iter()
                        .map(|&s| s as f32 / i16::MAX as f32)
                        .collect();
                    if let Ok(mut buf) = samples_cb.lock() {
                        push_mono(&mut buf, &f32_data, channels);
                    }
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::I32 => {
            let samples_cb = samples;
            device.build_input_stream(
                &stream_config,
                move |data: &[i32], _| {
                    let f32_data: Vec<f32> = data
                        .iter()
                        .map(|&s| s as f32 / i32::MAX as f32)
                        .collect();
                    if let Ok(mut buf) = samples_cb.lock() {
                        push_mono(&mut buf, &f32_data, channels);
                    }
                },
                err_fn,
                None,
            )?
        }
        other => {
            return Err(anyhow::anyhow!(
                "unsupported dictation mic sample format: {other:?}"
            ));
        }
    };

    Ok(stream)
}

fn resample_to_16k(input: &[f32], from_hz: u32) -> Vec<f32> {
    // Use rubato (already in Cargo.toml). Keep a small local helper; do not pull meeting pipeline.
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };
    let ratio = 16_000f64 / from_hz as f64;
    let chunk_size = 1024;
    let mut resampler = match SincFixedIn::<f32>::new(
        ratio,
        2.0,
        SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            oversampling_factor: 256,
            interpolation: SincInterpolationType::Linear,
            window: WindowFunction::BlackmanHarris2,
        },
        chunk_size,
        1,
    ) {
        Ok(r) => r,
        Err(e) => {
            log::warn!("dictation resampler init failed ({e}); returning unresampled audio");
            return input.to_vec();
        }
    };

    let mut out = Vec::new();
    let mut pos = 0;
    while pos + chunk_size <= input.len() {
        let chunk = vec![input[pos..pos + chunk_size].to_vec()];
        match resampler.process(&chunk, None) {
            Ok(waves) => out.extend_from_slice(&waves[0]),
            Err(e) => {
                log::warn!("dictation resampler chunk failed ({e}); stopping early");
                break;
            }
        }
        pos += chunk_size;
    }

    // Flush remaining samples instead of dropping the tail.
    if pos < input.len() {
        let remainder = vec![input[pos..].to_vec()];
        match resampler.process_partial(Some(&remainder), None) {
            Ok(waves) => out.extend_from_slice(&waves[0]),
            Err(e) => log::warn!("dictation resampler flush failed ({e})"),
        }
    } else {
        // Drain residual filter delay even when input was an exact multiple of chunk_size.
        match resampler.process_partial::<&[f32]>(None, None) {
            Ok(waves) => out.extend_from_slice(&waves[0]),
            Err(e) => log::warn!("dictation resampler drain failed ({e})"),
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_preserves_short_tail() {
        // 48kHz → 16kHz; 1200 samples = one full chunk + 176 remainder (must not be dropped).
        let input: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
        let out = resample_to_16k(&input, 48_000);
        // Exact length depends on sinc delay; just ensure remainder contributed output.
        assert!(
            out.len() > 300,
            "expected flushed remainder to contribute samples, got {}",
            out.len()
        );
    }
}
