use std::sync::{Arc, Mutex};

use anyhow::Result;

pub struct MicCapture {
    samples: Arc<Mutex<Vec<f32>>>,
    // hold cpal stream in struct so it stays alive
    _stream: Option<cpal::Stream>,
    sample_rate: u32,
}

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
        let samples_cb = samples.clone();
        let stream = device.build_input_stream(
            &config.into(),
            move |data: &[f32], _| {
                let mut buf = samples_cb.lock().unwrap();
                if channels <= 1 {
                    buf.extend_from_slice(data);
                } else {
                    for frame in data.chunks(channels) {
                        buf.push(frame[0]);
                    }
                }
            },
            |err| log::error!("dictation mic stream error: {err}"),
            None,
        )?;
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

fn resample_to_16k(input: &[f32], from_hz: u32) -> Vec<f32> {
    // Use rubato (already in Cargo.toml). Keep a small local helper; do not pull meeting pipeline.
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };
    let ratio = 16_000f64 / from_hz as f64;
    let chunk_size = 1024;
    let mut resampler = SincFixedIn::<f32>::new(
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
    )
    .expect("resampler");
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + chunk_size <= input.len() {
        let chunk = vec![input[pos..pos + chunk_size].to_vec()];
        if let Ok(waves) = resampler.process(&chunk, None) {
            out.extend_from_slice(&waves[0]);
        }
        pos += chunk_size;
    }
    out
}
