//! Qwen3-ASR ONNX inference (encoder + decoder_init + decoder_step).

use ndarray::{Array1, Array2, Array3, ArrayD, IxDyn};
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{DynValue, TensorRef};
use std::path::Path;
use tokenizers::Tokenizer;

use crate::qwen_engine::config::{EmbedDtype, QwenAsrConfig};
use crate::qwen_engine::mel::{self, log_mel_spectrogram, MelConfig};
use crate::qwen_engine::prompt::{
    build_prompt_ids_with_language, get_audio_pad_range, get_feat_extract_output_lengths,
};
use crate::qwen_engine::text::strip_language_prefix;

pub struct QwenModel {
    encoder: Session,
    decoder_init: Session,
    decoder_step: Session,
    /// FP32 cache of `embed_tokens.bin` `[vocab, hidden]` for decoder_step lookups.
    embed_tokens: Array2<f32>,
    config: QwenAsrConfig,
    tokenizer: Tokenizer,
    max_new_tokens: usize,
}

impl QwenModel {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let config = QwenAsrConfig::load(model_dir)?;
        validate_special_tokens(&config)?;
        validate_mel_params(&config)?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("load tokenizer {}: {e}", tokenizer_path.display()))?;

        let encoder = init_session(model_dir, "encoder.int4.onnx")
            .map_err(|e| hint_1_7b(model_dir, format!("load encoder: {e}")))?;
        let decoder_init = init_session(model_dir, "decoder_init.int4.onnx")
            .map_err(|e| hint_1_7b(model_dir, format!("load decoder_init: {e}")))?;
        let decoder_step = init_session(model_dir, "decoder_step.int4.onnx")
            .map_err(|e| hint_1_7b(model_dir, format!("load decoder_step: {e}")))?;

        let embed_path = model_dir.join("embed_tokens.bin");
        let embed_tokens = load_embed_cache(&embed_path, &config)
            .map_err(|e| hint_1_7b(model_dir, format!("load embed_tokens: {e}")))?;

        let max_new_tokens = config.max_new_tokens_or_default();
        log::info!(
            "QwenModel loaded from {} (vocab={}, hidden={}, max_new_tokens={})",
            model_dir.display(),
            config.decoder.vocab_size,
            config.decoder.hidden_size,
            max_new_tokens
        );

        Ok(Self {
            encoder,
            decoder_init,
            decoder_step,
            embed_tokens,
            config,
            tokenizer,
            max_new_tokens,
        })
    }

    /// Transcribe 16 kHz mono `f32` samples. `language` is a full name hint
    /// (e.g. `"English"`); `None` = auto-detect.
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<String, String> {
        const MAX_SAMPLES: usize = 60 * 16_000;
        if samples.len() > MAX_SAMPLES {
            return Err(format!(
                "audio too long: {} samples ({:.1}s); max 60s",
                samples.len(),
                samples.len() as f32 / 16_000.0
            ));
        }

        let mel_cfg = MelConfig {
            sample_rate: self.config.mel.sample_rate,
            n_fft: self.config.mel.n_fft,
            hop_length: self.config.mel.hop_length,
            n_mels: self.config.mel.n_mels,
        };
        let mel = log_mel_spectrogram(samples, &mel_cfg)?;
        let mel_frames = mel.shape()[2];
        let expected_tokens = get_feat_extract_output_lengths(mel_frames);
        if expected_tokens == 0 {
            return Ok(String::new());
        }

        let audio_features = self.encode(&mel)?;
        let got = audio_features.shape()[1];
        if got != expected_tokens {
            return Err(format!(
                "encoder produced {got} tokens, expected {expected_tokens} (mel_frames={mel_frames})"
            ));
        }

        let lang_ids = match language.map(str::trim).filter(|s| !s.is_empty()) {
            Some(name) => Some(self.encode_language_hint(name)?),
            None => None,
        };
        let lang_ref = lang_ids.as_deref();

        let raw = self.greedy_decode(&audio_features, self.max_new_tokens, lang_ref)?;
        Ok(strip_language_prefix(&raw))
    }

    fn encode_language_hint(&self, language_name: &str) -> Result<Vec<i64>, String> {
        let spaced = format!(" {language_name}");
        let encoding = self
            .tokenizer
            .encode(spaced.as_str(), false)
            .map_err(|e| format!("tokenize language hint {spaced:?}: {e}"))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
    }

    fn encode(&mut self, mel: &Array3<f32>) -> Result<Array3<f32>, String> {
        let mel_dyn = mel.view().into_dyn();
        let inputs = inputs![
            "mel" => TensorRef::from_array_view(mel_dyn.view())
                .map_err(|e| format!("encoder input: {e}"))?,
        ];
        let outputs = self
            .encoder
            .run(inputs)
            .map_err(|e| format!("encoder run: {e}"))?;

        let features = outputs
            .get("audio_features")
            .ok_or_else(|| "missing encoder output 'audio_features'".to_string())?
            .try_extract_array::<f32>()
            .map_err(|e| format!("extract audio_features: {e}"))?;

        features
            .to_owned()
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| format!("audio_features shape: {e}"))
    }

    fn greedy_decode(
        &mut self,
        audio_features: &Array3<f32>,
        max_tokens: usize,
        language_token_ids: Option<&[i64]>,
    ) -> Result<String, String> {
        let max_tokens = max_tokens.min(4096);
        let audio_token_count = audio_features.shape()[1];
        let prompt_ids = build_prompt_ids_with_language(
            &self.config.special_tokens,
            audio_token_count,
            language_token_ids,
        );
        let seq_len = prompt_ids.len();

        let (audio_start, _) =
            get_audio_pad_range(&prompt_ids, self.config.special_tokens.audio_pad_token_id)?;

        let input_ids = Array2::<i64>::from_shape_vec((1, seq_len), prompt_ids.clone())
            .map_err(|e| e.to_string())?;
        let position_ids = Array2::<i64>::from_shape_fn((1, seq_len), |(_, j)| j as i64);
        let audio_offset = Array1::<i64>::from_elem(1, audio_start as i64);

        let ids_dyn = input_ids.into_dyn();
        let pos_dyn = position_ids.into_dyn();
        let audio_dyn = audio_features.view().into_dyn();
        let offset_dyn = audio_offset.into_dyn();

        let (mut current_token, mut keys, mut values) = {
            let inputs = inputs![
                "input_ids" => TensorRef::from_array_view(ids_dyn.view())
                    .map_err(|e| format!("decoder_init input_ids: {e}"))?,
                "position_ids" => TensorRef::from_array_view(pos_dyn.view())
                    .map_err(|e| format!("decoder_init position_ids: {e}"))?,
                "audio_features" => TensorRef::from_array_view(audio_dyn.view())
                    .map_err(|e| format!("decoder_init audio_features: {e}"))?,
                "audio_offset" => TensorRef::from_array_view(offset_dyn.view())
                    .map_err(|e| format!("decoder_init audio_offset: {e}"))?,
            ];
            let mut init_outputs = self
                .decoder_init
                .run(inputs)
                .map_err(|e| format!("decoder_init run: {e}"))?;

            let logits = init_outputs
                .get("logits")
                .ok_or_else(|| "missing 'logits' from decoder_init".to_string())?
                .try_extract_array::<f32>()
                .map_err(|e| format!("extract logits: {e}"))?;
            let last_pos = logits
                .shape()[1]
                .checked_sub(1)
                .ok_or_else(|| "decoder_init returned empty logits".to_string())?;
            let token = argmax_slice(&logits, last_pos)?;
            drop(logits);

            let keys: DynValue = init_outputs
                .remove("present_keys")
                .ok_or_else(|| "missing 'present_keys' from decoder_init".to_string())?;
            let values: DynValue = init_outputs
                .remove("present_values")
                .ok_or_else(|| "missing 'present_values' from decoder_init".to_string())?;
            (token, keys, values)
        };

        let mut output_tokens = vec![current_token];
        if self
            .config
            .special_tokens
            .eos_token_ids
            .contains(&current_token)
        {
            return self.decode_tokens(&output_tokens);
        }

        let hidden_size = self.config.decoder.hidden_size;
        let mut pos = seq_len as i64;

        for _ in 1..max_tokens {
            let token_embed = {
                if current_token < 0 {
                    return Err(format!("negative token id: {current_token}"));
                }
                let id = current_token as usize;
                if id >= self.embed_tokens.nrows() {
                    return Err(format!(
                        "token {id} exceeds embed rows {}",
                        self.embed_tokens.nrows()
                    ));
                }
                let row = self.embed_tokens.row(id);
                let mut arr = Array3::<f32>::zeros((1, 1, hidden_size));
                arr.slice_mut(ndarray::s![0, 0, ..]).assign(&row);
                arr.into_dyn()
            };

            let step_pos = ArrayD::<i64>::from_shape_vec(IxDyn(&[1, 1]), vec![pos])
                .map_err(|e| e.to_string())?;

            let step_inputs = inputs![
                "input_embeds" => TensorRef::from_array_view(token_embed.view())
                    .map_err(|e| format!("decoder_step input_embeds: {e}"))?,
                "position_ids" => TensorRef::from_array_view(step_pos.view())
                    .map_err(|e| format!("decoder_step position_ids: {e}"))?,
                "past_keys" => keys,
                "past_values" => values,
            ];
            let mut step_outputs = self
                .decoder_step
                .run(step_inputs)
                .map_err(|e| format!("decoder_step run: {e}"))?;

            let step_logits = step_outputs
                .get("logits")
                .ok_or_else(|| "missing 'logits' from decoder_step".to_string())?
                .try_extract_array::<f32>()
                .map_err(|e| format!("extract step logits: {e}"))?;

            current_token = argmax_slice(&step_logits, 0)?;
            output_tokens.push(current_token);
            pos += 1;
            drop(step_logits);

            keys = step_outputs
                .remove("present_keys")
                .ok_or_else(|| "missing 'present_keys' from decoder_step".to_string())?;
            values = step_outputs
                .remove("present_values")
                .ok_or_else(|| "missing 'present_values' from decoder_step".to_string())?;

            if self
                .config
                .special_tokens
                .eos_token_ids
                .contains(&current_token)
            {
                break;
            }
        }

        let eos_reached = self
            .config
            .special_tokens
            .eos_token_ids
            .contains(&current_token);

        let asr_text_id = self.config.special_tokens.asr_text_token_id;
        if language_token_ids.is_none() && eos_reached && !output_tokens.contains(&asr_text_id) {
            log::warn!(
                "Qwen ASR: no <asr_text> in output ({} tokens); returning empty",
                output_tokens.len()
            );
            return Ok(String::new());
        }

        if !eos_reached {
            log::warn!("Qwen ASR: max_new_tokens ({max_tokens}) reached without EOS");
        }

        self.decode_tokens(&output_tokens)
    }

    fn decode_tokens(&self, token_ids: &[i64]) -> Result<String, String> {
        let ids_u32: Vec<u32> = token_ids
            .iter()
            .copied()
            .filter(|&t| t >= 0)
            .map(|t| t as u32)
            .collect();
        self.tokenizer
            .decode(&ids_u32, true)
            .map_err(|e| format!("tokenizer decode: {e}"))
    }
}

fn init_session(model_dir: &Path, filename: &str) -> Result<Session, String> {
    let path = model_dir.join(filename);
    if !path.is_file() {
        return Err(format!("missing model file {}", path.display()));
    }

    let providers = vec![CPUExecutionProvider::default().build()];
    log::info!("Loading Qwen ONNX session from {}...", filename);

    Session::builder()
        .map_err(|e| format!("session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("optimization level: {e}"))?
        .with_execution_providers(providers)
        .map_err(|e| format!("execution providers: {e}"))?
        .with_parallel_execution(true)
        .map_err(|e| format!("parallel execution: {e}"))?
        .commit_from_file(&path)
        .map_err(|e| format!("commit {}: {e}", path.display()))
}

fn validate_special_tokens(config: &QwenAsrConfig) -> Result<(), String> {
    let st = &config.special_tokens;
    let ok = st.pad_token_id >= 0
        && st.im_start_token_id >= 0
        && st.im_end_token_id >= 0
        && st.audio_start_token_id >= 0
        && st.audio_end_token_id >= 0
        && st.audio_pad_token_id >= 0
        && st.asr_text_token_id >= 0
        && !st.eos_token_ids.is_empty()
        && st.eos_token_ids.iter().all(|&id| id >= 0);
    if ok {
        Ok(())
    } else {
        Err("config.json: special_tokens contains negative or missing IDs".into())
    }
}

fn validate_mel_params(config: &QwenAsrConfig) -> Result<(), String> {
    if config.mel.n_mels != mel::N_MELS {
        return Err(format!(
            "expected n_mels={}, got {}",
            mel::N_MELS,
            config.mel.n_mels
        ));
    }
    if config.mel.n_fft != mel::N_FFT {
        return Err(format!(
            "expected n_fft={}, got {}",
            mel::N_FFT,
            config.mel.n_fft
        ));
    }
    if config.mel.hop_length != mel::HOP_LENGTH {
        return Err(format!(
            "expected hop_length={}, got {}",
            mel::HOP_LENGTH,
            config.mel.hop_length
        ));
    }
    Ok(())
}

fn hint_1_7b(model_dir: &Path, msg: String) -> String {
    let dir = model_dir.to_string_lossy();
    if dir.contains("1.7") {
        format!(
            "{msg} (1.7B pack is large — try qwen3-asr-0.6b-int4 if this is an OOM/load failure)"
        )
    } else {
        msg
    }
}

fn load_embed_cache(path: &Path, config: &QwenAsrConfig) -> Result<Array2<f32>, String> {
    let vocab_size = config.decoder.vocab_size;
    let hidden_size = config.decoder.hidden_size;
    let n_elements = vocab_size * hidden_size;
    let bpe = config.embed_tokens_dtype.bytes_per_element();
    let expected_bytes = n_elements * bpe;

    let file_size = path
        .metadata()
        .map_err(|e| format!("stat {}: {e}", path.display()))?
        .len() as usize;
    if file_size != expected_bytes {
        return Err(format!(
            "embed_tokens.bin size {file_size} != expected {expected_bytes} \
             ({vocab_size}x{hidden_size}x{bpe}, dtype={:?})",
            config.embed_tokens_dtype
        ));
    }

    let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;

    let float_data: Vec<f32> = match config.embed_tokens_dtype {
        EmbedDtype::Float16 => {
            log::info!(
                "Loading FP16 embed_tokens ({} MB)",
                file_size / (1024 * 1024)
            );
            data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
        EmbedDtype::Float32 => {
            log::info!(
                "Loading FP32 embed_tokens ({} MB)",
                file_size / (1024 * 1024)
            );
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
    };

    Array2::from_shape_vec((vocab_size, hidden_size), float_data).map_err(|e| e.to_string())
}

#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut e = exp;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e += 1;
            }
            f &= 0x3FF;
            let exp32 = 127 - 15 - e + 1;
            f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13))
        }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13))
    } else {
        let exp32 = exp + 127 - 15;
        f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
    }
}

fn argmax_slice(logits: &ndarray::ArrayViewD<'_, f32>, pos: usize) -> Result<i64, String> {
    let shape = logits.shape();
    if logits.ndim() != 3 || shape[0] != 1 {
        return Err(format!("argmax: expected [1, T, vocab], got {shape:?}"));
    }
    if pos >= shape[1] {
        return Err(format!(
            "argmax: pos {pos} out of range for T={}",
            shape[1]
        ));
    }

    let row = logits.slice(ndarray::s![0, pos, ..]);
    if let Some(slice) = row.as_slice() {
        let best = slice
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            })
            .0;
        return Ok(best as i64);
    }

    let mut best_idx = 0i64;
    let mut best_val = f32::NEG_INFINITY;
    for v in 0..shape[2] {
        let val = logits[[0, pos, v]];
        if val > best_val {
            best_val = val;
            best_idx = v as i64;
        }
    }
    Ok(best_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn f16_one() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
    }

    #[test]
    #[ignore = "requires downloaded qwen3-asr-0.6b-int4 pack under models/"]
    fn transcribes_fixture_when_model_present() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/qwen3-asr-0.6b-int4");
        assert!(dir.is_dir(), "missing model dir {:?}", dir);
        let mut model = QwenModel::load(&dir).expect("load");
        let samples = vec![0.0f32; 16_000];
        let text = model.transcribe(&samples, Some("English")).expect("tx");
        let _ = text;
    }
}
