//! Prompt token construction for Qwen3-ASR.

use crate::qwen_engine::config::SpecialTokens;

/// Fixed GPT-2 / Qwen BPE IDs (stable across Qwen3-ASR packs).
const SYSTEM_TOKEN_ID: i64 = 9125; // "system"
const USER_TOKEN_ID: i64 = 882; // "user"
const ASSISTANT_TOKEN_ID: i64 = 77091; // "assistant"
const NEWLINE_TOKEN_ID: i64 = 198; // "\n"
const LANGUAGE_TOKEN_ID: i64 = 11528; // "language"

/// Encoder token count from mel frame count (100-frame windows, 3× stride-2).
pub fn get_feat_extract_output_lengths(input_lengths: usize) -> usize {
    let leave = input_lengths % 100;
    let mut t = leave.div_ceil(2);
    t = t.div_ceil(2);
    t = t.div_ceil(2);
    t + (input_lengths / 100) * 13
}

/// Build prompt IDs. When `language_token_ids` is set, append
/// `language` + ` {Name}` tokens + `<asr_text>`.
pub fn build_prompt_ids_with_language(
    special_tokens: &SpecialTokens,
    audio_token_count: usize,
    language_token_ids: Option<&[i64]>,
) -> Vec<i64> {
    let mut ids = Vec::with_capacity(audio_token_count + 24);

    ids.push(special_tokens.im_start_token_id);
    ids.push(SYSTEM_TOKEN_ID);
    ids.push(NEWLINE_TOKEN_ID);
    ids.push(special_tokens.im_end_token_id);
    ids.push(NEWLINE_TOKEN_ID);

    ids.push(special_tokens.im_start_token_id);
    ids.push(USER_TOKEN_ID);
    ids.push(NEWLINE_TOKEN_ID);
    ids.push(special_tokens.audio_start_token_id);
    ids.extend(std::iter::repeat(special_tokens.audio_pad_token_id).take(audio_token_count));
    ids.push(special_tokens.audio_end_token_id);
    ids.push(special_tokens.im_end_token_id);
    ids.push(NEWLINE_TOKEN_ID);

    ids.push(special_tokens.im_start_token_id);
    ids.push(ASSISTANT_TOKEN_ID);
    ids.push(NEWLINE_TOKEN_ID);

    if let Some(lang_ids) = language_token_ids {
        ids.push(LANGUAGE_TOKEN_ID);
        ids.extend_from_slice(lang_ids);
        ids.push(special_tokens.asr_text_token_id);
    }

    ids
}

pub fn get_audio_pad_range(
    prompt_ids: &[i64],
    audio_pad_token_id: i64,
) -> Result<(usize, usize), String> {
    let start = prompt_ids
        .iter()
        .position(|&id| id == audio_pad_token_id)
        .ok_or_else(|| "No audio_pad tokens in prompt".to_string())?;
    let end = prompt_ids
        .iter()
        .rposition(|&id| id == audio_pad_token_id)
        .ok_or_else(|| "No audio_pad tokens in prompt".to_string())?
        + 1;
    Ok((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st() -> SpecialTokens {
        SpecialTokens {
            eos_token_ids: vec![151643, 151645],
            pad_token_id: 151643,
            im_start_token_id: 151644,
            im_end_token_id: 151645,
            audio_start_token_id: 151669,
            audio_end_token_id: 151670,
            audio_pad_token_id: 151676,
            asr_text_token_id: 151704,
        }
    }

    #[test]
    fn feat_extract_lengths() {
        assert_eq!(get_feat_extract_output_lengths(0), 0);
        assert_eq!(get_feat_extract_output_lengths(1), 1);
        assert_eq!(get_feat_extract_output_lengths(99), 13);
        assert_eq!(get_feat_extract_output_lengths(100), 13);
        assert_eq!(get_feat_extract_output_lengths(200), 26);
        assert_eq!(get_feat_extract_output_lengths(997), 130);
    }

    #[test]
    fn language_hint_appends_asr_text() {
        let ids = build_prompt_ids_with_language(&st(), 5, Some(&[6364]));
        assert_eq!(ids[20], LANGUAGE_TOKEN_ID);
        assert_eq!(ids[21], 6364);
        assert_eq!(ids[22], 151704);
        assert_eq!(ids.len(), 23);
    }

    #[test]
    fn audio_pad_range() {
        let ids = build_prompt_ids_with_language(&st(), 10, None);
        let (start, end) = get_audio_pad_range(&ids, 151676).unwrap();
        assert_eq!((start, end), (9, 19));
    }
}
