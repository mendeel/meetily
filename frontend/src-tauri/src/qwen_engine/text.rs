//! Transcript text cleanup for Qwen3-ASR outputs.

/// Language names Qwen3-ASR may emit as a leading token before the transcript.
const LANG_NAMES: &[&str] = &[
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Hungarian",
    "Macedonian",
    "Romanian",
];

/// Qwen3-ASR often emits a leading language name before the transcript.
/// Strip a known language token prefix when present.
pub fn strip_language_prefix(raw: &str) -> String {
    let trimmed = raw.trim();
    let Some((first, rest)) = trimmed.split_once(char::is_whitespace) else {
        return trimmed.to_string();
    };
    if LANG_NAMES.iter().any(|l| l.eq_ignore_ascii_case(first)) {
        rest.trim().to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_english_prefix() {
        assert_eq!(
            strip_language_prefix("English Hello world"),
            "Hello world"
        );
    }

    #[test]
    fn strips_chinese_prefix() {
        assert_eq!(strip_language_prefix("Chinese 你好"), "你好");
    }

    #[test]
    fn leaves_plain_text() {
        assert_eq!(strip_language_prefix("Hello world"), "Hello world");
    }

    #[test]
    fn trims_whitespace() {
        assert_eq!(strip_language_prefix("  English  hi  "), "hi");
    }
}
