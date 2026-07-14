//! Map Meetily language preferences to Nemotron `target_lang` locales.
//!
//! Nemotron does **not** translate to English. `auto-translate` maps to `auto`
//! (native language), same practical limitation as Parakeet today.

/// Preferred Meetily ISO code → Nemotron locale for codes that need a specific region.
const PREFERRED_LOCALES: &[(&str, &str)] = &[
    ("en", "en-US"),
    ("es", "es-ES"),
    ("pt", "pt-BR"),
    ("zh", "zh-CN"),
    ("no", "nb-NO"),
    ("nb", "nb-NO"),
    ("nn", "nn-NO"),
    ("fr", "fr-FR"),
    ("de", "de-DE"),
    ("it", "it-IT"),
    ("nl", "nl-NL"),
    ("pl", "pl-PL"),
    ("ru", "ru-RU"),
    ("uk", "uk-UA"),
    ("ja", "ja-JP"),
    ("ko", "ko-KR"),
    ("ar", "ar-AR"),
    ("hi", "hi-IN"),
    ("tr", "tr-TR"),
    ("vi", "vi-VN"),
    ("sv", "sv-SE"),
    ("da", "da-DK"),
    ("fi", "fi-FI"),
    ("cs", "cs-CZ"),
    ("sk", "sk-SK"),
    ("hr", "hr-HR"),
    ("bg", "bg-BG"),
    ("ro", "ro-RO"),
    ("hu", "hu-HU"),
    ("el", "el-GR"),
    ("he", "he-IL"),
    ("th", "th-TH"),
    ("id", "id-ID"),
    ("ms", "ms-MY"),
    ("fa", "fa-IR"),
    ("ur", "ur-PK"),
    ("bn", "bn-IN"),
    ("ta", "ta-IN"),
    ("te", "te-IN"),
    ("mr", "mr-IN"),
    ("gu", "gu-IN"),
    ("kn", "kn-IN"),
    ("ml", "ml-IN"),
    ("et", "et-EE"),
    ("lv", "lv-LV"),
    ("lt", "lt-LT"),
    ("sl", "sl-SI"),
    ("mt", "mt-MT"),
];

/// Locales accepted by Nemotron's multilingual prompt dictionary (subset used for validation).
const KNOWN_NEMOTRON_LOCALES: &[&str] = &[
    "auto", "en-US", "en-GB", "es-ES", "es-US", "pt-BR", "pt-PT", "zh-CN", "zh-TW",
    "fr-FR", "fr-CA", "de-DE", "it-IT", "nl-NL", "pl-PL", "ru-RU", "uk-UA", "ja-JP",
    "ko-KR", "ar-AR", "hi-IN", "tr-TR", "vi-VN", "sv-SE", "da-DK", "fi-FI", "cs-CZ",
    "sk-SK", "hr-HR", "bg-BG", "ro-RO", "hu-HU", "el-GR", "he-IL", "th-TH", "id-ID",
    "ms-MY", "fa-IR", "ur-PK", "nb-NO", "nn-NO", "no-NO", "et-EE", "lv-LV", "lt-LT",
    "sl-SI", "mt-MT",
];

/// Map a Meetily language preference to a Nemotron `target_lang` string.
///
/// | Meetily pref              | Nemotron `target_lang` |
/// |---------------------------|------------------------|
/// | `auto`, `auto-translate`, unset | `auto`          |
/// | `en`                      | `en-US`                |
/// | `es`                      | `es-ES`                |
/// | `pt`                      | `pt-BR`                |
/// | `zh`                      | `zh-CN`                |
/// | `no`                      | `nb-NO`                |
/// | other known ISO           | preferred locale       |
/// | unsupported               | `auto` (+ warn)        |
pub fn map_language_preference(pref: Option<&str>) -> String {
    let Some(raw) = pref.map(str::trim).filter(|s| !s.is_empty()) else {
        return "auto".to_string();
    };

    let lower = raw.to_ascii_lowercase();
    if lower == "auto" || lower == "auto-translate" {
        return "auto".to_string();
    }

    // Already a locale like en-US / zh-CN
    if raw.contains('-') {
        if KNOWN_NEMOTRON_LOCALES
            .iter()
            .any(|l| l.eq_ignore_ascii_case(raw))
        {
            // Normalize to canonical casing from the known list when possible
            if let Some(canonical) = KNOWN_NEMOTRON_LOCALES
                .iter()
                .find(|l| l.eq_ignore_ascii_case(raw))
            {
                return canonical.to_string();
            }
            return raw.to_string();
        }
        log::warn!(
            "Unsupported Nemotron locale '{}', falling back to auto",
            raw
        );
        return "auto".to_string();
    }

    // ISO 639-1 / short code
    if let Some((_, locale)) = PREFERRED_LOCALES.iter().find(|(code, _)| *code == lower) {
        return locale.to_string();
    }

    log::warn!(
        "Unsupported Meetily language preference '{}' for Nemotron, falling back to auto",
        raw
    );
    "auto".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_auto_variants() {
        assert_eq!(map_language_preference(None), "auto");
        assert_eq!(map_language_preference(Some("auto")), "auto");
        assert_eq!(map_language_preference(Some("auto-translate")), "auto");
        assert_eq!(map_language_preference(Some("")), "auto");
    }

    #[test]
    fn maps_preferred_iso_codes() {
        assert_eq!(map_language_preference(Some("en")), "en-US");
        assert_eq!(map_language_preference(Some("es")), "es-ES");
        assert_eq!(map_language_preference(Some("pt")), "pt-BR");
        assert_eq!(map_language_preference(Some("zh")), "zh-CN");
        assert_eq!(map_language_preference(Some("no")), "nb-NO");
    }

    #[test]
    fn maps_existing_locales() {
        assert_eq!(map_language_preference(Some("en-US")), "en-US");
        assert_eq!(map_language_preference(Some("zh-cn")), "zh-CN");
    }

    #[test]
    fn unsupported_falls_back_to_auto() {
        assert_eq!(map_language_preference(Some("xx")), "auto");
        assert_eq!(map_language_preference(Some("zz-ZZ")), "auto");
    }
}
