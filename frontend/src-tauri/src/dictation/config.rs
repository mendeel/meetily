use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Runtime};
use tauri_plugin_store::StoreExt;

use super::profiles::PolishProfile;
use super::session::TriggerMode;

const STORE_FILE: &str = "dictation_config.json";
const STORE_KEY: &str = "config";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DictationConfig {
    pub enabled: bool,
    pub trigger_mode: TriggerMode,
    pub hold_shortcut: String,
    pub toggle_shortcut: String,
    pub stt_engine: String, // "nemotron" | "parakeet" | "whisper"
    pub polish_enabled: bool,
    pub default_profile: PolishProfile,
    /// Lowercase app name or bundle id → profile override
    pub profile_overrides: std::collections::HashMap<String, PolishProfile>,
}

impl Default for DictationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            trigger_mode: TriggerMode::PushToTalk,
            hold_shortcut: "CommandOrControl+Shift+Space".into(),
            toggle_shortcut: "CommandOrControl+Shift+D".into(),
            stt_engine: "nemotron".into(),
            polish_enabled: true,
            default_profile: PolishProfile::Default,
            profile_overrides: std::collections::HashMap::new(),
        }
    }
}

impl DictationConfig {
    pub fn resolve_profile(&self, app_name: &str, bundle_id: Option<&str>) -> PolishProfile {
        let name_key = app_name.to_lowercase();
        if let Some(profile) = self.profile_overrides.get(&name_key) {
            return *profile;
        }
        if let Some(bid) = bundle_id {
            let bid_key = bid.to_lowercase();
            if let Some(profile) = self.profile_overrides.get(&bid_key) {
                return *profile;
            }
        }
        let builtin = super::profiles::profile_for_app(app_name, bundle_id);
        if builtin == PolishProfile::Default {
            self.default_profile
        } else {
            builtin
        }
    }
}

/// Load dictation config from the Tauri store (falls back to defaults).
pub fn load_persisted<R: Runtime>(app: &AppHandle<R>) -> DictationConfig {
    let store = match app.store(STORE_FILE) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("dictation config store unavailable: {e}");
            return DictationConfig::default();
        }
    };
    match store.get(STORE_KEY) {
        Some(value) => match serde_json::from_value::<DictationConfig>(value) {
            Ok(config) => config,
            Err(e) => {
                log::warn!("Failed to deserialize dictation config: {e}");
                DictationConfig::default()
            }
        },
        None => DictationConfig::default(),
    }
}

/// Persist dictation config to the Tauri store.
pub fn save_persisted<R: Runtime>(app: &AppHandle<R>, config: &DictationConfig) -> Result<(), String> {
    let store = app
        .store(STORE_FILE)
        .map_err(|e| format!("Failed to open dictation config store: {e}"))?;
    let value = serde_json::to_value(config)
        .map_err(|e| format!("Failed to serialize dictation config: {e}"))?;
    store.set(STORE_KEY, value);
    store
        .save()
        .map_err(|e| format!("Failed to save dictation config: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_prefer_nemotron_and_ptt() {
        let c = DictationConfig::default();
        assert_eq!(c.stt_engine, "nemotron");
        assert_eq!(c.trigger_mode, TriggerMode::PushToTalk);
        assert!(c.polish_enabled);
    }

    #[test]
    fn override_beats_builtin_map() {
        let mut c = DictationConfig::default();
        c.profile_overrides
            .insert("slack".into(), PolishProfile::Ide);
        assert_eq!(
            c.resolve_profile("Slack", Some("com.tinyspeck.slackmacgap")),
            PolishProfile::Ide
        );
    }

    #[test]
    fn default_profile_applies_when_builtin_is_default() {
        let mut c = DictationConfig::default();
        c.default_profile = PolishProfile::Email;
        assert_eq!(
            c.resolve_profile("WeirdApp", Some("com.example.weird")),
            PolishProfile::Email
        );
    }

    #[test]
    fn builtin_mapping_beats_default_profile() {
        let mut c = DictationConfig::default();
        c.default_profile = PolishProfile::Email;
        assert_eq!(
            c.resolve_profile("Slack", Some("com.tinyspeck.slackmacgap")),
            PolishProfile::Chat
        );
    }
}
