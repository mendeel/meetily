use serde::{Deserialize, Serialize};

use super::profiles::PolishProfile;
use super::session::TriggerMode;

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
        super::profiles::profile_for_app(app_name, bundle_id)
    }
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
}
