#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolishProfile {
    Ide,
    Chat,
    Email,
    Default,
}

pub fn profile_for_app(app_name: &str, bundle_id: Option<&str>) -> PolishProfile {
    let name = app_name.to_lowercase();
    let bid = bundle_id.unwrap_or("").to_lowercase();

    if is_ide(&name, &bid) {
        return PolishProfile::Ide;
    }
    if is_chat(&name, &bid) {
        return PolishProfile::Chat;
    }
    if is_email(&name, &bid) {
        return PolishProfile::Email;
    }
    PolishProfile::Default
}

fn is_ide(name: &str, bid: &str) -> bool {
    const BUNDLES: &[&str] = &[
        "com.todesktop.230313mzl4w4u92", // Cursor
        "com.microsoft.vscode",
        "com.apple.dt.xcode",
        "com.jetbrains.",
        "com.visualstudio.code",
    ];
    if BUNDLES.iter().any(|b| bid == *b || bid.starts_with(b)) {
        return true;
    }
    ["cursor", "code", "xcode", "visual studio", "intellij", "webstorm", "pycharm", "goland"]
        .iter()
        .any(|s| name.contains(s))
}

fn is_chat(name: &str, bid: &str) -> bool {
    if bid == "com.tinyspeck.slackmacgap"
        || bid.contains("slack")
        || bid.contains("discord")
        || bid.contains("whatsapp")
        || bid.contains("messages")
    {
        return true;
    }
    ["slack", "discord", "messages", "whatsapp", "telegram", "signal"]
        .iter()
        .any(|s| name.contains(s))
}

fn is_email(name: &str, bid: &str) -> bool {
    if bid == "com.apple.mail" || bid == "com.microsoft.outlook" || bid.contains("outlook") {
        return true;
    }
    ["mail", "outlook", "spark", "airmail", "superhuman"]
        .iter()
        .any(|s| name.contains(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_maps_to_ide() {
        assert_eq!(
            profile_for_app("Cursor", Some("com.todesktop.230313mzl4w4u92")),
            PolishProfile::Ide
        );
        assert_eq!(profile_for_app("Code", Some("com.microsoft.VSCode")), PolishProfile::Ide);
    }

    #[test]
    fn slack_maps_to_chat() {
        assert_eq!(
            profile_for_app("Slack", Some("com.tinyspeck.slackmacgap")),
            PolishProfile::Chat
        );
    }

    #[test]
    fn mail_maps_to_email() {
        assert_eq!(profile_for_app("Mail", Some("com.apple.mail")), PolishProfile::Email);
    }

    #[test]
    fn unknown_maps_to_default() {
        assert_eq!(profile_for_app("WeirdApp", Some("com.example.weird")), PolishProfile::Default);
    }
}
