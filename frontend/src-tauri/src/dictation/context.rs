#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrontmostApp {
    pub name: String,
    pub bundle_id: Option<String>,
}

/// Meetily's macOS bundle identifier (tauri.conf.json `identifier`).
pub const MEETILY_BUNDLE_ID: &str = "com.meetily.ai";

/// Pure gate: after a focus-restore attempt, may we synthesize Cmd+V?
///
/// - Requires successful activation of a known target.
/// - Refuses when the frontmost app is Meetily but the target was not (avoids
///   accidentally pasting into Meetily).
/// - When frontmost is known and differs from the target, refuses.
/// - When frontmost cannot be read after a successful activate, trusts activation.
pub fn may_paste_after_activation(
    target_bundle_id: &str,
    activation_succeeded: bool,
    frontmost_bundle_id: Option<&str>,
    self_bundle_id: &str,
) -> bool {
    if !activation_succeeded || target_bundle_id.is_empty() {
        return false;
    }
    match frontmost_bundle_id {
        Some(front) if front == target_bundle_id => true,
        Some(front) if front == self_bundle_id => false,
        Some(_) => false,
        None => true,
    }
}

#[cfg(not(target_os = "macos"))]
pub fn frontmost_app() -> Result<FrontmostApp, String> {
    Err("frontmost app detection is macOS-only in v1".into())
}

#[cfg(not(target_os = "macos"))]
pub fn activate_app_by_bundle_id(_bundle_id: &str) -> Result<(), String> {
    Err("app activation is macOS-only in v1".into())
}

#[cfg(target_os = "macos")]
pub fn frontmost_app() -> Result<FrontmostApp, String> {
    use cocoa::base::{id, nil};
    use objc::{class, msg_send, sel, sel_impl};

    unsafe {
        let workspace: id = msg_send![class!(NSWorkspace), sharedWorkspace];
        let app: id = msg_send![workspace, frontmostApplication];
        if app == nil {
            return Err("no frontmost application".into());
        }
        let name_ns: id = msg_send![app, localizedName];
        let name = if name_ns != nil {
            let bytes: *const i8 = msg_send![name_ns, UTF8String];
            if bytes.is_null() {
                "Unknown".into()
            } else {
                std::ffi::CStr::from_ptr(bytes).to_string_lossy().into_owned()
            }
        } else {
            "Unknown".into()
        };
        let bid_ns: id = msg_send![app, bundleIdentifier];
        let bundle_id = if bid_ns != nil {
            let bytes: *const i8 = msg_send![bid_ns, UTF8String];
            if bytes.is_null() {
                None
            } else {
                Some(std::ffi::CStr::from_ptr(bytes).to_string_lossy().into_owned())
            }
        } else {
            None
        };
        Ok(FrontmostApp { name, bundle_id })
    }
}

/// Activate a running app by bundle id (NSRunningApplication match).
#[cfg(target_os = "macos")]
pub fn activate_app_by_bundle_id(bundle_id: &str) -> Result<(), String> {
    use cocoa::appkit::{
        NSApplicationActivateIgnoringOtherApps, NSRunningApplication,
    };
    use cocoa::base::{id, nil};
    use cocoa::foundation::NSString;
    use objc::{class, msg_send, sel, sel_impl};

    if bundle_id.is_empty() {
        return Err("empty bundle id".into());
    }

    unsafe {
        let ns_bid: id = NSString::alloc(nil).init_str(bundle_id);
        if ns_bid == nil {
            return Err("failed to allocate bundle id string".into());
        }
        let apps: id = msg_send![
            class!(NSRunningApplication),
            runningApplicationsWithBundleIdentifier: ns_bid
        ];
        let _: () = msg_send![ns_bid, release];
        if apps == nil {
            return Err(format!("no running application for bundle id {bundle_id}"));
        }
        let count: usize = msg_send![apps, count];
        if count == 0 {
            return Err(format!("no running application for bundle id {bundle_id}"));
        }
        // Prefer the first match (typically the primary instance).
        let app: id = msg_send![apps, objectAtIndex: 0usize];
        if app == nil {
            return Err(format!("nil NSRunningApplication for {bundle_id}"));
        }
        let ok = app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps);
        if ok == cocoa::base::NO {
            return Err(format!("activateWithOptions failed for {bundle_id}"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paste_allowed_when_frontmost_matches_target() {
        assert!(may_paste_after_activation(
            "com.apple.TextEdit",
            true,
            Some("com.apple.TextEdit"),
            MEETILY_BUNDLE_ID,
        ));
    }

    #[test]
    fn paste_refused_when_activation_fails() {
        assert!(!may_paste_after_activation(
            "com.apple.TextEdit",
            false,
            Some("com.apple.TextEdit"),
            MEETILY_BUNDLE_ID,
        ));
    }

    #[test]
    fn paste_refused_when_meetily_frontmost_but_not_target() {
        assert!(!may_paste_after_activation(
            "com.apple.TextEdit",
            true,
            Some(MEETILY_BUNDLE_ID),
            MEETILY_BUNDLE_ID,
        ));
    }

    #[test]
    fn paste_allowed_when_meetily_was_intentional_target() {
        assert!(may_paste_after_activation(
            MEETILY_BUNDLE_ID,
            true,
            Some(MEETILY_BUNDLE_ID),
            MEETILY_BUNDLE_ID,
        ));
    }

    #[test]
    fn paste_refused_when_wrong_app_frontmost() {
        assert!(!may_paste_after_activation(
            "com.apple.TextEdit",
            true,
            Some("com.apple.Safari"),
            MEETILY_BUNDLE_ID,
        ));
    }

    #[test]
    fn paste_allowed_when_frontmost_unknown_after_activate() {
        assert!(may_paste_after_activation(
            "com.apple.TextEdit",
            true,
            None,
            MEETILY_BUNDLE_ID,
        ));
    }

    #[test]
    fn paste_refused_for_empty_target() {
        assert!(!may_paste_after_activation(
            "",
            true,
            None,
            MEETILY_BUNDLE_ID,
        ));
    }
}
