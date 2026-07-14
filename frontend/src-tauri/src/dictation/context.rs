#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrontmostApp {
    pub name: String,
    pub bundle_id: Option<String>,
}

#[cfg(not(target_os = "macos"))]
pub fn frontmost_app() -> Result<FrontmostApp, String> {
    Err("frontmost app detection is macOS-only in v1".into())
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
