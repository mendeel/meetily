#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InjectResult {
    Inserted,
    CopiedToClipboard,
}

pub fn copy_to_clipboard(text: &str) -> Result<(), String> {
    let mut clipboard = arboard::Clipboard::new().map_err(|e| e.to_string())?;
    clipboard.set_text(text.to_string()).map_err(|e| e.to_string())
}

#[cfg(target_os = "macos")]
pub fn accessibility_trusted() -> bool {
    // AXIsProcessTrusted with prompt=false. Link ApplicationServices.
    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXIsProcessTrusted() -> bool;
    }
    unsafe { AXIsProcessTrusted() }
}

#[cfg(not(target_os = "macos"))]
pub fn accessibility_trusted() -> bool {
    false
}

/// Copy text then synthesize Cmd+V when Accessibility trusted.
pub fn inject_or_clipboard(text: &str) -> Result<InjectResult, String> {
    if text.trim().is_empty() {
        return Err("empty text".into());
    }
    copy_to_clipboard(text)?;
    #[cfg(target_os = "macos")]
    {
        if accessibility_trusted() {
            simulate_paste_cmd_v()?;
            return Ok(InjectResult::Inserted);
        }
    }
    Ok(InjectResult::CopiedToClipboard)
}

#[cfg(target_os = "macos")]
fn simulate_paste_cmd_v() -> Result<(), String> {
    use core_graphics::event::{CGEvent, CGEventFlags, CGKeyCode};
    use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};

    // core-graphics 0.23 returns Result<T, ()> (not Option) for these constructors.
    const KEY_V: CGKeyCode = 9; // kVK_ANSI_V
    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .map_err(|_| "CGEventSource unavailable".to_string())?;
    let key_down = CGEvent::new_keyboard_event(source.clone(), KEY_V, true)
        .map_err(|_| "key down failed".to_string())?;
    let key_up = CGEvent::new_keyboard_event(source, KEY_V, false)
        .map_err(|_| "key up failed".to_string())?;
    key_down.set_flags(CGEventFlags::CGEventFlagCommand);
    key_up.set_flags(CGEventFlags::CGEventFlagCommand);
    key_down.post(core_graphics::event::CGEventTapLocation::HID);
    key_up.post(core_graphics::event::CGEventTapLocation::HID);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_errors() {
        assert!(inject_or_clipboard("").is_err());
        assert!(inject_or_clipboard("   ").is_err());
    }
}
