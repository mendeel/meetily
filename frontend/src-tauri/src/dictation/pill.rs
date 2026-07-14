use tauri::{AppHandle, Manager, Runtime};

/// Contract for surfacing the dictation pill above other applications.
///
/// Static config (`alwaysOnTop`, etc.) is necessary but not sufficient on macOS:
/// Tauri's `WebviewWindow::show` maps to `makeKeyAndOrderFront:`, which does not
/// reliably order an inactive app's window above other apps. Dictation is almost
/// always invoked while another app is focused, so macOS must use a native
/// AppKit overlay path (visibility + window level + collectionBehavior +
/// `orderFrontRegardless:`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PillShowPlan {
    pub reassert_always_on_top: bool,
    pub reassert_visible_on_all_workspaces: bool,
    /// macOS-only: order above other apps while Meetily is inactive.
    pub macos_order_front_regardless: bool,
    /// AppKit (`ns_window` / `setLevel` / `orderFrontRegardless`) must run on the
    /// main thread. Dictation show paths are often spawned from async hotkey tasks.
    pub macos_order_front_on_main_thread: bool,
}

pub fn pill_show_plan() -> PillShowPlan {
    PillShowPlan {
        // On macOS the native overlay path sets window level + collectionBehavior
        // itself. Calling Tauri's `set_always_on_top` would async-clobber a higher
        // level back to NSFloatingWindowLevel (tao `set_level_async`).
        reassert_always_on_top: !cfg!(target_os = "macos"),
        reassert_visible_on_all_workspaces: !cfg!(target_os = "macos"),
        macos_order_front_regardless: cfg!(target_os = "macos"),
        macos_order_front_on_main_thread: cfg!(target_os = "macos"),
    }
}

/// AppKit knobs applied on every macOS pill show (testable without a live NSWindow).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PillMacOverlaySpec {
    /// `NSWindow` level. Prefer status (25+) over floating (3).
    pub window_level: i64,
    /// `NSWindowCollectionBehavior` bitfield.
    pub collection_behavior: u64,
    /// Must make the window visible before `orderFrontRegardless` (no-op if hidden).
    pub make_visible_before_order_front: bool,
    pub hides_on_deactivate: bool,
    /// Place the HUD near the bottom-center of the main screen.
    pub position_bottom_center: bool,
}

/// macOS overlay policy for the dictation pill.
///
/// Why the previous `NSFloatingWindowLevel` + bare `orderFrontRegardless` fix failed:
/// 1. The pill starts with `visible: false` and `hide_pill` orderOuts it.
///    `orderFrontRegardless` does not show a hidden window — without an explicit
///    visibility step the HUD never appears while another app is focused.
/// 2. `NSFloatingWindowLevel` (3) is too low for a reliable cross-app HUD;
///    status-level (25) sits above normal and floating app windows.
/// 3. Tauri only sets `CanJoinAllSpaces`; fullscreen-auxiliary / stationary /
///    transient behavior was missing.
/// 4. Re-calling Tauri `set_always_on_top` races and can async-reset level to 3.
///
/// Remaining limitation: a standard Tauri `NSWindow` still cannot overlay another
/// app's *native fullscreen Space*. That needs an `NSPanel` (e.g. tauri-nspanel)
/// with nonactivating + fullScreenAuxiliary — document for a follow-up if required.
pub fn pill_mac_overlay_spec() -> PillMacOverlaySpec {
    // NSWindowCollectionBehavior bits (AppKit).
    const CAN_JOIN_ALL_SPACES: u64 = 1 << 0;
    const TRANSIENT: u64 = 1 << 3;
    const STATIONARY: u64 = 1 << 4;
    const IGNORES_CYCLE: u64 = 1 << 6;
    const FULL_SCREEN_AUXILIARY: u64 = 1 << 8;

    // NSStatusWindowLevel — above floating panels; below screensaver shielding.
    const NS_STATUS_WINDOW_LEVEL: i64 = 25;

    PillMacOverlaySpec {
        window_level: NS_STATUS_WINDOW_LEVEL,
        collection_behavior: CAN_JOIN_ALL_SPACES
            | TRANSIENT
            | STATIONARY
            | IGNORES_CYCLE
            | FULL_SCREEN_AUXILIARY,
        make_visible_before_order_front: true,
        hides_on_deactivate: false,
        position_bottom_center: true,
    }
}

/// Show the always-on-top dictation pill HUD (non-activating window).
pub fn show_pill<R: Runtime>(app: &AppHandle<R>) -> Result<(), String> {
    let Some(window) = app.get_webview_window("dictation-pill") else {
        log::warn!("dictation pill window missing — HUD cannot show");
        return Err("dictation-pill window not found".into());
    };

    let plan = pill_show_plan();

    #[cfg(target_os = "macos")]
    {
        if plan.macos_order_front_regardless {
            // Native path owns visibility, level, and collectionBehavior on the
            // main thread — do not call Tauri set_always_on_top / show here.
            return order_front_regardless(&window);
        }
    }

    if plan.reassert_always_on_top {
        window
            .set_always_on_top(true)
            .map_err(|e| e.to_string())?;
    }
    if plan.reassert_visible_on_all_workspaces {
        // Unsupported on Windows; ignore errors so show still proceeds.
        let _ = window.set_visible_on_all_workspaces(true);
    }

    window.show().map_err(|e| e.to_string())?;
    Ok(())
}

/// Hide the dictation pill HUD.
pub fn hide_pill<R: Runtime>(app: &AppHandle<R>) -> Result<(), String> {
    if let Some(w) = app.get_webview_window("dictation-pill") {
        w.hide().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Order the pill above other apps without activating Meetily.
///
/// AppKit requires the main thread. Hotkey / finalize paths often call
/// `show_pill` from `tauri::async_runtime` worker threads — calling
/// AppKit APIs off-main can abort.
#[cfg(target_os = "macos")]
fn order_front_regardless<R: Runtime>(
    window: &tauri::WebviewWindow<R>,
) -> Result<(), String> {
    if !pill_show_plan().macos_order_front_on_main_thread {
        return apply_macos_overlay(window);
    }
    run_on_main_thread_sync(window, apply_macos_overlay)
}

#[cfg(target_os = "macos")]
fn apply_macos_overlay<R: Runtime>(
    window: &tauri::WebviewWindow<R>,
) -> Result<(), String> {
    use cocoa::base::{id, nil, BOOL, NO, YES};
    use objc::{msg_send, sel, sel_impl};

    let spec = pill_mac_overlay_spec();
    let ns_window = window.ns_window().map_err(|e| e.to_string())? as id;
    if ns_window == nil {
        return Err("dictation pill ns_window was nil".into());
    }

    unsafe {
        if spec.position_bottom_center {
            position_pill_bottom_center(ns_window);
        }

        let hides: BOOL = if spec.hides_on_deactivate { YES } else { NO };
        let _: () = msg_send![ns_window, setHidesOnDeactivate: hides];
        let _: () = msg_send![ns_window, setLevel: spec.window_level];
        let _: () = msg_send![ns_window, setCollectionBehavior: spec.collection_behavior];
        let _: () = msg_send![ns_window, setAlphaValue: 1.0_f64];

        // Critical: orderFrontRegardless alone is a no-op while isVisible == NO.
        // orderFront: makes the window visible without making it key; then
        // orderFrontRegardless lifts it above other apps while Meetily is inactive.
        if spec.make_visible_before_order_front {
            let _: () = msg_send![ns_window, orderFront: nil];
        }
        let _: () = msg_send![ns_window, orderFrontRegardless];
    }

    log::info!(
        "dictation pill: ordered front (level={}, collectionBehavior={:#x})",
        spec.window_level,
        spec.collection_behavior
    );
    Ok(())
}

/// Bottom-center of the main screen's visible frame (Cocoa coords, y from bottom).
#[cfg(target_os = "macos")]
unsafe fn position_pill_bottom_center(ns_window: cocoa::base::id) {
    use cocoa::base::{id, nil};
    use cocoa::foundation::{NSPoint, NSRect};
    use objc::{class, msg_send, sel, sel_impl};

    let screen: id = msg_send![class!(NSScreen), mainScreen];
    if screen == nil {
        return;
    }
    let visible: NSRect = msg_send![screen, visibleFrame];
    let frame: NSRect = msg_send![ns_window, frame];
    let x = visible.origin.x + (visible.size.width - frame.size.width) / 2.0;
    let y = visible.origin.y + 36.0;
    let origin = NSPoint { x, y };
    let _: () = msg_send![ns_window, setFrameOrigin: origin];
}

/// Run `f` on the AppKit main thread. Avoids deadlock when already on main
/// (channel + `run_on_main_thread` would block waiting for itself).
#[cfg(target_os = "macos")]
fn run_on_main_thread_sync<R, F>(window: &tauri::WebviewWindow<R>, f: F) -> Result<(), String>
where
    R: Runtime,
    F: FnOnce(&tauri::WebviewWindow<R>) -> Result<(), String> + Send + 'static,
{
    if is_main_thread() {
        return f(window);
    }

    let (tx, rx) = std::sync::mpsc::channel();
    let window = window.clone();
    window
        .clone()
        .run_on_main_thread(move || {
            let _ = tx.send(f(&window));
        })
        .map_err(|e| e.to_string())?;
    rx.recv()
        .map_err(|_| "main-thread orderFrontRegardless did not complete".to_string())?
}

#[cfg(target_os = "macos")]
fn is_main_thread() -> bool {
    use cocoa::base::{BOOL, YES};
    use objc::{class, msg_send, sel, sel_impl};
    let main: BOOL = unsafe { msg_send![class!(NSThread), isMainThread] };
    main == YES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pill_is_configured_as_nonactivating_cross_workspace_overlay() {
        let config: serde_json::Value =
            serde_json::from_str(include_str!("../../tauri.conf.json")).unwrap();
        let pill = config["app"]["windows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|window| window["label"] == "dictation-pill")
            .unwrap();

        assert_eq!(pill["alwaysOnTop"], true);
        assert_eq!(pill["visibleOnAllWorkspaces"], true);
        assert_eq!(pill["focus"], false);
        assert_eq!(pill["focusable"], false);
        assert_eq!(pill["visible"], false);
    }

    #[test]
    fn pill_show_plan_reasserts_overlay_flags() {
        let plan = pill_show_plan();
        // Non-macOS still uses Tauri flags; macOS native path owns them.
        assert_eq!(plan.reassert_always_on_top, !cfg!(target_os = "macos"));
        assert_eq!(
            plan.reassert_visible_on_all_workspaces,
            !cfg!(target_os = "macos")
        );
    }

    #[test]
    fn pill_show_plan_orders_front_regardless_on_macos() {
        let plan = pill_show_plan();
        assert_eq!(plan.macos_order_front_regardless, cfg!(target_os = "macos"));
    }

    #[test]
    fn pill_show_plan_requires_main_thread_for_macos_order_front() {
        let plan = pill_show_plan();
        assert_eq!(
            plan.macos_order_front_on_main_thread,
            plan.macos_order_front_regardless
        );
    }

    #[test]
    fn pill_show_plan_owns_native_level_on_macos_to_avoid_tao_clobber() {
        // tao's set_always_on_top dispatches setLevel(NSFloatingWindowLevel)
        // asynchronously and would clobber a higher sync level if we still call it.
        let plan = pill_show_plan();
        assert_eq!(
            plan.reassert_always_on_top,
            !cfg!(target_os = "macos"),
            "macOS overlay path must set level itself; skip Tauri set_always_on_top"
        );
    }

    #[test]
    fn macos_overlay_uses_status_window_level_not_floating() {
        let spec = pill_mac_overlay_spec();
        // NSFloatingWindowLevel == 3 is too easy to lose under other floating UIs.
        // NSStatusWindowLevel == 25 sits above normal + floating app windows.
        assert!(
            spec.window_level >= 25,
            "expected NSStatusWindowLevel or higher, got {}",
            spec.window_level
        );
        assert_ne!(spec.window_level, 3, "must not use NSFloatingWindowLevel alone");
    }

    #[test]
    fn macos_overlay_makes_visible_before_order_front_regardless() {
        // orderFrontRegardless is a no-op while isVisible == NO (pill starts
        // visible:false and hide_pill orderOuts it). Previous fix skipped show().
        let spec = pill_mac_overlay_spec();
        assert!(spec.make_visible_before_order_front);
        assert!(!spec.hides_on_deactivate);
        assert!(spec.position_bottom_center);
    }

    #[test]
    fn macos_overlay_collection_behavior_joins_spaces_and_fullscreen_aux() {
        let spec = pill_mac_overlay_spec();
        const CAN_JOIN_ALL_SPACES: u64 = 1 << 0;
        const FULL_SCREEN_AUXILIARY: u64 = 1 << 8;
        assert_eq!(
            spec.collection_behavior & CAN_JOIN_ALL_SPACES,
            CAN_JOIN_ALL_SPACES
        );
        assert_eq!(
            spec.collection_behavior & FULL_SCREEN_AUXILIARY,
            FULL_SCREEN_AUXILIARY
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn is_main_thread_helper_is_false_on_cargo_test_workers() {
        // Dictation show_pill often runs on async workers; the helper must
        // distinguish those from the AppKit main thread.
        assert!(!is_main_thread());
    }

    /// Regression: bare `sleep` in clean_run/clean_build prints macOS usage text
    /// right after `tauri` exits (exit 0), which looks like an app crash.
    #[test]
    fn clean_run_and_build_scripts_have_no_bare_sleep() {
        for (name, script) in [
            ("clean_run.sh", include_str!("../../../clean_run.sh")),
            ("clean_build.sh", include_str!("../../../clean_build.sh")),
        ] {
            let bare = script
                .lines()
                .any(|line| line.trim() == "sleep" || line.trim().starts_with("sleep\t"));
            assert!(
                !bare,
                "{name} must not invoke bare `sleep` (macOS prints usage and mimics a crash)"
            );
        }
    }
}
