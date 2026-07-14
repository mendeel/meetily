# System-wide AI Dictation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a macOS-first, always-available Dictation mode inside Meetily: global hotkey → mic-only STT (Nemotron preferred) → context-aware LLM polish → insert at cursor (clipboard fallback), with a floating pill HUD and a dedicated settings page—without using the meeting recorder pipeline.

**Architecture:** New Rust module `frontend/src-tauri/src/dictation/` owns session state, mic capture, polish profiles, context detection, and text injection. It reuses existing STT engines (`nemotron_engine`, `parakeet_engine`, `whisper_engine`) and `summary::llm_client::generate_summary`. Frontend adds a Dictation settings tab and a tiny always-on-top pill window. Meeting recording and dictation soft-block each other via `is_recording()`.

**Tech Stack:** Tauri 2, Rust, `tauri-plugin-global-shortcut`, `arboard`, existing `objc`/`core-graphics`, cpal mic capture, Nemotron/Parakeet/Whisper, existing LLM providers, Next.js settings UI, plugin-store for preferences.

**Spec:** `docs/superpowers/specs/2026-07-14-system-dictation-design.md`

---

## File map

| Path | Responsibility |
|---|---|
| `frontend/src-tauri/src/dictation/mod.rs` | Module exports |
| `frontend/src-tauri/src/dictation/session.rs` | Hold/toggle state machine |
| `frontend/src-tauri/src/dictation/profiles.rs` | App bundle/name → polish profile |
| `frontend/src-tauri/src/dictation/polish.rs` | Rule-based cleanup + LLM polish prompts |
| `frontend/src-tauri/src/dictation/context.rs` | Frontmost macOS app detection |
| `frontend/src-tauri/src/dictation/injector.rs` | Clipboard + Accessibility paste (macOS) |
| `frontend/src-tauri/src/dictation/capture.rs` | Mic-only PCM capture to `Vec<f32>` |
| `frontend/src-tauri/src/dictation/config.rs` | `DictationConfig` types + defaults |
| `frontend/src-tauri/src/dictation/commands.rs` | Tauri commands + orchestration |
| `frontend/src-tauri/src/dictation/hotkeys.rs` | Register/unregister global shortcuts |
| `frontend/src-tauri/src/dictation/pill.rs` | Show/hide/update pill window from Rust |
| `frontend/src-tauri/src/lib.rs` | `mod dictation`; register commands/plugins |
| `frontend/src-tauri/Cargo.toml` | New deps: global-shortcut, arboard |
| `frontend/src-tauri/tauri.conf.json` | Pill window + capabilities |
| `frontend/src-tauri/entitlements.plist` | Docs note; Accessibility is TCC not entitlement |
| `frontend/src/components/DictationSettings.tsx` | Settings UI |
| `frontend/src/app/settings/page.tsx` | Add Dictation tab |
| `frontend/src/app/dictation-pill/page.tsx` | Minimal pill UI |
| `frontend/package.json` | `@tauri-apps/plugin-global-shortcut` if JS APIs needed |

---

### Task 1: Session state machine (pure Rust, TDD)

**Files:**
- Create: `frontend/src-tauri/src/dictation/session.rs`
- Create: `frontend/src-tauri/src/dictation/mod.rs`
- Modify: `frontend/src-tauri/src/lib.rs` (add `pub mod dictation;`)

- [ ] **Step 1: Write the failing tests**

Create `session.rs` with tests only first (types stubbed so it compiles after Step 3):

```rust
// frontend/src-tauri/src/dictation/session.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DictationPhase {
    Idle,
    Listening,
    Transcribing,
    Polishing,
    Inserting,
    Done,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TriggerMode {
    PushToTalk,
    Toggle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DictationSession {
    pub phase: DictationPhase,
    pub mode: TriggerMode,
    pub meeting_recording_active: bool,
}

impl DictationSession {
    pub fn new(mode: TriggerMode) -> Self {
        Self {
            phase: DictationPhase::Idle,
            mode,
            meeting_recording_active: false,
        }
    }

    pub fn on_hotkey_pressed(&mut self) -> Result<(), &'static str> {
        todo!()
    }

    pub fn on_hotkey_released(&mut self) -> Result<(), &'static str> {
        todo!()
    }

    pub fn advance_after_transcript(&mut self) {
        todo!()
    }

    pub fn advance_after_polish(&mut self) {
        todo!()
    }

    pub fn finish_insert(&mut self) {
        todo!()
    }

    pub fn fail(&mut self) {
        todo!()
    }

    pub fn reset(&mut self) {
        self.phase = DictationPhase::Idle;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptt_press_starts_listening_when_idle() {
        let mut s = DictationSession::new(TriggerMode::PushToTalk);
        s.on_hotkey_pressed().unwrap();
        assert_eq!(s.phase, DictationPhase::Listening);
    }

    #[test]
    fn ptt_release_moves_to_transcribing() {
        let mut s = DictationSession::new(TriggerMode::PushToTalk);
        s.on_hotkey_pressed().unwrap();
        s.on_hotkey_released().unwrap();
        assert_eq!(s.phase, DictationPhase::Transcribing);
    }

    #[test]
    fn blocks_when_meeting_recording_active() {
        let mut s = DictationSession::new(TriggerMode::PushToTalk);
        s.meeting_recording_active = true;
        assert!(s.on_hotkey_pressed().is_err());
        assert_eq!(s.phase, DictationPhase::Idle);
    }

    #[test]
    fn toggle_press_starts_then_stops() {
        let mut s = DictationSession::new(TriggerMode::Toggle);
        s.on_hotkey_pressed().unwrap();
        assert_eq!(s.phase, DictationPhase::Listening);
        s.on_hotkey_pressed().unwrap();
        assert_eq!(s.phase, DictationPhase::Transcribing);
    }

    #[test]
    fn ptt_release_while_idle_is_noop_ok() {
        let mut s = DictationSession::new(TriggerMode::PushToTalk);
        assert!(s.on_hotkey_released().is_ok());
        assert_eq!(s.phase, DictationPhase::Idle);
    }
}
```

- [ ] **Step 2: Add module and run tests (expect fail)**

```rust
// frontend/src-tauri/src/dictation/mod.rs
pub mod session;
```

In `lib.rs` near other `pub mod` lines:

```rust
pub mod dictation;
```

Run:

```bash
cd frontend/src-tauri && cargo test dictation::session -- --nocapture
```

Expected: FAIL (`todo!` panics) or compile errors on unimplemented bodies.

- [ ] **Step 3: Implement state machine**

```rust
impl DictationSession {
    pub fn on_hotkey_pressed(&mut self) -> Result<(), &'static str> {
        if self.meeting_recording_active {
            return Err("Stop meeting recording first");
        }
        match (self.mode, self.phase) {
            (_, DictationPhase::Idle) => {
                self.phase = DictationPhase::Listening;
                Ok(())
            }
            (TriggerMode::Toggle, DictationPhase::Listening) => {
                self.phase = DictationPhase::Transcribing;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    pub fn on_hotkey_released(&mut self) -> Result<(), &'static str> {
        match (self.mode, self.phase) {
            (TriggerMode::PushToTalk, DictationPhase::Listening) => {
                self.phase = DictationPhase::Transcribing;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    pub fn advance_after_transcript(&mut self) {
        if self.phase == DictationPhase::Transcribing {
            self.phase = DictationPhase::Polishing;
        }
    }

    pub fn advance_after_polish(&mut self) {
        if self.phase == DictationPhase::Polishing {
            self.phase = DictationPhase::Inserting;
        }
    }

    pub fn finish_insert(&mut self) {
        self.phase = DictationPhase::Done;
    }

    pub fn fail(&mut self) {
        self.phase = DictationPhase::Error;
    }
}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd frontend/src-tauri && cargo test dictation::session -- --nocapture
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src-tauri/src/dictation/ frontend/src-tauri/src/lib.rs
git commit -m "feat(dictation): add session state machine"
```

---

### Task 2: Polish profiles + rule-based cleanup (TDD)

**Files:**
- Create: `frontend/src-tauri/src/dictation/profiles.rs`
- Create: `frontend/src-tauri/src/dictation/polish.rs`
- Modify: `frontend/src-tauri/src/dictation/mod.rs`

- [ ] **Step 1: Write failing tests in `profiles.rs` and `polish.rs`**

```rust
// profiles.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolishProfile {
    Ide,
    Chat,
    Email,
    Default,
}

pub fn profile_for_app(app_name: &str, bundle_id: Option<&str>) -> PolishProfile {
    let _ = (app_name, bundle_id);
    todo!()
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
```

```rust
// polish.rs
pub fn rule_based_cleanup(text: &str) -> String {
    todo!()
}

pub fn system_prompt_for(profile: super::profiles::PolishProfile) -> &'static str {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictation::profiles::PolishProfile;

    #[test]
    fn strips_filler_and_collapses_space() {
        let out = rule_based_cleanup("  um, hello   uh world  ");
        assert_eq!(out, "hello world");
    }

    #[test]
    fn empty_stays_empty() {
        assert_eq!(rule_based_cleanup("   "), "");
        assert_eq!(rule_based_cleanup("um"), "");
    }

    #[test]
    fn ide_prompt_mentions_technical() {
        let p = system_prompt_for(PolishProfile::Ide);
        assert!(p.to_lowercase().contains("technical") || p.to_lowercase().contains("code"));
    }

    #[test]
    fn chat_prompt_mentions_grammar_or_message() {
        let p = system_prompt_for(PolishProfile::Chat);
        assert!(p.to_lowercase().contains("grammar") || p.to_lowercase().contains("message"));
    }
}
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd frontend/src-tauri && cargo test dictation::profiles dictation::polish -- --nocapture
```

- [ ] **Step 3: Implement**

`profile_for_app`: match known bundle IDs and lowercase app name substrings (`cursor`, `code`, `xcode`, `slack`, `discord`, `messages`, `mail`, `outlook`, etc.) → profile; else `Default`.

`rule_based_cleanup`: trim; lowercase-token remove exact fillers `um`, `uh`, `erm`, `like` only when they appear as whole tokens separated by punctuation/space; collapse whitespace; strip leading leftover commas.

`system_prompt_for`:
- Ide: light cleanup, keep identifiers, no fluff
- Chat/Email: grammar, paragraphs, professional tone
- Default: moderate cleanup

Export modules from `mod.rs`.

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd frontend/src-tauri && cargo test dictation::profiles dictation::polish -- --nocapture
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src-tauri/src/dictation/
git commit -m "feat(dictation): add polish profiles and rule-based cleanup"
```

---

### Task 3: DictationConfig defaults (TDD)

**Files:**
- Create: `frontend/src-tauri/src/dictation/config.rs`
- Modify: `frontend/src-tauri/src/dictation/mod.rs`

- [ ] **Step 1: Write tests + types**

```rust
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
        todo!()
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
        assert_eq!(c.resolve_profile("Slack", Some("com.tinyspeck.slackmacgap")), PolishProfile::Ide);
    }
}
```

Also ensure `TriggerMode` already has `Serialize`/`Deserialize` (added in Task 1).

- [ ] **Step 2: Run — expect FAIL on `resolve_profile`**

```bash
cd frontend/src-tauri && cargo test dictation::config -- --nocapture
```

- [ ] **Step 3: Implement `resolve_profile`**

Check `profile_overrides` by lowercase app name and bundle id; else `super::profiles::profile_for_app`; if polish disabled callers skip LLM (not this method).

- [ ] **Step 4: Tests PASS + commit**

```bash
git add frontend/src-tauri/src/dictation/
git commit -m "feat(dictation): add DictationConfig with Nemotron default"
```

---

### Task 4: Text injector (macOS clipboard + paste)

**Files:**
- Create: `frontend/src-tauri/src/dictation/injector.rs`
- Modify: `frontend/src-tauri/Cargo.toml` (add `arboard`)
- Modify: `frontend/src-tauri/src/dictation/mod.rs`

- [ ] **Step 1: Add dependency**

In `Cargo.toml` under `[dependencies]`:

```toml
arboard = "3"
```

- [ ] **Step 2: Write injector with unit-testable pure helpers + macOS paste**

```rust
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
    // Minimal check via `osascript`/Swift is fragile; prefer FFI:
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
    const KEY_V: CGKeyCode = 9; // kVK_ANSI_V
    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .ok_or_else(|| "CGEventSource unavailable".to_string())?;
    let key_down = CGEvent::new_keyboard_event(source.clone(), KEY_V, true)
        .ok_or_else(|| "key down failed".to_string())?;
    let key_up = CGEvent::new_keyboard_event(source, KEY_V, false)
        .ok_or_else(|| "key up failed".to_string())?;
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
```

- [ ] **Step 3: Compile check**

```bash
cd frontend/src-tauri && cargo test dictation::injector -- --nocapture
```

Expected: PASS for empty-text test; paste path exercised manually later.

- [ ] **Step 4: Commit**

```bash
git add frontend/src-tauri/Cargo.toml frontend/src-tauri/Cargo.lock frontend/src-tauri/src/dictation/
git commit -m "feat(dictation): add clipboard inject with Accessibility paste"
```

---

### Task 5: Frontmost app context (macOS)

**Files:**
- Create: `frontend/src-tauri/src/dictation/context.rs`
- Modify: `frontend/src-tauri/src/dictation/mod.rs`

- [ ] **Step 1: Define API + stub non-macOS**

```rust
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
    use cocoa::foundation::NSString;
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
```

If `cocoa` is not already a dependency, add `cocoa = "0.26"` under macOS target deps in `Cargo.toml`, or implement the same via raw `objc` only (no cocoa crate) matching other files in this repo.

- [ ] **Step 2: Compile `frontmost_app`**

Match `msg_send!` patterns already used in the macOS audio code. Return name + optional bundle id.

- [ ] **Step 3: Manual smoke (optional in CI)**

```bash
cd frontend/src-tauri && cargo check
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src-tauri/src/dictation/
git commit -m "feat(dictation): detect frontmost macOS app for polish profile"
```

---

### Task 6: Mic-only capture

**Files:**
- Create: `frontend/src-tauri/src/dictation/capture.rs`
- Modify: `frontend/src-tauri/src/dictation/mod.rs`

- [ ] **Step 1: Implement a small capture handle**

```rust
use std::sync::{Arc, Mutex};
use anyhow::Result;

pub struct MicCapture {
    samples: Arc<Mutex<Vec<f32>>>,
    // hold cpal stream in struct so it stays alive
    _stream: Option<cpal::Stream>,
    sample_rate: u32,
}

impl MicCapture {
    pub fn start_default_input() -> Result<Self> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("no default input device"))?;
        let config = device.default_input_config()?;
        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        let samples = Arc::new(Mutex::new(Vec::<f32>::new()));
        let samples_cb = samples.clone();
        let stream = device.build_input_stream(
            &config.into(),
            move |data: &[f32], _| {
                let mut buf = samples_cb.lock().unwrap();
                if channels <= 1 {
                    buf.extend_from_slice(data);
                } else {
                    for frame in data.chunks(channels) {
                        buf.push(frame[0]);
                    }
                }
            },
            |err| log::error!("dictation mic stream error: {err}"),
            None,
        )?;
        stream.play()?;
        Ok(Self {
            samples,
            _stream: Some(stream),
            sample_rate,
        })
    }

    pub fn stop(self) -> Vec<f32> {
        // Drop stream by consuming self; resample to 16 kHz mono for STT if sample_rate != 16000
        let raw = self.samples.lock().ok().map(|g| g.clone()).unwrap_or_default();
        if self.sample_rate == 16_000 || raw.is_empty() {
            return raw;
        }
        resample_to_16k(&raw, self.sample_rate)
    }
}

fn resample_to_16k(input: &[f32], from_hz: u32) -> Vec<f32> {
    // Use rubato (already in Cargo.toml). Keep a small local helper; do not pull meeting pipeline.
    use rubato::{Resampler, SincFixedIn, SincInterpolationType, WindowFunction};
    let ratio = 16_000f64 / from_hz as f64;
    let mut resampler = SincFixedIn::<f32>::new(
        ratio,
        2.0,
        rubato::SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            oversampling_factor: 256,
            interpolation: SincInterpolationType::Linear,
            window: WindowFunction::BlackmanHarris2,
        },
        1024,
        1,
    )
    .expect("resampler");
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + 1024 <= input.len() {
        let chunk = vec![input[pos..pos + 1024].to_vec()];
        if let Ok(waves) = resampler.process(&chunk, None) {
            out.extend_from_slice(&waves[0]);
        }
        pos += 1024;
    }
    out
}
```

- [ ] **Step 2: `cargo check` the module**

```bash
cd frontend/src-tauri && cargo check
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src-tauri/src/dictation/
git commit -m "feat(dictation): add mic-only PCM capture for dictation sessions"
```

---

### Task 7: Orchestration commands (start/stop/finalize)

**Files:**
- Create: `frontend/src-tauri/src/dictation/commands.rs`
- Modify: `frontend/src-tauri/src/dictation/mod.rs`
- Modify: `frontend/src-tauri/src/lib.rs` (`generate_handler!`)

- [ ] **Step 1: Global dictation state**

```rust
use once_cell::sync::Lazy;
use std::sync::Mutex;
use super::session::{DictationSession, TriggerMode};
use super::config::DictationConfig;
use super::capture::MicCapture;

pub struct DictationRuntime {
    pub session: DictationSession,
    pub config: DictationConfig,
    pub capture: Option<MicCapture>,
    pub target_app_name: Option<String>,
    pub target_bundle_id: Option<String>,
}

pub static DICTATION: Lazy<Mutex<DictationRuntime>> = Lazy::new(|| {
    Mutex::new(DictationRuntime {
        session: DictationSession::new(TriggerMode::PushToTalk),
        config: DictationConfig::default(),
        capture: None,
        target_app_name: None,
        target_bundle_id: None,
    })
});
```

- [ ] **Step 2: Implement commands**

```rust
#[tauri::command]
pub async fn dictation_get_config() -> Result<DictationConfig, String> { ... }

#[tauri::command]
pub async fn dictation_set_config(config: DictationConfig) -> Result<(), String> { ... }

#[tauri::command]
pub async fn dictation_start() -> Result<(), String> {
    // if audio::recording_commands::is_recording().await { return Err(...); }
    // frontmost_app → store
    // session.on_hotkey_pressed
    // MicCapture::start
    // emit "dictation-phase" { phase: "listening", app }
}

#[tauri::command]
pub async fn dictation_stop<R: tauri::Runtime>(app: tauri::AppHandle<R>) -> Result<(), String> {
    // session release → Transcribing
    // stop capture → samples
    // if silence/empty → reset, emit done quietly
    // STT via match config.stt_engine:
    //   "nemotron" => use NemotronEngine global / nemotron_transcribe_audio path
    //   "parakeet" | "whisper" => existing one-shot commands
    // rule_based_cleanup
    // if polish_enabled → llm_client::generate_summary with system_prompt_for(profile)
    //   on LLM err → keep rule-based text
    // inject_or_clipboard
    // emit phase updates for pill: polishing, done/error
    // reset session
}

#[tauri::command]
pub async fn dictation_accessibility_status() -> bool {
    super::injector::accessibility_trusted()
}

#[tauri::command]
pub async fn dictation_open_accessibility_settings() -> Result<(), String> {
    // open x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility
    // or `open "x-apple.systempreferences:..."` via std::process::Command
}
```

Wire STT by calling into existing engine singletons the same way `nemotron_transcribe_audio` does (read that command and reuse the engine handle—do not spawn a meeting).

For LLM polish: reuse whatever provider/model the app already stores for summaries (read from existing summary config commands / DB). Keep a short timeout (e.g. 15–20s) via `CancellationToken` + `tokio::time::timeout`. Max tokens ~512, temperature ~0.2.

Emit events:
- `dictation-phase` payload: `{ phase: string, app: string | null, message: string | null }`
- `dictation-result` payload: `{ text: string, delivery: "inserted" | "clipboard" }`

- [ ] **Step 3: Register in `lib.rs` generate_handler**

Add: `dictation_get_config`, `dictation_set_config`, `dictation_start`, `dictation_stop`, `dictation_accessibility_status`, `dictation_open_accessibility_settings`.

- [ ] **Step 4: `cargo check`**

```bash
cd frontend/src-tauri && cargo check
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src-tauri/src/dictation/ frontend/src-tauri/src/lib.rs
git commit -m "feat(dictation): wire start/stop orchestration with STT and polish"
```

---

### Task 8: Global hotkeys

**Files:**
- Create: `frontend/src-tauri/src/dictation/hotkeys.rs`
- Modify: `frontend/src-tauri/Cargo.toml`
- Modify: `frontend/src-tauri/src/lib.rs` (plugin setup)
- Modify: `frontend/src-tauri/tauri.conf.json` capabilities permissions

- [ ] **Step 1: Add plugin dependency**

```toml
tauri-plugin-global-shortcut = "2"
```

Frontend (if needed):

```bash
cd frontend && pnpm add @tauri-apps/plugin-global-shortcut
```

- [ ] **Step 2: Register in setup with Pressed/Released**

Follow Tauri 2 docs: `ShortcutState::Pressed` → `dictation_start` logic; `Released` → `dictation_stop` for PTT. For Toggle mode, only handle Pressed.

Load shortcuts from `DictationConfig` (defaults `CommandOrControl+Shift+Space` hold, `CommandOrControl+Shift+D` toggle). When `trigger_mode` is PTT, register hold shortcut with press/release. When Toggle, register toggle shortcut.

- [ ] **Step 3: Add capability permissions**

In `tauri.conf.json` capabilities for main (and pill) windows:

```json
"global-shortcut:allow-is-registered",
"global-shortcut:allow-register",
"global-shortcut:allow-unregister"
```

- [ ] **Step 4: `cargo check` + commit**

```bash
git add frontend/src-tauri/Cargo.toml frontend/src-tauri/Cargo.lock frontend/src-tauri/src/ frontend/src-tauri/tauri.conf.json frontend/package.json frontend/pnpm-lock.yaml
git commit -m "feat(dictation): register global push-to-talk and toggle hotkeys"
```

---

### Task 9: Floating pill window

**Files:**
- Create: `frontend/src/app/dictation-pill/page.tsx`
- Create: `frontend/src-tauri/src/dictation/pill.rs`
- Modify: `frontend/src-tauri/tauri.conf.json`
- Modify: `frontend/src-tauri/src/lib.rs` / dictation commands to show/update pill

- [ ] **Step 1: Add window config**

In `tauri.conf.json` `app.windows`, add:

```json
{
  "label": "dictation-pill",
  "title": "Dictation",
  "url": "/dictation-pill",
  "width": 280,
  "height": 48,
  "resizable": false,
  "decorations": false,
  "transparent": true,
  "alwaysOnTop": true,
  "visible": false,
  "focus": false,
  "skipTaskbar": true
}
```

Add `"dictation-pill"` to a capability that allows event listen + window show/hide.

- [ ] **Step 2: Pill page UI**

Minimal client page: listen to `dictation-phase`, show red dot + label (`Listening…` / `Polishing…` / `Done` / error). No cards; single pill row matching the approved mockup.

- [ ] **Step 3: Rust helpers**

```rust
pub fn show_pill<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> Result<(), String> {
    if let Some(w) = app.get_webview_window("dictation-pill") {
        w.show().map_err(|e| e.to_string())?;
        // center near bottom of current monitor if easy; else default position
    }
    Ok(())
}

pub fn hide_pill<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> Result<(), String> { ... }
```

Call show on start, hide shortly after Done/Error.

- [ ] **Step 4: Manual check in `./clean_run.sh`** — hotkey shows pill.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/dictation-pill/ frontend/src-tauri/src/dictation/ frontend/src-tauri/tauri.conf.json
git commit -m "feat(dictation): add always-on-top floating pill HUD"
```

---

### Task 10: Dictation settings UI

**Files:**
- Create: `frontend/src/components/DictationSettings.tsx`
- Modify: `frontend/src/app/settings/page.tsx`

- [ ] **Step 1: Build settings panel**

Controls:
- Enable dictation
- Trigger mode: Push-to-talk / Toggle
- Hold shortcut + Toggle shortcut (text inputs; save via `dictation_set_config` then re-register hotkeys)
- STT engine select: Nemotron (default) / Parakeet / Whisper
- Polish enabled toggle
- Default profile select
- Accessibility status + “Open Accessibility Settings” button
- Short help text: distinct from meeting recording; mic required

Persist: call Rust `dictation_set_config`. Optionally also mirror into `@tauri-apps/plugin-store` key `dictation-config.json` for frontend-only reads—prefer Rust as source of truth (save file under app data via `dirs` or existing store path used by summary config).

- [ ] **Step 2: Add tab to settings page**

Add `{ value: 'dictation', label: 'Dictation', icon: Keyboard or Mic2 }` to `TABS` and render `<DictationSettings />`.

- [ ] **Step 3: Manual UI pass + commit**

```bash
git add frontend/src/components/DictationSettings.tsx frontend/src/app/settings/page.tsx
git commit -m "feat(dictation): add dedicated Dictation settings tab"
```

---

### Task 11: Soft-block meeting ↔ dictation + permissions UX

**Files:**
- Modify: `frontend/src-tauri/src/dictation/commands.rs`
- Modify: `frontend/src-tauri/src/audio/recording_commands.rs` (optional guard)
- Modify: tray / notification toast path if one exists

- [ ] **Step 1: On `dictation_start`, call `is_recording().await` and Err with `"Stop meeting recording first"`**

Emit a notification/toast if a helper exists (`tauri-plugin-notification` or frontend listener on `dictation-phase` error).

- [ ] **Step 2: On meeting `start_recording`, if dictation session phase is `Listening`, stop/cancel dictation first** (or refuse meeting start with clear error). Prefer: cancel dictation quietly and allow meeting start, OR refuse meeting—choose **refuse meeting start while dictating** for safety:

```rust
// in start_recording path
if dictation_is_listening() {
    return Err("Finish dictation before starting a meeting recording".into());
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src-tauri/src/dictation/ frontend/src-tauri/src/audio/recording_commands.rs
git commit -m "feat(dictation): soft-block dictation against meeting recording"
```

---

### Task 12: End-to-end verification

- [ ] **Step 1: Unit suite**

```bash
cd frontend/src-tauri && cargo test dictation -- --nocapture
```

Expected: all `dictation::*` tests PASS.

- [ ] **Step 2: Manual macOS checklist**

1. Load Nemotron model in Transcription settings.
2. Configure Summary LLM (Ollama/Claude/etc.) for polish.
3. Grant Mic; leave Accessibility off → speak → text on clipboard + toast.
4. Enable Accessibility → speak in Notes → text inserts at cursor.
5. Speak in Cursor → light polish (technical).
6. Speak in Slack → heavier polish.
7. Start meeting recording → hotkey blocked.
8. Toggle mode works from settings.
9. Pill appears/hides correctly; main window stays unfocused.

- [ ] **Step 3: Final commit if any fixes**

```bash
git add -A
git commit -m "fix(dictation): address e2e verification issues"
```

---

## Spec coverage checklist

| Spec requirement | Task(s) |
|---|---|
| Dedicated module, not meeting pipeline | 1, 6, 7 |
| PTT default + toggle option | 1, 3, 8, 10 |
| Insert + clipboard fallback | 4, 7 |
| Context-aware polish | 2, 3, 5, 7 |
| Nemotron preferred, other engines selectable | 3, 7, 10 |
| Floating pill | 9 |
| Own settings + permissions | 10, 4, 7 |
| Soft-block vs meeting recording | 1, 7, 11 |
| macOS only v1 | 4, 5, 8 (cfg gates) |
| LLM failure → rule-based text | 2, 7 |
| Empty/silence → no paste | 7 |
| Future companion-ready boundaries | file map (`dictation/` isolation) |

## Notes for implementers

- Do **not** call `start_recording` / create meetings / write WAVs for dictation.
- Prefer existing `nemotron_transcribe_audio` patterns for one-shot STT.
- Keep pill non-activating (`focus: false`) so the target app keeps the insertion point.
- Accessibility on macOS is a TCC prompt (`AXIsProcessTrusted`), not an entitlements.plist flag—document in settings copy.
- YAGNI: no wake word, no preview confirm, no Windows injector in this plan.
