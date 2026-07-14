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

    pub fn reset(&mut self) {
        self.phase = DictationPhase::Idle;
    }
}

/// Whether Listening should open a new mic stream or keep the existing one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureStartDecision {
    StartNew,
    KeepExisting,
    NotListening,
}

pub fn decide_capture_start(phase: DictationPhase, has_capture: bool) -> CaptureStartDecision {
    match phase {
        DictationPhase::Listening if has_capture => CaptureStartDecision::KeepExisting,
        DictationPhase::Listening => CaptureStartDecision::StartNew,
        _ => CaptureStartDecision::NotListening,
    }
}

/// How stop / PTT release should behave given current runtime flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopDecision {
    /// Enter Transcribing (if needed) and run finalize now.
    FinalizeNow,
    /// Start is still in flight (Idle but starting); finalize after start reaches Listening.
    PendingStop,
    /// Already finalizing or nothing actionable.
    Noop,
}

/// Decide stop behavior.
///
/// `starting` is true only while a start attempt is in flight. Idle release with no
/// start in-flight is a no-op — avoids a stale `pending_stop` after a failed start.
pub fn decide_stop(phase: DictationPhase, finalizing: bool, starting: bool) -> StopDecision {
    if finalizing {
        return StopDecision::Noop;
    }
    match phase {
        DictationPhase::Listening | DictationPhase::Transcribing => StopDecision::FinalizeNow,
        DictationPhase::Idle if starting => StopDecision::PendingStop,
        _ => StopDecision::Noop,
    }
}

/// Whether a scheduled pill auto-hide should still run.
///
/// `scheduled_generation` is captured when the hide was scheduled; if a newer session
/// (or terminal status) bumped `current_generation`, the hide is skipped.
pub fn should_hide_pill(scheduled_generation: u64, current_generation: u64) -> bool {
    scheduled_generation == current_generation
}

/// Whether a finalize entry may proceed, no-op, or only clean up idle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalizeDecision {
    Proceed,
    Noop,
    IdleCleanup,
}

pub fn decide_finalize(phase: DictationPhase, finalizing: bool) -> FinalizeDecision {
    if finalizing {
        FinalizeDecision::Noop
    } else if phase == DictationPhase::Transcribing {
        FinalizeDecision::Proceed
    } else {
        FinalizeDecision::IdleCleanup
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

    #[test]
    fn reentrant_listening_keeps_existing_capture() {
        assert_eq!(
            decide_capture_start(DictationPhase::Listening, true),
            CaptureStartDecision::KeepExisting
        );
        assert_eq!(
            decide_capture_start(DictationPhase::Listening, false),
            CaptureStartDecision::StartNew
        );
        assert_eq!(
            decide_capture_start(DictationPhase::Idle, false),
            CaptureStartDecision::NotListening
        );
    }

    #[test]
    fn ptt_release_while_idle_is_noop_without_start_in_flight() {
        assert_eq!(
            decide_stop(DictationPhase::Idle, false, false),
            StopDecision::Noop
        );
        assert_eq!(
            decide_stop(DictationPhase::Listening, false, false),
            StopDecision::FinalizeNow
        );
        assert_eq!(
            decide_stop(DictationPhase::Idle, true, true),
            StopDecision::Noop
        );
        assert_eq!(
            decide_stop(DictationPhase::Polishing, false, false),
            StopDecision::Noop
        );
    }

    #[test]
    fn ptt_release_while_idle_defers_only_when_starting() {
        assert_eq!(
            decide_stop(DictationPhase::Idle, false, true),
            StopDecision::PendingStop
        );
        // After a failed start, starting is cleared — release must not leave a stale pending_stop.
        assert_eq!(
            decide_stop(DictationPhase::Idle, false, false),
            StopDecision::Noop
        );
    }

    #[test]
    fn stale_pill_hide_skipped_when_generation_advances() {
        assert!(should_hide_pill(3, 3));
        assert!(!should_hide_pill(3, 4));
        assert!(!should_hide_pill(1, 2));
    }

    #[test]
    fn concurrent_finalize_is_noop_while_finalizing() {
        assert_eq!(
            decide_finalize(DictationPhase::Transcribing, true),
            FinalizeDecision::Noop
        );
        assert_eq!(
            decide_finalize(DictationPhase::Transcribing, false),
            FinalizeDecision::Proceed
        );
        assert_eq!(
            decide_finalize(DictationPhase::Idle, false),
            FinalizeDecision::IdleCleanup
        );
        assert_eq!(
            decide_finalize(DictationPhase::Listening, false),
            FinalizeDecision::IdleCleanup
        );
    }
}
