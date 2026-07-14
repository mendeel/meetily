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
