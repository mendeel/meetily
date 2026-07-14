# System-wide AI Dictation — Design Spec

**Date:** 2026-07-14  
**Status:** Approved for planning  
**Platform (v1):** macOS only  
**Approach:** Dedicated dictation engine inside Meetily (not meeting-pipeline reuse)

## Problem

Meetily already has local realtime STT and LLM summarization for meetings. Users also want **system-wide speech-to-text**: speak while focused in another app (IDE, Slack, email, browser chat), get cleaned/formatted text inserted at the cursor—similar to ChatGPT’s voice input polish, but local and available everywhere.

## Goals

- Dictation into **any focused app** (general system dictation)
- Especially good for **coding IDEs and AI/chat UIs** (well-formatted prompts/messages)
- **Always available** while Meetily runs (tray + global hotkey), clearly separate from meeting recording
- Prefer **Nemotron** for STT; allow Parakeet/Whisper as selectable engines
- **Context-aware polish**: lighter for IDEs, heavier for chat/email
- Ship **inside Meetily** now; keep boundaries clean enough for a future companion app

## Non-goals (v1)

- Windows / Linux
- Wake-word always-listening
- Preview/confirm panel before insert
- Streaming partial transcripts in the HUD
- Separate companion binary
- Dictation history / library
- System audio capture (mic-only)

## Product decisions

| Decision | Choice |
|---|---|
| Primary jobs | Both: general insert-at-cursor + polished coding/chat text |
| Trigger | Hold-to-talk default; optional toggle mode in settings |
| Delivery | Inject into focused app; clipboard + toast fallback |
| Polish | Context-aware by frontmost app class |
| Product surface | Always-on inside Meetily; own settings + permissions; future companion-ready |
| Live UI | Floating always-on-top pill (Listening / Polishing / Done / Error) |
| STT default | Nemotron (user may pick any existing engine) |

## Architecture

Dedicated Rust module under `frontend/src-tauri/src/dictation/`. Shares STT model loaders and LLM client with meetings; **does not** use meeting recording lifecycle, dual-stream mixing, SQLite meeting rows, or WAV saving.

```
Global hotkey (hold / toggle)
        │
        ▼
┌───────────────────┐
│  DictationSession │  mic-only · no meeting DB
└─────────┬─────────┘
          │
          ▼
   Floating pill HUD
          │
          ▼
  STT engine (shared)
  Nemotron / Parakeet / Whisper
          │
          ▼
  Context detector (frontmost app)
          │
          ▼
  LLM polish (shared client)
  light ↔ heavy by app class
          │
          ▼
  Inject at cursor  ──fail──▶  Clipboard + toast
```

## Components

| Unit | Responsibility |
|---|---|
| **DictationSession** | Hold/toggle state machine: start mic → collect speech → finalize → teardown |
| **HotkeyController** | Global shortcuts via Tauri global-shortcut (or equivalent); PTT + optional toggle |
| **PillWindow** | Tiny always-on-top Tauri window for session status |
| **ContextDetector** | Frontmost macOS app → polish profile (`ide` / `chat` / `email` / `default`) |
| **PolishService** | Prompt layer on existing `llm_client`; profile selects light vs heavy cleanup |
| **TextInjector** | Accessibility-based paste into focused app; clipboard fallback |
| **DictationSettings** | Hotkeys, engine, polish on/off, app→profile overrides, permission status |

**Shared (reuse):** Whisper/Parakeet/Nemotron engines, LLM providers, tray, mic permission helpers.

**Hard rule:** If meeting recording is active, dictation soft-blocks with a toast (“Stop meeting recording first”) so both modes never contend for the mic.

## Session flow

1. User holds hotkey (or toggles on) with another app focused.
2. Capture frontmost app → select polish profile.
3. Pill shows **Listening…**
4. Mic-only capture → default Nemotron transcript.
5. On release / toggle off → stop capture → final transcript.
6. Pill shows **Polishing…** → LLM formats per profile (unless polish disabled).
7. Restore focus if needed → inject at cursor.
8. On inject failure → clipboard + toast “Copied — paste with ⌘V”.
9. Pill shows **Done** briefly, then hides.

### Polish profiles

- **`ide` (light):** Punctuation, filler removal (“um”, “uh”), preserve technical terms/code identifiers; no fluffy rewriting.
- **`chat` / `email` (heavy):** Grammar, sentence structure, paragraphs, professional/clear tone suitable for messages.
- **`default`:** Between light and heavy.
- User can override per-app mapping or disable polish entirely (STT text with rule-based cleanup only: trim, collapse whitespace, strip obvious fillers—no LLM).

v1 pill does **not** require live partials; Listening until finalize is enough. Partials are a later enhancement.

If polish is enabled but no LLM provider/model is configured, behave like LLM failure: inject rule-based-cleaned STT text.

## Permissions (macOS)

| Permission | Required for | Behavior if missing |
|---|---|---|
| Microphone | Capture | Block start; pill/settings guide to System Settings |
| Accessibility | Inject keystrokes/paste | Allow session; force clipboard fallback; one-time prompt to enable |
| Screen Recording | Meetings only | Not required for dictation |

Update entitlements/docs accordingly; keep Screen Recording out of the dictation permission story.

## Error handling

| Situation | Behavior |
|---|---|
| Mic denied | Block; guide to settings |
| Accessibility denied | Clipboard path + prompt |
| No STT model loaded | Block; “Load a model in Dictation settings” |
| Meeting recording active | Soft-block toast |
| Silence / empty utterance | Quiet dismiss; no paste |
| STT failure | Pill error; no paste |
| LLM timeout/failure / no LLM configured | Inject rule-based-cleaned STT text |
| Inject failure | Clipboard + toast |
| Hotkey conflict | Surface in settings; allow remap |

## Settings surface

Own **Dictation** settings page (not buried inside meeting transcript settings):

- Enable dictation / run in background with tray
- Hotkey: hold binding + toggle binding
- STT engine (default Nemotron)
- Polish: on/off, default profile, app→profile overrides
- Permission status (Mic, Accessibility) with fix actions
- Test dictation control

## Testing

- **Unit:** Session state machine; app→profile mapping; polish failure fallback; soft-block when recording
- **Integration:** Hotkey start/stop without creating meetings; inject vs clipboard paths
- **Manual:** Cursor/IDE (light), Slack/Messages (heavy), Notes, browser chat; Accessibility denied path; concurrent meeting recording blocked

## Future (explicitly deferred)

- Windows/Linux injectors
- Companion app split (module boundaries should allow extraction)
- Wake word
- Preview/confirm UI
- Streaming partials in pill
- Dictation history

## Success criteria

1. From Cursor (or any app), hold hotkey → speak → release → polished text appears at cursor without opening Meetily’s main window.
2. IDE output stays terse/technical; chat/email output is readable and well-formatted.
3. Meeting recording and dictation never run at the same time.
4. Without Accessibility, dictation still works via clipboard fallback.
5. Feature is discoverable as **Dictation**, not confused with meeting transcription.
