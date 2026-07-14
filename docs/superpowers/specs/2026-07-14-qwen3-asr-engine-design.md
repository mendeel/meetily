# Qwen3-ASR Local Transcription Engine — Design Spec

**Date:** 2026-07-14  
**Status:** Approved for planning  
**Approach:** Fully local in-app ONNX engine (same UX as Parakeet / Whisper)  
**Models:** Both `Qwen3-ASR-0.6B` and `Qwen3-ASR-1.7B` downloadable

## Problem

Meetily already ships three local STT engines (Whisper via GGML, Parakeet and Nemotron via ONNX). Users want another high-quality multilingual option based on [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), with the **same product UX**: download a model in settings, select it as the transcript provider, and use it for live recording, import, and retranscription.

The official Qwen3-ASR release is a Python / transformers / vLLM stack. That does not fit Meetily’s Tauri + Rust local runtime. We need a **Rust-native variant** that plugs into the existing transcription plumbing.

## Goals

- Add a fourth local provider, `qwen`, selectable in Transcript Settings
- Ship **both** quantized ONNX packs: **0.6B** (default) and **1.7B** (optional quality upgrade)
- Support **live VAD chunks**, **import**, **retranscription**, and **dictation** STT selection
- Reuse existing `ort` (ONNX Runtime) dependency — same family as Parakeet / Nemotron
- Mirror Parakeet’s engine module shape so download / load / unload / validate feel familiar

## Non-goals (v1)

- Official Python `qwen-asr` package, vLLM, or any Python sidecar
- GGUF / llama.cpp / candle backends
- Forced-aligner word timestamps (`Qwen3-ForcedAligner`)
- Nemotron-style token streaming / partials (utterance-level results only)
- Refactoring all engines to trait-only dispatch (nice-to-have later; not required for v1)
- Cloud Qwen API
- Making Qwen the default provider (Parakeet remains recommended default)

## Product decisions

| Decision | Choice |
|---|---|
| Runtime | ONNX via existing `ort` crate |
| Model sources | Community ONNX exports derived from official HF weights (e.g. andrewleech `qwen3-asr-*-onnx` / int4–int8 packs) |
| Sizes shipped | Both 0.6B and 1.7B downloadable |
| Default model | `qwen3-asr-0.6b` quantized pack (live-friendly) |
| Provider ID | `qwen` |
| Live behavior | Offline / utterance decode on VAD speech chunks (Parakeet-like) |
| Timestamps | Segment-level from VAD only; no forced-aligner in v1 |
| Confidence | Optional / omit if unavailable (`None`, like Parakeet) |

## Architecture

New module `frontend/src-tauri/src/qwen_engine/`, parallel to `parakeet_engine/` and `nemotron_engine/`.

```
Transcript settings (provider=qwen, model=0.6b|1.7b)
        │
        ▼
┌──────────────────┐
│  QwenEngine      │  discover · download · load · unload · transcribe
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  QwenModel (ort) │  encoder + decoder_init + decoder_step
│  mel + tokenizer │
└────────┬─────────┘
         │
    ┌────┴────────────────────────────┐
    ▼                                 ▼
Live worker / import /               Dictation STT
retranscription                      (optional engine)
```

**Inference contract (same as other providers):** 16 kHz mono `f32` samples in → text out.

**Decode path:**
1. Log-mel spectrogram frontend (params from pack `config.json`)
2. `encoder.onnx` → audio features
3. `decoder_init.onnx` prefill with audio features + language/prompt tokens
4. Autoregressive `decoder_step.onnx` until EOS / max tokens
5. HF BPE decode via `tokenizer.json`; strip leading language tag if present

## Components

| Unit | Responsibility |
|---|---|
| **`qwen_engine/model.rs`** | ORT sessions, mel, tokenizer, autoregressive decode |
| **`qwen_engine/qwen_engine.rs`** | Catalog, download, load/unload, `transcribe` API |
| **`qwen_engine/commands.rs`** | Tauri commands + global `Mutex<Option<Arc<QwenEngine>>>` |
| **`audio/transcription/qwen_provider.rs`** | `TranscriptionProvider` thin wrapper |
| **`TranscriptionEngine::Qwen`** | Enum arm in `engine.rs` + worker match arms |
| **Import / retranscription / common unload** | Provider string `qwen` + init/transcribe/unload |
| **Dictation** | Map `stt_engine == "qwen"` to Qwen transcribe commands |
| **`QwenModelManager.tsx` + `lib/qwen.ts`** | Download / select UI for both sizes |
| **`TranscriptSettings.tsx`** | Provider option + embed manager |
| **`useTranscriptionModels.ts` / `modelDefaults.ts`** | Aggregate catalog + default sync |

## Model catalog

| Model ID (app) | Approx. source pack | Role |
|---|---|---|
| `qwen3-asr-0.6b-int4` (or int8 if int4 unavailable/unstable) | ONNX export of `Qwen/Qwen3-ASR-0.6B` | **Default** — live + dictation |
| `qwen3-asr-1.7b-int4` (or int8) | ONNX export of `Qwen/Qwen3-ASR-1.7B` | Optional higher accuracy |

Storage layout under `app_data_dir/models/`:

```
models/
  qwen3-asr-0.6b-int4/
    encoder*.onnx
    decoder_init*.onnx
    decoder_step*.onnx
    decoder_weights*.data   # if external
    embed_tokens.bin        # if required by pack
    tokenizer.json
    config.json
  qwen3-asr-1.7b-int4/
    …same shape…
```

Exact filenames follow the chosen exporter’s convention; discovery validates required files before marking a model `Available`.

Download UX matches Parakeet: progress events, cancel, incomplete-pack → “re-download”.

Only one Qwen model is loaded at a time; switching 0.6 ↔ 1.7 unloads the previous session.

## Integration touch points

Must update every place the three existing engines are enumerated:

**Rust:** `lib.rs` (mod + init + commands), `config.rs` defaults, `audio/transcription/{mod,engine,worker}.rs`, `audio/import.rs`, `audio/retranscription.rs`, `audio/common.rs`, `dictation/commands.rs` (+ config if needed), `Cargo.toml` (tokenizer / mel deps if not already present).

**Frontend:** `TranscriptSettings.tsx`, new model manager + `lib/qwen.ts`, `useTranscriptionModels.ts`, `constants/modelDefaults.ts`, types for provider union, onboarding optional.

## Error handling

| Failure | User-facing behavior |
|---|---|
| Incomplete / corrupt pack | Model status Missing/Error; prompt re-download |
| ORT load failure (esp. 1.7B RAM) | Clear error; suggest switching to 0.6B |
| Audio too short | Same `AudioTooShort` path as other providers |
| Decode / tokenizer failure | `EngineFailed` with logged detail; no crash of recording loop |

## Testing / verification

- Unit or integration: load 0.6B pack and transcribe a short fixture WAV → non-empty text
- Manual: download both sizes; live meeting on 0.6B; import on 1.7B; switch models without app restart
- Regression: Parakeet / Whisper / Nemotron still work; unload-after-batch does not free a model while recording

## Implementation notes

- Prefer adapting proven ONNX decode logic (e.g. from `transcribe-rs` Qwen3 engine) over re-deriving from Python, but **vendor or copy** into `qwen_engine/` rather than taking a hard dependency if `ort` versions conflict (`Meetily` pins `ort = 2.0.0-rc.12`).
- Quantization choice (int4 vs int8) is an implementation detail finalized during spike; catalog IDs above may adjust to the pack that passes quality + latency checks on macOS CPU/Metal EP.
- Do not change the default transcript provider away from Parakeet in v1.

## Open items resolved in design

| Question | Resolution |
|---|---|
| Product shape | Fully local in-app engine (same UX as Parakeet/Whisper) |
| Runtime | ONNX / `ort`, not Python or GGUF |
| Model sizes | Both 0.6B and 1.7B downloadable |
| Default | Quantized 0.6B |
)
