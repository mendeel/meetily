# Qwen3-ASR Local ONNX Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fourth fully local STT provider (`qwen`) that downloads and runs Qwen3-ASR **0.6B** and **1.7B** ONNX packs in-process via existing `ort`, with the same settings / live / import / dictation UX as Parakeet.

**Architecture:** New `qwen_engine/` module (catalog, download, ORT encoder+decoder_init+decoder_step, tokenizer) mirrored on `parakeet_engine/`. Wire `TranscriptionEngine::Qwen` through live worker, import, retranscription, unload-after-batch, and dictation. Frontend gets `QwenModelManager` + provider option. Default model is quantized 0.6B; Parakeet remains the app-wide recommended default provider.

**Tech Stack:** Tauri 2, Rust, `ort` 2.0.0-rc.12, `ndarray`, HuggingFace `tokenizers`, ONNX packs from `andrewleech/qwen3-asr-*-onnx`, Next.js settings UI.

**Spec:** `docs/superpowers/specs/2026-07-14-qwen3-asr-engine-design.md`

**Inference reference (copy/adapt, do not hard-depend):** [transcribe-rs Qwen3 ONNX engine](https://github.com/cjpais/transcribe-rs) / [andrewleech/qwen3-asr-onnx](https://github.com/andrewleech/qwen3-asr-onnx) — Meetily pins `ort = 2.0.0-rc.12`; prefer vendoring adapted logic into `qwen_engine/model.rs` over adding `transcribe-rs` if versions conflict.

---

## File map

| Path | Responsibility |
|---|---|
| `frontend/src-tauri/src/qwen_engine/mod.rs` | Module exports |
| `frontend/src-tauri/src/qwen_engine/catalog.rs` | Model IDs, HF repos, required files, size estimates |
| `frontend/src-tauri/src/qwen_engine/text.rs` | Language-prefix stripping / light cleanup |
| `frontend/src-tauri/src/qwen_engine/mel.rs` | Log-mel frontend from `config.json` / preprocessor params |
| `frontend/src-tauri/src/qwen_engine/model.rs` | ORT sessions + autoregressive decode |
| `frontend/src-tauri/src/qwen_engine/qwen_engine.rs` | Discover / download / load / unload / `transcribe_audio` |
| `frontend/src-tauri/src/qwen_engine/commands.rs` | Tauri commands + `QWEN_ENGINE` singleton |
| `frontend/src-tauri/src/audio/transcription/qwen_provider.rs` | `TranscriptionProvider` wrapper |
| `frontend/src-tauri/src/audio/transcription/{mod,engine,worker}.rs` | Enum + validate + live match arms |
| `frontend/src-tauri/src/audio/{import,retranscription,common}.rs` | Batch paths + unload |
| `frontend/src-tauri/src/dictation/{commands,config}.rs` | `stt_engine == "qwen"` |
| `frontend/src-tauri/src/config.rs` | `DEFAULT_QWEN_MODEL` |
| `frontend/src-tauri/src/lib.rs` | `mod qwen_engine`; init; register commands |
| `frontend/src-tauri/Cargo.toml` | `tokenizers` (+ FFT dep if needed for mel) |
| `frontend/src/lib/qwen.ts` | Types, display config, invoke helpers |
| `frontend/src/components/QwenModelManager.tsx` | Download / select UI (two models) |
| `frontend/src/components/TranscriptSettings.tsx` | Provider option + embed manager |
| `frontend/src/hooks/useTranscriptionModels.ts` | Include `qwen` models |
| `frontend/src/constants/modelDefaults.ts` | Sync default with Rust |
| `frontend/src/types/index.ts` (or equivalent) | Provider union includes `qwen` |

---

### Task 1: Catalog + pack completeness checks (TDD)

**Files:**
- Create: `frontend/src-tauri/src/qwen_engine/catalog.rs`
- Create: `frontend/src-tauri/src/qwen_engine/mod.rs`
- Modify: `frontend/src-tauri/src/lib.rs` (add `pub mod qwen_engine;`)

- [ ] **Step 1: Write failing tests in `catalog.rs`**

```rust
// frontend/src-tauri/src/qwen_engine/catalog.rs
pub const DEFAULT_QWEN_MODEL: &str = "qwen3-asr-0.6b-int4";
pub const QWEN_0_6B_INT4: &str = "qwen3-asr-0.6b-int4";
pub const QWEN_1_7B_INT4: &str = "qwen3-asr-1.7b-int4";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenModelSpec {
    pub id: &'static str,
    pub hf_repo: &'static str,
    pub size_mb: u32,
    pub files: &'static [&'static str],
}

pub fn model_spec(id: &str) -> Option<&'static QwenModelSpec> {
    None // stub — Step 3 fills this
}

pub fn required_files(id: &str) -> Option<&'static [&'static str]> {
    model_spec(id).map(|s| s.files)
}

pub fn pack_is_complete(dir: &std::path::Path, id: &str) -> bool {
    let Some(files) = required_files(id) else {
        return false;
    };
    files.iter().all(|f| dir.join(f).is_file())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn known_models_have_specs() {
        assert!(model_spec(QWEN_0_6B_INT4).is_some());
        assert!(model_spec(QWEN_1_7B_INT4).is_some());
        assert!(model_spec("nope").is_none());
    }

    #[test]
    fn pack_complete_requires_all_int4_files() {
        let dir = tempdir().unwrap();
        let id = QWEN_0_6B_INT4;
        let files = required_files(id).unwrap();
        assert!(!pack_is_complete(dir.path(), id));
        for f in files {
            fs::write(dir.path().join(f), b"x").unwrap();
        }
        assert!(pack_is_complete(dir.path(), id));
    }

    #[test]
    fn default_model_is_0_6b() {
        assert_eq!(DEFAULT_QWEN_MODEL, QWEN_0_6B_INT4);
    }
}
```

If `tempfile` is not in `[dev-dependencies]`, add it to `frontend/src-tauri/Cargo.toml`:

```toml
[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 2: Run tests — expect fail**

```bash
cd frontend/src-tauri && cargo test -p app-lib catalog::tests -- --nocapture
```

Expected: compile error or `assert!(model_spec(...).is_some())` fails because `model_spec` returns `None`.

(If the crate name is not `app-lib`, use the package name from `Cargo.toml` `[package].name`.)

- [ ] **Step 3: Implement catalog**

```rust
const INT4_FILES: &[&str] = &[
    "encoder.int4.onnx",
    "decoder_init.int4.onnx",
    "decoder_step.int4.onnx",
    "decoder_weights.int4.data",
    "embed_tokens.bin",
    "config.json",
    "tokenizer.json",
];

// Note: some HF packs also ship `decoder_init.int4.onnx.data` /
// `decoder_step.int4.onnx.data`. During Task 5 spike, if ORT requires
// those external data files, add them to INT4_FILES before shipping.

static SPECS: &[QwenModelSpec] = &[
    QwenModelSpec {
        id: QWEN_0_6B_INT4,
        hf_repo: "andrewleech/qwen3-asr-0.6b-onnx",
        size_mb: 1950, // ~encoder+decoder+embed; refine after first real download
        files: INT4_FILES,
    },
    QwenModelSpec {
        id: QWEN_1_7B_INT4,
        hf_repo: "andrewleech/qwen3-asr-1.7b-onnx",
        size_mb: 4000, // refine after first real download
        files: INT4_FILES,
    },
];

pub fn model_spec(id: &str) -> Option<&'static QwenModelSpec> {
    SPECS.iter().find(|s| s.id == id)
}

pub fn all_specs() -> &'static [QwenModelSpec] {
    SPECS
}

pub fn hf_file_url(repo: &str, filename: &str) -> String {
    format!(
        "https://huggingface.co/{repo}/resolve/main/{filename}"
    )
}
```

Also create `mod.rs`:

```rust
//! Qwen3-ASR ONNX speech recognition engine.
pub mod catalog;
pub mod text;
// mel, model, qwen_engine, commands added in later tasks

pub use catalog::{DEFAULT_QWEN_MODEL, QWEN_0_6B_INT4, QWEN_1_7B_INT4};
```

Add `pub mod qwen_engine;` near other engine mods in `lib.rs`.

- [ ] **Step 4: Re-run tests — expect pass**

```bash
cd frontend/src-tauri && cargo test catalog::tests -- --nocapture
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src-tauri/src/qwen_engine/catalog.rs \
  frontend/src-tauri/src/qwen_engine/mod.rs \
  frontend/src-tauri/src/lib.rs \
  frontend/src-tauri/Cargo.toml
git commit -m "$(cat <<'EOF'
feat(qwen): add ONNX model catalog and pack checks

EOF
)"
```

---

### Task 2: Language-prefix text cleanup (TDD)

**Files:**
- Create: `frontend/src-tauri/src/qwen_engine/text.rs`
- Modify: `frontend/src-tauri/src/qwen_engine/mod.rs`

- [ ] **Step 1: Write failing tests**

```rust
// frontend/src-tauri/src/qwen_engine/text.rs
/// Qwen3-ASR often emits a leading language name before the transcript.
/// Strip a known language token prefix when present.
pub fn strip_language_prefix(raw: &str) -> String {
    raw.trim().to_string() // stub
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_english_prefix() {
        assert_eq!(
            strip_language_prefix("English Hello world"),
            "Hello world"
        );
    }

    #[test]
    fn strips_chinese_prefix() {
        assert_eq!(
            strip_language_prefix("Chinese 你好"),
            "你好"
        );
    }

    #[test]
    fn leaves_plain_text() {
        assert_eq!(strip_language_prefix("Hello world"), "Hello world");
    }

    #[test]
    fn trims_whitespace() {
        assert_eq!(strip_language_prefix("  English  hi  "), "hi");
    }
}
```

Use the language name list from transcribe-rs / Qwen docs (at least: Chinese, English, Cantonese, Japanese, Korean, French, German, Spanish, Portuguese, Russian, Arabic, … — include the 30 languages from the model card). Match first token case-insensitively against that set; if match, return the remainder trimmed.

- [ ] **Step 2: Run — expect fail**

```bash
cd frontend/src-tauri && cargo test text::tests -- --nocapture
```

Expected: FAIL on `strips_english_prefix`

- [ ] **Step 3: Implement real stripper**

```rust
const LANG_NAMES: &[&str] = &[
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay",
    "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech",
    "Filipino", "Persian", "Greek", "Hungarian", "Macedonian", "Romanian",
];

pub fn strip_language_prefix(raw: &str) -> String {
    let trimmed = raw.trim();
    let Some((first, rest)) = trimmed.split_once(char::is_whitespace) else {
        return trimmed.to_string();
    };
    if LANG_NAMES.iter().any(|l| l.eq_ignore_ascii_case(first)) {
        rest.trim().to_string()
    } else {
        trimmed.to_string()
    }
}
```

Export `pub mod text;` from `mod.rs`.

- [ ] **Step 4: Run — expect pass**

```bash
cd frontend/src-tauri && cargo test text::tests -- --nocapture
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src-tauri/src/qwen_engine/text.rs frontend/src-tauri/src/qwen_engine/mod.rs
git commit -m "$(cat <<'EOF'
feat(qwen): strip ASR language prefix from transcripts

EOF
)"
```

---

### Task 3: Mel frontend + Cargo deps

**Files:**
- Create: `frontend/src-tauri/src/qwen_engine/mel.rs`
- Modify: `frontend/src-tauri/Cargo.toml`
- Modify: `frontend/src-tauri/src/qwen_engine/mod.rs`

- [ ] **Step 1: Add dependencies**

In `frontend/src-tauri/Cargo.toml` `[dependencies]`:

```toml
tokenizers = { version = "0.21", default-features = false, features = ["onig"] }
rustfft = "6"
```

(Adjust tokenizer features if build fails on Windows; prefer the same feature set used by other local HF tokenizers crates if you find a working example in the ecosystem.)

- [ ] **Step 2: Write a shape test for mel**

```rust
// mel.rs — public API
pub struct MelConfig {
    pub sample_rate: u32,      // 16000
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,         // typically 80 / 128 per config.json
    // fill remaining fields from pack config.json during model load
}

impl MelConfig {
    pub fn from_qwen_defaults() -> Self {
        Self {
            sample_rate: 16_000,
            n_fft: 400,
            hop_length: 160,
            n_mels: 128, // confirm against downloaded config.json in Task 5
        }
    }
}

/// Returns mel features shaped [1, n_mels, time] as f32.
pub fn log_mel_spectrogram(samples: &[f32], cfg: &MelConfig) -> Result<ndarray::Array3<f32>, String> {
    let _ = (samples, cfg);
    Err("not implemented".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_has_expected_channel_and_mel_dims() {
        let cfg = MelConfig::from_qwen_defaults();
        let samples = vec![0.0f32; cfg.sample_rate as usize]; // 1s silence
        let mel = log_mel_spectrogram(&samples, &cfg).unwrap();
        assert_eq!(mel.shape()[0], 1);
        assert_eq!(mel.shape()[1], cfg.n_mels);
        assert!(mel.shape()[2] > 0);
    }
}
```

- [ ] **Step 3: Run — expect fail**, then implement using Whisper-style log-mel (adapt from transcribe-rs qwen3 mel / whisper feature extractor). Match mean/std normalization from pack `config.json` / `preprocessor_config.json` once available.

- [ ] **Step 4: Tests pass**

```bash
cd frontend/src-tauri && cargo test mel::tests -- --nocapture
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src-tauri/Cargo.toml frontend/src-tauri/src/qwen_engine/mel.rs frontend/src-tauri/src/qwen_engine/mod.rs Cargo.lock
git commit -m "$(cat <<'EOF'
feat(qwen): add log-mel frontend and tokenizer deps

EOF
)"
```

---

### Task 4: `QwenModel` ORT inference (spike + implement)

**Files:**
- Create: `frontend/src-tauri/src/qwen_engine/model.rs`
- Modify: `frontend/src-tauri/src/qwen_engine/mod.rs`

**Spike (do this first in the task, before coding decode):**

1. Manually download the 0.6B int4 pack once into a local dir (or `models/qwen3-asr-0.6b-int4/`).
2. Confirm exact filenames present (update `catalog::INT4_FILES` if `.onnx.data` sidecars are required).
3. Open `config.json` and lock mel params + special token ids + max new tokens defaults.
4. Skim transcribe-rs `onnx/qwen3` for session I/O names (`input_ids`, `audio_features`, `input_embeds`, KV cache tensors).

- [ ] **Step 1: Define the public model API**

```rust
use std::path::Path;
use ort::session::Session;

pub struct QwenModel {
    encoder: Session,
    decoder_init: Session,
    decoder_step: Session,
    // tokenizer, embed_tokens, special ids, mel config, kv metadata…
}

impl QwenModel {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        Err("not implemented".into())
    }

    /// Transcribe 16 kHz mono f32 audio. `language` is optional hint
    /// (e.g. "English"); `None` = auto.
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<String, String> {
        let _ = (samples, language);
        Err("not implemented".into())
    }
}
```

- [ ] **Step 2: Implement `load` using Parakeet’s ORT session builder pattern**

Mirror `ParakeetModel::init_session` in `parakeet_engine/model.rs` (CPU EP first; GraphOptimizationLevel::Level3). Load:
- `encoder.int4.onnx`
- `decoder_init.int4.onnx`
- `decoder_step.int4.onnx`
- `embed_tokens.bin` (FP16 → f32 lookup table)
- `tokenizer.json` via `tokenizers::Tokenizer::from_file`
- `config.json` via `serde_json`

- [ ] **Step 3: Implement `transcribe`**

Pipeline:
1. `mel::log_mel_spectrogram`
2. Encoder run → audio features
3. Build prompt / `input_ids` (include language hint if provided — follow transcribe-rs prompt format)
4. `decoder_init` → logits + KV cache
5. Greedy loop on `decoder_step` with `embed_tokens` lookup until EOS or `max_new_tokens`
6. Decode tokens → `text::strip_language_prefix`

- [ ] **Step 4: Add ignored integration test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    #[ignore = "requires downloaded qwen3-asr-0.6b-int4 pack under models/"]
    fn transcribes_fixture_when_model_present() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../models/qwen3-asr-0.6b-int4");
        assert!(dir.is_dir(), "missing model dir {:?}", dir);
        let mut model = QwenModel::load(&dir).expect("load");
        // 1s of quiet noise is enough to smoke-test the graph; replace with
        // a real WAV fixture when available.
        let samples = vec![0.0f32; 16_000];
        let text = model.transcribe(&samples, Some("English")).expect("tx");
        // silence may yield empty or short text — assert call succeeded
        let _ = text;
    }
}
```

Run ignored test locally after download:

```bash
cd frontend/src-tauri && cargo test -- --ignored transcribes_fixture --nocapture
```

Expected: completes without ORT panic; refine asserts once you have a spoken WAV fixture.

- [ ] **Step 5: Commit**

```bash
git add frontend/src-tauri/src/qwen_engine/model.rs \
  frontend/src-tauri/src/qwen_engine/catalog.rs \
  frontend/src-tauri/src/qwen_engine/mod.rs
git commit -m "$(cat <<'EOF'
feat(qwen): implement ONNX encoder/decoder inference

EOF
)"
```

---

### Task 5: `QwenEngine` lifecycle (discover / load / unload / transcribe)

**Files:**
- Create: `frontend/src-tauri/src/qwen_engine/qwen_engine.rs`
- Modify: `frontend/src-tauri/src/qwen_engine/mod.rs`
- Modify: `frontend/src-tauri/src/config.rs`

- [ ] **Step 1: Add default to `config.rs`**

```rust
pub const DEFAULT_QWEN_MODEL: &str = "qwen3-asr-0.6b-int4";
```

Re-export from catalog or keep single source — prefer `config.rs` re-exports catalog constant to avoid drift:

```rust
pub use crate::qwen_engine::catalog::DEFAULT_QWEN_MODEL;
```

(Only if that does not create a circular module dependency; if it does, duplicate the string literal and add a unit test that both equal.)

- [ ] **Step 2: Implement engine shell**

Follow `ParakeetEngine` field layout:
- `models_dir: PathBuf`
- `current_model: Arc<RwLock<Option<String>>>`
- `model: Arc<RwLock<Option<QwenModel>>>` (or `Mutex` if `QwenModel` is not Sync — ORT Session often requires `Mutex`)
- `available_models: Arc<RwLock<Vec<ModelInfo>>>`
- download cancellation / active set (wired in Task 6)

Public methods (signatures to match other engines so commands stay thin):

```rust
impl QwenEngine {
    pub fn new_with_models_dir(models_dir: PathBuf) -> Result<Self, QwenEngineError> { … }
    pub async fn discover_models(&self) -> Result<(), QwenEngineError> { … }
    pub async fn get_available_models(&self) -> Vec<ModelInfo> { … }
    pub async fn load_model(&self, model_name: &str) -> Result<(), QwenEngineError> { … }
    pub async fn unload_model(&self) { … }
    pub async fn is_model_loaded(&self) -> bool { … }
    pub async fn get_current_model(&self) -> Option<String> { … }
    pub async fn transcribe_audio(&self, samples: Vec<f32>) -> Result<String, QwenEngineError> { … }
    pub async fn transcribe_audio_with_language(
        &self,
        samples: Vec<f32>,
        language: Option<String>,
    ) -> Result<String, QwenEngineError> { … }
}
```

`discover_models`: for each `catalog::all_specs()`, status = `Available` if `pack_is_complete(models_dir.join(id))`, else `Missing`.

`load_model`: unload previous; `QwenModel::load(&models_dir.join(name))?`; set current.

`transcribe_audio`: error if not loaded; call `QwenModel::transcribe`.

Serialize `ModelInfo` / `ModelStatus` like Parakeet (`Available`, `Missing`, `{ Downloading: u8 }`, `{ Error: String }`, …).

- [ ] **Step 3: Unit-test discover without ORT**

```rust
#[tokio::test]
async fn discover_marks_missing_when_dir_empty() {
    let dir = tempfile::tempdir().unwrap();
    let engine = QwenEngine::new_with_models_dir(dir.path().to_path_buf()).unwrap();
    engine.discover_models().await.unwrap();
    let models = engine.get_available_models().await;
    assert_eq!(models.len(), 2);
    assert!(models.iter().all(|m| matches!(m.status, ModelStatus::Missing)));
}
```

(Add `tokio` test feature if not already available via crate.)

- [ ] **Step 4: Commit**

```bash
git add frontend/src-tauri/src/qwen_engine/qwen_engine.rs \
  frontend/src-tauri/src/qwen_engine/mod.rs \
  frontend/src-tauri/src/config.rs
git commit -m "$(cat <<'EOF'
feat(qwen): add engine discover/load/unload/transcribe shell

EOF
)"
```

---

### Task 6: Model download + cancel

**Files:**
- Modify: `frontend/src-tauri/src/qwen_engine/qwen_engine.rs`

- [ ] **Step 1: Implement `download_model_detailed`**

Copy the streaming / resume / weighted-progress loop from `ParakeetEngine::download_model_detailed` (`parakeet_engine/parakeet_engine.rs`), then change:

1. Resolve `spec = catalog::model_spec(model_name)?`
2. Target dir = `models_dir.join(model_name)` (create if needed)
3. Files = `spec.files`
4. URL = `catalog::hf_file_url(spec.hf_repo, filename)`
5. Progress type can reuse a local `DownloadProgress` identical to Parakeet’s
6. On success, re-run `discover_models` and mark `Available`

Also implement `cancel_download` mirroring Parakeet’s cancel flag + active set.

- [ ] **Step 2: Manual smoke (not CI)**

Download 0.6B only once:

```bash
# From a small Rust test or temporary command — or trigger via Task 7 UI later.
# Verify pack_is_complete(models/qwen3-asr-0.6b-int4) == true
```

Update `size_mb` in catalog if the real total differs by >20%.

- [ ] **Step 3: Commit**

```bash
git add frontend/src-tauri/src/qwen_engine/qwen_engine.rs frontend/src-tauri/src/qwen_engine/catalog.rs
git commit -m "$(cat <<'EOF'
feat(qwen): download ONNX packs from Hugging Face

EOF
)"
```

---

### Task 7: Tauri commands + app init

**Files:**
- Create: `frontend/src-tauri/src/qwen_engine/commands.rs`
- Modify: `frontend/src-tauri/src/qwen_engine/mod.rs`
- Modify: `frontend/src-tauri/src/lib.rs`

- [ ] **Step 1: Mirror Parakeet command surface**

In `commands.rs`, provide:

```rust
pub static QWEN_ENGINE: Mutex<Option<Arc<QwenEngine>>> = Mutex::new(None);

#[tauri::command]
pub async fn qwen_init() -> Result<(), String> { … }

#[tauri::command]
pub async fn qwen_get_available_models() -> Result<Vec<ModelInfo>, String> { … }

#[tauri::command]
pub async fn qwen_load_model(model_name: String) -> Result<(), String> { … }

#[tauri::command]
pub async fn qwen_get_current_model() -> Result<Option<String>, String> { … }

#[tauri::command]
pub async fn qwen_is_model_loaded() -> Result<bool, String> { … }

#[tauri::command]
pub async fn qwen_transcribe_audio(audio_data: Vec<f32>) -> Result<String, String> { … }

#[tauri::command]
pub async fn qwen_download_model<R: Runtime>(
    app: AppHandle<R>,
    model_name: String,
) -> Result<(), String> { … }  // emit qwen-model-download-progress / complete / error

#[tauri::command]
pub async fn qwen_cancel_download(model_name: String) -> Result<(), String> { … }

#[tauri::command]
pub async fn qwen_validate_model_ready() -> Result<String, String> { … }

#[tauri::command]
pub async fn qwen_validate_model_ready_with_config<R: Runtime>(
    app: &AppHandle<R>,
) -> Result<String, String> { … }
```

`qwen_validate_model_ready_with_config`: if transcript config provider is `qwen`, load `config.model` (or `DEFAULT_QWEN_MODEL`); else behave like Parakeet’s validate (auto-pick first Available).

Event payload fields must match what the frontend listener expects (copy Parakeet’s progress JSON keys, rename event prefix to `qwen-`).

- [ ] **Step 2: Register in `lib.rs`**

1. Call `qwen_engine::commands::qwen_init().await` next to parakeet/nemotron init (after models dir is set).
2. Add all `qwen_*` commands to `invoke_handler![…]`.

- [ ] **Step 3: `cargo check`**

```bash
cd frontend/src-tauri && cargo check
```

Expected: success (warnings OK).

- [ ] **Step 4: Commit**

```bash
git add frontend/src-tauri/src/qwen_engine/commands.rs \
  frontend/src-tauri/src/qwen_engine/mod.rs \
  frontend/src-tauri/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(qwen): register Tauri commands and startup init

EOF
)"
```

---

### Task 8: Wire live transcription path

**Files:**
- Create: `frontend/src-tauri/src/audio/transcription/qwen_provider.rs`
- Modify: `frontend/src-tauri/src/audio/transcription/mod.rs`
- Modify: `frontend/src-tauri/src/audio/transcription/engine.rs`
- Modify: `frontend/src-tauri/src/audio/transcription/worker.rs`

- [ ] **Step 1: Provider wrapper**

Copy `parakeet_provider.rs` → `qwen_provider.rs`, rename types, call `engine.transcribe_audio_with_language(audio, language)`, `provider_name() -> "Qwen"`. Pass language through (Qwen supports it).

- [ ] **Step 2: `TranscriptionEngine` enum**

Add:

```rust
Qwen(Arc<crate::qwen_engine::QwenEngine>),
```

Update `is_model_loaded`, `get_current_model`, `provider_name` match arms.

- [ ] **Step 3: `validate_transcription_model_ready` + `get_or_init_transcription_engine`**

Add `"qwen" => { qwen_init; qwen_validate_model_ready_with_config; … }` arms parallel to `"parakeet"`. Update the unsupported-provider error string to list `qwen`.

- [ ] **Step 4: `worker.rs`**

1. Clone arm for `TranscriptionEngine::Qwen`
2. In `transcribe_chunk_with_provider`, add a Parakeet-like arm calling `qwen_engine.transcribe_audio_with_language(speech_samples, language)`
3. Confidence threshold: treat like Parakeet (`0.0` / no confidence)

- [ ] **Step 5: `cargo check` + commit**

```bash
cd frontend/src-tauri && cargo check
git add frontend/src-tauri/src/audio/transcription/
git commit -m "$(cat <<'EOF'
feat(qwen): wire live transcription engine and worker

EOF
)"
```

---

### Task 9: Wire import, retranscription, unload

**Files:**
- Modify: `frontend/src-tauri/src/audio/common.rs`
- Modify: `frontend/src-tauri/src/audio/import.rs`
- Modify: `frontend/src-tauri/src/audio/retranscription.rs`

- [ ] **Step 1: `unload_engine_after_batch`**

```rust
Some("qwen") => {
    use crate::qwen_engine::commands::QWEN_ENGINE;
    // same pattern as parakeet arm
}
```

- [ ] **Step 2: `import.rs`**

1. `use crate::config::DEFAULT_QWEN_MODEL` and `QwenEngine`
2. `let use_qwen = provider.as_deref() == Some("qwen");`
3. Add `get_or_init_qwen` helper (copy `get_or_init_parakeet`)
4. In the per-segment transcribe match, add qwen branch
5. Default model mapping: `"qwen" => DEFAULT_QWEN_MODEL.to_string()`

- [ ] **Step 3: `retranscription.rs`**

Same provider flag / init / per-segment / default model updates as import.

- [ ] **Step 4: `cargo check` + commit**

```bash
cd frontend/src-tauri && cargo check
git add frontend/src-tauri/src/audio/common.rs \
  frontend/src-tauri/src/audio/import.rs \
  frontend/src-tauri/src/audio/retranscription.rs
git commit -m "$(cat <<'EOF'
feat(qwen): support import and retranscription batch paths

EOF
)"
```

---

### Task 10: Dictation STT engine

**Files:**
- Modify: `frontend/src-tauri/src/dictation/commands.rs`
- Modify: `frontend/src-tauri/src/dictation/config.rs`
- Modify: dictation settings UI if it hard-codes engine options

- [ ] **Step 1: Extend preparation + transcribe match**

```rust
"qwen" => TranscriptionPreparation::Qwen,
// …
TranscriptionPreparation::Qwen => {
    crate::qwen_engine::commands::qwen_init().await?;
    crate::qwen_engine::commands::qwen_validate_model_ready().await?;
}
// …
"qwen" => crate::qwen_engine::commands::qwen_transcribe_audio(samples).await,
```

Update comment on `stt_engine` to `"nemotron" | "parakeet" | "whisper" | "qwen"`.

- [ ] **Step 2: Unit test**

```rust
#[test]
fn qwen_maps_to_qwen_preparation() {
    assert_eq!(
        transcription_preparation("qwen"),
        TranscriptionPreparation::Qwen
    );
}
```

- [ ] **Step 3: If Dictation settings has a hard-coded select list, add Qwen option** (search `stt_engine` / `nemotron` in `frontend/src`).

- [ ] **Step 4: Commit**

```bash
git add frontend/src-tauri/src/dictation/ frontend/src/
git commit -m "$(cat <<'EOF'
feat(qwen): allow Qwen as dictation STT engine

EOF
)"
```

---

### Task 11: Frontend model manager + settings

**Files:**
- Create: `frontend/src/lib/qwen.ts`
- Create: `frontend/src/components/QwenModelManager.tsx`
- Modify: `frontend/src/components/TranscriptSettings.tsx`
- Modify: `frontend/src/hooks/useTranscriptionModels.ts`
- Modify: `frontend/src/constants/modelDefaults.ts`
- Modify: provider type unions wherever `parakeet | nemotron | localWhisper` is listed

- [ ] **Step 1: `lib/qwen.ts`**

Mirror `lib/parakeet.ts`:
- Types `QwenModelInfo`, status union
- `MODEL_DISPLAY_CONFIG` for both IDs (0.6B recommended “Fast multilingual”, 1.7B “Higher accuracy”)
- Helpers: `qwenGetAvailableModels`, `qwenDownloadModel`, `qwenLoadModel`, listen to `qwen-model-download-progress`

- [ ] **Step 2: `QwenModelManager.tsx`**

Copy `ParakeetModelManager.tsx` structure; swap invokes/events to `qwen_*`; show exactly the two catalog models.

- [ ] **Step 3: `TranscriptSettings.tsx`**

1. Extend provider type with `'qwen'`
2. Add to `LOCAL_PROVIDERS`
3. `<SelectItem value="qwen">Qwen3-ASR (Multilingual)</SelectItem>`
4. Embed `<QwenModelManager …>` when `uiProvider === 'qwen'`
5. `onModelSelect` sets `provider: 'qwen'`

- [ ] **Step 4: `useTranscriptionModels.ts`**

Fetch `qwen_get_available_models`, map `provider: 'qwen'`, include in configured-model match.

- [ ] **Step 5: `modelDefaults.ts`**

```ts
export const DEFAULT_QWEN_MODEL = 'qwen3-asr-0.6b-int4';
// PROVIDER_DEFAULT_MODELS.qwen = DEFAULT_QWEN_MODEL
```

- [ ] **Step 6: Typecheck**

```bash
cd frontend && pnpm exec tsc --noEmit
```

Expected: no errors related to `qwen` provider.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/lib/qwen.ts \
  frontend/src/components/QwenModelManager.tsx \
  frontend/src/components/TranscriptSettings.tsx \
  frontend/src/hooks/useTranscriptionModels.ts \
  frontend/src/constants/modelDefaults.ts \
  frontend/src/types/
git commit -m "$(cat <<'EOF'
feat(qwen): add settings UI and model download manager

EOF
)"
```

---

### Task 12: End-to-end verification

**Files:** none (manual + ignored test)

- [ ] **Step 1: Build and run**

```bash
cd frontend && ./clean_run.sh
```

- [ ] **Step 2: Checklist**

1. Settings → Transcript → select **Qwen3-ASR**
2. Download **0.6B** — progress events complete; status Available
3. Download **1.7B** — same
4. Select 0.6B → start a short meeting → live transcript lines appear
5. Import a short audio file with provider Qwen / 1.7B → segments saved
6. Switch back to Parakeet → recording still works
7. Dictation settings: pick Qwen → dictate a phrase → text inserts
8. After import, confirm model unload does not break a subsequent live session

- [ ] **Step 3: Run unit tests**

```bash
cd frontend/src-tauri && cargo test qwen_engine -- --nocapture
```

Expected: catalog/text/mel/discover tests PASS; ignored ORT test remains ignored unless pack present.

- [ ] **Step 4: Final commit only if verification fixes were needed**; otherwise done.

---

## Spec coverage (self-review)

| Spec requirement | Task |
|---|---|
| Provider `qwen`, Parakeet-like UX | 7, 11 |
| Both 0.6B + 1.7B downloadable | 1, 6, 11 |
| Default quantized 0.6B | 1, 5, 11 |
| ONNX via existing `ort` | 4 |
| Live VAD chunks | 8 |
| Import + retranscription | 9 |
| Dictation STT | 10 |
| No Python / GGUF / forced-aligner / streaming partials | honored (non-goals) |
| Unload-after-batch | 9 |
| Error: incomplete pack / suggest 0.6B on 1.7B OOM | 5–7 (load/download errors surface strings; include “try 0.6B” in 1.7B load failure message) |

## Placeholder / consistency notes

- Catalog file list may gain `.onnx.data` sidecars after Task 4 spike — update `INT4_FILES` before download ships.
- `size_mb` values are estimates until first real download (Task 6).
- Crate package name for `cargo test -p …` must match `[package].name` in `frontend/src-tauri/Cargo.toml`.
)
