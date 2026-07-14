Vendored from emotechlab/silero-rs @ 26a6460.

Local changes for Meetily Nemotron ASR integration:
- bump `ort` from `=2.0.0-rc.10` to `=2.0.0-rc.12`
- bump `ndarray` from `0.16` to `0.17`
- map `ort::Error<SessionBuilder>` via `anyhow!("{}", e)` during session construction
  (rc.12's builder errors are not `Send`/`Sync`, so they cannot convert into `anyhow::Error` via `?`)

Required so Meetily can share a single ONNX Runtime with `parakeet-rs` 0.3.6.
