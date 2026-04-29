# TODO

Tracked gaps before this is a real MVP. Ordered by impact.

## P0 — must do before trusting it

- [x] **API drift fixed (2026-04-29).** `DiarizationPipeline` lives in
  `whisperx.diarize` (not top-level); constructor uses `token=` not
  `use_auth_token=`; default model is `pyannote/speaker-diarization-community-1`.
  Real run completes end-to-end; output shape not yet asserted in code.
- [ ] **Integration test.** Add opt-in test (skip unless `TRANSCRIBE_INTEGRATION=1`)
  that runs a short WAV through the real pipeline and asserts segment shape.

## P1 — ergonomics that matter day one

- [ ] **Stdout pollution.** WhisperX prints progress to stdout. When `-o` is
  omitted, that chatter mixes into the rendered transcript. Either route
  WhisperX progress to stderr or always write rendered output to a buffer and
  flush at the end.
- [ ] **`--list-speakers` mode.** Without it the name-mapping workflow is
  "run, scan output for SPEAKER_xx, re-run with `--names`." Add a mode that
  prints `SPEAKER_00 (first heard at 00:01:23) ...` and exits.
- [ ] **`mps` device for Apple Silicon.** `--device` choices are `{cpu, cuda}`
  only.

## P2 — polish

- [ ] **Memory cleanup between stages.** `del model; gc.collect()` between
  Whisper and pyannote (Gemini flagged this as important on smaller GPUs).
- [ ] **CUDA default `compute_type`.** Currently `int8` regardless of device.
  When `--device cuda` and `--compute-type` is not set, default to `float16`.
- [ ] **`--version` flag.**
- [ ] **Empty-audio-track mp4.** ffmpeg succeeds with an empty WAV; pipeline
  then raises a confusing whisperx error. Detect and report clearly.

## Nice to have

- [ ] **Multiple input files** (`transcribe a.mp3 b.mp3 -o out_dir/`).
- [ ] **`--json` output** for programmatic consumers.
- [ ] **Persistent HF token cache** via `huggingface_hub` login, so `--hf-token`
  isn't required every run.
