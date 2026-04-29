# TODO

Tracked gaps before this is a real MVP. Ordered by impact.

## P0 — must do before trusting it

- [ ] **End-to-end smoke test against real WhisperX.** The pipeline has only been
  exercised against mocks that reflect my reading of Gemini's 2024 snippet.
  Add an opt-in integration test (skip unless `TRANSCRIBE_INTEGRATION=1`) that
  runs a short WAV through the real pipeline and asserts segment shape. First
  real run will likely surface keyword/argument drift in
  `whisperx.load_model`, `DiarizationPipeline(...)`, or
  `assign_word_speakers(...)`. Fix as it surfaces, then revisit P1+.

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
