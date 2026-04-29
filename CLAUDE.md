# CLAUDE.md

Instructions for Claude Code working in this repo.

## Project

Diarized audio transcription CLI. Wraps WhisperX (Whisper + pyannote) behind
`transcribe input.mp3 -o out.vtt`. See `README.md` for user-facing docs and
`TODO.md` for known gaps.

## Hard rules

- **Python 3.13+ via uv.** No other package managers. No `pip install` outside
  `uv pip`.
- **TDD is required.** Red → green for every change. Write the failing test
  first, run it, then implement. Do not batch tests after the fact.
- **stdlib `unittest` only.** No pytest, no test plugins, no fixtures
  libraries. `uv run --python 3.13 python -m unittest discover -s tests`.
- **No new dependencies without justification.** The user said "no
  unnecessary dependencies." `whisperx` is the only runtime dep, kept in
  `[project.optional-dependencies] run` so tests don't pay the torch install
  cost. Tests run with zero deps.
- **No emojis** in code, commits, or docs unless explicitly asked.
- **No new docs files** unless the user asks. README/CLAUDE/TODO already
  exist; update them rather than spawning new ones.

## Architecture

```
src/transcribe/
  formats.py    # Segment dataclass + pure VTT/text/md renderers
  audio.py      # ffmpeg shell-out (16kHz mono wav)
  pipeline.py   # WhisperX orchestration; whisperx is lazy-imported
  speakers.py   # SPEAKER_xx -> name remap
  cli.py        # argparse entrypoint
  __main__.py   # `python -m transcribe`
tests/
  test_formats.py test_audio.py test_pipeline.py
  test_speakers.py test_cli.py
```

Boundaries are isolated so tests stay fast and offline:

- `pipeline.py` lazy-imports both `whisperx` and `whisperx.diarize` inside
  the function. Tests must patch both: `mock.patch.dict(sys.modules,
  {"whisperx": fake, "whisperx.diarize": fake_diarize})`.
  `DiarizationPipeline` is NOT exported from `whisperx.__init__` — it lives
  only in `whisperx.diarize`. Constructor takes `token=` and `device=`.
  `transcribe.pipeline.extract_wav` is patched separately.
- `audio.py` uses `subprocess.run`. Tests patch
  `transcribe.audio.subprocess.run`.
- Formatters are pure functions over `Segment` — no I/O, no mocks needed.

If you add a new external integration, keep it behind a similar boundary so
tests don't need the real thing installed.

## Running things

```bash
# tests (44 tests, ~30ms, no whisperx required)
uv run --python 3.13 python -m unittest discover -s tests

# tests + coverage (matches CI; uses the dev dependency group)
uv run --python 3.13 --group dev coverage run -m unittest discover -s tests
uv run --python 3.13 --group dev coverage report

# CLI help
uv run --python 3.13 transcribe --help

# real run (requires `uv pip install -e .[run]`, ffmpeg, HF token)
uv run --python 3.13 transcribe audio.mp3 -o out.vtt --hf-token "$HF_TOKEN"
```

## Git

- Active feature branch: `claude/audio-transcription-tool-ZLtuf`.
- Commit messages: imperative mood, brief subject, body explains the *why*.
- Never push to `main`.

## When picking up work

1. Read `TODO.md` first — P0 items block trusting the tool.
2. WhisperX API verified (2026-04-29). Key findings: `DiarizationPipeline` is
   in `whisperx.diarize` (not top-level); constructor uses `token=` not
   `use_auth_token=`; default model is `pyannote/speaker-diarization-community-1`
   (user must accept HF gated access). `torchcodec` warning on macOS is benign.
