# transcribe

Speaker-diarized audio transcription CLI. Pass an `mp3`, `mp4`, or `wav`,
get back a transcript labeled by speaker (`SPEAKER_00`, `SPEAKER_01`, …) in
WebVTT, plain text, or Markdown.

Built on [WhisperX](https://github.com/m-bain/whisperX), which combines
OpenAI Whisper for transcription with `pyannote.audio` for diarization.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` on `PATH` (used to extract 16 kHz mono audio from mp3/mp4)
- A Hugging Face access token, with terms accepted on:
  - [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)

## Install

```bash
uv pip install -e .[run]
```

The `[run]` extra pulls in WhisperX (and transitively `torch`, `pyannote`).
Without it you can still run the test suite, but not the CLI against real
audio.

## Usage

```bash
# default: text transcript to stdout
transcribe meeting.mp3 --hf-token "$HF_TOKEN"

# WebVTT subtitle file (format inferred from .vtt extension)
transcribe meeting.mp3 -o meeting.vtt --hf-token "$HF_TOKEN"

# Markdown, GPU, named speakers
transcribe call.mp4 -o call.md \
  --device cuda --compute-type float16 \
  --names "SPEAKER_00=Dave,SPEAKER_01=Sarah" \
  --hf-token "$HF_TOKEN"

# already 16kHz wav: ffmpeg step is skipped
transcribe podcast.wav -f text --hf-token "$HF_TOKEN"
```

`HF_TOKEN` env var is read if `--hf-token` is omitted.

### Output formats

| flag           | extension              | example                                     |
|----------------|------------------------|---------------------------------------------|
| `-f text`      | `.txt`, `.text`        | `[Dave] Hello there.`                       |
| `-f vtt`       | `.vtt`                 | `00:00:00.000 --> 00:00:02.000\n<v Dave>Hello there.` |
| `-f md`        | `.md`, `.markdown`     | `**Dave** _(00:00:00 – 00:00:02)_\n\nHello there.` |

If `-o` is given and `-f` is not, the format is inferred from the output
extension.

### Speaker names

Either inline:

```
--names "SPEAKER_00=Dave,SPEAKER_01=Sarah"
```

Or a JSON file:

```json
{ "SPEAKER_00": "Dave", "SPEAKER_01": "Sarah" }
```

```
--names speakers.json
```

Unmapped speakers are left as `SPEAKER_xx`.

## Development

```bash
# tests (stdlib unittest, no extra deps)
uv run --python 3.13 python -m unittest discover -s tests
```

44 tests, ~30ms. WhisperX, ffmpeg, and the network are all stubbed at
their boundaries, so tests pass without any of them installed.

See [`CLAUDE.md`](CLAUDE.md) for the working agreement (TDD, layout,
boundaries) and [`TODO.md`](TODO.md) for known gaps before MVP.

## Status

Pre-MVP. The WhisperX integration was written against a recipe and tested
through mocks; it has not yet been exercised against the real library.
See `TODO.md` P0.
