from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import IO, Sequence

from transcribe.formats import to_markdown, to_text, to_vtt
from transcribe.pipeline import TranscriptionConfig, transcribe_audio
from transcribe.speakers import apply_names, parse_names

_FORMATTERS = {
    "vtt": to_vtt,
    "text": to_text,
    "md": to_markdown,
}

_EXTENSION_FORMATS = {
    ".vtt": "vtt",
    ".txt": "text",
    ".text": "text",
    ".md": "md",
    ".markdown": "md",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe audio (mp3/mp4/wav) with speaker diarization.",
    )
    p.add_argument("input", help="path to mp3, mp4, or wav file")
    p.add_argument("-o", "--output", help="write to file (default: stdout)")
    p.add_argument(
        "-f",
        "--format",
        choices=sorted(_FORMATTERS),
        help="output format; inferred from -o extension if omitted, else 'text'",
    )
    p.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face access token (or set HF_TOKEN env var)",
    )
    p.add_argument("--model", default="large-v3", help="Whisper model name")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--language", default=None, help="ISO language code (auto-detect if omitted)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument(
        "--names",
        default=None,
        help='speaker remap, e.g. "SPEAKER_00=Dave,SPEAKER_01=Sarah" or path to JSON object',
    )
    return p


def _resolve_format(explicit: str | None, output: str | None) -> str | None:
    if explicit:
        return explicit
    if output:
        return _EXTENSION_FORMATS.get(Path(output).suffix.lower())
    return "text"


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr

    parser = _build_parser()
    args = parser.parse_args(argv)

    src = Path(args.input)
    if not src.is_file():
        print(f"transcribe: input not found: {src}", file=err)
        return 2

    fmt = _resolve_format(args.format, args.output)
    if fmt is None:
        print(
            f"transcribe: cannot infer format from {args.output!r}; pass -f vtt|text|md",
            file=err,
        )
        return 2

    try:
        names = parse_names(args.names)
    except ValueError as exc:
        print(f"transcribe: {exc}", file=err)
        return 2

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    cfg = TranscriptionConfig(
        hf_token=hf_token,
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        language=args.language,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    try:
        segments = transcribe_audio(src, cfg)
    except ValueError as exc:
        print(f"transcribe: {exc}", file=err)
        return 2
    except Exception as exc:
        print(f"transcribe: {exc.__class__.__name__}: {exc}", file=err)
        return 1

    segments = apply_names(segments, names)
    rendered = _FORMATTERS[fmt](segments)

    if args.output:
        Path(args.output).write_text(rendered)
    else:
        out.write(rendered)
    return 0
