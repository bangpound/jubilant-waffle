from __future__ import annotations

import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def extract_wav(src: Path, dst: Path) -> Path:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel", "error",
        "-i", str(src),
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        "-y",
        str(dst),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError as exc:
        raise FFmpegError("ffmpeg not found on PATH") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise FFmpegError(f"ffmpeg failed (exit {result.returncode}): {stderr}")
    return dst
