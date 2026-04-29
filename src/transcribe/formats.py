from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str


def _format_timestamp(seconds: float) -> str:
    if seconds < 0:
        raise ValueError("timestamp must be non-negative")
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _format_clock(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _coalesce(segments: Iterable[Segment]) -> Iterator[Segment]:
    current: Segment | None = None
    for seg in segments:
        text = seg.text.strip()
        if current is None:
            current = Segment(seg.start, seg.end, text, seg.speaker)
            continue
        if seg.speaker == current.speaker:
            joined = f"{current.text} {text}".strip() if text else current.text
            current = Segment(current.start, seg.end, joined, current.speaker)
        else:
            yield current
            current = Segment(seg.start, seg.end, text, seg.speaker)
    if current is not None:
        yield current


def to_vtt(segments: Iterable[Segment]) -> str:
    parts = ["WEBVTT\n"]
    for seg in segments:
        parts.append(
            f"\n{_format_timestamp(seg.start)} --> {_format_timestamp(seg.end)}\n"
            f"<v {seg.speaker}>{seg.text.strip()}\n"
        )
    return "".join(parts)


def to_text(segments: Iterable[Segment]) -> str:
    return "".join(f"[{s.speaker}] {s.text}\n" for s in _coalesce(segments))


def to_markdown(segments: Iterable[Segment]) -> str:
    blocks = [
        f"**{s.speaker}** _({_format_clock(s.start)} – {_format_clock(s.end)})_\n\n{s.text}\n"
        for s in _coalesce(segments)
    ]
    return "\n".join(blocks)
