from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from transcribe.formats import Segment


def apply_names(segments: Iterable[Segment], mapping: Mapping[str, str]) -> list[Segment]:
    if not mapping:
        return list(segments)
    return [
        Segment(s.start, s.end, s.text, mapping.get(s.speaker, s.speaker))
        for s in segments
    ]


def parse_names(spec: str | None) -> dict[str, str]:
    if spec is None:
        return {}
    candidate = Path(spec)
    if candidate.is_file():
        data = json.loads(candidate.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{spec}: expected a JSON object of speaker→name")
        return {str(k): str(v) for k, v in data.items()}

    out: dict[str, str] = {}
    for pair in spec.split(","):
        if "=" not in pair:
            raise ValueError(f"invalid --names entry: {pair!r} (expected KEY=VALUE)")
        key, value = pair.split("=", 1)
        out[key.strip()] = value.strip()
    return out
