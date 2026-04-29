from __future__ import annotations

import importlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

from transcribe.audio import extract_wav
from transcribe.formats import Segment


@dataclass(frozen=True)
class TranscriptionConfig:
    hf_token: str | None
    model: str = "large-v3"
    device: str = "cpu"
    compute_type: str = "int8"
    batch_size: int = 16
    language: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


def _to_segments(raw: list[dict]) -> list[Segment]:
    out: list[Segment] = []
    for item in raw:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        out.append(
            Segment(
                start=float(item["start"]),
                end=float(item["end"]),
                text=text,
                speaker=str(item.get("speaker") or "UNKNOWN"),
            )
        )
    return out


def transcribe_audio(src: Path, cfg: TranscriptionConfig) -> list[Segment]:
    if not cfg.hf_token:
        raise ValueError(
            "hf_token is required for speaker diarization. "
            "Generate one at https://huggingface.co/settings/tokens and "
            "accept the terms for pyannote/speaker-diarization-3.1 and "
            "pyannote/segmentation-3.0."
        )

    whisperx = importlib.import_module("whisperx")

    with tempfile.TemporaryDirectory(prefix="transcribe-") as tmp:
        if src.suffix.lower() == ".wav":
            wav_path = src
        else:
            wav_path = extract_wav(src, Path(tmp) / "audio.wav")

        audio = whisperx.load_audio(str(wav_path))

        model = whisperx.load_model(
            cfg.model,
            cfg.device,
            compute_type=cfg.compute_type,
            language=cfg.language,
        )
        result = model.transcribe(audio, batch_size=cfg.batch_size)

        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=cfg.device
        )
        aligned = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            cfg.device,
            return_char_alignments=False,
        )

        diar_kwargs: dict[str, object] = {
            "use_auth_token": cfg.hf_token,
            "device": cfg.device,
        }
        diarize_pipeline = whisperx.DiarizationPipeline(**diar_kwargs)

        diarize_call_kwargs: dict[str, object] = {}
        if cfg.min_speakers is not None:
            diarize_call_kwargs["min_speakers"] = cfg.min_speakers
        if cfg.max_speakers is not None:
            diarize_call_kwargs["max_speakers"] = cfg.max_speakers
        diarize_segments = diarize_pipeline(audio, **diarize_call_kwargs)

        final = whisperx.assign_word_speakers(diarize_segments, aligned)
        return _to_segments(final["segments"])
