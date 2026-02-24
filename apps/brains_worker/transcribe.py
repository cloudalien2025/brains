from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


class TranscriptionError(RuntimeError):
    pass


class TranscriptionTimeout(RuntimeError):
    pass


@dataclass
class TranscriptionResult:
    text: str
    language: str
    segments: list[dict[str, Any]]


def ffprobe_duration(path: Path) -> float | None:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
        ).strip()
        return float(out)
    except Exception:
        return None


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise TranscriptionError("faster-whisper not available") from exc
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def transcribe_audio(
    audio_path: Path,
    language: str | None,
    model_name: str,
    beam_size: int,
    vad_filter: bool,
    timeout_s: int,
) -> TranscriptionResult:
    if not Path(audio_path).exists():
        raise TranscriptionError(f"audio file not found: {audio_path}")
    started = time.perf_counter()
    model = _load_model(model_name)
    lang = None if (language in {None, "", "auto"}) else language
    try:
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=lang,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
    except Exception as exc:
        raise TranscriptionError(str(exc)) from exc
    segments: list[dict[str, Any]] = []
    texts: list[str] = []
    for seg in segments_iter:
        segments.append({"id": seg.id, "start": seg.start, "end": seg.end, "text": seg.text})
        if seg.text:
            texts.append(seg.text.strip())
        if timeout_s and (time.perf_counter() - started) > timeout_s:
            raise TranscriptionTimeout("transcription timeout")
    return TranscriptionResult(text="\n".join(texts).strip(), language=info.language or "", segments=segments)
