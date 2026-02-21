from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import _errors as transcript_errors


TranscriptsDisabled = getattr(transcript_errors, "TranscriptsDisabled", Exception)
NoTranscriptFound = getattr(transcript_errors, "NoTranscriptFound", Exception)
VideoUnavailable = getattr(transcript_errors, "VideoUnavailable", Exception)
TooManyRequests = getattr(transcript_errors, "TooManyRequests", Exception)


class TranscriptionError(RuntimeError):
    def __init__(self, message: str, *, video_id: str, url: str, code: str = "transcription_error"):
        super().__init__(message)
        self.video_id = video_id
        self.url = url
        self.code = code


class TranscriptUnavailableError(TranscriptionError):
    pass


def _video_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.lstrip("/")
    return parse_qs(parsed.query).get("v", [""])[0]


def _normalize_segments(items: list[dict]) -> list[dict]:
    segments = []
    for item in items:
        segments.append(
            {
                "start": float(item["start"]) if item.get("start") is not None else None,
                "duration": float(item["duration"]) if item.get("duration") is not None else None,
                "text": (item.get("text") or "").strip(),
            }
        )
    return [s for s in segments if s["text"]]


def _fetch_youtube_transcript(video_id: str) -> tuple[list[dict], str | None]:
    try:
        return YouTubeTranscriptApi.get_transcript(video_id, languages=["en"]), "en"
    except Exception:
        fetched = YouTubeTranscriptApi.get_transcript(video_id)
        return fetched, None


def _full_text(segments: list[dict]) -> str:
    return " ".join(seg["text"] for seg in segments).strip()


def _transcribe_with_openai_audio(source: dict) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for audio transcription fallback.")

    from openai import OpenAI
    from yt_dlp import YoutubeDL

    client = OpenAI(api_key=api_key)
    video_id = _video_id_from_url(source["url"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        outtmpl = str(Path(tmp_dir) / f"{video_id}.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "noplaylist": True,
            "no_warnings": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source["url"], download=True)
            downloaded = Path(ydl.prepare_filename(info))

        with downloaded.open("rb") as audio_handle:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_handle,
                response_format="verbose_json",
            )

        segments = []
        for segment in getattr(transcription, "segments", []) or []:
            segments.append(
                {
                    "start": float(getattr(segment, "start", 0.0)),
                    "duration": float(getattr(segment, "end", 0.0) - getattr(segment, "start", 0.0)),
                    "text": getattr(segment, "text", "").strip(),
                }
            )

        text = getattr(transcription, "text", "") or _full_text(segments)
        return {
            "url": source["url"],
            "video_id": video_id,
            "method": "openai_audio_transcription",
            "language": getattr(transcription, "language", None),
            "segments": [seg for seg in segments if seg["text"]],
            "full_text": text.strip(),
        }


def get_transcript_for_source(source: dict, allow_audio_fallback: bool, openai_api_key_present: bool) -> dict:
    video_id = _video_id_from_url(source["url"])
    if not video_id:
        raise TranscriptionError(
            f"Unable to extract YouTube video id from URL: {source['url']}",
            video_id="",
            url=source["url"],
            code="invalid_video_url",
        )

    try:
        fetched, language = _fetch_youtube_transcript(video_id)
        segments = _normalize_segments(fetched)
        if not segments:
            raise TranscriptionError(
                "Transcript returned no usable segments.",
                video_id=video_id,
                url=source["url"],
                code="empty_transcript",
            )

        return {
            "url": source["url"],
            "video_id": video_id,
            "method": "youtube_transcript",
            "language": language,
            "segments": segments,
            "full_text": _full_text(segments),
        }
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable, TooManyRequests) as exc:
        if not allow_audio_fallback:
            raise TranscriptUnavailableError(
                "No transcript available (audio fallback disabled).",
                video_id=video_id,
                url=source["url"],
                code="transcript_unavailable_disabled",
            ) from exc

        if not openai_api_key_present:
            raise TranscriptionError(
                "transcript_unavailable (will attempt audio fallback): Audio fallback requires OPENAI_API_KEY.",
                video_id=video_id,
                url=source["url"],
                code="audio_fallback_requires_openai_key",
            ) from exc

        try:
            return _transcribe_with_openai_audio(source)
        except Exception as audio_exc:
            raise TranscriptionError(
                f"transcript_unavailable (will attempt audio fallback); audio_fallback_failed: {audio_exc}",
                video_id=video_id,
                url=source["url"],
                code="audio_fallback_failed",
            ) from audio_exc
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(str(exc), video_id=video_id, url=source["url"], code="transcription_error") from exc
