from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled


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
                "start": float(item.get("start", 0.0)),
                "duration": float(item.get("duration", 0.0)),
                "text": (item.get("text") or "").strip(),
            }
        )
    return [s for s in segments if s["text"]]


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


def transcribe_youtube_source(source: dict, allow_audio_fallback: bool = False) -> dict:
    video_id = _video_id_from_url(source["url"])
    if not video_id:
        raise RuntimeError(f"Unable to extract YouTube video id from URL: {source['url']}")

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(["en"])
        except NoTranscriptFound:
            transcript = transcript_list.find_transcript([t.language_code for t in transcript_list])
        fetched = transcript.fetch()
        segments = _normalize_segments(fetched)
        return {
            "url": source["url"],
            "video_id": video_id,
            "method": "youtube_transcript",
            "language": transcript.language_code,
            "segments": segments,
            "full_text": _full_text(segments),
        }
    except (NoTranscriptFound, TranscriptsDisabled):
        if allow_audio_fallback and os.getenv("OPENAI_API_KEY"):
            return _transcribe_with_openai_audio(source)
        raise
    except Exception:
        if allow_audio_fallback and os.getenv("OPENAI_API_KEY"):
            return _transcribe_with_openai_audio(source)
        raise


def transcribe(source: dict, allow_audio_fallback: bool = False) -> dict:
    return transcribe_youtube_source(source=source, allow_audio_fallback=allow_audio_fallback)
