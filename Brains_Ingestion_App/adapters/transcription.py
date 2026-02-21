from __future__ import annotations

import os
import tempfile
from html import unescape
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import parse_qs, urlparse
import xml.etree.ElementTree as ET

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import _errors as transcript_errors


TranscriptsDisabled = getattr(transcript_errors, "TranscriptsDisabled", Exception)
NoTranscriptFound = getattr(transcript_errors, "NoTranscriptFound", Exception)
VideoUnavailable = getattr(transcript_errors, "VideoUnavailable", Exception)
TooManyRequests = getattr(transcript_errors, "TooManyRequests", Exception)


TIMEDTEXT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


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


def _parse_timestamp_seconds(value: str) -> float:
    part = value.strip().replace(",", ".")
    units = part.split(":")
    if len(units) == 3:
        hours, minutes, seconds = units
    elif len(units) == 2:
        hours = "0"
        minutes, seconds = units
    else:
        return float(part)
    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)


def _parse_webvtt_segments(body: str) -> list[dict]:
    segments: list[dict] = []
    blocks = body.replace("\r\n", "\n").split("\n\n")
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        cue_index = 0
        if "-->" not in lines[cue_index] and len(lines) > 1 and "-->" in lines[1]:
            cue_index = 1
        if "-->" not in lines[cue_index]:
            continue

        start_raw, end_raw = [part.strip() for part in lines[cue_index].split("-->", 1)]
        text = " ".join(lines[cue_index + 1 :]).strip()
        if not text:
            continue

        start = _parse_timestamp_seconds(start_raw.split(" ")[0])
        end = _parse_timestamp_seconds(end_raw.split(" ")[0])
        duration = max(0.0, end - start)
        segments.append({"start": start, "duration": duration, "text": text})

    return segments


def _parse_timedtext_segments(body: str) -> list[dict]:
    trimmed = body.strip()
    if not trimmed:
        return []

    if "<text" in trimmed:
        try:
            root = ET.fromstring(trimmed)
        except ET.ParseError:
            return []

        segments: list[dict] = []
        for node in root.findall(".//text"):
            raw_text = "".join(node.itertext()) if node is not None else ""
            text = unescape(raw_text).strip()
            if not text:
                continue
            start = float(node.attrib.get("start", 0.0))
            duration = float(node.attrib.get("dur", 0.0))
            segments.append({"start": start, "duration": duration, "text": text})
        return segments

    if "-->" in trimmed or trimmed.startswith("WEBVTT"):
        return _parse_webvtt_segments(trimmed)

    return []


def fetch_timedtext(video_id: str, lang: str = "en") -> dict | None:
    candidate_langs = [lang]
    if lang == "en":
        candidate_langs.extend(["en-US", "en-GB"])

    for language in candidate_langs:
        urls = [
            f"https://www.youtube.com/api/timedtext?lang={language}&v={video_id}",
            f"https://www.youtube.com/api/timedtext?lang={language}&v={video_id}&kind=asr",
        ]
        for timedtext_url in urls:
            request = Request(timedtext_url, headers=TIMEDTEXT_HEADERS)
            try:
                with urlopen(request, timeout=10) as response:
                    if response.status != 200:
                        continue
                    body = response.read().decode("utf-8", errors="replace")
            except Exception:
                continue

            if not body.strip():
                continue
            lowered = body.lower()
            if "<text" not in lowered and "-->" not in body and "webvtt" not in lowered:
                continue

            segments = _parse_timedtext_segments(body)
            if not segments:
                continue
            return {
                "video_id": video_id,
                "method": "youtube_timedtext",
                "language": language,
                "segments": segments,
            }

    return None


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
        timedtext = fetch_timedtext(video_id, "en")
        if timedtext:
            timedtext["url"] = source["url"]
            timedtext["full_text"] = _full_text(timedtext["segments"])
            return timedtext

        if not allow_audio_fallback:
            raise TranscriptUnavailableError(
                "transcript_unavailable_yta; transcript_unavailable_timedtext; No transcript available (audio fallback disabled).",
                video_id=video_id,
                url=source["url"],
                code="transcript_unavailable_disabled",
            ) from exc

        if not openai_api_key_present:
            raise TranscriptionError(
                "transcript_unavailable_yta; transcript_unavailable_timedtext; Audio fallback requires OPENAI_API_KEY.",
                video_id=video_id,
                url=source["url"],
                code="audio_fallback_requires_openai_key",
            ) from exc

        try:
            return _transcribe_with_openai_audio(source)
        except Exception as audio_exc:
            raise TranscriptionError(
                f"transcript_unavailable_yta; transcript_unavailable_timedtext; audio_fallback_failed: {audio_exc}",
                video_id=video_id,
                url=source["url"],
                code="audio_fallback_failed",
            ) from audio_exc
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(str(exc), video_id=video_id, url=source["url"], code="transcription_error") from exc
