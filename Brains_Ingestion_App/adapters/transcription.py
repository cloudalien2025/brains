from __future__ import annotations

import os
import tempfile
from html import unescape
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
import xml.etree.ElementTree as ET

import requests
import youtube_transcript_api as yta
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import _errors as transcript_errors

from brains.net.http_client import HttpClient


TranscriptsDisabled = getattr(transcript_errors, "TranscriptsDisabled", Exception)
NoTranscriptFound = getattr(transcript_errors, "NoTranscriptFound", Exception)
VideoUnavailable = getattr(transcript_errors, "VideoUnavailable", Exception)
TooManyRequests = getattr(transcript_errors, "TooManyRequests", Exception)


def _yta_runtime_info() -> dict:
    return {
        "yta_module_file": getattr(yta, "__file__", getattr(yta, "file", None)),
        "yta_version": getattr(yta, "__version__", getattr(yta, "version", None)),
        "has_get_transcript": hasattr(YouTubeTranscriptApi, "get_transcript"),
        "has_list_transcripts": hasattr(YouTubeTranscriptApi, "list_transcripts"),
        "has_fetch": hasattr(YouTubeTranscriptApi, "fetch"),
        "has_list": hasattr(YouTubeTranscriptApi, "list"),
    }


def get_yta_runtime_info() -> dict:
    return _yta_runtime_info()


TIMEDTEXT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def _http_client() -> HttpClient:
    return HttpClient()



class TranscriptionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        video_id: str,
        url: str,
        code: str = "transcription_error",
        diagnostics: dict | None = None,
    ):
        super().__init__(message)
        self.video_id = video_id
        self.url = url
        self.code = code
        self.diagnostics = diagnostics or {}


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
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        try:
            return YouTubeTranscriptApi.get_transcript(video_id, languages=["en"]), "en"
        except Exception:
            fetched = YouTubeTranscriptApi.get_transcript(video_id)
            return fetched, None

    if hasattr(YouTubeTranscriptApi, "list_transcripts"):
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        selected = None
        try:
            selected = transcripts.find_manually_created_transcript(["en", "en-US", "en-GB"])
        except Exception:
            selected = None
        if selected is None:
            try:
                selected = transcripts.find_generated_transcript(["en", "en-US", "en-GB"])
            except Exception:
                selected = None
        if selected is None:
            for transcript in transcripts:
                selected = transcript
                break
        if selected is None:
            raise NoTranscriptFound("No transcripts available from list_transcripts().")
        fetched = selected.fetch()
        language = getattr(selected, "language_code", None)
        return fetched, language

    if hasattr(YouTubeTranscriptApi, "fetch"):
        api = YouTubeTranscriptApi()
        try:
            return list(api.fetch(video_id, languages=["en", "en-US", "en-GB"])), "en"
        except Exception:
            fetched = list(api.fetch(video_id))
            return fetched, None

    if hasattr(YouTubeTranscriptApi, "list"):
        api = YouTubeTranscriptApi()
        transcripts = api.list(video_id)
        selected = None
        try:
            selected = transcripts.find_manually_created_transcript(["en", "en-US", "en-GB"])
        except Exception:
            selected = None
        if selected is None:
            try:
                selected = transcripts.find_generated_transcript(["en", "en-US", "en-GB"])
            except Exception:
                selected = None
        if selected is None:
            try:
                selected = transcripts.find_transcript(["en", "en-US", "en-GB"])
            except Exception:
                selected = None
        if selected is None:
            for transcript in transcripts:
                selected = transcript
                break
        if selected is None:
            raise NoTranscriptFound("No transcripts available from list().")
        fetched = list(selected.fetch())
        language = getattr(selected, "language_code", None)
        return fetched, language

    raise TranscriptionError(
        "youtube-transcript-api missing expected methods (import/version issue)",
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        code="yta_missing_methods",
        diagnostics={"yta_runtime": _yta_runtime_info()},
    )


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

    if "<text" in trimmed or "<p" in trimmed:
        try:
            root = ET.fromstring(trimmed)
        except ET.ParseError:
            return []

        segments: list[dict] = []
        for node in root.findall(".//text") + root.findall(".//p"):
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


def fetch_timedtext(video_id: str, lang: str = "en", proxy_session_key: str | None = None) -> dict | None:
    candidate_langs = [lang]
    if lang == "en":
        candidate_langs.extend(["en-US", "en-GB"])

    for language in candidate_langs:
        urls = [
            f"https://www.youtube.com/api/timedtext?lang={language}&v={video_id}",
            f"https://www.youtube.com/api/timedtext?lang={language}&v={video_id}&kind=asr",
        ]
        for timedtext_url in urls:
            response = _http_client().get(
                timedtext_url,
                headers=TIMEDTEXT_HEADERS,
                timeout_seconds=10,
                proxy_session_key=proxy_session_key,
                treat_empty_as_block=True,
            )
            if response.status_code != 200:
                continue
            body = response.text
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


def _timedtext_list_tracks(video_id: str, proxy_session_key: str | None = None) -> tuple[list[dict], dict]:
    response = _http_client().get(
        f"https://www.youtube.com/api/timedtext?type=list&v={video_id}",
        headers={
            **TIMEDTEXT_HEADERS,
            "Accept": "text/plain,text/vtt,application/xml,text/xml,*/*",
        },
        timeout_seconds=10,
        proxy_session_key=proxy_session_key,
        treat_empty_as_block=True,
    )
    body = response.text
    diagnostics = {
        "timedtext_list_http_status": response.status_code,
        "timedtext_list_body_len": len(body),
        "timedtext_list_body_head": body[:200],
    }

    if response.status_code != 200:
        return [], diagnostics

    trimmed = body.strip()
    if not trimmed or "<track" not in trimmed:
        return [], diagnostics

    try:
        root = ET.fromstring(trimmed)
    except ET.ParseError:
        return [], diagnostics

    tracks: list[dict] = []
    for node in root.findall(".//track"):
        kind = (node.attrib.get("kind") or "").strip()
        tracks.append(
            {
                "lang_code": (node.attrib.get("lang_code") or "").strip(),
                "lang_translated": (node.attrib.get("lang_translated") or "").strip() or None,
                "name": (node.attrib.get("name") or "").strip(),
                "kind": kind or None,
                "vss_id": (node.attrib.get("vss_id") or "").strip() or None,
                "is_auto": kind == "asr",
            }
        )
    return [track for track in tracks if track["lang_code"]], diagnostics


def _pick_best_track(tracks: list[dict]) -> dict | None:
    if not tracks:
        return None

    english_codes = {"en", "en-us", "en-gb"}

    def _is_english(track: dict) -> bool:
        return (track.get("lang_code") or "").lower() in english_codes

    manual_english = [t for t in tracks if _is_english(t) and t.get("kind") != "asr"]
    if manual_english:
        return manual_english[0]

    auto_english = [t for t in tracks if _is_english(t) and t.get("kind") == "asr"]
    if auto_english:
        return auto_english[0]

    any_manual = [t for t in tracks if t.get("kind") != "asr"]
    if any_manual:
        return any_manual[0]

    any_auto = [t for t in tracks if t.get("kind") == "asr"]
    return any_auto[0] if any_auto else None


def _timedtext_fetch_track(video_id: str, track: dict, proxy_session_key: str | None = None) -> dict | None:
    params = {
        "v": video_id,
        "lang": track.get("lang_code") or "",
    }
    if track.get("kind") == "asr":
        params["kind"] = "asr"
    if track.get("name"):
        params["name"] = track["name"]

    fetch_attempts = [
        ("vtt", {**params, "fmt": "vtt"}),
        ("srv3", {**params, "fmt": "srv3"}),
        ("xml", params),
    ]

    for fmt, attempt_params in fetch_attempts:
        query = urlencode(attempt_params)
        timedtext_url = f"https://www.youtube.com/api/timedtext?{query}"
        response = _http_client().get(
            timedtext_url,
            headers={
                **TIMEDTEXT_HEADERS,
                "Accept": "text/plain,text/vtt,application/xml,text/xml,*/*",
            },
            timeout_seconds=10,
            proxy_session_key=proxy_session_key,
            treat_empty_as_block=True,
        )
        status = response.status_code
        body = response.text

        if status != 200 or len(body) <= 50:
            continue

        lowered = body.lower()
        if "webvtt" not in lowered and "<text" not in lowered and "<p" not in lowered:
            continue

        segments = _parse_timedtext_segments(body)
        if not segments:
            continue

        return {
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "video_id": video_id,
            "method": "youtube_timedtext",
            "language": track.get("lang_code") or None,
            "segments": segments,
            "full_text": _full_text(segments),
            "timedtext_fetch_format": fmt,
            "timedtext_fetch_status": status,
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

    diagnostics = {
        "yta_error": None,
        "timedtext_tracks_found": 0,
        "timedtext_best_track": None,
        "timedtext_fetch_status": None,
        "timedtext_list_http_status": None,
        "timedtext_list_body_len": None,
        "timedtext_list_body_head": None,
        "transcript_method_final": None,
    }

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

        result = {
            "url": source["url"],
            "video_id": video_id,
            "method": "youtube_transcript",
            "language": language,
            "segments": segments,
            "full_text": _full_text(segments),
        }
        diagnostics["transcript_method_final"] = "youtube_transcript"
        result["diagnostics"] = diagnostics
        return result
    except Exception as exc:
        diagnostics["yta_error"] = str(exc)

    try:
        tracks, timedtext_list_diagnostics = _timedtext_list_tracks(video_id, proxy_session_key=video_id)
        diagnostics.update(timedtext_list_diagnostics)
        diagnostics["timedtext_tracks_found"] = len(tracks)
        best_track = _pick_best_track(tracks)
        diagnostics["timedtext_best_track"] = (
            {
                "lang_code": best_track.get("lang_code"),
                "kind": best_track.get("kind"),
                "name": best_track.get("name"),
            }
            if best_track
            else None
        )
        if best_track:
            timedtext = _timedtext_fetch_track(video_id, best_track, proxy_session_key=video_id)
            diagnostics["timedtext_fetch_status"] = (
                timedtext.get("timedtext_fetch_status") if timedtext else "empty"
            )
            if timedtext and timedtext.get("segments"):
                timedtext["url"] = source["url"]
                diagnostics["transcript_method_final"] = "youtube_timedtext"
                timedtext["diagnostics"] = diagnostics
                return timedtext
        else:
            diagnostics["timedtext_fetch_status"] = "no_track"
    except Exception as exc:
        diagnostics["timedtext_fetch_status"] = f"error: {exc}"

        if not allow_audio_fallback:
            raise TranscriptUnavailableError(
                "transcript_unavailable_yta; transcript_unavailable_timedtext; No transcript available (audio fallback disabled).",
                video_id=video_id,
                url=source["url"],
                code="transcript_unavailable_disabled",
                diagnostics=diagnostics,
            )

        if not openai_api_key_present:
            raise TranscriptionError(
                "transcript_unavailable_yta; transcript_unavailable_timedtext; Audio fallback requires OPENAI_API_KEY.",
                video_id=video_id,
                url=source["url"],
                code="audio_fallback_requires_openai_key",
                diagnostics=diagnostics,
            )

        try:
            audio_result = _transcribe_with_openai_audio(source)
            diagnostics["transcript_method_final"] = "audio_fallback"
            audio_result["diagnostics"] = diagnostics
            return audio_result
        except Exception as audio_exc:
            raise TranscriptionError(
                f"transcript_unavailable_yta; transcript_unavailable_timedtext; audio_fallback_failed: {audio_exc}",
                video_id=video_id,
                url=source["url"],
                code="audio_fallback_failed",
                diagnostics=diagnostics,
            ) from audio_exc
    except TranscriptionError:
        raise

    if not allow_audio_fallback:
        raise TranscriptUnavailableError(
            "transcript_unavailable_yta; transcript_unavailable_timedtext; No transcript available (audio fallback disabled).",
            video_id=video_id,
            url=source["url"],
            code="transcript_unavailable_disabled",
            diagnostics=diagnostics,
        )

    if not openai_api_key_present:
        raise TranscriptionError(
            "transcript_unavailable_yta; transcript_unavailable_timedtext; Audio fallback requires OPENAI_API_KEY.",
            video_id=video_id,
            url=source["url"],
            code="audio_fallback_requires_openai_key",
            diagnostics=diagnostics,
        )

    try:
        audio_result = _transcribe_with_openai_audio(source)
        diagnostics["transcript_method_final"] = "audio_fallback"
        audio_result["diagnostics"] = diagnostics
        return audio_result
    except Exception as audio_exc:
        raise TranscriptionError(
            f"transcript_unavailable_yta; transcript_unavailable_timedtext; audio_fallback_failed: {audio_exc}",
            video_id=video_id,
            url=source["url"],
            code="audio_fallback_failed",
            diagnostics=diagnostics,
        ) from audio_exc


def get_transcript_from_worker(
    source: dict,
    *,
    worker_url: str,
    worker_api_key: str,
    allow_audio_fallback: bool,
    prefer_lang: list[str] | None = None,
) -> dict:
    response = requests.post(
        f"{worker_url.rstrip('/')}/transcript",
        headers={"X-Api-Key": worker_api_key, "Content-Type": "application/json"},
        json={
            "source_id": f"yt:{_video_id_from_url(source['url'])}",
            "url": source["url"],
            "language": (prefer_lang or ["en"])[0] if (prefer_lang or ["en"]) else None,
            "allow_audio_fallback": allow_audio_fallback,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "url": source["url"],
        "video_id": payload.get("video_id") or _video_id_from_url(source["url"]),
        "method": payload.get("method", "worker"),
        "language": payload.get("language"),
        "segments": payload.get("segments", []),
        "full_text": payload.get("text", "").strip(),
        "diagnostics": payload.get("diagnostics", {}),
    }
