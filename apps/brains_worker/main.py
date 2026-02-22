from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urlencode
from urllib.parse import parse_qs, urlparse

import requests
from fastapi import FastAPI, Header, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    RequestBlocked,
    TooManyRequests,
    TranscriptsDisabled,
    VideoUnavailable,
)

from Brains_Ingestion_App.brains.net.proxy_manager import ProxyManager


class TranscriptRequest(BaseModel):
    source_id: str | None = Field(default=None, examples=["yt:5lQf89-AeFo"])
    url: str | None = None
    preferred_langs: list[str] | None = None
    preferred_language: str | None = None
    allow_audio_fallback: bool = True
    proxy_enabled: bool | None = None
    proxy_country: str | None = None
    proxy_sticky: bool = True


class TranscriptResponse(BaseModel):
    method: str
    text: str
    segments: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: dict[str, Any]


app = FastAPI(title="Brains Worker")


def _read_expected_api_key() -> str:
    return (os.getenv("BRAINS_API_KEY") or os.getenv("BRAINS_WORKER_API_KEY") or "").strip()


def _require_api_key(header_key: str | None) -> None:
    expected = _read_expected_api_key()
    if not expected:
        raise HTTPException(status_code=500, detail="BRAINS_API_KEY is not configured")
    if header_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def build_proxy_url(proxy_manager: ProxyManager, proxy_session_key: str | None) -> str | None:
    proxies = proxy_manager.get_proxies(proxy_session_key=proxy_session_key)
    if not proxies:
        return None
    return proxies.get("https") or proxies.get("http")


def build_decodo_proxy_url(country: str | None, sticky: bool, session_id: str | None) -> str | None:
    enabled_flag = (os.getenv("BRAINS_PROXY_ENABLED") or "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled_flag:
        return None

    host = (os.getenv("DECODO_HOST") or os.getenv("DECDO_HOST") or "").strip() or "gate.decodo.com"
    port = (os.getenv("DECODO_PORT") or os.getenv("DECDO_PORT") or "").strip() or "7000"
    username = (os.getenv("DECODO_USERNAME") or os.getenv("DECDO_USERNAME") or "").strip()
    password = (os.getenv("DECODO_PASSWORD") or os.getenv("DECDO_PASSWORD") or "").strip()
    if not (username and password):
        return None

    if country:
        username = f"{username}-country-{country.lower()}"
    if sticky and session_id:
        safe_session = "".join(ch for ch in session_id if ch.isalnum())[:24] or "sticky"
        username = f"{username}-session-{safe_session}"

    return f"http://{username}:{password}@{host}:{port}"


def extract_video_id(source_id: str | None, url: str | None) -> str:
    if source_id:
        cleaned = source_id.strip()
        if cleaned.startswith("yt:"):
            cleaned = cleaned.split(":", 1)[1].strip()
        if len(cleaned) >= 6:
            return cleaned

    if url:
        parsed = urlparse(url.strip())
        host = parsed.netloc.lower()
        if "youtu.be" in host:
            candidate = parsed.path.lstrip("/").strip().split("/")[0]
            if len(candidate) >= 6:
                return candidate
        candidate = (parse_qs(parsed.query).get("v") or [""])[0].strip()
        if len(candidate) >= 6:
            return candidate

    return ""


def _normalize_segments(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for item in items:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            {
                "start": float(item["start"]) if item.get("start") is not None else None,
                "duration": float(item["duration"]) if item.get("duration") is not None else None,
                "text": text,
            }
        )
    return segments


def _full_text(segments: list[dict[str, Any]]) -> str:
    return " ".join((segment.get("text") or "").strip() for segment in segments).strip()


def _parse_timedtext_segments(body: str) -> list[dict[str, Any]]:
    trimmed = body.strip()
    if not trimmed:
        return []
    try:
        root = ET.fromstring(trimmed)
    except ET.ParseError:
        return []

    segments: list[dict[str, Any]] = []
    for node in root.findall(".//text") + root.findall(".//p"):
        text = "".join(node.itertext()).strip()
        if not text:
            continue
        start = float(node.attrib.get("start", 0.0) or 0.0)
        dur = float(node.attrib.get("dur", 0.0) or 0.0)
        segments.append({"start": start, "duration": dur, "text": text})
    return segments


def timedtext_list_tracks(video_id: str, *, proxy: str | None) -> tuple[int | None, list[dict], str | None]:
    url = f"https://www.youtube.com/api/timedtext?type=list&v={video_id}"
    proxies = {"http": proxy, "https": proxy} if proxy else None
    last_error = None
    status_code: int | None = None

    for _ in range(3):
        try:
            response = requests.get(url, timeout=15, proxies=proxies)
            status_code = response.status_code
            body = (response.text or "").strip()
            if response.status_code != 200:
                return status_code, [], f"timedtext_list_http_{response.status_code}"
            if "<track" not in body:
                return status_code, [], None
            root = ET.fromstring(body)
            tracks: list[dict[str, Any]] = []
            for node in root.findall(".//track"):
                tracks.append(
                    {
                        "lang_code": (node.attrib.get("lang_code") or "").strip(),
                        "name": (node.attrib.get("name") or "").strip(),
                        "kind": ((node.attrib.get("kind") or "").strip() or None),
                        "is_auto": (node.attrib.get("kind") or "").strip() == "asr",
                    }
                )
            tracks = [track for track in tracks if track.get("lang_code")]
            return status_code, tracks, None
        except requests.RequestException as exc:
            last_error = str(exc)[:120]
        except ET.ParseError:
            return status_code, [], "timedtext_list_parse_error"
    return status_code, [], (last_error or "timedtext_list_failed")


def _pick_best_track(tracks: list[dict], preferred_language: str | None) -> dict | None:
    if not tracks:
        return None
    preferred = (preferred_language or "").strip().lower()

    def sort_key(track: dict) -> tuple[int, int]:
        lang = (track.get("lang_code") or "").lower()
        lang_match = 0 if preferred and (lang == preferred or lang.startswith(f"{preferred}-")) else 1
        asr_penalty = 1 if track.get("is_auto") else 0
        return (lang_match, asr_penalty)

    return sorted(tracks, key=sort_key)[0]


def timedtext_fetch_track(video_id: str, track: dict, *, proxy: str | None) -> tuple[int | None, str | None, str | None]:
    proxies = {"http": proxy, "https": proxy} if proxy else None
    params = {"v": video_id, "lang": track.get("lang_code") or ""}
    if track.get("kind") == "asr":
        params["kind"] = "asr"
    if track.get("name"):
        params["name"] = track.get("name")

    status_code: int | None = None
    last_error = None
    for fmt in ("srv3", "vtt", None):
        attempt_params = dict(params)
        if fmt:
            attempt_params["fmt"] = fmt
        url = f"https://www.youtube.com/api/timedtext?{urlencode(attempt_params)}"
        for _ in range(2):
            try:
                response = requests.get(url, timeout=15, proxies=proxies)
                status_code = response.status_code
                if response.status_code != 200:
                    continue
                body = (response.text or "").strip()
                segments = _parse_timedtext_segments(body)
                if not segments:
                    continue
                return status_code, _full_text(segments), None
            except requests.RequestException as exc:
                last_error = str(exc)[:120]
    if status_code:
        return status_code, None, f"timedtext_fetch_http_{status_code}"
    return status_code, None, (last_error or "timedtext_fetch_failed")


def _fetch_transcript_from_yta(video_id: str, preferred_langs: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
    selected = None
    for lang in preferred_langs:
        try:
            selected = transcripts.find_manually_created_transcript([lang])
            break
        except Exception:
            continue
    if selected is None:
        for lang in preferred_langs:
            try:
                selected = transcripts.find_generated_transcript([lang])
                break
            except Exception:
                continue
    if selected is None:
        selected = next(iter(transcripts), None)
    if selected is None:
        raise NoTranscriptFound(video_id, preferred_langs, transcripts)

    fetched = selected.fetch()
    segments = _normalize_segments(fetched)
    return segments, {
        "language": getattr(selected, "language_code", None),
        "is_generated": bool(getattr(selected, "is_generated", False)),
    }


def ytdlp_download_audio(video_id: str, *, proxy: str | None) -> tuple[str | None, str | None]:
    yt_dlp_bin = shutil.which("yt-dlp")
    if not yt_dlp_bin:
        return None, "ytdlp_missing"

    tmp_dir = tempfile.mkdtemp(prefix="brains-audio-")
    output_template = os.path.join(tmp_dir, f"{video_id}.%(ext)s")
    attempts = [None, proxy] if proxy else [None]
    for attempt_proxy in attempts:
        cmd = [
            yt_dlp_bin,
            "--no-playlist",
            "-f",
            "bestaudio/best",
            "--extract-audio",
            "--audio-format",
            "mp3",
            "--output",
            output_template,
        ]
        if attempt_proxy:
            cmd.extend(["--proxy", attempt_proxy])
        cmd.append(f"https://www.youtube.com/watch?v={video_id}")

        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        except subprocess.TimeoutExpired:
            continue
        if completed.returncode != 0:
            stderr = (completed.stderr or "").lower()
            if "403" in stderr:
                continue
            continue

        for name in os.listdir(tmp_dir):
            if not (name.startswith(video_id) and name.endswith(".mp3")):
                continue
            path = os.path.join(tmp_dir, name)
            if os.path.getsize(path) < 50 * 1024:
                return None, "ytdlp_no_file"
            return path, None

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return None, "ytdlp_failed"


def transcribe_audio(path: str) -> tuple[str | None, list[dict] | None, str | None]:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None, None, "asr_not_configured"

    client = OpenAI(api_key=api_key)
    try:
        with open(path, "rb") as audio_handle:
            response = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_handle,
                response_format="verbose_json",
            )
    except Exception:
        return None, None, "asr_failed"

    segments = []
    for segment in getattr(response, "segments", []) or []:
        start = float(getattr(segment, "start", 0.0))
        end = float(getattr(segment, "end", start))
        text = (getattr(segment, "text", "") or "").strip()
        if text:
            segments.append({"start": start, "duration": max(0.0, end - start), "text": text})

    text = (getattr(response, "text", "") or _full_text(segments)).strip()
    if not text:
        return None, None, "asr_empty"
    return text, segments, None


@app.get("/health")
def health() -> dict[str, Any]:
    proxy_manager = ProxyManager()
    return {
        "ok": True,
        "proxy": proxy_manager.safe_diagnostics(),
    }


@app.get("/proxy/health")
def proxy_health(x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    _require_api_key(x_api_key)
    proxy_manager = ProxyManager()
    proxy_url = build_proxy_url(proxy_manager, proxy_session_key="proxy-health")
    return {
        "ok": True,
        "proxy": proxy_manager.safe_diagnostics(),
        "proxy_url_present": bool(proxy_url),
        "health": proxy_manager.health_check(proxy_session_key="proxy-health"),
    }


@app.post("/transcript", response_model=TranscriptResponse)
def transcript(payload: TranscriptRequest, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> TranscriptResponse:
    _require_api_key(x_api_key)

    video_id = extract_video_id(payload.source_id, payload.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Unable to determine YouTube video_id from source_id/url")

    preferred_langs = payload.preferred_langs or ["en", "en-US", "en-GB"]
    preferred_language = payload.preferred_language or (preferred_langs[0] if preferred_langs else "en")
    proxy_manager = ProxyManager()
    proxy_url = None
    if payload.proxy_enabled is None:
        proxy_url = build_proxy_url(proxy_manager, proxy_session_key=video_id) or build_decodo_proxy_url(
            payload.proxy_country, payload.proxy_sticky, video_id
        )
    elif payload.proxy_enabled:
        proxy_url = build_decodo_proxy_url(payload.proxy_country, payload.proxy_sticky, video_id) or build_proxy_url(
            proxy_manager, proxy_session_key=video_id
        )
    diagnostics: dict[str, Any] = {
        "video_id": video_id,
        "proxy_enabled": bool(proxy_url),
        "preferred_language": preferred_language,
        "timedtext_tracks_found": 0,
        "timedtext_best_track": None,
        "timedtext_list_http_status": None,
        "timedtext_fetch_http_status": None,
        "error": None,
    }

    list_status, tracks, list_error = timedtext_list_tracks(video_id, proxy=proxy_url)
    diagnostics["timedtext_list_http_status"] = list_status
    diagnostics["timedtext_tracks_found"] = len(tracks)
    if list_error:
        diagnostics["error"] = list_error

    best_track = _pick_best_track(tracks, preferred_language)
    if best_track:
        diagnostics["timedtext_best_track"] = {
            "lang": best_track.get("lang_code"),
            "kind": best_track.get("kind"),
            "name": (best_track.get("name") or "")[:40],
        }
        fetch_status, transcript_text, fetch_error = timedtext_fetch_track(video_id, best_track, proxy=proxy_url)
        diagnostics["timedtext_fetch_http_status"] = fetch_status
        if transcript_text:
            return TranscriptResponse(method="timedtext", text=transcript_text, segments=[], diagnostics=diagnostics)
        diagnostics["error"] = fetch_error or diagnostics["error"] or "timedtext_failed"

    if not payload.allow_audio_fallback:
        diagnostics["error"] = diagnostics["error"] or "no_caption_tracks"
        return TranscriptResponse(method="none", text="", segments=[], diagnostics=diagnostics)

    audio_path, dl_error = ytdlp_download_audio(video_id, proxy=proxy_url)
    if not audio_path:
        diagnostics["error"] = dl_error or "ytdlp_failed"
        return TranscriptResponse(method="none", text="", segments=[], diagnostics=diagnostics)

    try:
        text, segments, asr_error = transcribe_audio(audio_path)
    finally:
        shutil.rmtree(os.path.dirname(audio_path), ignore_errors=True)

    if asr_error or not text:
        diagnostics["error"] = asr_error or "asr_failed"
        return TranscriptResponse(method="none", text="", segments=[], diagnostics=diagnostics)

    return TranscriptResponse(method="asr", text=text, segments=segments or [], diagnostics=diagnostics)
