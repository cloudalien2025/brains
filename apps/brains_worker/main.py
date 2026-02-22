from __future__ import annotations

import os
import logging
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

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
    debug_keep_files: bool = False


app = FastAPI(title="Brains Worker")
logger = logging.getLogger(__name__)


HTTPStatusValue = int | str


def _read_expected_api_key() -> str:
    return (os.getenv("BRAINS_API_KEY") or os.getenv("BRAINS_WORKER_API_KEY") or "").strip()


def _require_api_key(header_key: str | None) -> None:
    expected = _read_expected_api_key()
    if not expected:
        raise HTTPException(status_code=500, detail={"error_code": "WORKER_API_KEY_MISSING", "error": "BRAINS_API_KEY is not configured"})
    if not (header_key or "").strip():
        raise HTTPException(status_code=401, detail={"error_code": "MISSING_API_KEY", "error": "Missing x-api-key"})
    if header_key != expected:
        raise HTTPException(status_code=401, detail={"error_code": "INVALID_API_KEY", "error": "Invalid x-api-key"})


def _tail(value: str | None, limit: int = 1800) -> str:
    return (value or "")[-limit:]


def _detect_git_commit() -> str:
    env_commit = (os.getenv("BRAINS_GIT_COMMIT") or os.getenv("GIT_COMMIT") or "").strip()
    if env_commit:
        return env_commit[:12]
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            return (result.stdout or "").strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def _safe_ytdlp_available() -> bool:
    if shutil.which("yt-dlp"):
        return True
    try:
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True, timeout=1)
        return result.returncode == 0
    except Exception:
        return False


def _safe_ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg"))


def _version_payload() -> dict[str, Any]:
    return {
        "service": "brains-worker",
        "git_commit": _detect_git_commit(),
        "build_time_utc": os.getenv("BRAINS_BUILD_TIME_UTC", "unknown"),
        "audio_fallback_supported": True,
        "yt_dlp_available": _safe_ytdlp_available(),
        "ffmpeg_available": _safe_ffmpeg_available(),
        "openai_key_present": bool((os.getenv("OPENAI_API_KEY") or "").strip()),
    }


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


def _full_text(segments: list[dict[str, Any]]) -> str:
    return " ".join((segment.get("text") or "").strip() for segment in segments).strip()


def timedtext_list_tracks(video_id: str, *, proxy: str | None) -> tuple[HTTPStatusValue, list[dict], str | None]:
    url = f"https://www.youtube.com/api/timedtext?type=list&v={video_id}"
    proxies = {"http": proxy, "https": proxy} if proxy else None
    status_code: HTTPStatusValue = "exception"
    for _ in range(3):
        try:
            response = requests.get(url, timeout=15, proxies=proxies)
            status_code = response.status_code
            body = (response.text or "").strip()
            if response.status_code != 200:
                return status_code, [], "TIMEDTEXT_LIST_FAILED"
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
        except requests.RequestException:
            status_code = "exception"
            continue
        except ET.ParseError:
            return status_code, [], "TIMEDTEXT_LIST_PARSE_ERROR"
    return status_code, [], "TIMEDTEXT_LIST_FAILED"


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


def timedtext_fetch_track(video_id: str, track: dict, *, proxy: str | None) -> tuple[HTTPStatusValue, str | None, str | None]:
    proxies = {"http": proxy, "https": proxy} if proxy else None
    params = {"v": video_id, "lang": track.get("lang_code") or ""}
    if track.get("kind") == "asr":
        params["kind"] = "asr"
    if track.get("name"):
        params["name"] = track.get("name")

    status_code: HTTPStatusValue = "exception"
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
            except requests.RequestException:
                status_code = "exception"
                continue
    if isinstance(status_code, int):
        return status_code, None, f"TIMEDTEXT_FETCH_HTTP_{status_code}"
    return status_code, None, "TIMEDTEXT_FETCH_FAILED"


def _http_error(status_code: int, error_code: str, error: str, diagnostics: dict[str, Any]) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"error_code": error_code, "error": error, "diagnostics": diagnostics})


def _run_ytdlp_download(video_id: str, proxy: str | None, work_dir: str, diagnostics: dict[str, Any]) -> Path:
    start = time.perf_counter()
    yt_dlp_bin = shutil.which("yt-dlp")
    if not yt_dlp_bin:
        diagnostics["audio_download_status"] = "failed"
        diagnostics["audio_download_elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
        raise _http_error(502, "YTDLP_DOWNLOAD_FAILED", "yt-dlp is not available", diagnostics)

    output_template = str(Path(work_dir) / "%(id)s.%(ext)s")
    cmd = [
        yt_dlp_bin,
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "-x",
        "--audio-format",
        "mp3",
        "-o",
        output_template,
        "--restrict-filenames",
        "--no-warnings",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    if proxy:
        cmd.extend(["--proxy", proxy])

    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    diagnostics["audio_download_elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
    if completed.returncode != 0:
        diagnostics["audio_download_status"] = "failed"
        diagnostics["audio_download_error_code"] = "YTDLP_DOWNLOAD_FAILED"
        diagnostics["audio_download_stderr_tail"] = _tail(completed.stderr)
        logger.error("yt-dlp failed for video_id=%s", video_id)
        raise _http_error(502, "YTDLP_DOWNLOAD_FAILED", "YouTube audio download failed", diagnostics)

    audio_files = [p for p in Path(work_dir).glob("*") if p.is_file() and p.name != "audio.wav"]
    if not audio_files:
        diagnostics["audio_download_status"] = "failed"
        diagnostics["audio_download_error_code"] = "YTDLP_DOWNLOAD_FAILED"
        raise _http_error(502, "YTDLP_DOWNLOAD_FAILED", "yt-dlp succeeded but produced no audio file", diagnostics)

    audio_path = audio_files[0]
    diagnostics["audio_download_status"] = "success"
    diagnostics["audio_file_bytes"] = audio_path.stat().st_size
    logger.info("yt-dlp succeeded for video_id=%s bytes=%s", video_id, diagnostics["audio_file_bytes"])
    return audio_path


def _transcribe_with_openai(audio_path: Path, diagnostics: dict[str, Any]) -> str:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("BRAINS_TRANSCRIBE_MODEL") or "gpt-4o-mini-transcribe").strip()
    diagnostics["stt_provider"] = "openai"
    diagnostics["stt_model"] = model
    if not api_key:
        diagnostics["stt_status"] = "failed"
        diagnostics["stt_error_code"] = "OPENAI_KEY_MISSING"
        raise _http_error(503, "STT_FAILED", "OpenAI API key missing on worker", diagnostics)

    start = time.perf_counter()
    with audio_path.open("rb") as audio_handle:
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (audio_path.name, audio_handle, "audio/mpeg")},
            data={"model": model},
            timeout=180,
        )
    diagnostics["stt_elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)

    if response.status_code != 200:
        diagnostics["stt_status"] = "failed"
        diagnostics["stt_http_status"] = response.status_code
        diagnostics["stt_error_code"] = f"OPENAI_{response.status_code}"
        diagnostics["stt_error_tail"] = _tail(response.text)
        logger.error("STT failed for audio_path=%s status=%s", audio_path.name, response.status_code)
        raise _http_error(503, "STT_FAILED", "STT provider failed", diagnostics)

    payload = response.json()
    text = (payload.get("text") or "").strip()
    if not text:
        diagnostics["stt_status"] = "failed"
        diagnostics["stt_error_code"] = "OPENAI_EMPTY_TRANSCRIPT"
        raise _http_error(503, "STT_FAILED", "STT returned empty transcript", diagnostics)
    diagnostics["stt_status"] = "success"
    logger.info("STT succeeded for audio_path=%s", audio_path.name)
    return text


def _audio_fallback(video_id: str, proxy_url: str | None, diagnostics: dict[str, Any], debug_keep_files: bool) -> str:
    base_tmp_dir = Path("/tmp/brains-worker")
    base_tmp_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{video_id}-", dir=str(base_tmp_dir)))
    diagnostics["audio_work_dir"] = str(work_dir) if debug_keep_files else None
    try:
        audio_path = _run_ytdlp_download(video_id, proxy_url, str(work_dir), diagnostics)
        return _transcribe_with_openai(audio_path, diagnostics)
    finally:
        if not debug_keep_files:
            shutil.rmtree(work_dir, ignore_errors=True)


@app.get("/health")
def health() -> dict[str, Any]:
    proxy_manager = ProxyManager()
    return {
        "ok": True,
        "version": os.getenv("BRAINS_WORKER_VERSION", "dev"),
        "time": datetime.now(timezone.utc).isoformat(),
        "proxy": proxy_manager.safe_diagnostics(),
    }


@app.get("/transcript/version")
def transcript_version() -> dict[str, Any]:
    return _version_payload()


@app.post("/transcript")
def transcript(
    payload: TranscriptRequest,
    x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
    x_api_key_lower: str | None = Header(default=None, alias="x-api-key"),
) -> dict[str, Any]:
    _require_api_key(x_api_key or x_api_key_lower)

    diagnostics: dict[str, Any] = {
        "proxy_enabled": False,
        "preferred_language": None,
        "timedtext_tracks_found": 0,
        "timedtext_list_http_status": "exception",
        "timedtext_fetch_http_status": 0,
        "timedtext_best_track": None,
        "audio_fallback_enabled": payload.allow_audio_fallback,
        "audio_fallback_attempted": False,
        "audio_download_status": "not_attempted",
        "audio_file_bytes": 0,
        "audio_download_elapsed_ms": 0,
        "stt_provider": "openai",
        "stt_model": (os.getenv("BRAINS_TRANSCRIBE_MODEL") or "gpt-4o-mini-transcribe").strip(),
        "stt_status": "not_attempted",
        "stt_elapsed_ms": 0,
    }

    try:
        video_id = extract_video_id(payload.source_id, payload.url)
        if not video_id:
            raise _http_error(400, "INVALID_VIDEO_ID", "Unable to determine YouTube video_id from source_id/url", diagnostics)

        preferred_langs = payload.preferred_langs or ["en", "en-US", "en-GB"]
        preferred_language = payload.preferred_language or (preferred_langs[0] if preferred_langs else "en")

        proxy_manager = ProxyManager()
        proxy_url = None
        if payload.proxy_enabled is None:
            proxy_url = build_proxy_url(proxy_manager, proxy_session_key=video_id) or build_decodo_proxy_url(payload.proxy_country, payload.proxy_sticky, video_id)
        elif payload.proxy_enabled:
            proxy_url = build_decodo_proxy_url(payload.proxy_country, payload.proxy_sticky, video_id) or build_proxy_url(proxy_manager, proxy_session_key=video_id)

        diagnostics["proxy_enabled"] = bool(proxy_url)
        diagnostics["preferred_language"] = preferred_language

        logger.info("Caption attempt started for video_id=%s fallback_enabled=%s", video_id, payload.allow_audio_fallback)

        list_status, tracks, list_error = timedtext_list_tracks(video_id, proxy=proxy_url)
        diagnostics["timedtext_list_http_status"] = list_status
        diagnostics["timedtext_tracks_found"] = len(tracks)
        if list_error and list_error == "TIMEDTEXT_LIST_FAILED":
            raise _http_error(502, "TIMEDTEXT_LIST_FAILED", "Failed to list caption tracks", diagnostics)

        best_track = _pick_best_track(tracks, preferred_language)
        if best_track:
            diagnostics["timedtext_best_track"] = {
                "lang": best_track.get("lang_code"),
                "kind": best_track.get("kind"),
            }
            fetch_status, transcript_text, fetch_error = timedtext_fetch_track(video_id, best_track, proxy=proxy_url)
            diagnostics["timedtext_fetch_http_status"] = fetch_status
            if transcript_text:
                logger.info("Caption attempt succeeded for video_id=%s", video_id)
                return {
                    "video_id": video_id,
                    "transcript_source": "captions",
                    "transcript_text": transcript_text,
                    "diagnostics": diagnostics,
                }
            if fetch_error and fetch_error.startswith("TIMEDTEXT_FETCH_HTTP_4"):
                raise _http_error(502, "TIMEDTEXT_FETCH_FAILED", "Failed to fetch captions", diagnostics)

        if not payload.allow_audio_fallback:
            logger.warning("No caption tracks and fallback disabled for video_id=%s", video_id)
            raise _http_error(422, "NO_CAPTIONS_FALLBACK_DISABLED", "No captions available and audio fallback disabled", diagnostics)

        diagnostics["audio_fallback_attempted"] = True
        logger.info("Fallback triggered for video_id=%s", video_id)
        text = _audio_fallback(video_id, proxy_url, diagnostics, debug_keep_files=payload.debug_keep_files)
        if not text:
            logger.error("Transcript missing after fallback for video_id=%s", video_id)
            raise _http_error(500, "UNEXPECTED_SERVER_ERROR", "Transcript generation returned empty text", diagnostics)
        logger.info("Final outcome success for video_id=%s source=audio", video_id)
        return {
            "video_id": video_id,
            "transcript_source": "audio",
            "transcript_text": text,
            "diagnostics": diagnostics,
        }
    except HTTPException:
        logger.exception("Final outcome error for transcript request")
        raise
    except Exception:
        logger.exception("Final outcome unexpected server error")
        raise _http_error(500, "UNEXPECTED_SERVER_ERROR", "Unexpected server error", diagnostics)

