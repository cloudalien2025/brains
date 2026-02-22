from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from Brains_Ingestion_App.brains.net.proxy_manager import ProxyManager

app = FastAPI(title="Brains Worker")
logger = logging.getLogger(__name__)

ALLOWED_STT_MODELS = {"gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"}
HTTPStatusValue = int | str


class TranscriptRequest(BaseModel):
    video_id: str | None = Field(default=None, examples=["yt:5lQf89-AeFo"])
    source_id: str | None = Field(default=None, examples=["yt:5lQf89-AeFo"])
    url: str | None = None
    preferred_language: str | None = None
    captions_enabled: bool = True
    yt_dlp_subs_enabled: bool = True
    audio_fallback_enabled: bool = False
    allow_audio_fallback: bool | None = None
    stt_provider: str = "openai"
    stt_model: str = "gpt-4o-mini-transcribe"
    stt_language: str | None = None
    stt_prompt: str | None = None
    debug_keep_files: bool = False
    debug_include_artifact_paths: bool = False
    proxy_enabled: bool | None = None
    proxy_country: str | None = None
    proxy_sticky: bool = True
    proxy_url: str | None = None


def _tail(value: str | None, limit: int = 300) -> str:
    return (value or "")[-limit:]


def _safe_ytdlp_available() -> bool:
    path = shutil.which("yt-dlp")
    if path:
        return True
    try:
        return subprocess.run(["yt-dlp", "--version"], capture_output=True, timeout=2).returncode == 0
    except Exception:
        return False


def _safe_ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg"))


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


def _version_payload() -> dict[str, Any]:
    return {
        "service": "brains-worker",
        "git_commit": _detect_git_commit(),
        "build_time_utc": os.getenv("BRAINS_BUILD_TIME_UTC", "unknown"),
        "pipeline": {
            "player_json_captions": True,
            "ytdlp_subtitles": True,
            "audio_openai_stt": True,
        },
        "yt_dlp_available": _safe_ytdlp_available(),
        "ffmpeg_available": _safe_ffmpeg_available(),
        "openai_key_present": bool((os.getenv("OPENAI_API_KEY") or "").strip()),
    }


def _expected_api_key() -> str:
    return (os.getenv("BRAINS_API_KEY") or os.getenv("BRAINS_WORKER_API_KEY") or "").strip()


def _auth_error() -> JSONResponse:
    return JSONResponse(status_code=401, content={"error_code": "WORKER_AUTH_FAILED", "error": "Missing or invalid x-api-key", "diagnostics": {}})


def _error(status_code: int, code: str, message: str, diagnostics: dict[str, Any]) -> JSONResponse:
    diagnostics["final_error_code"] = code
    diagnostics["final_error"] = message
    return JSONResponse(status_code=status_code, content={"error_code": code, "error": message, "diagnostics": diagnostics})


def _build_proxy_url(payload: TranscriptRequest, video_id: str) -> str | None:
    if payload.proxy_url:
        return payload.proxy_url
    proxy_manager = ProxyManager()
    try:
        managed = proxy_manager.get_proxies(proxy_session_key=video_id)
        managed_url = (managed or {}).get("https") or (managed or {}).get("http")
    except Exception:
        managed_url = None
    if payload.proxy_enabled is False:
        return None
    if managed_url and payload.proxy_enabled is not True:
        return managed_url
    enabled_flag = payload.proxy_enabled is True or (os.getenv("BRAINS_PROXY_ENABLED") or "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled_flag:
        return managed_url
    host = (os.getenv("DECODO_HOST") or os.getenv("DECDO_HOST") or "").strip() or "gate.decodo.com"
    port = (os.getenv("DECODO_PORT") or os.getenv("DECDO_PORT") or "").strip() or "7000"
    username = (os.getenv("DECODO_USERNAME") or os.getenv("DECDO_USERNAME") or "").strip()
    password = (os.getenv("DECODO_PASSWORD") or os.getenv("DECDO_PASSWORD") or "").strip()
    if not (username and password):
        return managed_url
    if payload.proxy_country:
        username = f"{username}-country-{payload.proxy_country.lower()}"
    if payload.proxy_sticky:
        safe_session = "".join(ch for ch in video_id if ch.isalnum())[:24] or "sticky"
        username = f"{username}-session-{safe_session}"
    return f"http://{username}:{password}@{host}:{port}"


def _extract_video_id(video_id: str | None, source_id: str | None, url: str | None) -> str:
    candidates = [video_id, source_id]
    for candidate in candidates:
        if not candidate:
            continue
        cleaned = candidate.strip()
        if cleaned.startswith("yt:"):
            cleaned = cleaned.split(":", 1)[1].strip()
        if "youtube.com" in cleaned or "youtu.be" in cleaned:
            url = cleaned
            break
        if re.match(r"^[A-Za-z0-9_-]{6,}$", cleaned):
            return cleaned
    if url:
        parsed = urlparse(url.strip())
        host = parsed.netloc.lower()
        if "youtu.be" in host:
            short_id = parsed.path.strip("/").split("/")[0]
            if re.match(r"^[A-Za-z0-9_-]{6,}$", short_id):
                return short_id
        qv = (parse_qs(parsed.query).get("v") or [""])[0]
        if re.match(r"^[A-Za-z0-9_-]{6,}$", qv):
            return qv
    return ""


def _youtube_watch_player_json(video_id: str, proxy: str | None) -> tuple[HTTPStatusValue, dict[str, Any] | None, str | None]:
    proxies = {"http": proxy, "https": proxy} if proxy else None
    url = f"https://www.youtube.com/watch?v={video_id}&hl=en"
    try:
        response = requests.get(url, timeout=20, proxies=proxies, headers={"User-Agent": "Mozilla/5.0"})
        status = response.status_code
        if status in {429, 403}:
            return status, None, "PLAYER_JSON_BLOCKED"
        if status != 200:
            return status, None, "PLAYER_JSON_FETCH_FAILED"
        text = response.text or ""
        if "consent.youtube.com" in text:
            return status, None, "PLAYER_JSON_CONSENT_REQUIRED"
        match = re.search(r"ytInitialPlayerResponse\s*=\s*(\{.+?\})\s*;", text)
        if not match:
            return status, None, "PLAYER_JSON_MISSING"
        return status, json.loads(match.group(1)), None
    except Exception:
        return "exception", None, "PLAYER_JSON_EXCEPTION"


def _choose_caption_track(tracks: list[dict[str, Any]], preferred_language: str | None) -> dict[str, Any] | None:
    if not tracks:
        return None
    pref = (preferred_language or "").lower()

    def rank(track: dict[str, Any]) -> tuple[int, int, str]:
        lang = (track.get("languageCode") or "").lower()
        exact = 0 if pref and (lang == pref or lang.startswith(f"{pref}-")) else 1
        english = 0 if lang.startswith("en") else 1
        auto = 1 if (track.get("kind") == "asr") else 0
        return exact, english, auto

    return sorted(tracks, key=rank)[0]


def _strip_caption_markup(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_subtitle_to_text(raw: str, ext: str) -> str:
    content = raw.strip()
    if not content:
        return ""
    lines: list[str] = []
    if ext in {"vtt", "srt"}:
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.upper() == "WEBVTT" or stripped.isdigit() or "-->" in stripped:
                continue
            lines.append(_strip_caption_markup(stripped))
        return re.sub(r"\s+", " ", " ".join(lines)).strip()
    if ext in {"ttml", "xml"}:
        try:
            root = ET.fromstring(content)
            for node in root.findall(".//{*}p") + root.findall(".//text"):
                val = _strip_caption_markup("".join(node.itertext()))
                if val:
                    lines.append(val)
            return re.sub(r"\s+", " ", " ".join(lines)).strip()
        except ET.ParseError:
            return ""
    return ""


def _download_text(url: str, proxy: str | None) -> tuple[HTTPStatusValue, str | None]:
    proxies = {"http": proxy, "https": proxy} if proxy else None
    try:
        response = requests.get(url, timeout=20, proxies=proxies, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return response.status_code, None
        return response.status_code, response.text
    except Exception:
        return "exception", None


def _run_ytdlp_subtitles(youtube_url: str, preferred_language: str | None, proxy: str | None, work_dir: Path) -> tuple[str | None, dict[str, Any]]:
    start = time.perf_counter()
    diag = {
        "ytdlp_subs_status": "failed",
        "ytdlp_subs_error_code": None,
        "ytdlp_subs_elapsed_ms": 0,
        "ytdlp_subs_lang": None,
        "ytdlp_subs_format": None,
        "ytdlp_subs_file_bytes": 0,
        "ytdlp_subs_file_path": None,
    }
    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        diag["ytdlp_subs_error_code"] = "YTDLP_SUBS_FAILED"
        diag["ytdlp_subs_elapsed_ms"] = int((time.perf_counter() - start) * 1000)
        return None, diag
    langs = ",".join([p for p in [preferred_language, "en.*"] if p])
    cmd = [
        yt_dlp,
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        langs or "en.*",
        "--sub-format",
        "vtt/srt/ttml/best",
        "-o",
        str(work_dir / "%(id)s.%(ext)s"),
        youtube_url,
    ]
    if proxy:
        cmd.extend(["--proxy", proxy])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except Exception:
        diag["ytdlp_subs_error_code"] = "YTDLP_SUBS_FAILED"
        diag["ytdlp_subs_elapsed_ms"] = int((time.perf_counter() - start) * 1000)
        return None, diag

    diag["ytdlp_subs_elapsed_ms"] = int((time.perf_counter() - start) * 1000)
    if proc.returncode != 0:
        diag["ytdlp_subs_error_code"] = "YTDLP_SUBS_FAILED"
        return None, diag

    files = [p for p in work_dir.glob("*.*") if p.suffix.lower().lstrip(".") in {"vtt", "srt", "ttml"}]
    if not files:
        diag["ytdlp_subs_error_code"] = "NO_SUBTITLE_FILES"
        return None, diag

    pref = (preferred_language or "").lower()

    def score(path: Path) -> tuple[int, int, int]:
        name = path.name.lower()
        auto = 1 if ".live_chat." in name or ".asr." in name else 0
        lang_match = 0 if pref and f".{pref}" in name else 1
        english = 0 if ".en" in name else 1
        return auto, lang_match, english

    selected = sorted(files, key=score)[0]
    text = _parse_subtitle_to_text(selected.read_text(encoding="utf-8", errors="ignore"), selected.suffix.lower().lstrip("."))
    if not text:
        diag["ytdlp_subs_error_code"] = "SUBTITLE_PARSE_FAILED"
        return None, diag

    diag["ytdlp_subs_status"] = "success"
    diag["ytdlp_subs_error_code"] = None
    diag["ytdlp_subs_file_bytes"] = selected.stat().st_size
    diag["ytdlp_subs_lang"] = next((part for part in selected.stem.split(".") if len(part) in {2, 5} and part[:2].isalpha()), None)
    diag["ytdlp_subs_format"] = selected.suffix.lower().lstrip(".")
    diag["ytdlp_subs_file_path"] = str(selected)
    return text, diag


def _download_audio(youtube_url: str, proxy: str | None, work_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    start = time.perf_counter()
    diag = {
        "audio_download_status": "failed",
        "audio_download_error_code": None,
        "audio_download_elapsed_ms": 0,
        "audio_file_ext": None,
        "audio_file_bytes": 0,
        "audio_file_path": None,
    }
    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        diag["audio_download_error_code"] = "YTDLP_AUDIO_DOWNLOAD_FAILED"
        return None, diag
    cmd = [
        yt_dlp,
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "-x",
        "--audio-format",
        "mp3",
        "-o",
        str(work_dir / "%(id)s.%(ext)s"),
        youtube_url,
    ]
    if proxy:
        cmd.extend(["--proxy", proxy])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except Exception:
        diag["audio_download_elapsed_ms"] = int((time.perf_counter() - start) * 1000)
        diag["audio_download_error_code"] = "YTDLP_AUDIO_DOWNLOAD_FAILED"
        return None, diag
    diag["audio_download_elapsed_ms"] = int((time.perf_counter() - start) * 1000)
    if proc.returncode != 0:
        diag["audio_download_error_code"] = "YTDLP_AUDIO_DOWNLOAD_FAILED"
        return None, diag
    candidates = sorted([p for p in work_dir.glob("*.*") if p.suffix.lower().lstrip(".") in {"mp3", "m4a", "webm", "opus"}])
    if not candidates:
        diag["audio_download_error_code"] = "YTDLP_AUDIO_DOWNLOAD_FAILED"
        return None, diag
    audio_path = candidates[0]
    diag["audio_download_status"] = "success"
    diag["audio_file_ext"] = audio_path.suffix.lower().lstrip(".")
    diag["audio_file_bytes"] = audio_path.stat().st_size
    diag["audio_file_path"] = str(audio_path)
    return audio_path, diag


def _transcribe_openai(audio_path: Path, model: str, language: str | None, prompt: str | None) -> tuple[str | None, dict[str, Any], int, str | None, str | None]:
    start = time.perf_counter()
    diag = {
        "stt_provider": "openai",
        "stt_model": model,
        "stt_language": language,
        "stt_status": "failed",
        "stt_error_code": None,
        "stt_elapsed_ms": 0,
        "transcript_chars": 0,
    }
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        diag["stt_error_code"] = "OPENAI_KEY_MISSING"
        return None, diag, 500, "OPENAI_KEY_MISSING", "OpenAI API key is missing"
    if model not in ALLOWED_STT_MODELS:
        model = "gpt-4o-mini-transcribe"
        diag["stt_model"] = model
    headers = {"Authorization": f"Bearer {api_key}"}
    data: dict[str, str] = {"model": model}
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt
    mime = "audio/mpeg" if audio_path.suffix.lower() == ".mp3" else "audio/mp4"
    try:
        with audio_path.open("rb") as handle:
            resp = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=headers,
                files={"file": (audio_path.name, handle, mime)},
                data=data,
                timeout=300,
            )
    except Exception:
        diag["stt_elapsed_ms"] = int((time.perf_counter() - start) * 1000)
        diag["stt_error_code"] = "OPENAI_STT_FAILED"
        return None, diag, 503, "OPENAI_STT_FAILED", "OpenAI transcription failed"

    diag["stt_elapsed_ms"] = int((time.perf_counter() - start) * 1000)
    if resp.status_code == 429:
        diag["stt_error_code"] = "OPENAI_RATE_LIMIT"
        return None, diag, 503, "OPENAI_RATE_LIMIT", "OpenAI rate limit"
    if resp.status_code != 200:
        diag["stt_error_code"] = "OPENAI_STT_FAILED"
        logger.error("openai stt failed: %s", _tail(resp.text))
        return None, diag, 503, "OPENAI_STT_FAILED", "OpenAI transcription failed"
    text = (resp.json().get("text") or "").strip()
    if not text:
        diag["stt_error_code"] = "OPENAI_STT_FAILED"
        return None, diag, 503, "OPENAI_STT_FAILED", "OpenAI transcription returned empty transcript"
    diag["stt_status"] = "success"
    diag["transcript_chars"] = len(text)
    return text, diag, 200, None, None


def _base_diagnostics(video_id: str, payload: TranscriptRequest, proxy_enabled: bool, preferred_language: str | None) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "proxy_enabled": proxy_enabled,
        "preferred_language": preferred_language,
        "pipeline_stage_attempts": [],
        "pipeline_stage_success": None,
        "elapsed_ms_total": 0,
        "player_json_attempted": False,
        "player_json_http_status": "exception",
        "player_json_error_code": None,
        "caption_tracks_found": 0,
        "caption_best_track": None,
        "caption_download_http_status": "exception",
        "caption_parse_status": "skipped",
        "caption_chars": 0,
        "ytdlp_subs_attempted": False,
        "ytdlp_subs_status": "skipped",
        "ytdlp_subs_error_code": None,
        "ytdlp_subs_elapsed_ms": 0,
        "ytdlp_subs_lang": None,
        "ytdlp_subs_format": None,
        "ytdlp_subs_file_bytes": 0,
        "ytdlp_subs_file_path": None,
        "audio_fallback_enabled": payload.audio_fallback_enabled if payload.allow_audio_fallback is None else payload.allow_audio_fallback,
        "audio_fallback_attempted": False,
        "audio_download_status": "skipped",
        "audio_download_error_code": None,
        "audio_download_elapsed_ms": 0,
        "audio_file_ext": None,
        "audio_file_bytes": 0,
        "audio_file_path": None,
        "stt_provider": None,
        "stt_model": payload.stt_model,
        "stt_language": payload.stt_language,
        "stt_status": "skipped",
        "stt_error_code": None,
        "stt_elapsed_ms": 0,
        "transcript_chars": 0,
        "final_error_code": None,
        "final_error": None,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    proxy_manager = ProxyManager()
    return {"ok": True, "version": os.getenv("BRAINS_WORKER_VERSION", "dev"), "time": datetime.now(timezone.utc).isoformat(), "proxy": proxy_manager.safe_diagnostics()}


@app.get("/transcript/version")
def transcript_version() -> dict[str, Any]:
    return _version_payload()


@app.post("/transcript")
def transcript(payload: TranscriptRequest, x_api_key: str | None = Header(default=None, alias="x-api-key"), x_api_key_2: str | None = Header(default=None, alias="X-Api-Key")) -> Any:
    expected_key = _expected_api_key()
    provided = (x_api_key or x_api_key_2 or "").strip()
    if expected_key and provided != expected_key:
        return _auth_error()

    started = time.perf_counter()
    canonical_id = _extract_video_id(payload.video_id, payload.source_id, payload.url)
    preferred_language = (payload.preferred_language or "").strip() or None
    if not canonical_id:
        diagnostics = _base_diagnostics("", payload, False, preferred_language)
        diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
        return _error(400, "INVALID_VIDEO_ID", "Unable to determine YouTube video id", diagnostics)

    fallback_enabled = payload.audio_fallback_enabled if payload.allow_audio_fallback is None else payload.allow_audio_fallback
    payload.audio_fallback_enabled = fallback_enabled
    youtube_url = f"https://www.youtube.com/watch?v={canonical_id}"
    proxy_url = _build_proxy_url(payload, canonical_id)

    diagnostics = _base_diagnostics(canonical_id, payload, bool(proxy_url), preferred_language)

    temp_root = Path("/tmp/brains-worker")
    temp_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{canonical_id}-", dir=str(temp_root)))

    try:
        if payload.captions_enabled:
            diagnostics["pipeline_stage_attempts"].append("captions_player_json")
            diagnostics["player_json_attempted"] = True
            status, player, player_err = _youtube_watch_player_json(canonical_id, proxy_url)
            diagnostics["player_json_http_status"] = status
            diagnostics["player_json_error_code"] = player_err
            tracks = ((player or {}).get("captions") or {}).get("playerCaptionsTracklistRenderer", {}).get("captionTracks", [])
            diagnostics["caption_tracks_found"] = len(tracks)
            if tracks:
                best = _choose_caption_track(tracks, preferred_language)
                if best:
                    diagnostics["caption_best_track"] = {
                        "lang": best.get("languageCode"),
                        "kind": best.get("kind"),
                        "name": ((best.get("name") or {}).get("simpleText") if isinstance(best.get("name"), dict) else best.get("name")),
                    }
                    base_url = best.get("baseUrl")
                    if base_url:
                        sep = "&" if "?" in base_url else "?"
                        cap_status, cap_body = _download_text(f"{base_url}{sep}fmt=vtt", proxy_url)
                        diagnostics["caption_download_http_status"] = cap_status
                        if cap_body:
                            parsed = _parse_subtitle_to_text(cap_body, "vtt")
                            if parsed:
                                diagnostics["caption_parse_status"] = "success"
                                diagnostics["caption_chars"] = len(parsed)
                                diagnostics["pipeline_stage_success"] = "captions_player_json"
                                diagnostics["transcript_chars"] = len(parsed)
                                diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
                                return {"video_id": canonical_id, "transcript_source": "captions_player_json", "transcript_text": parsed, "diagnostics": diagnostics}
                            diagnostics["caption_parse_status"] = "failed"
                        else:
                            diagnostics["caption_parse_status"] = "failed"
            if diagnostics["caption_parse_status"] == "skipped" and diagnostics["caption_tracks_found"] == 0:
                diagnostics["caption_download_http_status"] = "exception"
        else:
            diagnostics["caption_parse_status"] = "skipped"

        if payload.yt_dlp_subs_enabled:
            diagnostics["pipeline_stage_attempts"].append("subs_ytdlp")
            diagnostics["ytdlp_subs_attempted"] = True
            text, diag = _run_ytdlp_subtitles(youtube_url, preferred_language, proxy_url, work_dir)
            diagnostics.update(diag)
            if payload.debug_include_artifact_paths is False:
                diagnostics["ytdlp_subs_file_path"] = None
            if text:
                diagnostics["pipeline_stage_success"] = "subs_ytdlp"
                diagnostics["transcript_chars"] = len(text)
                diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
                return {"video_id": canonical_id, "transcript_source": "subs_ytdlp", "transcript_text": text, "diagnostics": diagnostics}
            if diagnostics["ytdlp_subs_error_code"] == "YTDLP_SUBS_FAILED":
                diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
                return _error(502, "YTDLP_SUBS_FAILED", "yt-dlp subtitles extraction failed", diagnostics)

        if not fallback_enabled:
            diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
            return _error(422, "NO_CAPTIONS_AND_FALLBACK_DISABLED", "No captions/subtitles found and audio fallback disabled", diagnostics)

        diagnostics["pipeline_stage_attempts"].append("audio_openai_stt")
        diagnostics["audio_fallback_attempted"] = True
        audio_path, audio_diag = _download_audio(youtube_url, proxy_url, work_dir)
        diagnostics.update(audio_diag)
        if payload.debug_include_artifact_paths is False:
            diagnostics["audio_file_path"] = None
        if not audio_path:
            diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
            return _error(502, "YTDLP_AUDIO_DOWNLOAD_FAILED", "yt-dlp audio download failed", diagnostics)
        text, stt_diag, status_code, err_code, err_message = _transcribe_openai(audio_path, payload.stt_model, payload.stt_language, payload.stt_prompt)
        diagnostics.update(stt_diag)
        if not text:
            diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
            return _error(status_code, err_code or "OPENAI_STT_FAILED", err_message or "OpenAI transcription failed", diagnostics)

        diagnostics["pipeline_stage_success"] = "audio_openai_stt"
        diagnostics["transcript_chars"] = len(text)
        diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
        return {"video_id": canonical_id, "transcript_source": "audio_openai_stt", "transcript_text": text, "diagnostics": diagnostics}
    except Exception:
        logger.exception("unexpected error")
        diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
        return _error(500, "UNEXPECTED_SERVER_ERROR", "Unexpected server error", diagnostics)
    finally:
        if not payload.debug_keep_files:
            shutil.rmtree(work_dir, ignore_errors=True)
