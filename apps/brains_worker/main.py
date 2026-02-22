from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urlencode
from urllib.parse import parse_qs, urlparse

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound

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
    if not (header_key or "").strip():
        raise HTTPException(status_code=401, detail="Missing x-api-key")
    if header_key != expected:
        raise HTTPException(status_code=401, detail="Invalid x-api-key")


def _tail(value: str | None, limit: int = 2000) -> str:
    return (value or "")[-limit:]


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
                return status_code, [], "timedtext_list_failed"
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


def timedtext_fetch_track(
    video_id: str,
    track: dict,
    *,
    proxy: str | None,
) -> tuple[int | None, str | None, list[dict[str, Any]] | None, str | None]:
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
                return status_code, _full_text(segments), segments, None
            except requests.RequestException as exc:
                last_error = str(exc)[:120]
    if status_code:
        return status_code, None, None, f"timedtext_fetch_http_{status_code}"
    return status_code, None, None, (last_error or "timedtext_fetch_failed")


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


def ytdlp_download_audio(video_id: str, *, proxy: str | None, work_dir: str) -> tuple[str | None, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "audio_download_ok": False,
        "audio_download_exit_code": None,
        "audio_download_stderr_tail": "",
        "audio_download_path": None,
    }
    yt_dlp_bin = "/opt/brains-worker/.venv/bin/yt-dlp"
    if not os.path.exists(yt_dlp_bin):
        yt_dlp_bin = shutil.which("yt-dlp") or ""
    if not yt_dlp_bin:
        diagnostics["error"] = "audio_download_failed"
        return None, diagnostics

    output_template = os.path.join(work_dir, "%(id)s.%(ext)s")
    attempts = [None, proxy] if proxy else [None]
    for attempt_proxy in attempts:
        cmd = [
            yt_dlp_bin,
            "--no-playlist",
            "-f",
            "bestaudio/best",
            "-o",
            output_template,
            "--restrict-filenames",
            "--quiet",
            "--no-warnings",
        ]
        if attempt_proxy:
            cmd.extend(["--proxy", attempt_proxy])
        cmd.append(f"https://www.youtube.com/watch?v={video_id}")

        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            diagnostics["audio_download_exit_code"] = -1
            diagnostics["audio_download_stderr_tail"] = "yt-dlp timeout"
            continue
        diagnostics["audio_download_exit_code"] = completed.returncode
        diagnostics["audio_download_stderr_tail"] = _tail(completed.stderr)
        if completed.returncode != 0:
            continue

        for name in os.listdir(work_dir):
            path = os.path.join(work_dir, name)
            if not os.path.isfile(path) or name == "audio.wav":
                continue
            diagnostics["audio_download_ok"] = True
            diagnostics["audio_download_path"] = path
            diagnostics["audio_file_bytes"] = os.path.getsize(path)
            return path, diagnostics

    diagnostics["error"] = "audio_download_failed"
    return None, diagnostics


def convert_audio_for_transcription(input_path: str, work_dir: str) -> tuple[str | None, int | None, str | None, str]:
    output_path = os.path.join(work_dir, "audio.wav")
    ffmpeg_bin = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    cmd = [ffmpeg_bin, "-y", "-i", input_path, "-ac", "1", "-ar", "16000", "-vn", output_path]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return None, -1, "audio_convert_failed", "ffmpeg timeout"
    if completed.returncode != 0:
        return None, completed.returncode, "audio_convert_failed", _tail(completed.stderr)
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
        return None, completed.returncode, "audio_convert_failed", "audio.wav missing or too small"
    return output_path, completed.returncode, None, _tail(completed.stderr)


def _transcribe_with_openai(path: str, diagnostics: dict[str, Any]) -> tuple[str | None, list[dict] | None, str | None]:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None, None, "audio_fallback_missing_api_key"
    model = (os.getenv("BRAINS_TRANSCRIBE_MODEL") or "gpt-4o-mini-transcribe").strip()
    with open(path, "rb") as audio_handle:
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (os.path.basename(path), audio_handle, "audio/wav")},
            data={"model": model, "response_format": "verbose_json"},
            timeout=180,
        )
    diagnostics["openai_http_status"] = response.status_code
    diagnostics["transcription_method"] = "openai"
    if response.status_code != 200:
        diagnostics["openai_error"] = _tail(response.text)
        return None, None, "audio_transcribe_failed"
    payload = response.json()
    segments = []
    for segment in payload.get("segments") or []:
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", start) or start)
        text = (segment.get("text") or "").strip()
        if text:
            segments.append({"start": start, "duration": max(0.0, end - start), "text": text})
    text = (payload.get("text") or _full_text(segments)).strip()
    if not text:
        return None, None, "audio_transcribe_failed"
    return text, segments, None


def _transcribe_with_faster_whisper(path: str, diagnostics: dict[str, Any]) -> tuple[str | None, list[dict] | None, str | None]:
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return None, None, "audio_fallback_dependency_missing"
    diagnostics["transcription_method"] = "faster-whisper"
    model_size = (os.getenv("BRAINS_FASTER_WHISPER_MODEL") or "base").strip()
    model = WhisperModel(model_size, device="cpu")
    items, _ = model.transcribe(path)
    segments: list[dict[str, Any]] = []
    for item in items:
        text = (getattr(item, "text", "") or "").strip()
        if text:
            start = float(getattr(item, "start", 0.0) or 0.0)
            end = float(getattr(item, "end", start) or start)
            segments.append({"start": start, "duration": max(0.0, end - start), "text": text})
    text = _full_text(segments)
    if not text:
        return None, None, "audio_transcribe_failed"
    return text, segments, None


def transcribe_youtube_audio(video_id: str, diagnostics: dict[str, Any], proxy_url: str | None) -> tuple[str | None, list[dict] | None]:
    base_tmp_dir = "/opt/brains-worker/tmp"
    os.makedirs(base_tmp_dir, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix=f"{video_id}-", dir=base_tmp_dir)
    keep_tmp = (os.getenv("BRAINS_KEEP_AUDIO_TMP") or "").strip() == "1"
    try:
        audio_path, download_diag = ytdlp_download_audio(video_id, proxy=proxy_url, work_dir=work_dir)
        diagnostics["audio_download_ok"] = bool(download_diag.get("audio_download_ok"))
        diagnostics["audio_file_bytes"] = download_diag.get("audio_file_bytes")
        if not audio_path:
            diagnostics["error"] = download_diag.get("error") or "audio_download_failed"
            diagnostics["transcription_engine"] = "none"
            return None, None

        if (download_diag.get("audio_file_bytes") or 0) < 50 * 1024:
            diagnostics["error"] = "audio_download_too_small"
            diagnostics["transcription_engine"] = "none"
            return None, None

        transcribe_path, ffmpeg_exit_code, convert_error, ffmpeg_stderr_tail = convert_audio_for_transcription(audio_path, work_dir)
        diagnostics["ffmpeg_exit_code"] = ffmpeg_exit_code
        diagnostics["ffmpeg_stderr_tail"] = ffmpeg_stderr_tail
        if convert_error or not transcribe_path:
            diagnostics["error"] = "audio_convert_failed"
            diagnostics["transcription_engine"] = "none"
            return None, None

        if (os.getenv("OPENAI_API_KEY") or "").strip():
            diagnostics["transcription_engine"] = "openai"
            text, segments, asr_error = _transcribe_with_openai(transcribe_path, diagnostics)
        else:
            diagnostics["transcription_engine"] = "faster_whisper"
            text, segments, asr_error = _transcribe_with_faster_whisper(transcribe_path, diagnostics)
            if asr_error == "audio_fallback_dependency_missing":
                diagnostics["hint"] = "Set OPENAI_API_KEY on worker OR install faster-whisper"
        if asr_error or not text:
            diagnostics["error"] = asr_error or "audio_transcribe_failed"
            diagnostics["transcription_engine"] = "none"
            return None, None
        return text, segments or []
    finally:
        if not keep_tmp:
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
def transcript(
    payload: TranscriptRequest,
    x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
    x_api_key_lower: str | None = Header(default=None, alias="x-api-key"),
) -> TranscriptResponse:
    _require_api_key(x_api_key or x_api_key_lower)

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
        "yta_error": None,
        "timedtext_tracks_found": 0,
        "timedtext_best_track": None,
        "timedtext_list_http_status": None,
        "timedtext_fetch_http_status": None,
        "audio_download_ok": False,
        "audio_file_bytes": None,
        "transcription_engine": "none",
        "error": None,
    }

    try:
        yta_segments, yta_meta = _fetch_transcript_from_yta(video_id, preferred_langs)
        yta_text = _full_text(yta_segments)
        if yta_text:
            diagnostics["timedtext_tracks_found"] = 0
            diagnostics["timedtext_best_track"] = {
                "lang": yta_meta.get("language"),
                "kind": "asr" if yta_meta.get("is_generated") else "manual",
                "name": "youtube_transcript_api",
            }
            return TranscriptResponse(method="captions", text=yta_text, segments=yta_segments, diagnostics=diagnostics)
    except Exception as exc:
        diagnostics["yta_error"] = str(exc)[:300]

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
        fetch_status, transcript_text, transcript_segments, fetch_error = timedtext_fetch_track(
            video_id,
            best_track,
            proxy=proxy_url,
        )
        diagnostics["timedtext_fetch_http_status"] = fetch_status
        if transcript_text:
            return TranscriptResponse(
                method="captions",
                text=transcript_text,
                segments=transcript_segments or [],
                diagnostics=diagnostics,
            )
        diagnostics["error"] = fetch_error or diagnostics["error"] or "timedtext_failed"

    if not payload.allow_audio_fallback:
        diagnostics["error"] = diagnostics["error"] or "no_caption_tracks"
        return TranscriptResponse(method="audio_fallback_failed", text="", segments=[], diagnostics=diagnostics)

    text, segments = transcribe_youtube_audio(video_id, diagnostics, proxy_url=proxy_url)
    if not text:
        return TranscriptResponse(method="audio_fallback_failed", text="", segments=[], diagnostics=diagnostics)
    return TranscriptResponse(method="audio_fallback", text=text, segments=segments or [], diagnostics=diagnostics)
