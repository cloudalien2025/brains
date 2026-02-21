from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any
from urllib.parse import parse_qs, urlparse

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


def _download_audio(video_id: str, proxy_url: str | None) -> tuple[str, dict[str, Any]]:
    yt_dlp_bin = shutil.which("yt-dlp")
    if not yt_dlp_bin:
        raise HTTPException(status_code=500, detail="yt-dlp is not installed on worker")

    tmp_dir = tempfile.mkdtemp(prefix="brains-audio-")
    output_template = os.path.join(tmp_dir, f"{video_id}.%(ext)s")
    cmd = [
        yt_dlp_bin,
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "--output",
        output_template,
    ]
    if proxy_url:
        cmd.extend(["--proxy", proxy_url])
    cmd.append(f"https://www.youtube.com/watch?v={video_id}")

    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    stderr_head = (completed.stderr or "").strip()[:400]
    diagnostics = {
        "ok": completed.returncode == 0,
        "stderr_head": stderr_head,
        "stdout_head": (completed.stdout or "").strip()[:300],
    }
    if completed.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=502, detail=f"audio_download_failed: {stderr_head or 'yt-dlp failed'}")

    audio_path = None
    for name in os.listdir(tmp_dir):
        if name.startswith(video_id) and name.endswith(".mp3"):
            audio_path = os.path.join(tmp_dir, name)
            break
    if not audio_path:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=502, detail="audio_download_failed: no mp3 output from yt-dlp")

    return audio_path, diagnostics


def _transcribe_audio_with_openai(audio_path: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="No captions and OPENAI_API_KEY not configured for ASR fallback",
        )

    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as audio_handle:
        response = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_handle,
            response_format="verbose_json",
        )

    segments = []
    for segment in getattr(response, "segments", []) or []:
        start = float(getattr(segment, "start", 0.0))
        end = float(getattr(segment, "end", start))
        text = (getattr(segment, "text", "") or "").strip()
        if text:
            segments.append({"start": start, "duration": max(0.0, end - start), "text": text})

    text = (getattr(response, "text", "") or _full_text(segments)).strip()
    return text, segments, {"model": "gpt-4o-mini-transcribe", "ok": bool(text)}


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
    proxy_manager = ProxyManager()
    proxy_url = build_proxy_url(proxy_manager, proxy_session_key=video_id)
    diagnostics: dict[str, Any] = {
        "video_id": video_id,
        "proxy_enabled": bool(proxy_url),
        "yta_error": None,
        "audio_download": {"ok": False, "stderr_head": None},
        "asr": {"model": "gpt-4o-mini-transcribe", "ok": False},
    }

    try:
        segments, selected = _fetch_transcript_from_yta(video_id, preferred_langs)
        text = _full_text(segments)
        if text:
            diagnostics["selected_language"] = selected.get("language")
            diagnostics["is_generated"] = selected.get("is_generated")
            return TranscriptResponse(
                method="yta",
                text=text,
                segments=segments,
                diagnostics=diagnostics,
            )
        diagnostics["yta_error"] = "empty_transcript"
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, TooManyRequests, RequestBlocked) as exc:
        diagnostics["yta_error"] = str(exc)
    except Exception as exc:
        diagnostics["yta_error"] = f"unexpected_yta_error: {exc}"

    audio_path = None
    audio_dir = None
    try:
        audio_path, dl_diag = _download_audio(video_id, proxy_url)
        diagnostics["audio_download"] = dl_diag
        audio_dir = os.path.dirname(audio_path)
        text, segments, asr_diag = _transcribe_audio_with_openai(audio_path)
        diagnostics["asr"] = asr_diag
        return TranscriptResponse(
            method="asr_openai",
            text=text,
            segments=segments,
            diagnostics=diagnostics,
        )
    finally:
        if audio_dir:
            shutil.rmtree(audio_dir, ignore_errors=True)
