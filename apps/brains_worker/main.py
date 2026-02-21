from __future__ import annotations

import os
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from Brains_Ingestion_App.adapters.transcription import (
    TranscriptionError,
    get_transcript_for_source,
)
from Brains_Ingestion_App.brains.net.proxy_manager import ProxyManager


class TranscriptRequest(BaseModel):
    source_id: str = Field(..., examples=["yt:5lQf89-AeFo"])
    url: str | None = None
    allow_audio_fallback: bool = False
    language: str | None = None


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


def _video_id_from_payload(payload: TranscriptRequest) -> str:
    if payload.source_id.startswith("yt:"):
        video_id = payload.source_id.split(":", 1)[1].strip()
        if video_id:
            return video_id

    if payload.url:
        parsed = urlparse(payload.url)
        if parsed.netloc.endswith("youtu.be"):
            return parsed.path.lstrip("/").strip()
        return (parse_qs(parsed.query).get("v") or [""])[0].strip()

    if payload.source_id:
        return payload.source_id.strip()

    return ""


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
    return {
        "ok": True,
        "proxy": proxy_manager.safe_diagnostics(),
        "health": proxy_manager.health_check(proxy_session_key="proxy-health"),
    }


@app.post("/transcript", response_model=TranscriptResponse)
def transcript(payload: TranscriptRequest, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> TranscriptResponse:
    _require_api_key(x_api_key)

    video_id = _video_id_from_payload(payload)
    if not video_id:
        raise HTTPException(status_code=400, detail="Unable to determine YouTube video_id from source_id/url")

    source_url = payload.url or f"https://www.youtube.com/watch?v={video_id}"
    source = {"url": source_url, "source_id": payload.source_id}

    proxy_manager = ProxyManager()
    proxy_health = proxy_manager.health_check(proxy_session_key=video_id) if proxy_manager.is_enabled() else {}
    diagnostics: dict[str, Any] = {
        "video_id": video_id,
        "transcript_available": False,
        "yta_error": None,
        "timedtext_error": None,
        "used_proxy": proxy_manager.is_enabled(),
        "exit_ip": proxy_health.get("exit_ip"),
        "audio_fallback_attempted": False,
        "audio_download_error": None,
    }

    try:
        result = get_transcript_for_source(
            source,
            allow_audio_fallback=payload.allow_audio_fallback,
            openai_api_key_present=bool(os.getenv("OPENAI_API_KEY")),
        )
        method = result.get("method") or "timedtext"
        diagnostics.update(result.get("diagnostics") or {})
        diagnostics["video_id"] = video_id
        diagnostics["transcript_available"] = bool(result.get("full_text", "").strip())
        diagnostics["used_proxy"] = proxy_manager.is_enabled()
        diagnostics["exit_ip"] = diagnostics.get("exit_ip") or proxy_health.get("exit_ip")
        diagnostics["audio_fallback_attempted"] = method == "audio_fallback" or payload.allow_audio_fallback

        return TranscriptResponse(
            method=method,
            text=result.get("full_text", "").strip(),
            segments=result.get("segments") or [],
            diagnostics=diagnostics,
        )
    except TranscriptionError as exc:
        exc_diag = exc.diagnostics or {}
        diagnostics.update(exc_diag)
        diagnostics["video_id"] = video_id
        diagnostics["transcript_available"] = False
        diagnostics["used_proxy"] = proxy_manager.is_enabled()
        diagnostics["exit_ip"] = diagnostics.get("exit_ip") or proxy_health.get("exit_ip")
        diagnostics["audio_fallback_attempted"] = payload.allow_audio_fallback
        if payload.allow_audio_fallback and exc.code == "audio_fallback_requires_openai_key":
            diagnostics["audio_download_error"] = "audio_fallback_requires_transcriber"
            raise HTTPException(
                status_code=424,
                detail={
                    "error": "audio_fallback_requires_transcriber",
                    "message": "audio_fallback_requires_transcriber",
                    "diagnostics": diagnostics,
                },
            ) from exc

        if exc.code.startswith("audio_fallback"):
            diagnostics["audio_download_error"] = str(exc)

        if not diagnostics.get("yta_error"):
            diagnostics["yta_error"] = str(exc)
        if not diagnostics.get("timedtext_error"):
            diagnostics["timedtext_error"] = exc_diag.get("timedtext_fetch_status")

        raise HTTPException(
            status_code=424,
            detail={
                "error": exc.code,
                "message": str(exc),
                "diagnostics": diagnostics,
            },
        ) from exc
