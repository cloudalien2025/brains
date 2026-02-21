from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from Brains_Ingestion_App.adapters.transcription import get_transcript_for_source
from Brains_Ingestion_App.brains.net.proxy_manager import ProxyManager


class ProxyOverrides(BaseModel):
    enabled: bool | None = None
    country: str | None = None
    sticky_mode: str | None = None


class TranscriptRequest(BaseModel):
    video_id: str
    prefer_lang: list[str] = Field(default_factory=lambda: ["en"])
    allow_audio_fallback: bool = False
    proxy: ProxyOverrides | None = None


app = FastAPI(title="Brains Worker")


def _require_api_key(header_key: str | None) -> None:
    expected = os.getenv("BRAINS_WORKER_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=500, detail="BRAINS_WORKER_API_KEY is not configured")
    if header_key != expected:
        raise HTTPException(status_code=401, detail="Invalid worker API key")


@app.get("/health")
def health() -> dict[str, Any]:
    proxy_manager = ProxyManager()
    return {
        "ok": True,
        "proxy": proxy_manager.safe_diagnostics(),
    }


@app.post("/transcript")
def transcript(payload: TranscriptRequest, x_brains_worker_key: str | None = Header(default=None)) -> dict[str, Any]:
    _require_api_key(x_brains_worker_key)

    if payload.proxy:
        if payload.proxy.enabled is not None:
            os.environ["DECODO_ENABLED"] = "true" if payload.proxy.enabled else "false"
        if payload.proxy.country is not None:
            os.environ["DECODO_COUNTRY"] = payload.proxy.country
        if payload.proxy.sticky_mode is not None:
            os.environ["DECODO_STICKY_MODE"] = payload.proxy.sticky_mode

    source = {"url": f"https://www.youtube.com/watch?v={payload.video_id}", "source_id": payload.video_id}
    openai_key_present = bool(os.getenv("OPENAI_API_KEY"))
    result = get_transcript_for_source(
        source,
        allow_audio_fallback=payload.allow_audio_fallback,
        openai_api_key_present=openai_key_present,
    )

    return {
        "video_id": payload.video_id,
        "method": result.get("method"),
        "language": result.get("language"),
        "text": result.get("full_text", ""),
        "segments": result.get("segments", []),
        "diagnostics": result.get("diagnostics", {}),
    }
