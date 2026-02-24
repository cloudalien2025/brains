from __future__ import annotations

import asyncio
import random
import html
import csv
import http.cookiejar
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import threading
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, parse_qsl, quote, urlencode, urlparse, urlsplit, urlunsplit
import xml.etree.ElementTree as ET

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator
import requests

from apps.brains_worker.ingest_multisource import run_ingest_multisource
from apps.brains_worker.ingest_types import RunContext
from apps.brains_worker.brain_stats import compute_brain_stats
from apps.brains_worker.webdocs_discovery import is_serpapi_configured
from apps.brains_worker.status import RunStatus, StatusWriter
from apps.brains_worker.transcribe import transcribe_audio, TranscriptionError, TranscriptionTimeout
from apps.brains_worker.yt_audio import download_audio, AudioDownloadError
from apps.brains_worker.db import db_session, get_engine
from apps.brains_worker.models import Artifact, Brain, BrainSource, IngestionAttempt, Source, TranscriptIndex
from apps.brains_worker.storage import upload_transcript, build_s3_uri

app = FastAPI(title="Brains Worker v1")
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

WORKER_API_KEY = (os.getenv("WORKER_API_KEY") or "").strip()
REPO_ROOT = Path(__file__).resolve().parents[2]
BRAINS_DATA_DIR = Path(os.getenv("BRAINS_DATA_DIR", "/opt/brains-data"))
BRAINS_ROOT = BRAINS_DATA_DIR / "brains"
GLOBAL_TRANSCRIPT_CACHE = BRAINS_DATA_DIR / "global_cache" / "transcripts"
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "3"))
MAX_CONCURRENT_STT = int(os.getenv("MAX_CONCURRENT_STT", "1"))
MAX_CONCURRENT_SYNTHESIS = int(os.getenv("MAX_CONCURRENT_SYNTHESIS", "1"))
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "600"))
OVERLAP_SECONDS = int(os.getenv("OVERLAP_SECONDS", "15"))
ARCHIVE_AUDIO = (os.getenv("ARCHIVE_AUDIO", "false").lower() in {"1", "true", "yes", "on"})
AUDIO_FALLBACK_ENABLED = os.getenv("AUDIO_FALLBACK_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
BRAINS_AUDIO_TRANSCRIBE_PRIMARY = os.getenv("BRAINS_AUDIO_TRANSCRIBE_PRIMARY", "1").lower() in {"1", "true", "yes", "on"}
DEFAULT_COOKIES_DIR = Path(os.getenv("BRAINS_COOKIES_DIR", REPO_ROOT / "cookies"))
_cookies_paths_env = [p.strip() for p in os.getenv("BRAINS_COOKIES_PATHS", "").split(",") if p.strip()]
COOKIES_PATHS = [Path(p) for p in _cookies_paths_env] if _cookies_paths_env else [
    DEFAULT_COOKIES_DIR / "youtube_cookies.txt",
    DEFAULT_COOKIES_DIR / "youtube_storage_state.json",
]
CAPTION_PROBE_CONCURRENCY = max(1, int(os.getenv("CAPTION_PROBE_CONCURRENCY", "2")))
CAPTION_PROBE_DELAY_MIN_MS = int(os.getenv("CAPTION_PROBE_DELAY_MIN_MS", "350"))
CAPTION_PROBE_DELAY_MAX_MS = int(os.getenv("CAPTION_PROBE_DELAY_MAX_MS", "800"))
CAPTION_PROBE_429_COOLDOWN_AFTER = int(os.getenv("CAPTION_PROBE_429_COOLDOWN_AFTER", "5"))
CAPTION_PROBE_429_COOLDOWN_SECONDS = int(os.getenv("CAPTION_PROBE_429_COOLDOWN_SECONDS", "45"))
CAPTION_PROBE_CACHE_DAYS = int(os.getenv("CAPTION_PROBE_CACHE_DAYS", "14"))
CAPTION_PROBE_CACHE_BLOCKED_DAYS = int(os.getenv("CAPTION_PROBE_CACHE_BLOCKED_DAYS", "1"))
CAPTION_PROBE_CACHE_NO_CAPTIONS_DAYS = int(os.getenv("CAPTION_PROBE_CACHE_NO_CAPTIONS_DAYS", "3"))

YOUTUBE_MAX_CANDIDATES_DEFAULT = int(os.getenv("YOUTUBE_MAX_CANDIDATES_DEFAULT", "50"))
YOUTUBE_REQUESTED_NEW_DEFAULT = int(os.getenv("YOUTUBE_REQUESTED_NEW_DEFAULT", "1"))
AUDIO_DL_TIMEOUT_S_DEFAULT = int(os.getenv("AUDIO_DL_TIMEOUT_S_DEFAULT", "180"))
TRANSCRIBE_TIMEOUT_S_DEFAULT = int(os.getenv("TRANSCRIBE_TIMEOUT_S_DEFAULT", "600"))
RUN_MAX_WALL_S_DEFAULT = int(os.getenv("RUN_MAX_WALL_S_DEFAULT", "1800"))
TRANSCRIBE_MODEL_DEFAULT = os.getenv("TRANSCRIBE_MODEL_DEFAULT", "small")
TRANSCRIBE_LANGUAGE_DEFAULT = os.getenv("TRANSCRIBE_LANGUAGE_DEFAULT", "auto")

WEBDOC_DISCOVERY_PRIMARY_DEFAULT = os.getenv("WEBDOC_DISCOVERY_PRIMARY_DEFAULT", "ddg")
WEBDOC_DISCOVERY_SECONDARY_DEFAULT = os.getenv("WEBDOC_DISCOVERY_SECONDARY_DEFAULT", "serpapi")
WEBDOC_MAX_CANDIDATES_DEFAULT = int(os.getenv("WEBDOC_MAX_CANDIDATES_DEFAULT", "50"))
WEBDOC_REQUESTED_NEW_DEFAULT = int(os.getenv("WEBDOC_REQUESTED_NEW_DEFAULT", "3"))
WEBDOC_MIN_RESULTS_DEFAULT = int(os.getenv("WEBDOC_MIN_RESULTS_DEFAULT", "8"))
WEBDOC_MIN_PDF_RESULTS_DEFAULT = int(os.getenv("WEBDOC_MIN_PDF_RESULTS_DEFAULT", "3"))
WEBDOC_ALLOW_PDF_DEFAULT = os.getenv("WEBDOC_ALLOW_PDF_DEFAULT", "true").lower() in {"1", "true", "yes", "on"}
WEBDOC_ALLOW_HTML_DEFAULT = os.getenv("WEBDOC_ALLOW_HTML_DEFAULT", "true").lower() in {"1", "true", "yes", "on"}
WEBDOC_MAX_FILE_MB_DEFAULT = int(os.getenv("WEBDOC_MAX_FILE_MB_DEFAULT", "50"))
WEBDOC_HTTP_TIMEOUT_S_DEFAULT = int(os.getenv("WEBDOC_HTTP_TIMEOUT_S_DEFAULT", "20"))
WEBDOC_THROTTLE_MS_DEFAULT = int(os.getenv("WEBDOC_THROTTLE_MS_DEFAULT", "750"))
WEBDOC_OCR_ENABLED_DEFAULT = os.getenv("WEBDOC_OCR_ENABLED_DEFAULT", "false").lower() in {"1", "true", "yes", "on"}
DOC_EXTRACT_TIMEOUT_S_DEFAULT = int(os.getenv("DOC_EXTRACT_TIMEOUT_S_DEFAULT", "60"))



RUN_QUEUE: asyncio.Queue[dict[str, Any]] | None = None
RUN_TASK: asyncio.Task | None = None
RUN_LOCK = asyncio.Lock()
RUN_INDEX_LOCK = threading.Lock()
DOWNLOAD_SEM = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
STT_SEM = asyncio.Semaphore(MAX_CONCURRENT_STT)
SYNTH_SEM = asyncio.Semaphore(MAX_CONCURRENT_SYNTHESIS)
CAPTION_PROBE_SEM = asyncio.Semaphore(CAPTION_PROBE_CONCURRENCY)

HTTPStatusValue = int | str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _webdoc_primary_provider() -> str:
    return os.getenv("WEBDOC_DISCOVERY_PRIMARY_DEFAULT", "ddg")


def _webdoc_secondary_provider() -> str:
    return os.getenv("WEBDOC_DISCOVERY_SECONDARY_DEFAULT", "serpapi")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
    return slug or "brain"


def require_api_key(x_api_key: str | None) -> None:
    if not WORKER_API_KEY or x_api_key != WORKER_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-Api-Key")


def ensure_dirs(brain_id: str) -> Path:
    root = BRAINS_ROOT / brain_id
    for folder in ["sources", "transcripts", "extractions", "synthesis", "diagnostics", "runs", "packs", "tmp"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    (BRAINS_ROOT).mkdir(parents=True, exist_ok=True)
    GLOBAL_TRANSCRIPT_CACHE.mkdir(parents=True, exist_ok=True)
    return root


def _db_enabled() -> bool:
    return get_engine() is not None


def _utcnow():
    return datetime.now(timezone.utc)


def _ensure_brain_row(session, brain_id: str, brain_type: str | None, keyword: str | None) -> Brain | None:
    if session is None:
        return None
    row = session.query(Brain).filter(Brain.brain_id == brain_id).one_or_none()
    now = _utcnow()
    if row:
        row.brain_type = brain_type or row.brain_type
        row.keyword = keyword or row.keyword
        row.updated_at = now
        return row
    row = Brain(brain_id=brain_id, brain_type=brain_type, keyword=keyword, created_at=now, updated_at=now)
    session.add(row)
    session.flush()
    return row


def _ensure_source_row(session, source_id: str, provider: str, url: str | None, title: str | None, channel: str | None, published_at: str | None) -> Source | None:
    if session is None:
        return None
    row = session.query(Source).filter(Source.source_id == source_id).one_or_none()
    now = _utcnow()
    if row:
        row.url = url or row.url
        row.title = title or row.title
        row.channel = channel or row.channel
        return row
    pub_dt = None
    if published_at:
        try:
            pub_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        except Exception:
            pub_dt = None
    row = Source(
        source_id=source_id,
        provider=provider,
        url=url,
        title=title,
        channel=channel,
        published_at=pub_dt,
        created_at=now,
    )
    session.add(row)
    session.flush()
    return row


def _link_brain_source(session, brain: Brain | None, source: Source | None) -> None:
    if session is None or not brain or not source:
        return
    existing = (
        session.query(BrainSource)
        .filter(BrainSource.brain_id == brain.id, BrainSource.source_id == source.id)
        .one_or_none()
    )
    if existing:
        return
    session.add(BrainSource(brain_id=brain.id, source_id=source.id, created_at=_utcnow()))


def _get_transcript_index(session, source: Source | None) -> TranscriptIndex | None:
    if session is None or not source:
        return None
    return session.query(TranscriptIndex).filter(TranscriptIndex.source_id == source.id).one_or_none()


def _repair_transcript_index(session, source: Source | None) -> TranscriptIndex | None:
    if session is None or not source:
        return None
    artifact = (
        session.query(Artifact)
        .filter(Artifact.source_id == source.id, Artifact.artifact_type == "transcript")
        .order_by(Artifact.id.desc())
        .first()
    )
    if not artifact:
        return None
    idx = _get_transcript_index(session, source)
    now = _utcnow()
    if idx:
        idx.transcript_status = "succeeded"
        idx.best_transcript_artifact_id = artifact.id
        idx.transcript_sha256 = artifact.sha256
        idx.updated_at = now
        idx.needs_repair = False
        return idx
    idx = TranscriptIndex(
        source_id=source.id,
        transcript_status="succeeded",
        best_transcript_artifact_id=artifact.id,
        provider=None,
        transcript_sha256=artifact.sha256,
        updated_at=now,
        needs_repair=False,
    )
    session.add(idx)
    session.flush()
    return idx


def _record_ingestion_attempt(session, run_id: str, source: Source | None, status: str, error_code: str | None = None, error_message: str | None = None) -> None:
    if session is None or not source:
        return
    session.add(
        IngestionAttempt(
            run_id=run_id,
            source_id=source.id,
            status=status,
            error_code=error_code,
            error_message=error_message,
            created_at=_utcnow(),
            completed_at=_utcnow() if status in {"succeeded", "failed", "skipped"} else None,
        )
    )


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def run_dir(root: Path, run_id: str) -> Path:
    path = root / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_run_dirs(root: Path, run_id: str) -> Path:
    path = run_dir(root, run_id)
    for folder in ["audio", "transcripts", "artifacts", "docs/pdf", "docs/html", "docs/text", "docs/meta"]:
        (path / folder).mkdir(parents=True, exist_ok=True)
    return path


def _list_files(root: Path, base: Path, pattern: str = "*") -> list[dict[str, Any]]:
    if not root.exists():
        return []
    files: list[dict[str, Any]] = []
    for path in root.glob(pattern):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        name = str(path.relative_to(base))
        files.append({"name": name, "bytes": size})
    return files


def run_index_path() -> Path:
    return BRAINS_ROOT / "run_index.json"


def register_run(run_id: str, brain_id: str) -> None:
    with RUN_INDEX_LOCK:
        index = load_json(run_index_path(), {})
        index[run_id] = {"brain_id": brain_id, "updated_at": utc_now()}
        write_json(run_index_path(), index)


def resolve_run_root(run_id: str) -> Path | None:
    index = load_json(run_index_path(), {})
    brain_id = (index.get(run_id) or {}).get("brain_id")
    if brain_id:
        root = BRAINS_ROOT / brain_id
        if (run_dir(root, run_id) / "run.json").exists():
            return root
    # Legacy fallback for historical runs.
    run_matches = list(BRAINS_ROOT.glob(f"*/runs/{run_id}/run.json"))
    if run_matches:
        return run_matches[0].parents[2]
    return None


def write_run(root: Path, run_id: str, run_payload: dict[str, Any]) -> None:
    write_json(run_dir(root, run_id) / "run.json", run_payload)


def load_run(root: Path, run_id: str) -> dict[str, Any]:
    current = run_dir(root, run_id) / "run.json"
    if current.exists():
        return load_json(current, {})
    legacy = root / "runs" / f"{run_id}.json"
    return load_json(legacy, {})


def status_from_run(run_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_payload.get("run_id"),
        "brain_slug": run_payload.get("brain_id"),
        "status": run_payload.get("status", "queued"),
        "step": run_payload.get("stage", "queued"),
        "candidates_found": int(run_payload.get("candidates_found") or 0),
        "requested_new": int(run_payload.get("requested_new") or 0),
        "selected_new": int(run_payload.get("selected_new") or 0),
        "eligible_count": int(run_payload.get("eligible_count") or 0),
        "eligible_shortfall": int(run_payload.get("eligible_shortfall") or 0),
        "completed": int(run_payload.get("completed") or 0),
        "failed": int(run_payload.get("failed") or 0),
        "discovery_method": (run_payload.get("discovery") or {}).get("method"),
        "youtube_api_http_status": (run_payload.get("discovery") or {}).get("youtube_api_http_status"),
        "audio_fallback_used": bool(run_payload.get("audio_fallback_used", False)),
        "error_code": run_payload.get("error_code"),
        "error_message": run_payload.get("error_message"),
        "updated_at": utc_now(),
    }


def write_status(root: Path, run_id: str, run_payload: dict[str, Any]) -> None:
    write_json(run_dir(root, run_id) / "status.json", status_from_run(run_payload))


def write_report(root: Path, run_id: str, run_payload: dict[str, Any]) -> None:
    artifacts = []
    for vid in run_payload.get("ingested_video_ids", []):
        artifacts.append({
            "video_id": vid,
            "source": f"sources/yt_{vid}.json",
            "transcript": f"transcripts/yt_{vid}.txt",
            "extraction": f"extractions/yt_{vid}.v1.json",
            "diagnostics": f"diagnostics/yt_{vid}.json",
        })
    attempts = load_transcript_attempts(root, run_id)
    transcript_summary = selected_transcript_summary(attempts)
    transcript_attempts_selected = transcript_summary["selected_attempts"]
    caption_probe_attempts = [x for x in attempts if x.get("phase") == "caption_probe"]
    blocked_probe = sum(1 for x in caption_probe_attempts if (x.get("probe_status") or "").startswith("blocked"))
    no_caption_probe = sum(1 for x in caption_probe_attempts if x.get("probe_status") == "no_captions")
    audio_attempted = sum(1 for x in attempts if x.get("phase") == "audio_fallback")
    audio_success = sum(1 for x in attempts if x.get("phase") == "audio_fallback" and x.get("success"))
    audio_failures = [x for x in attempts if x.get("phase") == "audio_fallback" and not x.get("success")]
    audio_failure_reasons = dict(Counter((x.get("error_code") or "audio_download_failed") for x in audio_failures).most_common(10))
    report_payload = {
        "run_id": run_id,
        "brain_id": run_payload.get("brain_id"),
        "status": run_payload.get("status"),
        "discovery_method": (run_payload.get("discovery") or {}).get("method"),
        "youtube_api_http_status": (run_payload.get("discovery") or {}).get("youtube_api_http_status"),
        "candidates_found": int(run_payload.get("candidates_found") or 0),
        "requested_new": int(run_payload.get("requested_new") or 0),
        "selected_new": int(run_payload.get("selected_new") or 0),
        "eligible_count": int(run_payload.get("eligible_count") or 0),
        "eligible_shortfall": int(run_payload.get("eligible_shortfall") or 0),
        "ingested_new": len(run_payload.get("ingested_video_ids", [])),
        "caption_probe_attempted": len(caption_probe_attempts),
        "caption_probe_blocked": blocked_probe,
        "caption_probe_no_captions": no_caption_probe,
        "transcripts_attempted_selected": transcript_summary["transcripts_attempted_selected"],
        "transcripts_succeeded": transcript_summary["transcripts_succeeded"],
        "transcripts_failed": transcript_summary["transcripts_failed"],
        "transcripts_failed_selected": transcript_summary["transcripts_failed"],
        "transcript_failure_reasons": summarize_transcript_failures(transcript_attempts_selected),
        "sample_failures": sample_failures(transcript_attempts_selected),
        "audio_attempted": audio_attempted,
        "audio_success": audio_success,
        "total_audio_minutes": run_payload.get("total_audio_minutes"),
        "audio_fallback_unavailable": run_payload.get("audio_fallback_unavailable", False),
        "audio_failure_reasons": audio_failure_reasons,
        "sample_audio_failures": [{"video_id": x.get("video_id"), "error_code": x.get("error_code"), "stderr_tail": x.get("stderr_tail")} for x in audio_failures[:3]],
        "proxy_enabled": any(bool(x.get("proxy_enabled")) for x in attempts),
        "brain_pack_id": (run_payload.get("brain_pack") or {}).get("brain_pack_id"),
        "transcript_attempts_jsonl": str(run_dir(root, run_id) / "transcript_attempts.jsonl"),
        "artifacts": artifacts,
        "brain_root": str(root),
        "updated_at": utc_now(),
    }
    write_json(run_dir(root, run_id) / "report.json", report_payload)


def diagnostics_summary_path(root: Path, run_id: str) -> Path:
    return run_dir(root, run_id) / "diagnostics_summary.json"


def init_diagnostics_summary(root: Path, run_id: str) -> None:
    write_json(
        diagnostics_summary_path(root, run_id),
        {
            "run_id": run_id,
            "_selected_video_ids": [],
            "counts": {
                "probe_blocked": 0,
                "probe_no_captions": 0,
                "transcripts_success": 0,
                "transcripts_failed": 0,
                "audio_attempted": 0,
                "audio_success": 0,
            },
            "transcripts_attempted_selected": 0,
            "transcript_failure_reasons": {},
            "sample_failures": [],
            "updated_at": utc_now(),
        },
    )


def update_diagnostics_summary(transcript_attempts_path: Path, payload: dict[str, Any]) -> None:
    run_root = transcript_attempts_path.parent
    summary_path = run_root / "diagnostics_summary.json"
    summary = load_json(summary_path, None)
    if not summary:
        summary = {
            "run_id": run_root.name,
            "_selected_video_ids": [],
            "counts": {
                "probe_blocked": 0,
                "probe_no_captions": 0,
                "transcripts_success": 0,
                "transcripts_failed": 0,
                "audio_attempted": 0,
                "audio_success": 0,
            },
            "transcripts_attempted_selected": 0,
            "transcript_failure_reasons": {},
            "sample_failures": [],
        }
    counts = summary["counts"]
    phase = payload.get("phase")
    if phase == "caption_probe":
        probe_status = payload.get("probe_status")
        if isinstance(probe_status, str) and probe_status.startswith("blocked"):
            counts["probe_blocked"] = int(counts.get("probe_blocked", 0)) + 1
        if probe_status == "no_captions":
            counts["probe_no_captions"] = int(counts.get("probe_no_captions", 0)) + 1
    if phase == "audio_fallback":
        counts["audio_attempted"] = int(counts.get("audio_attempted", 0)) + 1
        if payload.get("success"):
            counts["audio_success"] = int(counts.get("audio_success", 0)) + 1

    if phase in {"transcript", "fallback", "transcription"} and payload.get("selected_for_ingest"):
        attempts_by_video = int(summary.get("transcripts_attempted_selected", 0))
        seen = set(summary.get("_selected_video_ids") or [])
        video_id = str(payload.get("video_id") or "")
        if video_id and video_id not in seen:
            seen.add(video_id)
            summary["_selected_video_ids"] = sorted(seen)
            summary["transcripts_attempted_selected"] = attempts_by_video + 1
        if payload.get("success"):
            counts["transcripts_success"] = int(counts.get("transcripts_success", 0)) + 1
        else:
            counts["transcripts_failed"] = int(counts.get("transcripts_failed", 0)) + 1
            reasons = dict(summary.get("transcript_failure_reasons") or {})
            code = payload.get("error_code") or "unknown"
            reasons[code] = int(reasons.get(code, 0)) + 1
            summary["transcript_failure_reasons"] = reasons
            samples = list(summary.get("sample_failures") or [])
            if len(samples) < 5:
                samples.append(
                    {
                        "video_id": payload.get("video_id"),
                        "error_code": payload.get("error_code"),
                        "error_message": payload.get("error_message"),
                    }
                )
                summary["sample_failures"] = samples
    summary["updated_at"] = utc_now()
    write_json(summary_path, summary)


def read_ledger(root: Path) -> dict[str, Any]:
    return load_json(root / "ledger.json", {"ingested_video_ids": [], "ingested_doc_ids": [], "records": []})


def update_ledger(root: Path, record: dict[str, Any]) -> None:
    ledger = read_ledger(root)
    vid = record["video_id"]
    if vid not in ledger["ingested_video_ids"]:
        ledger["ingested_video_ids"].append(vid)
    ledger["records"].append(record)
    write_json(root / "ledger.json", ledger)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    if path.name == "transcript_attempts.jsonl":
        update_diagnostics_summary(path, payload)


def load_transcript_attempts(root: Path, run_id: str) -> list[dict[str, Any]]:
    path = run_dir(root, run_id) / "transcript_attempts.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def selected_transcript_summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    selected_attempts = [x for x in attempts if x.get("phase") in {"transcript", "fallback"} and x.get("selected_for_ingest")]
    by_video: dict[str, list[dict[str, Any]]] = {}
    for row in selected_attempts:
        by_video.setdefault(str(row.get("video_id") or ""), []).append(row)
    succeeded_videos = {vid for vid, rows in by_video.items() if any(r.get("success") for r in rows)}
    failed_videos = set(by_video.keys()) - succeeded_videos
    return {
        "selected_attempts": selected_attempts,
        "transcripts_attempted_selected": len(by_video),
        "transcripts_succeeded": len(succeeded_videos),
        "transcripts_failed": len(failed_videos),
    }


def summarize_transcript_failures(attempts: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter((x.get("error_code") or "unknown") for x in attempts if not x.get("success"))
    return dict(counts.most_common(10))


def sample_failures(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in attempts:
        if row.get("success"):
            continue
        out.append({"video_id": row.get("video_id"), "method": row.get("method"), "http_status": row.get("http_status"), "error_code": row.get("error_code"), "error_message": row.get("error_message")})
        if len(out) >= 3:
            break
    return out


def _redact_proxy_url(proxy_url: str | None) -> str | None:
    if not proxy_url:
        return None
    return re.sub(r"//([^:@/]+):([^@/]+)@", r"//***:***@", proxy_url)


def _build_proxy() -> tuple[bool, str | None, str | None, dict[str, str] | None, str | None]:
    proxy_enabled_env = os.getenv("BRAINS_PROXY_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
    proxy_url = (
        os.getenv("BRAINS_PROXY_URL")
        or os.getenv("DECODO_PROXY_URL")
        or os.getenv("RESIDENTIAL_PROXY_URL")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("HTTP_PROXY")
        or ""
    ).strip()
    if not proxy_url:
        dec_host = (os.getenv("DECODO_HOST") or "").strip()
        dec_port = (os.getenv("DECODO_PORT") or "").strip()
        dec_user = (os.getenv("DECODO_USERNAME") or "").strip()
        dec_pass = (os.getenv("DECODO_PASSWORD") or "").strip()
        if dec_host and dec_port and dec_user and dec_pass:
            proxy_url = f"http://{quote(dec_user, safe='')}:{quote(dec_pass, safe='')}@{dec_host}:{dec_port}"
    if not proxy_url:
        return False, None, None, None, "proxy_url_missing"
    if not proxy_enabled_env and os.getenv("BRAINS_PROXY_ENABLED") is not None:
        return False, None, None, None, "proxy_disabled_by_env"
    provider = "decodo" if "decodo" in proxy_url.lower() else "custom"
    return True, provider, _redact_proxy_url(proxy_url), {"http": proxy_url, "https": proxy_url}, None


def _cookie_source() -> Path | None:
    for path in COOKIES_PATHS:
        if path.exists():
            return path
    return None


def _build_http_session() -> tuple[requests.Session, dict[str, Any]]:
    session = requests.Session()
    proxy_enabled, provider, proxy_url_redacted, proxies, proxy_reason = _build_proxy()
    if proxies:
        session.proxies.update(proxies)
    if proxy_reason == "proxy_disabled_by_env":
        logger.info("Proxy URL configured but disabled by BRAINS_PROXY_ENABLED=0")
    source = _cookie_source()
    cookies_enabled = False
    cookies_error = None
    if source and source.suffix == ".txt":
        jar = http.cookiejar.MozillaCookieJar(str(source))
        try:
            jar.load(ignore_discard=True, ignore_expires=True)
            session.cookies.update(jar)
            cookies_enabled = True
        except Exception as exc:
            cookies_enabled = False
            cookies_error = f"cookie_load_failed_txt: {exc}"
            logger.warning("Failed to load cookies from %s: %s", source, exc)
    elif source and source.suffix == ".json":
        try:
            state = json.loads(source.read_text(encoding="utf-8"))
            for cookie in state.get("cookies", []):
                if cookie.get("name") and cookie.get("value"):
                    session.cookies.set(cookie["name"], cookie["value"], domain=cookie.get("domain"), path=cookie.get("path", "/"))
            cookies_enabled = bool(state.get("cookies"))
        except Exception as exc:
            cookies_enabled = False
            cookies_error = f"cookie_load_failed_json: {exc}"
            logger.warning("Failed to load cookies from %s: %s", source, exc)

    proxy_test_ip = None
    if proxy_enabled and os.getenv("BRAINS_PROXY_DIAGNOSTICS", "0").lower() in {"1", "true", "yes", "on"}:
        try:
            proxy_test_ip = session.get("https://ipinfo.io/ip", timeout=10).text.strip()
        except Exception:
            proxy_test_ip = None

    return session, {
        "proxy_enabled": proxy_enabled,
        "proxy_provider": provider,
        "proxy_url_redacted": proxy_url_redacted,
        "proxy_test_ip": proxy_test_ip,
        "proxy_reason": proxy_reason,
        "cookies_enabled": cookies_enabled,
        "cookies_source": str(source) if source else None,
        "cookies_error": cookies_error,
    }


def _caption_cache_path(root: Path, video_id: str) -> Path:
    return root / "cache" / "caption_probe" / f"{video_id}.json"


def _load_caption_probe_cache(root: Path, video_id: str) -> dict[str, Any] | None:
    path = _caption_cache_path(root, video_id)
    if not path.exists():
        return None
    try:
        cached = load_json(path, {})
        probed_at = datetime.fromisoformat(cached.get("probed_at", "").replace("Z", "+00:00"))
        age = datetime.now(timezone.utc) - probed_at
        probe_status = str(cached.get("probe_status") or "")
        if probe_status.startswith("blocked"):
            ttl_days = CAPTION_PROBE_CACHE_BLOCKED_DAYS
        elif probe_status == "no_captions":
            ttl_days = CAPTION_PROBE_CACHE_NO_CAPTIONS_DAYS
        else:
            ttl_days = CAPTION_PROBE_CACHE_DAYS
        if age.days <= ttl_days:
            return cached
    except Exception:
        return None
    return None


def _save_caption_probe_cache(root: Path, video_id: str, payload: dict[str, Any]) -> None:
    write_json(_caption_cache_path(root, video_id), payload)


def _cookies_json_to_netscape(source: Path, out_path: Path) -> Path | None:
    try:
        state = json.loads(source.read_text(encoding="utf-8"))
        cookies = state.get("cookies", [])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Netscape HTTP Cookie File"]
        for cookie in cookies:
            name = cookie.get("name")
            value = cookie.get("value")
            domain = cookie.get("domain") or ""
            path = cookie.get("path") or "/"
            if not name or value is None:
                continue
            include_subdomains = "TRUE" if domain.startswith(".") else "FALSE"
            secure = "TRUE" if cookie.get("secure") else "FALSE"
            expires = int(cookie.get("expires") or 0)
            if cookie.get("httpOnly"):
                domain = f"#HttpOnly_{domain}"
            lines.append("\t".join([domain, include_subdomains, path, secure, str(expires), name, value]))
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_path
    except Exception as exc:
        logger.warning("Failed to convert cookies %s to Netscape format: %s", source, exc)
        return None


def _cookie_source_for_ytdlp() -> Path | None:
    source = _cookie_source()
    if not source:
        return None
    if source.suffix == ".txt":
        return source
    if source.suffix == ".json":
        out_path = BRAINS_DATA_DIR / "tmp" / "cookies_ytdlp.txt"
        try:
            if not out_path.exists() or source.stat().st_mtime > out_path.stat().st_mtime:
                return _cookies_json_to_netscape(source, out_path)
            return out_path
        except Exception as exc:
            logger.warning("Failed to prepare yt-dlp cookies from %s: %s", source, exc)
            return None
    return None


def _proxy_url_raw() -> str | None:
    proxy_enabled, _provider, _redacted, proxies, _reason = _build_proxy()
    if not proxy_enabled or not proxies:
        return None
    return proxies.get("https") or proxies.get("http")


def _url_with_query(url: str, params: dict[str, Any] | None) -> str:
    if not params:
        return url
    parts = urlsplit(url)
    query_items = parse_qsl(parts.query, keep_blank_values=True)
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            for item in value:
                query_items.append((key, str(item)))
        else:
            query_items.append((key, str(value)))
    query = urlencode(query_items, doseq=True)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def _request_with_backoff(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    timeout: int = 20,
    retry_statuses: set[int] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[requests.Response | None, dict[str, Any]]:
    if retry_statuses is None:
        retry_statuses = {403, 429, 500, 502, 503, 504}
    retries: list[dict[str, Any]] = []
    url_with_query = _url_with_query(url, params)
    for attempt in range(5):
        try:
            resp = session.get(url_with_query, headers=headers, timeout=timeout)
            if resp.status_code not in retry_statuses:
                return resp, {"retry_count": attempt, "retries": retries}
            sleep_s = min(8, 2 ** attempt) + random.uniform(0.1, 0.9)
            retries.append({"attempt": attempt + 1, "http_status": resp.status_code, "sleep_seconds": round(sleep_s, 2)})
            time.sleep(sleep_s)
        except requests.RequestException as exc:
            sleep_s = min(8, 2 ** attempt) + random.uniform(0.1, 0.9)
            retries.append({"attempt": attempt + 1, "error": str(exc), "sleep_seconds": round(sleep_s, 2)})
            time.sleep(sleep_s)
    return None, {"retry_count": len(retries), "retries": retries, "error": "retries_exhausted"}


def _caption_tracks(session: requests.Session, video_id: str) -> tuple[list[dict[str, str]], int | None, str | None, int, dict[str, Any]]:
    resp, retry_diag = _request_with_backoff(
        session,
        "https://www.youtube.com/api/timedtext",
        {"type": "list", "v": video_id},
        timeout=20,
    )
    if resp is None:
        return [], None, "blocked_error", 0, retry_diag
    response_len = len(resp.text or "")
    if resp.status_code in {403, 429}:
        return [], resp.status_code, "blocked", response_len, retry_diag
    if resp.status_code != 200:
        return [], resp.status_code, "blocked_error", response_len, retry_diag
    if not (resp.text or "").strip():
        return [], resp.status_code, "blocked_empty", response_len, retry_diag
    root = ET.fromstring(resp.text or "<transcript_list/>")
    tracks = []
    for node in root.findall("track"):
        tracks.append({
            "lang_code": node.attrib.get("lang_code", ""),
            "name": node.attrib.get("name", ""),
            "kind": node.attrib.get("kind", ""),
        })
    if tracks:
        return tracks, resp.status_code, "captions_available", response_len, retry_diag
    return tracks, resp.status_code, "no_captions", response_len, retry_diag


def _yt_dlp_exists() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return shutil.which("yt-dlp") is not None


def _ffmpeg_exists() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return shutil.which("ffmpeg") is not None


def _ffprobe_exists() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return shutil.which("ffprobe") is not None


def _transcription_engine_available() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        return False
    return True


def _webdocs_engine_available() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    try:
        import bs4  # noqa: F401
        import readability  # noqa: F401
        import pypdf  # noqa: F401
    except Exception:
        return False
    return True


def _download_audio_with_diagnostics(video_url: str, out_path: Path) -> tuple[Path | None, dict[str, Any]]:
    started = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    expected_id = _extract_video_id(None, None, video_url)
    proxy_url = _proxy_url_raw()
    cookie_source = _cookie_source_for_ytdlp()
    retries: list[dict[str, Any]] = []
    proc = None
    resolved = out_path
    def _to_str(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="ignore")
            except Exception:
                return ""
        return str(value)
    for attempt in range(3):
        cmd = [
            "yt-dlp",
            "-f", "bestaudio/best",
            "--extract-audio",
            "--audio-format", "mp3",
            "--no-playlist",
            "--force-ipv4",
            "--socket-timeout", "15",
            "--retries", "3",
            "--fragment-retries", "3",
            "--newline",
            "-o", str(out_path),
            video_url,
        ]
        if proxy_url:
            cmd.extend(["--proxy", proxy_url])
        if cookie_source:
            cmd.extend(["--cookies", str(cookie_source)])
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired as exc:
            stderr_full = _to_str(exc.stderr)
            stdout_full = _to_str(exc.stdout)
            stderr_tail = stderr_full[-1000:]
            stdout_tail = stdout_full[-1000:]
            stderr_tail = re.sub(r"//([^:@/]+):([^@/]+)@", "//***:***@", stderr_tail)
            stdout_tail = re.sub(r"//([^:@/]+):([^@/]+)@", "//***:***@", stdout_tail)
            retries.append({
                "attempt": attempt + 1,
                "exit_code": "timeout",
                "stderr_tail": stderr_tail,
                "stdout_tail": stdout_tail,
                "stderr": re.sub(r"//([^:@/]+):([^@/]+)@", "//***:***@", stderr_full),
                "stdout": re.sub(r"//([^:@/]+):([^@/]+)@", "//***:***@", stdout_full),
            })
            proc = None
        if "%(" in out_path.name:
            matched: list[Path] = []
            if expected_id:
                matched = sorted(out_path.parent.glob(f"{expected_id}.*"))
                if not matched:
                    matched = sorted(out_path.parent.glob(f"{expected_id}*"))
            if not matched:
                matched = sorted([
                    p for p in out_path.parent.glob("*.*")
                    if p.suffix.lower().lstrip(".") in {"mp3", "m4a", "webm", "opus"}
                ])
            resolved = matched[0] if matched else out_path
        else:
            matched = sorted(out_path.parent.glob(f"{out_path.stem.split('.')[0]}*"))
            resolved = matched[0] if matched else out_path
        if proc and proc.returncode == 0 and resolved.exists():
            break
        stderr_full = _to_str(proc.stderr) if proc else ""
        stdout_full = _to_str(proc.stdout) if proc else ""
        stderr_tail = stderr_full[-1000:] if proc else ""
        stdout_tail = stdout_full[-1000:] if proc else ""
        stderr_tail = re.sub(r"//([^:@/]+):([^@/]+)@", "//***:***@", stderr_tail)
        stdout_tail = re.sub(r"//([^:@/]+):([^@/]+)@", "//***:***@", stdout_tail)
        retries.append({
            "attempt": attempt + 1,
            "exit_code": proc.returncode if proc else None,
            "stderr_tail": stderr_tail,
            "stdout_tail": stdout_tail,
            "stderr": re.sub(r"//([^:@/]+):([^@/]+)@", "//***:***@", stderr_full),
            "stdout": re.sub(r"//([^:@/]+):([^@/]+)@", "//***:***@", stdout_full),
        })
        if attempt < 2:
            time.sleep(min(8, 2 ** attempt) + random.uniform(0.1, 0.9))
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    stderr_full = _to_str(proc.stderr) if proc else ""
    stdout_full = _to_str(proc.stdout) if proc else ""
    stderr_tail = stderr_full[-1000:] if proc else ""
    stdout_tail = stdout_full[-1000:] if proc else ""
    stderr_tail = re.sub(r"//([^:@/]+):([^@/]+)@", "//***:***@", stderr_tail)
    stdout_tail = re.sub(r"//([^:@/]+):([^@/]+)@", "//***:***@", stdout_tail)
    diag = {
        "phase": "audio_fallback",
        "method": "yt-dlp",
        "command": "yt-dlp -f bestaudio/best --extract-audio --audio-format mp3 --no-playlist -o <output> <url>",
        "exit_code": proc.returncode if proc else None,
        "stderr": re.sub(r"//([^:@/]+):([^@/]+)@", "//***:***@", stderr_full),
        "stdout": re.sub(r"//([^:@/]+):([^@/]+)@", "//***:***@", stdout_full),
        "stderr_tail": stderr_tail,
        "stdout_tail": stdout_tail,
        "elapsed_ms": elapsed_ms,
        "audio_file_path": str(resolved),
        "bytes": resolved.stat().st_size if resolved.exists() else 0,
        "proxy_enabled": bool(proxy_url),
        "cookies_source": str(cookie_source) if cookie_source else None,
        "retry_count": len(retries),
        "retries": retries,
        "success": bool(proc and proc.returncode == 0 and resolved.exists()),
    }
    if proc is None:
        diag["stage_failed"] = "download_audio"
    return (resolved if diag["success"] else None), diag


def _fetch_timedtext_transcript(session: requests.Session, video_id: str, track: dict[str, str]) -> tuple[str, int | None, dict[str, Any]]:
    params = {"v": video_id, "lang": track.get("lang_code") or "en", "fmt": "srv3"}
    if track.get("kind"):
        params["kind"] = track["kind"]
    if track.get("name"):
        params["name"] = track["name"]
    resp, retry_diag = _request_with_backoff(session, "https://www.youtube.com/api/timedtext", params, timeout=20)
    if resp is None:
        raise RuntimeError("HTTP unavailable after retries")
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    root = ET.fromstring(resp.text or "<transcript/>")
    lines = [html.unescape((n.text or "").strip()) for n in root.findall("text")]
    transcript = "\n".join(x for x in lines if x)
    return transcript, resp.status_code, retry_diag


def _track_key(track: dict[str, str]) -> tuple[str, str, str]:
    return (track.get("lang_code") or "", track.get("name") or "", track.get("kind") or "")


def _track_sequence(primary: dict[str, str] | None, tracks: list[dict[str, str]], preferred_language: str) -> list[dict[str, str]]:
    ordered: list[dict[str, str]] = []
    if primary:
        ordered.append(primary)
    for track in tracks:
        if _track_key(track) not in {_track_key(t) for t in ordered}:
            ordered.append(track)
    preferred = [t for t in ordered if (t.get("lang_code") or "").startswith(preferred_language)]
    english = [t for t in ordered if (t.get("lang_code") or "").startswith("en")]
    others = [t for t in ordered if t not in preferred and t not in english]
    # Keep primary first if it was provided; then prefer language matches.
    if primary:
        rest = [t for t in (preferred + english + others) if _track_key(t) != _track_key(primary)]
        return [primary] + rest
    return preferred + english + others


def _expected_api_key() -> str:
    return (os.getenv("BRAINS_API_KEY") or os.getenv("BRAINS_WORKER_API_KEY") or "").strip()


def _auth_error() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"error_code": "WORKER_AUTH_FAILED", "error": "Missing or invalid x-api-key", "diagnostics": {}},
    )


def _error(status_code: int, code: str, message: str, diagnostics: dict[str, Any]) -> JSONResponse:
    diagnostics["final_error_code"] = code
    diagnostics["final_error"] = message
    return JSONResponse(status_code=status_code, content={"error_code": code, "error": message, "diagnostics": diagnostics})


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
        match = re.search(r"ytInitialPlayerResponse\\s*=\\s*(\\{.+?\\})\\s*;", text)
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
    text = re.sub(r"\\s+", " ", text)
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
        return re.sub(r"\\s+", " ", " ".join(lines)).strip()
    if ext in {"ttml", "xml"}:
        try:
            root = ET.fromstring(content)
            for node in root.findall(".//{*}p") + root.findall(".//text"):
                val = _strip_caption_markup("".join(node.itertext()))
                if val:
                    lines.append(val)
            return re.sub(r"\\s+", " ", " ".join(lines)).strip()
        except ET.ParseError:
            return ""
    return ""


def _parse_caption_json3(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except Exception:
        return ""
    lines: list[str] = []
    for event in payload.get("events", []):
        for seg in event.get("segs", []) or []:
            text = (seg.get("utf8") or "").strip()
            if text:
                lines.append(text)
    return re.sub(r"\\s+", " ", " ".join(lines)).strip()


def _download_text(url: str, proxy: str | None) -> tuple[HTTPStatusValue, str | None, str | None]:
    proxies = {"http": proxy, "https": proxy} if proxy else None
    try:
        response = requests.get(url, timeout=20, proxies=proxies, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return response.status_code, None, response.headers.get("Content-Type")
        return response.status_code, response.text, response.headers.get("Content-Type")
    except Exception:
        return "exception", None, None


def _run_ytdlp_subtitles(
    youtube_url: str,
    preferred_language: str | None,
    proxy: str | None,
    work_dir: Path,
    cookies_path: Path | None = None,
) -> tuple[str | None, dict[str, Any]]:
    start = time.perf_counter()
    diag = {
        "ytdlp_subs_status": "failed",
        "ytdlp_subs_error_code": None,
        "ytdlp_subs_elapsed_ms": 0,
        "ytdlp_subs_lang": None,
        "ytdlp_subs_format": None,
        "ytdlp_subs_file_bytes": 0,
        "ytdlp_subs_file_path": None,
        "ytdlp_subs_stderr_sniff": None,
        "ytdlp_used_cookies": bool(cookies_path),
    }
    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        diag["ytdlp_subs_error_code"] = "YTDLP_SUBS_FAILED"
        diag["ytdlp_subs_elapsed_ms"] = int((time.perf_counter() - start) * 1000)
        return None, diag
    langs = ",".join([p for p in [preferred_language, "en.*"] if p])
    output_template = str(Path("/tmp") / "%(id)s.%(ext)s")
    cmd = [
        yt_dlp,
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--ignore-no-formats-error",
        "--sub-langs",
        langs or "en.*",
        "--sub-format",
        "vtt",
        "-o",
        output_template,
        youtube_url,
    ]
    if proxy:
        cmd.extend(["--proxy", proxy])
    if cookies_path:
        cmd.extend(["--cookies", str(cookies_path)])
    proc = None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except Exception:
        diag["ytdlp_subs_error_code"] = "YTDLP_SUBS_FAILED"
        diag["ytdlp_subs_elapsed_ms"] = int((time.perf_counter() - start) * 1000)
        return None, diag

    diag["ytdlp_subs_elapsed_ms"] = int((time.perf_counter() - start) * 1000)
    if proc and proc.stderr:
        diag["ytdlp_subs_stderr_sniff"] = (proc.stderr or "")[-200:]

    video_id = _extract_video_id(None, None, youtube_url)
    files = []
    if video_id:
        files = [
            p
            for p in Path("/tmp").glob(f"{video_id}.*")
            if p.suffix.lower().lstrip(".") in {"vtt", "srt", "ttml"}
        ]
    if not files:
        files = [p for p in work_dir.glob("*.*") if p.suffix.lower().lstrip(".") in {"vtt", "srt", "ttml"}]
    if not files:
        diag["ytdlp_subs_status"] = "no_subs"
        diag["ytdlp_subs_error_code"] = "NO_SUBTITLE_FILES"
        return None, diag

    pref = (preferred_language or "").lower()

    def score(path: Path) -> tuple[int, int, int]:
        name = path.name.lower()
        auto = 1 if ".live_chat." in name or ".asr." in name else 0
        lang_match = 0 if pref and f".{pref}" in name else 1
        english = 0 if ".en" in name else 1
        return auto, lang_match, english

    selected = sorted(files, key=lambda path: (score(path), -path.stat().st_mtime))[0]
    text = _parse_subtitle_to_text(
        selected.read_text(encoding="utf-8", errors="ignore"), selected.suffix.lower().lstrip(".")
    )
    if not text:
        diag["ytdlp_subs_error_code"] = "SUBTITLE_PARSE_FAILED"
        return None, diag

    diag["ytdlp_subs_status"] = "success"
    diag["ytdlp_subs_error_code"] = None
    diag["ytdlp_subs_file_bytes"] = selected.stat().st_size
    diag["ytdlp_subs_lang"] = next(
        (part for part in selected.stem.split(".") if len(part) in {2, 5} and part[:2].isalpha()), None
    )
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
    candidates = sorted(
        [p for p in work_dir.glob("*.*") if p.suffix.lower().lstrip(".") in {"mp3", "m4a", "webm", "opus"}]
    )
    if not candidates:
        diag["audio_download_error_code"] = "YTDLP_AUDIO_DOWNLOAD_FAILED"
        return None, diag
    audio_path = candidates[0]
    diag["audio_download_status"] = "success"
    diag["audio_file_ext"] = audio_path.suffix.lower().lstrip(".")
    diag["audio_file_bytes"] = audio_path.stat().st_size
    diag["audio_file_path"] = str(audio_path)
    return audio_path, diag


def _transcribe_openai(
    audio_path: Path, model: str, language: str | None, prompt: str | None
) -> tuple[str | None, dict[str, Any], int, str | None, str | None]:
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
    allowed_models = {"gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"}
    if model not in allowed_models:
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
        logger.error("openai stt failed: %s", (resp.text or "")[-300:])
        return None, diag, 503, "OPENAI_STT_FAILED", "OpenAI transcription failed"
    text = (resp.json().get("text") or "").strip()
    if not text:
        diag["stt_error_code"] = "OPENAI_STT_FAILED"
        return None, diag, 503, "OPENAI_STT_FAILED", "OpenAI transcription returned empty transcript"
    diag["stt_status"] = "success"
    diag["transcript_chars"] = len(text)
    return text, diag, 200, None, None


def _legacy_proxy_url(payload: TranscriptRequest) -> str | None:
    if payload.proxy_enabled is False:
        return None
    if payload.proxy_url:
        return payload.proxy_url
    return _proxy_url_raw()


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
        "caption_sniff_hint": None,
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


class BrainCreateRequest(BaseModel):
    name: str
    brain_type: str = Field(pattern="^(BD|UAP)$")
    description: str | None = None


class IngestRequest(BaseModel):
    keyword: str
    n_new_videos: int | None = Field(default=None, ge=1)
    selected_new: int | None = Field(default=None, ge=1)
    max_new: int | None = Field(default=None, ge=1)
    max_candidates: int = Field(default=YOUTUBE_MAX_CANDIDATES_DEFAULT, ge=1, le=50)
    youtube_requested_new: int | None = Field(default=None, ge=1)
    webdocs_requested_new: int | None = Field(default=None, ge=1)
    youtube_max_candidates: int | None = Field(default=None, ge=1, le=50)
    webdocs_max_candidates: int | None = Field(default=None, ge=1, le=50)
    discovery_order: str | None = Field(default=None, pattern="^(relevance|date)$")
    published_after: str | None = None
    mode: str = "audio_first"
    preferred_language: str = "en"
    longform: dict[str, int] = Field(default_factory=lambda: {"chunk_seconds": CHUNK_SECONDS, "overlap_seconds": OVERLAP_SECONDS})
    synthesis: dict[str, bool] = Field(default_factory=lambda: {"update": True})
    brain_pack: dict[str, bool] = Field(default_factory=lambda: {"build": True})
    force_refresh: bool = False


    @model_validator(mode="after")
    def validate_requested_new(self) -> "IngestRequest":
        requested = self.selected_new or self.max_new or self.n_new_videos
        if requested is None and self.youtube_requested_new is None and self.webdocs_requested_new is None:
            requested = YOUTUBE_REQUESTED_NEW_DEFAULT
        if requested is not None:
            self.n_new_videos = int(requested)
            self.selected_new = int(requested)
        return self


class TranscriptRequest(BaseModel):
    video_id: str | None = Field(default=None, examples=["yt:5lQf89-AeFo"])
    source_id: str | None = Field(default=None, examples=["yt:5lQf89-AeFo"])
    url: str | None = None
    preferred_language: str | None = None
    captions_enabled: bool = True
    yt_dlp_subs_enabled: bool = True
    audio_fallback_enabled: bool = False
    allow_audio_fallback: bool | None = None
    stt_model: str = "gpt-4o-mini-transcribe"
    stt_language: str | None = None
    stt_prompt: str | None = None
    debug_keep_files: bool = False
    debug_include_artifact_paths: bool = False
    proxy_enabled: bool | None = None
    proxy_url: str | None = None

def ffprobe_duration(path: Path) -> float | None:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)
        ], text=True).strip()
        return float(out)
    except Exception:
        return None


def chunk_ranges(duration: float, chunk_seconds: int, overlap_seconds: int) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    start = 0.0
    step = max(1, chunk_seconds - overlap_seconds)
    while start < duration:
        end = min(duration, start + chunk_seconds)
        ranges.append((start, end - start))
        start += step
    return ranges


def overlap_dedupe(prev: str, nxt: str) -> str:
    a = prev[-200:]
    b = nxt[:200]
    for n in range(min(len(a), len(b)), 20, -1):
        if a[-n:].strip() and a[-n:].strip() == b[:n].strip():
            return nxt[n:]
    return nxt


def transcribe_audio_chunks(video_id: str, audio_path: Path, chunk_seconds: int, overlap_seconds: int, preferred_language: str, tmp_dir: Path) -> tuple[str, dict[str, Any]]:
    raise RuntimeError("OpenAI transcription disabled; use local faster-whisper pipeline instead.")


def resolve_discovery_order(brain_type: str, requested_order: str | None) -> str:
    if requested_order in {"relevance", "date"}:
        return requested_order
    return "date" if brain_type == "UAP" else "relevance"


def build_extraction(brain_type: str, transcript: str, source_meta: dict[str, Any]) -> dict[str, Any]:
    lines = [ln.strip() for ln in transcript.splitlines() if ln.strip()]
    top = lines[:25]
    if brain_type == "BD":
        return {
            "extraction_version": "v1",
            "video_id": source_meta["video_id"],
            "seo_tips": [ln for ln in top if "seo" in ln.lower()][:10],
            "monetization_tactics": [ln for ln in top if any(k in ln.lower() for k in ["price", "revenue", "monet", "subscription"])][:10],
            "api_patterns": [ln for ln in top if "api" in ln.lower()][:10],
            "procedures": top[:8],
            "pitfalls_and_fixes": [ln for ln in lines if any(k in ln.lower() for k in ["mistake", "error", "fix", "issue"])][:10],
        }
    return {
        "extraction_version": "v1",
        "video_id": source_meta["video_id"],
        "timeline_candidates": [ln for ln in top if re.search(r"\b(19|20)\d{2}\b", ln)][:12],
        "key_claims": top[:12],
        "narrative_threads": top[12:24],
        "contradictions_disputes": [ln for ln in lines if any(k in ln.lower() for k in ["dispute", "contradict", "challenge"])][:10],
        "script_ready_beats": top[:15],
    }


def update_synthesis(root: Path, brain_type: str, extraction: dict[str, Any]) -> None:
    synthesis = root / "synthesis"
    if brain_type == "BD":
        mapping = {
            "bd_core_master.md": extraction.get("procedures", []),
            "seo_playbook.md": extraction.get("seo_tips", []),
            "monetization_playbook.md": extraction.get("monetization_tactics", []),
            "bd_api_contract.md": extraction.get("api_patterns", []),
            "faq_and_pitfalls.md": extraction.get("pitfalls_and_fixes", []),
        }
    else:
        mapping = {
            "timeline.md": extraction.get("timeline_candidates", []),
            "narratives.md": extraction.get("narrative_threads", []),
            "witnesses.md": extraction.get("key_claims", []),
            "disputed_points.md": extraction.get("contradictions_disputes", []),
            "script_beats.md": extraction.get("script_ready_beats", []),
        }
    for filename, items in mapping.items():
        path = synthesis / filename
        with path.open("a", encoding="utf-8") as f:
            f.write(f"\n\n## Update {utc_now()}\n")
            for line in items:
                f.write(f"- {line}\n")


def build_brain_pack(root: Path, brain_id: str, run_id: str, keyword: str, included_video_ids: list[str], brain_type: str) -> dict[str, Any]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pack_id = f"pack_{ts}_{brain_id}"
    pack_dir = root / "packs"
    pack_dir.mkdir(parents=True, exist_ok=True)
    zip_path = pack_dir / f"{pack_id}.zip"

    manifest = {
        "brain_id": brain_id,
        "brain_type": brain_type,
        "keyword": keyword,
        "run_id": run_id,
        "created_at": utc_now(),
        "included_video_ids": included_video_ids,
        "counts": {"videos": len(included_video_ids)},
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp) / f"BrainPack_{brain_id}_{ts}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        for name in ["brain.json", "ledger.json"]:
            src = root / name
            if src.exists():
                shutil.copy2(src, tmp_root / name)
        for folder in ["transcripts", "extractions", "synthesis", "diagnostics"]:
            src_dir = root / folder
            if src_dir.exists():
                shutil.copytree(src_dir, tmp_root / folder)
        with (tmp_root / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        ledger = read_ledger(root)
        with (tmp_root / "sources.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "title", "channel", "published_at", "ingested_at"])
            writer.writeheader()
            for rec in ledger.get("records", []):
                writer.writerow({k: rec.get(k, "") for k in writer.fieldnames})
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in tmp_root.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(tmp_root.parent))

    info = {"brain_pack_id": pack_id, "zip_path": str(zip_path), "manifest": manifest}
    write_json(pack_dir / f"{pack_id}.json", info)
    return info


async def process_run(job: dict[str, Any]) -> None:
    run_id = job["run_id"]
    brain_id = job["brain_id"]
    payload = IngestRequest(**job["payload"])
    root = ensure_dirs(brain_id)
    run = load_run(root, run_id)
    run["status"] = "processing"
    run["stage"] = "discovery"
    run["started_at"] = utc_now()
    run["payload"] = payload.model_dump()
    run["errors"] = []
    write_run(root, run_id, run)

    _proxy_enabled, _provider, _proxy_url_redacted, proxies, _reason = _build_proxy()
    proxy_url = _proxy_url_raw()
    cookies_path = str(_cookie_source_for_ytdlp()) if _cookie_source_for_ytdlp() else None

    youtube_requested_new = int(payload.youtube_requested_new or payload.selected_new or payload.n_new_videos or YOUTUBE_REQUESTED_NEW_DEFAULT)
    webdocs_requested_new = int(payload.webdocs_requested_new or WEBDOC_REQUESTED_NEW_DEFAULT)
    youtube_max_candidates = int(payload.youtube_max_candidates or payload.max_candidates or YOUTUBE_MAX_CANDIDATES_DEFAULT)
    webdocs_max_candidates = int(payload.webdocs_max_candidates or WEBDOC_MAX_CANDIDATES_DEFAULT)

    run_ctx = RunContext(
        run_id=run_id,
        brain_id=brain_id,
        keyword=payload.keyword,
        brain_root=root,
        run_dir=run_dir(root, run_id),
        payload=payload.model_dump(),
        proxies=proxies,
        proxy_url=proxy_url,
        cookies_path=cookies_path,
        config={
            "youtube_requested_new": youtube_requested_new,
            "webdoc_requested_new": webdocs_requested_new,
            "youtube_max_candidates": youtube_max_candidates,
            "webdoc_max_candidates": webdocs_max_candidates,
            "proxies": proxies,
            "audio_dl_timeout_s": AUDIO_DL_TIMEOUT_S_DEFAULT,
            "transcribe_timeout_s": TRANSCRIBE_TIMEOUT_S_DEFAULT,
            "transcribe_model": TRANSCRIBE_MODEL_DEFAULT,
            "transcribe_language": TRANSCRIBE_LANGUAGE_DEFAULT,
            "run_max_wall_s": RUN_MAX_WALL_S_DEFAULT,
            "webdoc_min_results": WEBDOC_MIN_RESULTS_DEFAULT,
            "webdoc_min_pdf_results": WEBDOC_MIN_PDF_RESULTS_DEFAULT,
            "webdoc_allow_pdf": WEBDOC_ALLOW_PDF_DEFAULT,
            "webdoc_allow_html": WEBDOC_ALLOW_HTML_DEFAULT,
            "webdoc_max_file_mb": WEBDOC_MAX_FILE_MB_DEFAULT,
            "webdoc_http_timeout_s": WEBDOC_HTTP_TIMEOUT_S_DEFAULT,
            "webdoc_throttle_ms": WEBDOC_THROTTLE_MS_DEFAULT,
            "webdoc_ocr_enabled": WEBDOC_OCR_ENABLED_DEFAULT,
            "doc_extract_timeout_s": DOC_EXTRACT_TIMEOUT_S_DEFAULT,
            "webdoc_primary": _webdoc_primary_provider(),
            "webdoc_secondary": _webdoc_secondary_provider(),
            "serpapi_api_key": os.getenv("SERPAPI_API_KEY"),
        },
        started_at=time.time(),
    )

    status = await asyncio.to_thread(run_ingest_multisource, run_ctx)

    run["completed_at"] = utc_now()
    run["status"] = status.status
    run["stage"] = status.step
    run["requested_new"] = status.requested_new_total
    run["selected_new"] = status.selected_new_total
    run["candidates_found"] = status.candidates_found_total
    run["total_audio_minutes"] = status.total_audio_minutes
    write_run(root, run_id, run)


async def queue_worker() -> None:
    global RUN_QUEUE
    if RUN_QUEUE is None:
        RUN_QUEUE = asyncio.Queue()
    while True:
        job = await RUN_QUEUE.get()
        try:
            await process_run(job)
        except Exception:
            logger.exception("Run processing failed")
        finally:
            RUN_QUEUE.task_done()


@app.post("/transcript")
def transcript(
    payload: TranscriptRequest,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    x_api_key_2: str | None = Header(default=None, alias="X-Api-Key"),
) -> Any:
    expected_key = _expected_api_key()
    provided = (x_api_key or x_api_key_2 or "").strip()
    if expected_key and provided != expected_key:
        return _auth_error()

    started = time.perf_counter()
    canonical_id = _extract_video_id(payload.video_id, payload.source_id, payload.url)
    if not canonical_id:
        diagnostics = {"pipeline_stage_attempts": ["audio_first"], "elapsed_ms_total": int((time.perf_counter() - started) * 1000)}
        return _error(400, "INVALID_VIDEO_ID", "Unable to determine YouTube video id", diagnostics)

    if not _yt_dlp_exists() or not _ffmpeg_exists() or not _ffprobe_exists() or not _transcription_engine_available():
        diagnostics = {"pipeline_stage_attempts": ["audio_first"], "elapsed_ms_total": int((time.perf_counter() - started) * 1000)}
        return _error(503, "DEPENDENCIES_MISSING", "Required audio dependencies missing", diagnostics)

    proxy_url = _legacy_proxy_url(payload)
    cookies_path = _cookie_source_for_ytdlp()

    temp_root = Path("/tmp/brains-worker")
    temp_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{canonical_id}-", dir=str(temp_root)))

    diagnostics: dict[str, Any] = {"pipeline_stage_attempts": ["audio_first"], "audio_download_status": "pending", "stt_status": "pending"}
    try:
        audio_path = download_audio(
            canonical_id,
            work_dir,
            proxy_url,
            str(cookies_path) if cookies_path else None,
            AUDIO_DL_TIMEOUT_S_DEFAULT,
            False,
        )
        diagnostics["audio_download_status"] = "success"

        result = transcribe_audio(
            audio_path,
            payload.stt_language or TRANSCRIBE_LANGUAGE_DEFAULT,
            payload.stt_model or TRANSCRIBE_MODEL_DEFAULT,
            5,
            True,
            TRANSCRIBE_TIMEOUT_S_DEFAULT,
        )
        diagnostics["stt_status"] = "success"
        diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
        return {
            "video_id": canonical_id,
            "transcript_source": "audio_local_whisper",
            "transcript_text": result.text,
            "diagnostics": diagnostics,
        }
    except AudioDownloadError as exc:
        diagnostics["audio_download_status"] = "failed"
        diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
        return _error(502, "AUDIO_DOWNLOAD_FAILED", str(exc), diagnostics)
    except (TranscriptionTimeout, TranscriptionError) as exc:
        diagnostics["stt_status"] = "failed"
        diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
        return _error(502, "TRANSCRIPTION_FAILED", str(exc), diagnostics)
    except Exception as exc:
        diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
        return _error(500, "UNEXPECTED_SERVER_ERROR", str(exc), diagnostics)
    finally:
        if not payload.debug_keep_files:
            shutil.rmtree(work_dir, ignore_errors=True)


@app.get("/transcript/subs-dry-run")
def transcript_subs_dry_run(video_id: str, preferred_language: str | None = None) -> dict[str, Any]:
    return {"ok": False, "error": "timedtext_disabled"}


@app.get("/v1/sources/{source_id}")
def get_source(source_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> Any:
    require_api_key(x_api_key)
    with db_session() as session:
        if session is None:
            return JSONResponse(status_code=501, content={"error_code": "DB_DISABLED", "error": "Database not configured"})
        row = session.query(Source).filter(Source.source_id == source_id).one_or_none()
        if not row:
            return JSONResponse(status_code=404, content={"error_code": "NOT_FOUND", "error": "source not found"})
        return {
            "source_id": row.source_id,
            "provider": row.provider,
            "url": row.url,
            "title": row.title,
            "channel": row.channel,
            "published_at": row.published_at.isoformat() if row.published_at else None,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }


@app.get("/v1/sources/{source_id}/transcript")
def get_source_transcript(source_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> Any:
    require_api_key(x_api_key)
    with db_session() as session:
        if session is None:
            return JSONResponse(status_code=501, content={"error_code": "DB_DISABLED", "error": "Database not configured"})
        source = session.query(Source).filter(Source.source_id == source_id).one_or_none()
        if not source:
            return JSONResponse(status_code=404, content={"error_code": "NOT_FOUND", "error": "source not found"})
        idx = session.query(TranscriptIndex).filter(TranscriptIndex.source_id == source.id).one_or_none()
        if not idx or not idx.best_transcript_artifact_id:
            return JSONResponse(status_code=404, content={"error_code": "NOT_FOUND", "error": "transcript not found"})
        artifact = session.query(Artifact).filter(Artifact.id == idx.best_transcript_artifact_id).one_or_none()
        if not artifact:
            return JSONResponse(status_code=404, content={"error_code": "NOT_FOUND", "error": "artifact not found"})
        return {
            "source_id": source.source_id,
            "transcript_status": idx.transcript_status,
            "artifact_id": artifact.id,
            "storage_uri": artifact.storage_uri,
            "sha256": artifact.sha256,
            "bytes": artifact.bytes,
            "provider": idx.provider,
            "updated_at": idx.updated_at.isoformat() if idx.updated_at else None,
        }


@app.on_event("startup")
async def startup() -> None:
    global RUN_TASK
    BRAINS_ROOT.mkdir(parents=True, exist_ok=True)
    global RUN_QUEUE
    if RUN_QUEUE is None:
        RUN_QUEUE = asyncio.Queue()
    if RUN_TASK is None:
        RUN_TASK = asyncio.create_task(queue_worker())


@app.get("/v1/health")
def health() -> dict[str, Any]:
    try:
        proxy_enabled, _provider, _proxy_url_redacted, _proxies, proxy_reason = _build_proxy()
        cookies_source = _cookie_source()
        db_configured = "configured" if _db_enabled() else "missing"
        spaces_configured = "configured" if (os.getenv("BRAINS_S3_ENDPOINT") and os.getenv("BRAINS_S3_BUCKET") and (os.getenv("BRAINS_S3_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID")) and (os.getenv("BRAINS_S3_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY"))) else "missing"
        webdocs_primary_provider = _webdoc_primary_provider()
        webdocs_secondary_provider = _webdoc_secondary_provider()
        serpapi_configured = is_serpapi_configured()
        return {
            "status": "ok",
            "time": utc_now(),
            "db": db_configured,
            "spaces": spaces_configured,
            "yt_dlp_available": _yt_dlp_exists(),
            "ffmpeg_available": _ffmpeg_exists(),
            "ffprobe_available": _ffprobe_exists(),
            "transcription_engine_available": _transcription_engine_available(),
            "webdocs_engine_available": _webdocs_engine_available(),
            "proxy_configured": bool(proxy_enabled),
            "proxy_reason": proxy_reason,
            "cookies_present": bool(cookies_source),
            "cookies_source": str(cookies_source) if cookies_source else None,
            "serpapi_configured": serpapi_configured,
            "webdocs_primary_provider": webdocs_primary_provider,
            "webdocs_secondary_provider": webdocs_secondary_provider,
            "webdocs_fallback_enabled": bool(serpapi_configured and webdocs_secondary_provider == "serpapi"),
            "brain_stats_endpoint": True,
        }
    except Exception:
        return {"status": "ok", "time": utc_now(), "db": "unknown", "spaces": "unknown"}


@app.get("/v1/brains")
def list_brains(x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> list[dict[str, Any]]:
    require_api_key(x_api_key)
    brains = []
    for brain_json in BRAINS_ROOT.glob("*/brain.json"):
        brains.append(load_json(brain_json, {}))
    return brains


@app.post("/v1/brains")
def create_brain(req: BrainCreateRequest, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    brain_id = slugify(req.name)
    root = ensure_dirs(brain_id)
    brain_json = root / "brain.json"
    if brain_json.exists():
        raise HTTPException(status_code=409, detail="Brain already exists")
    brain = {"brain_id": brain_id, "brain_name": req.name, "brain_type": req.brain_type, "description": req.description, "created_at": utc_now()}
    write_json(brain_json, brain)
    write_json(root / "ledger.json", {"ingested_video_ids": [], "ingested_doc_ids": [], "records": []})
    return brain


@app.get("/v1/brains/{brain_id}")
def get_brain(brain_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    root = BRAINS_ROOT / brain_id
    brain = load_json(root / "brain.json", None)
    if not brain:
        raise HTTPException(status_code=404, detail="Brain not found")
    return brain


@app.get("/v1/brains/{brain_slug}/stats")
def brain_stats(brain_slug: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    return compute_brain_stats(brain_slug)


@app.post("/v1/brains/{brain_id}/ingest", status_code=202)
async def ingest(brain_id: str, req: IngestRequest, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    root = BRAINS_ROOT / brain_id
    if not (root / "brain.json").exists():
        raise HTTPException(status_code=404, detail="Brain not found")
    global RUN_QUEUE
    if RUN_QUEUE is None:
        RUN_QUEUE = asyncio.Queue()
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{brain_id}"
    run = {
        "run_id": run_id,
        "brain_id": brain_id,
        "status": "queued",
        "created_at": utc_now(),
        "payload": req.model_dump(),
        "stage": "queued",
        "candidates_found": 0,
        "requested_new": int(req.selected_new or req.n_new_videos or 0),
        "selected_new": 0,
        "eligible_count": 0,
        "eligible_shortfall": 0,
        "skipped_duplicates": 0,
        "completed": 0,
        "failed": 0,
        "audio_fallback_used": False,
        "current": None,
        "progress": {"total": 0, "done": 0, "failed": 0, "stage": "queued"},
    }
    write_run(root, run_id, run)
    status = RunStatus(run_id=run_id, brain_slug=brain_id, keyword=req.keyword, status="queued", step="queued")
    StatusWriter(run_dir(root, run_id) / "status.json", status).update()
    init_diagnostics_summary(root, run_id)
    register_run(run_id, brain_id)
    await RUN_QUEUE.put({"run_id": run_id, "brain_id": brain_id, "payload": req.model_dump()})
    return {"run_id": run_id, "status": "queued"}


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    started = time.perf_counter()
    root = resolve_run_root(run_id)
    if not root:
        legacy_matches = list(BRAINS_ROOT.glob(f"*/runs/{run_id}.json"))
        if not legacy_matches:
            raise HTTPException(status_code=404, detail="Run not found")
        payload = status_from_run(load_json(legacy_matches[0], {}))
        return payload

    status_path = run_dir(root, run_id) / "status.json"
    payload = load_json(status_path, {})
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info("get_run_status end run_id=%s elapsed_ms=%s", run_id, elapsed_ms)
    return payload


@app.get("/v1/runs/{run_id}/diagnostics")
def run_diagnostics(run_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    root = resolve_run_root(run_id)
    if not root:
        raise HTTPException(status_code=404, detail="Run not found")
    status_path = run_dir(root, run_id) / "status.json"
    if status_path.exists():
        return load_json(status_path, {})
    return {"run_id": run_id, "status": "unknown"}


@app.get("/v1/runs/{run_id}/report")
def run_report(run_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    root = resolve_run_root(run_id)
    if root:
        report_path = run_dir(root, run_id) / "report.json"
        if report_path.exists():
            return load_json(report_path, {})
    raise HTTPException(status_code=202, detail="report_not_ready")


@app.get("/v1/runs/{run_id}/files")
def run_files(run_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    root = resolve_run_root(run_id)
    if not root:
        raise HTTPException(status_code=404, detail="Run not found")
    run_root = root / "runs" / run_id
    if not run_root.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    return {
        "run_id": run_id,
        "run_dir": str(run_root),
        "audio_files": _list_files(run_root / "audio", run_root),
        "transcript_files_txt": _list_files(run_root / "transcripts", run_root, "*.txt"),
        "transcript_files_json": _list_files(run_root / "transcripts", run_root, "*.json"),
        "doc_pdf_files": _list_files(run_root / "docs" / "pdf", run_root),
        "doc_html_files": _list_files(run_root / "docs" / "html", run_root),
        "doc_text_files": _list_files(run_root / "docs" / "text", run_root, "*.txt"),
        "artifact_files": _list_files(run_root / "artifacts", run_root),
    }


@app.post("/v1/runs/{run_id}/brain-pack")
def build_run_pack(run_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    run_matches = list(BRAINS_ROOT.glob(f"*/runs/{run_id}/run.json"))
    if not run_matches:
        raise HTTPException(status_code=404, detail="Run not found")
    run = load_json(run_matches[0], {})
    root = BRAINS_ROOT / run.get("brain_id")
    brain = load_json(root / "brain.json", {})
    info = build_brain_pack(root, run["brain_id"], run_id, run.get("payload", {}).get("keyword", ""), run.get("ingested_video_ids", []), brain.get("brain_type", "BD"))
    run["brain_pack"] = info
    write_run(root, run_id, run)
    attempts = load_transcript_attempts(root, run_id)
    info["diagnostics_summary"] = {
        "transcript_failure_reasons": summarize_transcript_failures(attempts),
        "sample_failures": sample_failures(attempts),
        "transcript_attempts_jsonl": str(run_dir(root, run_id) / "transcript_attempts.jsonl"),
    }
    return info


@app.get("/v1/runs/{run_id}/brain-pack")
def get_or_build_run_pack(run_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    run_matches = list(BRAINS_ROOT.glob(f"*/runs/{run_id}/run.json"))
    if not run_matches:
        raise HTTPException(status_code=404, detail="Run not found")
    run = load_json(run_matches[0], {})
    if run.get("brain_pack"):
        root = BRAINS_ROOT / run.get("brain_id")
        attempts = load_transcript_attempts(root, run_id)
        return {
            **run["brain_pack"],
            "diagnostics_summary": {
                "transcript_failure_reasons": summarize_transcript_failures(attempts),
                "sample_failures": sample_failures(attempts),
                "transcript_attempts_jsonl": str(run_dir(root, run_id) / "transcript_attempts.jsonl"),
            },
        }
    return build_run_pack(run_id, x_api_key)


@app.get("/v1/brain-packs/{brain_pack_id}")
def get_brain_pack(brain_pack_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    matches = list(BRAINS_ROOT.glob(f"*/packs/{brain_pack_id}.json"))
    if not matches:
        raise HTTPException(status_code=404, detail="Brain pack not found")
    return load_json(matches[0], {})


@app.get("/v1/brain-packs/{brain_pack_id}/download")
def download_brain_pack(brain_pack_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> FileResponse:
    require_api_key(x_api_key)
    info = get_brain_pack(brain_pack_id, x_api_key)
    zip_path = Path(info["zip_path"])
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Brain pack archive missing")
    return FileResponse(path=str(zip_path), filename=zip_path.name, media_type="application/zip")
