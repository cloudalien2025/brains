from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, Field

from apps.brains_worker.discovery import DiscoveryError, discover_youtube_videos

app = FastAPI(title="Brains Worker v1")
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

WORKER_API_KEY = (os.getenv("WORKER_API_KEY") or "").strip()
BRAINS_DATA_DIR = Path(os.getenv("BRAINS_DATA_DIR", "/opt/brains-data"))
BRAINS_ROOT = BRAINS_DATA_DIR / "brains"
GLOBAL_TRANSCRIPT_CACHE = BRAINS_DATA_DIR / "global_cache" / "transcripts"
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "3"))
MAX_CONCURRENT_STT = int(os.getenv("MAX_CONCURRENT_STT", "1"))
MAX_CONCURRENT_SYNTHESIS = int(os.getenv("MAX_CONCURRENT_SYNTHESIS", "1"))
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "600"))
OVERLAP_SECONDS = int(os.getenv("OVERLAP_SECONDS", "15"))
ARCHIVE_AUDIO = (os.getenv("ARCHIVE_AUDIO", "false").lower() in {"1", "true", "yes", "on"})



RUN_QUEUE: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
RUN_TASK: asyncio.Task | None = None
RUN_LOCK = asyncio.Lock()
DOWNLOAD_SEM = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
STT_SEM = asyncio.Semaphore(MAX_CONCURRENT_STT)
SYNTH_SEM = asyncio.Semaphore(MAX_CONCURRENT_SYNTHESIS)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def read_ledger(root: Path) -> dict[str, Any]:
    return load_json(root / "ledger.json", {"ingested_video_ids": [], "records": []})


def update_ledger(root: Path, record: dict[str, Any]) -> None:
    ledger = read_ledger(root)
    vid = record["video_id"]
    if vid not in ledger["ingested_video_ids"]:
        ledger["ingested_video_ids"].append(vid)
    ledger["records"].append(record)
    write_json(root / "ledger.json", ledger)


class BrainCreateRequest(BaseModel):
    name: str
    brain_type: str = Field(pattern="^(BD|UAP)$")
    description: str | None = None


class IngestRequest(BaseModel):
    keyword: str
    n_new_videos: int = Field(ge=1)
    max_candidates: int = Field(default=50, ge=1, le=50)
    discovery_order: str | None = Field(default=None, pattern="^(relevance|date)$")
    published_after: str | None = None
    mode: str = "audio_first"
    preferred_language: str = "en"
    longform: dict[str, int] = Field(default_factory=lambda: {"chunk_seconds": CHUNK_SECONDS, "overlap_seconds": OVERLAP_SECONDS})
    synthesis: dict[str, bool] = Field(default_factory=lambda: {"update": True})
    brain_pack: dict[str, bool] = Field(default_factory=lambda: {"build": True})


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
    diagnostics: dict[str, Any] = {"video_id": video_id, "chunks": [], "failed_chunks": 0}
    duration = ffprobe_duration(audio_path) or 0
    ranges = chunk_ranges(duration, chunk_seconds, overlap_seconds) if duration > chunk_seconds else [(0.0, duration or chunk_seconds)]
    stitched = ""
    for idx, (start, length) in enumerate(ranges, start=1):
        chunk_path = tmp_dir / f"yt_{video_id}_chunk_{idx:04d}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio_path), "-ss", str(start), "-t", str(length), "-ac", "1", "-ar", "16000", str(chunk_path)
        ], check=True, capture_output=True)
        chunk_text = ""
        chunk_diag = {"chunk": idx, "start": start, "length": length, "status": "ok", "retry": 0}
        for attempt in range(2):
            try:
                with chunk_path.open("rb") as audio_file:
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    rsp = client.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe",
                        file=audio_file,
                        language=preferred_language,
                    )
                chunk_text = (getattr(rsp, "text", "") or "").strip()
                break
            except Exception as exc:
                chunk_diag["retry"] = attempt + 1
                chunk_diag["status"] = "failed"
                chunk_diag["error"] = str(exc)
                if attempt == 0:
                    time.sleep(2)
        if not chunk_text:
            diagnostics["failed_chunks"] += 1
        if stitched and chunk_text:
            chunk_text = overlap_dedupe(stitched, chunk_text)
        stitched = f"{stitched}\n{chunk_text}".strip()
        diagnostics["chunks"].append(chunk_diag)
    diagnostics["duration_seconds"] = duration
    diagnostics["chunk_count"] = len(ranges)
    return stitched, diagnostics


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
    run_path = root / "runs" / f"{run_id}.json"
    brain = load_json(root / "brain.json", {})
    run = load_json(run_path, {})
    run["status"] = "processing"
    run["stage"] = "discovery"
    run["started_at"] = utc_now()
    run["completed"] = 0
    run["failed"] = 0
    run["errors"] = []
    write_json(run_path, run)

    errors: list[str] = []
    ingested_ids: list[str] = []
    ledger = read_ledger(root)
    order = resolve_discovery_order(brain.get("brain_type", "BD"), payload.discovery_order)

    published_after = payload.published_after
    if not published_after and brain.get("brain_type") == "UAP":
        published_after = "2023-01-01T00:00:00Z"

    try:
        candidates = discover_youtube_videos(
            keyword=payload.keyword,
            max_candidates=payload.max_candidates,
            published_after=published_after,
            language=payload.preferred_language,
            order=order,
        )
    except DiscoveryError as exc:
        run["status"] = "completed_with_errors"
        run["final_error_code"] = exc.code
        run["final_error"] = exc.message
        run["errors"] = [exc.message]
        run["stage"] = "completed"
        run["completed_at"] = utc_now()
        write_json(run_path, run)
        return

    existing_ids = set(ledger.get("ingested_video_ids", []))
    selected = [c for c in candidates if c.get("video_id") and c.get("video_id") not in existing_ids][: payload.n_new_videos]
    skipped_duplicates = max(0, len(candidates) - len([c for c in candidates if c.get("video_id") not in existing_ids]))

    run["discovery"] = {"method": "youtube_data_api", "order": order, "published_after": published_after}
    run["candidates_found"] = len(candidates)
    run["candidates"] = candidates[:50]
    run["selected_new"] = len(selected)
    run["skipped_duplicates"] = skipped_duplicates
    run["progress"] = {"total": len(selected), "done": 0, "failed": 0, "stage": "ingesting"}

    if len(candidates) == 0:
        run["status"] = "completed_with_errors"
        run["final_error_code"] = "DISCOVERY_ZERO_RESULTS"
        run["final_error"] = "YouTube API returned 0 candidates for keyword"
        run["stage"] = "completed"
        run["completed_at"] = utc_now()
        write_json(run_path, run)
        return

    if len(selected) == 0:
        run["status"] = "completed"
        run["message"] = "No new videos; all candidates already ingested"
        run["stage"] = "completed"
        run["completed_at"] = utc_now()
        write_json(run_path, run)
        return

    write_json(run_path, run)

    for src in selected:
        vid = src.get("video_id", "")
        title = src.get("title", "")
        run["current"] = {"video_id": vid, "stage": "ingesting", "detail": title}
        write_json(run_path, run)
        diag: dict[str, Any] = {"video_id": vid, "title": title, "stages": []}
        tmp_dir = root / "tmp" / vid
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            transcript_path = root / "transcripts" / f"yt_{vid}.txt"
            global_cache = GLOBAL_TRANSCRIPT_CACHE / f"yt_{vid}.txt"
            transcript = ""
            if transcript_path.exists():
                transcript = transcript_path.read_text(encoding="utf-8")
                diag["stages"].append("brain_cache_hit")
            elif global_cache.exists():
                transcript = global_cache.read_text(encoding="utf-8")
                transcript_path.write_text(transcript, encoding="utf-8")
                diag["stages"].append("global_cache_hit")
            else:
                async with DOWNLOAD_SEM:
                    audio_path = tmp_dir / f"yt_{vid}.audio.m4a"
                    subprocess.run(["yt-dlp", "-f", "bestaudio/best", "-o", str(audio_path), src.get("url")], check=True, capture_output=True)
                async with STT_SEM:
                    run["stage"] = "extracting"
                    run["current"] = {"video_id": vid, "stage": "extracting", "detail": "Transcribing audio"}
                    write_json(run_path, run)
                    transcript, stt_diag = transcribe_audio_chunks(
                        video_id=vid,
                        audio_path=audio_path,
                        chunk_seconds=payload.longform.get("chunk_seconds", CHUNK_SECONDS),
                        overlap_seconds=payload.longform.get("overlap_seconds", OVERLAP_SECONDS),
                        preferred_language=payload.preferred_language,
                        tmp_dir=tmp_dir,
                    )
                    diag["stt"] = stt_diag
                transcript_path.write_text(transcript, encoding="utf-8")
                global_cache.write_text(transcript, encoding="utf-8")
                diag["stages"].append("audio_stt")

            source_meta = {
                "video_id": vid,
                "title": title,
                "channel": src.get("channel_title"),
                "published_at": src.get("published_at"),
                "url": src.get("url"),
                "ingested_at": utc_now(),
            }
            write_json(root / "sources" / f"yt_{vid}.json", source_meta)
            run["stage"] = "synthesizing"
            run["current"] = {"video_id": vid, "stage": "synthesizing", "detail": "Updating synthesis"}
            write_json(run_path, run)
            extraction = build_extraction(brain.get("brain_type", "BD"), transcript, source_meta)
            write_json(root / "extractions" / f"yt_{vid}.v1.json", extraction)
            async with SYNTH_SEM:
                if payload.synthesis.get("update", True):
                    update_synthesis(root, brain.get("brain_type", "BD"), extraction)
            update_ledger(root, source_meta)
            write_json(root / "diagnostics" / f"yt_{vid}.json", diag)
            ingested_ids.append(vid)
        except Exception as exc:
            errors.append(f"{vid}: {exc}")
            run["progress"]["failed"] += 1
            run["failed"] = run["progress"]["failed"]
            diag["error"] = str(exc)
            write_json(root / "diagnostics" / f"yt_{vid}.json", diag)
        finally:
            run["progress"]["done"] += 1
            run["completed"] = run["progress"]["done"]
            write_json(run_path, run)
            if not ARCHIVE_AUDIO and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    pack_info = None
    if payload.brain_pack.get("build", True):
        run["progress"]["stage"] = "brain_pack"
        run["stage"] = "packing"
        write_json(run_path, run)
        pack_info = build_brain_pack(root, brain_id, run_id, payload.keyword, ingested_ids, brain.get("brain_type", "BD"))

    run["completed_at"] = utc_now()
    run["errors"] = errors
    run["ingested_video_ids"] = ingested_ids
    run["brain_pack"] = pack_info
    run["status"] = "completed" if not errors else ("completed_with_errors" if ingested_ids else "failed")
    run["progress"]["stage"] = "completed"
    run["stage"] = "completed"
    run["current"] = None
    run["failed"] = run["progress"].get("failed", 0)
    run["completed"] = run["progress"].get("done", 0)
    write_json(run_path, run)


async def queue_worker() -> None:
    while True:
        job = await RUN_QUEUE.get()
        try:
            await process_run(job)
        except Exception:
            logger.exception("Run processing failed")
        finally:
            RUN_QUEUE.task_done()


@app.on_event("startup")
async def startup() -> None:
    global RUN_TASK
    BRAINS_ROOT.mkdir(parents=True, exist_ok=True)
    if RUN_TASK is None:
        RUN_TASK = asyncio.create_task(queue_worker())


@app.get("/v1/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "time": utc_now()}


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
    write_json(root / "ledger.json", {"ingested_video_ids": [], "records": []})
    return brain


@app.get("/v1/brains/{brain_id}")
def get_brain(brain_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    root = BRAINS_ROOT / brain_id
    brain = load_json(root / "brain.json", None)
    if not brain:
        raise HTTPException(status_code=404, detail="Brain not found")
    return brain


@app.post("/v1/brains/{brain_id}/ingest", status_code=202)
async def ingest(brain_id: str, req: IngestRequest, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    root = BRAINS_ROOT / brain_id
    if not (root / "brain.json").exists():
        raise HTTPException(status_code=404, detail="Brain not found")
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{brain_id}"
    run = {
        "run_id": run_id,
        "brain_id": brain_id,
        "status": "queued",
        "created_at": utc_now(),
        "payload": req.model_dump(),
        "stage": "queued",
        "candidates_found": 0,
        "selected_new": 0,
        "skipped_duplicates": 0,
        "completed": 0,
        "failed": 0,
        "current": None,
        "progress": {"total": 0, "done": 0, "failed": 0, "stage": "queued"},
    }
    write_json(root / "runs" / f"{run_id}.json", run)
    await RUN_QUEUE.put({"run_id": run_id, "brain_id": brain_id, "payload": req.model_dump()})
    return {"run_id": run_id, "status": "queued"}


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    matches = list(BRAINS_ROOT.glob(f"*/runs/{run_id}.json"))
    if not matches:
        raise HTTPException(status_code=404, detail="Run not found")
    return load_json(matches[0], {})


@app.get("/v1/runs/{run_id}/report")
def run_report(run_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    run = get_run(run_id, x_api_key)
    brain_id = run["brain_id"]
    root = BRAINS_ROOT / brain_id
    artifacts = []
    for vid in run.get("ingested_video_ids", []):
        artifacts.append({
            "video_id": vid,
            "source": f"sources/yt_{vid}.json",
            "transcript": f"transcripts/yt_{vid}.txt",
            "extraction": f"extractions/yt_{vid}.v1.json",
            "diagnostics": f"diagnostics/yt_{vid}.json",
        })
    return {
        "run_id": run_id,
        "brain_id": brain_id,
        "status": run.get("status"),
        "ingested_new": len(run.get("ingested_video_ids", [])),
        "transcripts_succeeded": len(run.get("ingested_video_ids", [])),
        "transcripts_failed": len(run.get("errors", [])),
        "brain_pack_id": (run.get("brain_pack") or {}).get("brain_pack_id"),
        "artifacts": artifacts,
        "brain_root": str(root),
    }


@app.post("/v1/runs/{run_id}/brain-pack")
def build_run_pack(run_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    run = get_run(run_id, x_api_key)
    root = BRAINS_ROOT / run["brain_id"]
    brain = load_json(root / "brain.json", {})
    info = build_brain_pack(root, run["brain_id"], run_id, run.get("payload", {}).get("keyword", ""), run.get("ingested_video_ids", []), brain.get("brain_type", "BD"))
    run["brain_pack"] = info
    write_json(root / "runs" / f"{run_id}.json", run)
    return info


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
