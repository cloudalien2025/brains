from __future__ import annotations

import asyncio
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
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, Field, model_validator
import requests

from apps.brains_worker.discovery import DiscoveryError, DiscoveryOutcome, discover_youtube_videos

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
AUDIO_FALLBACK_ENABLED = os.getenv("AUDIO_FALLBACK_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
COOKIES_PATHS = [Path("/workspace/brains/cookies/youtube_cookies.txt"), Path("/workspace/brains/cookies/youtube_storage_state.json")]



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


def run_dir(root: Path, run_id: str) -> Path:
    path = root / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


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
        "selected_new": int(run_payload.get("selected_new") or 0),
        "completed": int(run_payload.get("completed") or 0),
        "failed": int(run_payload.get("failed") or 0),
        "discovery_method": (run_payload.get("discovery") or {}).get("method"),
        "youtube_api_http_status": (run_payload.get("discovery") or {}).get("youtube_api_http_status"),
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
    caption_probe_attempts = [x for x in attempts if x.get("phase") == "probe"]
    report_payload = {
        "run_id": run_id,
        "brain_id": run_payload.get("brain_id"),
        "status": run_payload.get("status"),
        "discovery_method": (run_payload.get("discovery") or {}).get("method"),
        "youtube_api_http_status": (run_payload.get("discovery") or {}).get("youtube_api_http_status"),
        "candidates_found": int(run_payload.get("candidates_found") or 0),
        "requested_new": int(run_payload.get("requested_new") or 0),
        "selected_new": int(run_payload.get("selected_new") or 0),
        "eligible_shortfall": int(run_payload.get("eligible_shortfall") or 0),
        "ingested_new": len(run_payload.get("ingested_video_ids", [])),
        "caption_probe_attempted": len(caption_probe_attempts),
        "caption_probe_failed": sum(1 for x in caption_probe_attempts if not x.get("success")),
        "transcripts_attempted_selected": transcript_summary["transcripts_attempted_selected"],
        "transcripts_succeeded": transcript_summary["transcripts_succeeded"],
        "transcripts_failed": transcript_summary["transcripts_failed"],
        "transcripts_failed_selected": transcript_summary["transcripts_failed"],
        "transcript_failure_reasons": summarize_transcript_failures(transcript_attempts_selected),
        "sample_failures": sample_failures(transcript_attempts_selected),
        "brain_pack_id": (run_payload.get("brain_pack") or {}).get("brain_pack_id"),
        "transcript_attempts_jsonl": str(run_dir(root, run_id) / "transcript_attempts.jsonl"),
        "artifacts": artifacts,
        "brain_root": str(root),
        "updated_at": utc_now(),
    }
    write_json(run_dir(root, run_id) / "report.json", report_payload)


def read_ledger(root: Path) -> dict[str, Any]:
    return load_json(root / "ledger.json", {"ingested_video_ids": [], "records": []})


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


def _build_proxy() -> tuple[bool, str | None, str | None, dict[str, str] | None]:
    proxy_url = (
        os.getenv("BRAINS_PROXY_URL")
        or os.getenv("DECODO_PROXY_URL")
        or os.getenv("RESIDENTIAL_PROXY_URL")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("HTTP_PROXY")
        or ""
    ).strip()
    if not proxy_url:
        return False, None, None, None
    provider = "decodo" if "decodo" in proxy_url.lower() else "custom"
    return True, provider, _redact_proxy_url(proxy_url), {"http": proxy_url, "https": proxy_url}


def _cookie_source() -> Path | None:
    for path in COOKIES_PATHS:
        if path.exists():
            return path
    return None


def _build_http_session() -> tuple[requests.Session, dict[str, Any]]:
    session = requests.Session()
    proxy_enabled, provider, proxy_url_redacted, proxies = _build_proxy()
    if proxies:
        session.proxies.update(proxies)
    source = _cookie_source()
    cookies_enabled = False
    if source and source.suffix == ".txt":
        jar = http.cookiejar.MozillaCookieJar(str(source))
        try:
            jar.load(ignore_discard=True, ignore_expires=True)
            session.cookies.update(jar)
            cookies_enabled = True
        except Exception:
            cookies_enabled = False
    elif source and source.suffix == ".json":
        try:
            state = json.loads(source.read_text(encoding="utf-8"))
            for cookie in state.get("cookies", []):
                if cookie.get("name") and cookie.get("value"):
                    session.cookies.set(cookie["name"], cookie["value"], domain=cookie.get("domain"), path=cookie.get("path", "/"))
            cookies_enabled = bool(state.get("cookies"))
        except Exception:
            cookies_enabled = False

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
        "cookies_enabled": cookies_enabled,
        "cookies_source": str(source) if source else None,
    }


def _caption_tracks(session: requests.Session, video_id: str) -> tuple[list[dict[str, str]], int | None, str | None]:
    resp = session.get("https://www.youtube.com/api/timedtext", params={"type": "list", "v": video_id}, timeout=20)
    if resp.status_code != 200:
        return [], resp.status_code, f"caption_probe_http_{resp.status_code}"
    root = ET.fromstring(resp.text or "<transcript_list/>")
    tracks = []
    for node in root.findall("track"):
        tracks.append({
            "lang_code": node.attrib.get("lang_code", ""),
            "name": node.attrib.get("name", ""),
            "kind": node.attrib.get("kind", ""),
        })
    return tracks, resp.status_code, None


def _fetch_timedtext_transcript(session: requests.Session, video_id: str, track: dict[str, str]) -> tuple[str, int | None]:
    params = {"v": video_id, "lang": track.get("lang_code") or "en", "fmt": "srv3"}
    if track.get("kind"):
        params["kind"] = track["kind"]
    if track.get("name"):
        params["name"] = track["name"]
    resp = session.get("https://www.youtube.com/api/timedtext", params=params, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    root = ET.fromstring(resp.text or "<transcript/>")
    lines = [html.unescape((n.text or "").strip()) for n in root.findall("text")]
    transcript = "\n".join(x for x in lines if x)
    return transcript, resp.status_code


class BrainCreateRequest(BaseModel):
    name: str
    brain_type: str = Field(pattern="^(BD|UAP)$")
    description: str | None = None


class IngestRequest(BaseModel):
    keyword: str
    n_new_videos: int | None = Field(default=None, ge=1)
    selected_new: int | None = Field(default=None, ge=1)
    max_new: int | None = Field(default=None, ge=1)
    max_candidates: int = Field(default=50, ge=1, le=50)
    discovery_order: str | None = Field(default=None, pattern="^(relevance|date)$")
    published_after: str | None = None
    mode: str = "audio_first"
    preferred_language: str = "en"
    longform: dict[str, int] = Field(default_factory=lambda: {"chunk_seconds": CHUNK_SECONDS, "overlap_seconds": OVERLAP_SECONDS})
    synthesis: dict[str, bool] = Field(default_factory=lambda: {"update": True})
    brain_pack: dict[str, bool] = Field(default_factory=lambda: {"build": True})


    @model_validator(mode="after")
    def validate_requested_new(self) -> "IngestRequest":
        requested = self.selected_new or self.max_new or self.n_new_videos
        if requested is None:
            raise ValueError("One of selected_new, max_new, or n_new_videos is required")
        self.n_new_videos = int(requested)
        self.selected_new = int(requested)
        return self

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
    brain = load_json(root / "brain.json", {})
    run = load_run(root, run_id)
    run["status"] = "processing"
    run["stage"] = "discovery"
    run["started_at"] = utc_now()
    run["completed"] = 0
    run["failed"] = 0
    run["errors"] = []
    write_run(root, run_id, run)
    write_status(root, run_id, run)

    errors: list[str] = []
    ingested_ids: list[str] = []
    ledger = read_ledger(root)
    order = resolve_discovery_order(brain.get("brain_type", "BD"), payload.discovery_order)

    published_after = payload.published_after
    if not published_after and brain.get("brain_type") == "UAP":
        published_after = "2023-01-01T00:00:00Z"

    try:
        discovery_outcome: DiscoveryOutcome = discover_youtube_videos(
            keyword=payload.keyword,
            max_candidates=payload.max_candidates,
            published_after=published_after,
            language=payload.preferred_language,
            order=order,
        )
        candidates = discovery_outcome.candidates
    except DiscoveryError as exc:
        run["status"] = "completed_with_errors"
        run["final_error_code"] = exc.code
        run["final_error"] = exc.message
        run["errors"] = [exc.message]
        run["stage"] = "completed"
        run["completed_at"] = utc_now()
        write_run(root, run_id, run)
        write_status(root, run_id, run)
        write_report(root, run_id, run)
        return

    existing_ids = set(ledger.get("ingested_video_ids", []))
    selected_pool = [c for c in candidates if c.get("video_id") and c.get("video_id") not in existing_ids]
    requested_new = int(payload.selected_new or payload.n_new_videos or 0)
    selected: list[dict[str, Any]] = []
    skipped_duplicates = max(0, len(candidates) - len([c for c in candidates if c.get("video_id") not in existing_ids]))

    run["discovery"] = {
        "method": discovery_outcome.method,
        "order": order,
        "published_after": published_after,
        "youtube_api_http_status": discovery_outcome.youtube_api_http_status,
    }
    run["candidates_found"] = len(candidates)
    run["candidates"] = candidates[:50]
    run["requested_new"] = requested_new
    run["selected_new"] = 0
    run["skipped_duplicates"] = skipped_duplicates
    run["progress"] = {"total": len(selected), "done": 0, "failed": 0, "stage": "ingesting"}
    run["request_contract"] = {"selected_new": requested_new, "payload": payload.model_dump()}

    if len(candidates) == 0:
        run["status"] = "completed_with_errors"
        run["final_error_code"] = "DISCOVERY_ZERO_RESULTS"
        run["final_error"] = "YouTube API returned 0 candidates for keyword"
        run["stage"] = "completed"
        run["completed_at"] = utc_now()
        write_run(root, run_id, run)
        write_status(root, run_id, run)
        write_report(root, run_id, run)
        return

    if len(selected_pool) == 0:
        run["status"] = "completed"
        run["message"] = "No new videos; all candidates already ingested"
        run["stage"] = "completed"
        run["completed_at"] = utc_now()
        write_run(root, run_id, run)
        write_status(root, run_id, run)
        write_report(root, run_id, run)
        return

    transcript_attempts_path = run_dir(root, run_id) / "transcript_attempts.jsonl"
    if transcript_attempts_path.exists():
        transcript_attempts_path.unlink()

    session, session_diag = _build_http_session()
    caption_eligible: list[dict[str, Any]] = []
    for src in selected_pool:
        if len(caption_eligible) >= requested_new:
            break
        vid = src.get("video_id", "")
        started = time.perf_counter()
        attempt = {
            "run_id": run_id,
            "brain_slug": brain_id,
            "video_id": vid,
            "video_url": src.get("url"),
            "title": src.get("title"),
            "phase": "probe",
            "method": "timedtext_probe",
            "selected_for_ingest": False,
            **session_diag,
            "http_status": None,
            "error_code": None,
            "error_message": None,
            "caption_tracks_found": 0,
            "selected_track": None,
            "elapsed_ms": 0,
            "success": False,
            "transcript_chars": 0,
        }
        try:
            tracks, status, probe_err = _caption_tracks(session, vid)
            attempt["http_status"] = status
            attempt["caption_tracks_found"] = len(tracks)
            if probe_err:
                attempt["error_code"] = probe_err
                attempt["error_message"] = probe_err
            elif tracks:
                track = next((t for t in tracks if (t.get("lang_code") or "").startswith("en")), tracks[0])
                attempt["selected_track"] = track
                attempt["success"] = True
                src["selected_track"] = track
                caption_eligible.append(src)
            else:
                attempt["error_code"] = "no_captions"
                attempt["error_message"] = "No caption tracks returned"
        except Exception as exc:
            attempt["error_code"] = "caption_probe_error"
            attempt["error_message"] = str(exc)
        attempt["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        append_jsonl(transcript_attempts_path, attempt)

    selected = caption_eligible
    if not selected and AUDIO_FALLBACK_ENABLED:
        selected = selected_pool[: requested_new]
    run["selected_new"] = len(selected)
    run["eligible_shortfall"] = max(0, requested_new - len(selected))
    run["caption_probe_attempted"] = len([x for x in load_transcript_attempts(root, run_id) if x.get("phase") == "probe"])
    run["progress"]["total"] = len(selected)
    write_run(root, run_id, run)
    write_status(root, run_id, run)

    for src in selected:
        src["selected_for_ingest"] = True
        vid = src.get("video_id", "")
        title = src.get("title", "")
        run["current"] = {"video_id": vid, "stage": "ingesting", "detail": title}
        write_run(root, run_id, run)
        write_status(root, run_id, run)
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
                track = src.get("selected_track") or {"lang_code": payload.preferred_language}
                transcript_ok = False
                for retry in range(4):
                    started = time.perf_counter()
                    attempt = {
                        "run_id": run_id,
                        "brain_slug": brain_id,
                        "video_id": vid,
                        "video_url": src.get("url"),
                        "title": src.get("title"),
                        "phase": "transcript",
                        "method": "timedtext",
                        "selected_for_ingest": True,
                        **session_diag,
                        "http_status": None,
                        "error_code": None,
                        "error_message": None,
                        "caption_tracks_found": 1,
                        "selected_track": track,
                        "elapsed_ms": 0,
                        "success": False,
                        "transcript_chars": 0,
                    }
                    try:
                        transcript, status = _fetch_timedtext_transcript(session, vid, track)
                        attempt["http_status"] = status
                        if transcript.strip():
                            attempt["success"] = True
                            attempt["transcript_chars"] = len(transcript)
                            transcript_ok = True
                            attempt["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
                            append_jsonl(transcript_attempts_path, attempt)
                            break
                        attempt["error_code"] = "empty_transcript"
                        attempt["error_message"] = "timedtext returned empty transcript"
                    except Exception as exc:
                        msg = str(exc)
                        attempt["error_message"] = msg
                        if "429" in msg:
                            attempt["error_code"] = "blocked_429"
                        elif "403" in msg:
                            attempt["error_code"] = "blocked_403"
                        else:
                            attempt["error_code"] = "timedtext_error"
                    attempt["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
                    append_jsonl(transcript_attempts_path, attempt)
                    if attempt["error_code"] in {"blocked_429", "blocked_403"} and retry < 3:
                        time.sleep(2**retry)
                        session, session_diag = _build_http_session()
                        continue
                    break

                if not transcript_ok and AUDIO_FALLBACK_ENABLED:
                    async with DOWNLOAD_SEM:
                        audio_path = tmp_dir / f"yt_{vid}.audio.m4a"
                        subprocess.run(["yt-dlp", "-f", "bestaudio/best", "-o", str(audio_path), src.get("url")], check=True, capture_output=True)
                    async with STT_SEM:
                        run["stage"] = "extracting"
                        run["current"] = {"video_id": vid, "stage": "extracting", "detail": "Transcribing audio"}
                        write_run(root, run_id, run)
                        write_status(root, run_id, run)
                        transcript, stt_diag = transcribe_audio_chunks(
                            video_id=vid,
                            audio_path=audio_path,
                            chunk_seconds=payload.longform.get("chunk_seconds", CHUNK_SECONDS),
                            overlap_seconds=payload.longform.get("overlap_seconds", OVERLAP_SECONDS),
                            preferred_language=payload.preferred_language,
                            tmp_dir=tmp_dir,
                        )
                        diag["stt"] = stt_diag
                elif not transcript_ok:
                    raise RuntimeError("No transcript available and audio fallback disabled")
                transcript_path.write_text(transcript, encoding="utf-8")
                global_cache.write_text(transcript, encoding="utf-8")
                diag["stages"].append("timedtext_or_fallback")

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
            write_run(root, run_id, run)
            write_status(root, run_id, run)
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
            write_run(root, run_id, run)
            write_status(root, run_id, run)
            if not ARCHIVE_AUDIO and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    pack_info = None
    if payload.brain_pack.get("build", True):
        run["progress"]["stage"] = "brain_pack"
        run["stage"] = "packing"
        write_run(root, run_id, run)
        write_status(root, run_id, run)
        pack_info = build_brain_pack(root, brain_id, run_id, payload.keyword, ingested_ids, brain.get("brain_type", "BD"))

    run["completed_at"] = utc_now()
    run["errors"] = errors
    run["ingested_video_ids"] = ingested_ids
    run["brain_pack"] = pack_info
    attempts = load_transcript_attempts(root, run_id)
    transcript_summary = selected_transcript_summary(attempts)
    selected_attempts = transcript_summary["selected_attempts"]
    transcripts_succeeded = transcript_summary["transcripts_succeeded"]
    blocked_codes = {"blocked_403", "blocked_429", "caption_probe_http_403", "caption_probe_http_429"}
    is_blocked = any((x.get("error_code") in blocked_codes) for x in selected_attempts)
    if transcripts_succeeded >= 1 and errors:
        run["status"] = "partial_success"
    elif transcripts_succeeded >= 1:
        run["status"] = "success"
    elif run.get("eligible_shortfall", 0) > 0 and run.get("selected_new", 0) == 0:
        run["status"] = "no_captions"
        run["message"] = "No caption-eligible videos found in selected candidates"
    elif is_blocked:
        run["status"] = "blocked"
        run["message"] = "Transcript requests were blocked (403/429). Verify proxy and cookie configuration."
    else:
        run["status"] = "failed" if errors else "completed"
    run["progress"]["stage"] = "completed"
    run["stage"] = "completed"
    run["current"] = None
    run["failed"] = run["progress"].get("failed", 0)
    run["completed"] = run["progress"].get("done", 0)
    write_run(root, run_id, run)
    write_status(root, run_id, run)
    write_report(root, run_id, run)


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
    write_run(root, run_id, run)
    write_status(root, run_id, run)
    await RUN_QUEUE.put({"run_id": run_id, "brain_id": brain_id, "payload": req.model_dump()})
    return {"run_id": run_id, "status": "queued"}


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    started = time.perf_counter()
    logger.info("get_run_status start run_id=%s", run_id)
    status_matches = list(BRAINS_ROOT.glob(f"*/runs/{run_id}/status.json"))
    if status_matches:
        payload = load_json(status_matches[0], {})
    else:
        legacy_matches = list(BRAINS_ROOT.glob(f"*/runs/{run_id}.json"))
        if not legacy_matches:
            raise HTTPException(status_code=404, detail="Run not found")
        payload = status_from_run(load_json(legacy_matches[0], {}))
    run_matches = list(BRAINS_ROOT.glob(f"*/runs/{run_id}/run.json"))
    if run_matches:
        run_payload = load_json(run_matches[0], {})
        root = run_matches[0].parents[2]
        attempts = load_transcript_attempts(root, run_id)
        transcript_summary = selected_transcript_summary(attempts)
        selected_attempts = transcript_summary["selected_attempts"]
        payload["transcript_failure_reasons"] = summarize_transcript_failures(selected_attempts)
        payload["sample_failures"] = sample_failures(selected_attempts)
        payload["transcript_attempts_jsonl"] = str(run_dir(root, run_id) / "transcript_attempts.jsonl")
        payload["transcripts_attempted_selected"] = transcript_summary["transcripts_attempted_selected"]
        payload["transcripts_succeeded"] = transcript_summary["transcripts_succeeded"]
        payload["transcripts_failed"] = transcript_summary["transcripts_failed"]
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info("get_run_status end run_id=%s elapsed_ms=%s", run_id, elapsed_ms)
    return payload


@app.get("/v1/runs/{run_id}/diagnostics")
def run_diagnostics(run_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    run_matches = list(BRAINS_ROOT.glob(f"*/runs/{run_id}/run.json"))
    if not run_matches:
        raise HTTPException(status_code=404, detail="Run not found")
    run_payload = load_json(run_matches[0], {})
    root = run_matches[0].parents[2]
    attempts = load_transcript_attempts(root, run_id)
    transcript_summary = selected_transcript_summary(attempts)
    selected_attempts = transcript_summary["selected_attempts"]
    return {
        "run_id": run_id,
        "brain_id": run_payload.get("brain_id"),
        "transcripts_attempted_selected": transcript_summary["transcripts_attempted_selected"],
        "transcript_failure_reasons": summarize_transcript_failures(selected_attempts),
        "sample_failures": sample_failures(selected_attempts),
        "diagnostics_path": str(run_dir(root, run_id) / "transcript_attempts.jsonl"),
    }


@app.get("/v1/runs/{run_id}/report")
def run_report(run_id: str, x_api_key: str | None = Header(default=None, alias="X-Api-Key")) -> dict[str, Any]:
    require_api_key(x_api_key)
    matches = list(BRAINS_ROOT.glob(f"*/runs/{run_id}/report.json"))
    if matches:
        return load_json(matches[0], {})
    raise HTTPException(status_code=202, detail="report_not_ready")


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
    write_status(root, run_id, run)
    write_report(root, run_id, run)
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
