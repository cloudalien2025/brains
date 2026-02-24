from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apps.brains_worker.ingest_types import RunContext, VideoCandidate, DocCandidate, ItemResult
from apps.brains_worker.status import RunStatus, StatusWriter
from apps.brains_worker.transcribe import TranscriptionError, TranscriptionTimeout, ffprobe_duration, transcribe_audio
from apps.brains_worker.yt_audio import AudioDownloadError, download_audio
from apps.brains_worker.yt_search import YTSearchError, ytsearch
from apps.brains_worker.webdocs_discovery import discover_webdocs
from apps.brains_worker.webdocs_ingest import process_doc




def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def ensure_run_dirs(run_dir: Path) -> None:
    for folder in [
        "audio",
        "transcripts",
        "artifacts",
        "docs/pdf",
        "docs/html",
        "docs/text",
        "docs/meta",
    ]:
        (run_dir / folder).mkdir(parents=True, exist_ok=True)


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ingested_video_ids": [], "ingested_doc_ids": [], "records": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("ingested_video_ids", [])
    payload.setdefault("ingested_doc_ids", [])
    payload.setdefault("records", [])
    return payload


def _save_ledger(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


def _check_wallclock(run_ctx: RunContext) -> bool:
    limit = int(run_ctx.config.get("run_max_wall_s", 1800))
    return (time.time() - run_ctx.started_at) <= limit


def _check_deps() -> list[str]:
    missing = []
    if shutil.which("yt-dlp") is None:
        missing.append("yt-dlp")
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    if shutil.which("ffprobe") is None:
        missing.append("ffprobe")
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        missing.append("faster-whisper")
    return missing


def _choose_top_new_youtube(candidates: list[VideoCandidate], existing: set[str], requested_new: int) -> list[VideoCandidate]:
    selected = []
    ordered = sorted(candidates, key=lambda c: c.duration if c.duration > 0 else 10**9)
    for cand in ordered:
        if cand.video_id in existing:
            continue
        selected.append(cand)
        if len(selected) >= requested_new:
            break
    return selected


def _choose_top_new_webdocs(candidates: list[DocCandidate], existing: set[str], requested_new: int) -> list[DocCandidate]:
    selected = []
    for cand in candidates:
        if cand.doc_id in existing:
            continue
        selected.append(cand)
        if len(selected) >= requested_new:
            break
    return selected


def _transcript_paths(run_ctx: RunContext, video_id: str) -> tuple[Path, Path]:
    return run_ctx.run_dir / "transcripts" / f"{video_id}.txt", run_ctx.run_dir / "transcripts" / f"{video_id}.json"


def _legacy_transcript_path(run_ctx: RunContext, video_id: str) -> Path:
    return run_ctx.brain_root / "transcripts" / f"yt_{video_id}.txt"


def process_video(video: VideoCandidate, run_ctx: RunContext, status: StatusWriter) -> ItemResult:
    status.update(step="audio", stage_detail=f"download:{video.video_id}")
    try:
        audio_path = download_audio(
            video.video_id,
            run_ctx.run_dir / "audio",
            run_ctx.proxy_url,
            run_ctx.cookies_path,
            int(run_ctx.config.get("audio_dl_timeout_s", 180)),
            bool(run_ctx.config.get("force_ipv4", False)),
        )
    except AudioDownloadError as exc:
        status.bump("items_failed_total")
        status.bump("items_failed_youtube")
        status.bump("transcripts_failed")
        status.add_failure_reason("transcript_failure_reasons", "audio_download_failed")
        status.add_sample_failure({"item": video.video_id, "stage": "audio", "error": str(exc)})
        return ItemResult(item_id=video.video_id, success=False, error_code="audio_download_failed", error_message=str(exc))

    duration = ffprobe_duration(audio_path) or 0.0
    status.update(total_audio_minutes=status.status.total_audio_minutes + (duration / 60.0))

    status.update(step="transcribe", stage_detail=f"transcribe:{video.video_id}")
    try:
        result = transcribe_audio(
            audio_path,
            run_ctx.config.get("transcribe_language", "auto"),
            run_ctx.config.get("transcribe_model", "small"),
            int(run_ctx.config.get("transcribe_beam_size", 5)),
            bool(run_ctx.config.get("transcribe_vad", True)),
            int(run_ctx.config.get("transcribe_timeout_s", 600)),
        )
    except (TranscriptionTimeout, TranscriptionError) as exc:
        status.bump("items_failed_total")
        status.bump("items_failed_youtube")
        status.bump("transcripts_failed")
        status.add_failure_reason("transcript_failure_reasons", "transcription_failed")
        status.add_sample_failure({"item": video.video_id, "stage": "transcribe", "error": str(exc)})
        return ItemResult(item_id=video.video_id, success=False, error_code="transcription_failed", error_message=str(exc))

    text_path, json_path = _transcript_paths(run_ctx, video.video_id)
    text_path.write_text(result.text, encoding="utf-8")
    json_path.write_text(
        json.dumps({"video_id": video.video_id, "language": result.language, "segments": result.segments}, indent=2),
        encoding="utf-8",
    )
    legacy_path = _legacy_transcript_path(run_ctx, video.video_id)
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(result.text, encoding="utf-8")

    status.bump("items_succeeded_total")
    status.bump("items_succeeded_youtube")
    status.bump("transcripts_succeeded")
    return ItemResult(item_id=video.video_id, success=True, error_code=None, error_message=None)


def run_ingest_multisource(run_ctx: RunContext) -> RunStatus:
    ensure_run_dirs(run_ctx.run_dir)

    status = RunStatus(
        run_id=run_ctx.run_id,
        brain_slug=run_ctx.brain_id,
        keyword=run_ctx.keyword,
        status="processing",
        step="discovery",
        requested_new=0,
        selected_new=0,
    )
    writer = StatusWriter(run_ctx.run_dir / "status.json", status)
    writer.update()

    missing = _check_deps()
    if missing:
        writer.update(status="failed", step="completed", stage_detail=f"missing_deps:{','.join(missing)}")
        return status

    requested_youtube = int(run_ctx.config.get("youtube_requested_new", 1))
    requested_webdocs = int(run_ctx.config.get("webdoc_requested_new", 3))
    max_candidates_youtube = int(run_ctx.config.get("youtube_max_candidates", 50))
    max_candidates_webdocs = int(run_ctx.config.get("webdoc_max_candidates", 50))

    writer.update(
        requested_new=requested_youtube + requested_webdocs,
        requested_new_total=requested_youtube + requested_webdocs,
        requested_new_youtube=requested_youtube,
        requested_new_webdocs=requested_webdocs,
    )

    if not _check_wallclock(run_ctx):
        writer.update(status="failed", step="completed", stage_detail="run_timeout")
        return status

    try:
        candidates_youtube = ytsearch(
            run_ctx.keyword,
            max_candidates_youtube,
            run_ctx.proxy_url,
            run_ctx.cookies_path,
            bool(run_ctx.config.get("force_ipv4", False)),
        )
    except YTSearchError as exc:
        writer.update(status="failed", step="completed", stage_detail=str(exc))
        return status

    webdocs_candidates, provider_used, fallback_reason = discover_webdocs(
        run_ctx.keyword,
        max_candidates_webdocs,
        {
            **run_ctx.config,
            "run_dir": run_ctx.run_dir,
        },
        writer,
    )

    writer.update(
        candidates_found=len(candidates_youtube) + len(webdocs_candidates),
        candidates_found_total=len(candidates_youtube) + len(webdocs_candidates),
        candidates_found_youtube=len(candidates_youtube),
        candidates_found_webdocs=len(webdocs_candidates),
        webdocs_discovery_provider_used=provider_used,
        webdocs_fallback_reason=fallback_reason,
    )

    ledger_path = run_ctx.brain_root / "ledger.json"
    ledger = _load_ledger(ledger_path)
    existing_video_ids = set(ledger.get("ingested_video_ids", []))
    existing_doc_ids = set(ledger.get("ingested_doc_ids", []))

    selected_youtube = _choose_top_new_youtube(candidates_youtube, existing_video_ids, requested_youtube)
    selected_webdocs = _choose_top_new_webdocs(webdocs_candidates, existing_doc_ids, requested_webdocs)

    selected_youtube_payload = [
        {
            "video_id": cand.video_id,
            "title": cand.title,
            "uploader": cand.uploader,
            "duration": cand.duration,
            "url": cand.url,
        }
        for cand in selected_youtube
    ]
    selected_webdocs_payload = [
        {
            "doc_id": cand.doc_id,
            "title": cand.title,
            "url": cand.url,
            "domain": cand.domain,
            "provider": cand.provider,
        }
        for cand in selected_webdocs
    ]

    writer.update(
        selected_new=len(selected_youtube) + len(selected_webdocs),
        selected_new_total=len(selected_youtube) + len(selected_webdocs),
        selected_new_youtube=len(selected_youtube),
        selected_new_webdocs=len(selected_webdocs),
        selected_youtube=selected_youtube_payload,
        selected_webdocs=selected_webdocs_payload,
    )

    if not selected_youtube and not selected_webdocs:
        writer.update(status="completed", step="completed", stage_detail="noop")
        return status

    for video in selected_youtube:
        if not _check_wallclock(run_ctx):
            writer.update(status="failed", step="completed", stage_detail="run_timeout")
            return status
        result = process_video(video, run_ctx, writer)
        if result.success:
            existing_video_ids.add(result.item_id)

    for doc in selected_webdocs:
        if not _check_wallclock(run_ctx):
            writer.update(status="failed", step="completed", stage_detail="run_timeout")
            return status
        writer.update(step="webdocs", stage_detail=f"doc:{doc.doc_id}")
        result = process_doc(doc, run_ctx, run_ctx.config)
        if result.success:
            writer.bump("items_succeeded_total")
            writer.bump("items_succeeded_webdocs")
            existing_doc_ids.add(result.item_id)
        else:
            writer.bump("items_failed_total")
            writer.bump("items_failed_webdocs")
            writer.add_failure_reason("webdocs_failure_reasons", result.error_code or "doc_failed")
            writer.add_sample_failure({"item": doc.doc_id, "stage": "webdocs", "error": result.error_message})

    ledger["ingested_video_ids"] = sorted(existing_video_ids)
    ledger["ingested_doc_ids"] = sorted(existing_doc_ids)
    _save_ledger(ledger_path, ledger)

    if status.selected_new_total > 0 and status.items_succeeded_total == 0:
        writer.update(status="failed", step="completed", stage_detail="no_items_succeeded")
    else:
        writer.update(status="completed", step="completed", stage_detail="done")
    return status
