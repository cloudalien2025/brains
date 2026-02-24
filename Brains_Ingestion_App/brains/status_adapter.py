from __future__ import annotations

from typing import Any


def _coalesce_int(value: Any, fallback: Any = 0) -> int:
    try:
        if value is None or value == "":
            return int(fallback or 0)
        return int(value)
    except (TypeError, ValueError):
        return int(fallback or 0)


def _coalesce_float(value: Any, fallback: Any = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(fallback or 0.0)
        return float(value)
    except (TypeError, ValueError):
        return float(fallback or 0.0)


def normalize_run_status(payload: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(payload or {})
    data["transcripts_succeeded"] = _coalesce_int(data.get("transcripts_succeeded"), data.get("items_succeeded_youtube"))
    data["transcripts_failed"] = _coalesce_int(data.get("transcripts_failed"), data.get("items_failed_youtube"))
    data["total_audio_minutes"] = _coalesce_float(data.get("total_audio_minutes"), 0.0)
    data.setdefault("webdocs_failure_reasons", {})
    data.setdefault("webdocs_discovery_diagnostics", {})
    return data


def normalize_report(report: dict[str, Any] | None, run_data: dict[str, Any] | None = None) -> dict[str, Any]:
    data = dict(report or {})
    run_data = run_data or {}
    if data.get("transcripts_succeeded") is None:
        data["transcripts_succeeded"] = _coalesce_int(run_data.get("items_succeeded_youtube"), 0)
    if data.get("transcripts_failed") is None:
        data["transcripts_failed"] = _coalesce_int(run_data.get("items_failed_youtube"), 0)
    if data.get("total_audio_minutes") is None:
        data["total_audio_minutes"] = _coalesce_float(run_data.get("total_audio_minutes"), 0.0)
    return data


def report_not_ready(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    detail = report.get("detail")
    if not isinstance(detail, str):
        return False
    return detail.strip().lower() == "report_not_ready"


def report_counters(report: dict[str, Any] | None, run_data: dict[str, Any] | None = None) -> dict[str, Any]:
    report = dict(report or {})
    run_data = dict(run_data or {})

    items_succeeded_total = _coalesce_int(
        report.get("items_succeeded_total"),
        report.get("transcripts_succeeded"),
    )
    if items_succeeded_total == 0:
        items_succeeded_total = _coalesce_int(
            run_data.get("items_succeeded_total"),
            run_data.get("transcripts_succeeded"),
        )

    items_failed_total = _coalesce_int(
        report.get("items_failed_total"),
        report.get("transcripts_failed"),
    )
    if items_failed_total == 0:
        items_failed_total = _coalesce_int(
            run_data.get("items_failed_total"),
            run_data.get("transcripts_failed"),
        )

    total_audio_minutes = _coalesce_float(
        report.get("total_audio_minutes"),
        run_data.get("total_audio_minutes"),
    )

    return {
        "items_succeeded_total": items_succeeded_total,
        "items_failed_total": items_failed_total,
        "total_audio_minutes": total_audio_minutes,
    }


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _coalesce_str(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    try:
        text = str(value)
    except Exception:
        return fallback
    return text


def _extract_selected_list(run_json: dict[str, Any], keys: list[str]) -> list[Any]:
    for key in keys:
        value = run_json.get(key)
        if isinstance(value, list):
            return value
    return []


def _normalize_selected_youtube(items: list[Any]) -> list[dict[str, Any]]:
    normalized = []
    for item in items:
        if isinstance(item, str):
            vid = item
            normalized.append(
                {
                    "video_id": vid,
                    "title": "",
                    "uploader": "",
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "duration_s": None,
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        vid = _coalesce_str(item.get("video_id") or item.get("id") or item.get("source_id"))
        if not vid:
            continue
        url = item.get("url") or (f"https://www.youtube.com/watch?v={vid}" if vid else "")
        normalized.append(
            {
                "video_id": vid,
                "title": _coalesce_str(item.get("title")),
                "uploader": _coalesce_str(item.get("uploader") or item.get("channel") or item.get("channel_title")),
                "url": _coalesce_str(url),
                "duration_s": item.get("duration") or item.get("duration_s") or item.get("duration_seconds"),
            }
        )
    return normalized


def _normalize_selected_webdocs(items: list[Any]) -> list[dict[str, Any]]:
    normalized = []
    for item in items:
        if isinstance(item, str):
            doc_id = item
            normalized.append(
                {
                    "doc_id": doc_id,
                    "title": "",
                    "url": "",
                    "domain": "",
                    "provider": "",
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        doc_id = _coalesce_str(item.get("doc_id") or item.get("id"))
        if not doc_id:
            continue
        normalized.append(
            {
                "doc_id": doc_id,
                "title": _coalesce_str(item.get("title")),
                "url": _coalesce_str(item.get("url") or item.get("canonical_url")),
                "domain": _coalesce_str(item.get("domain")),
                "provider": _coalesce_str(item.get("provider")),
            }
        )
    return normalized


def _files_list(files_json: dict[str, Any], key: str) -> list[str]:
    items = _safe_list(files_json.get(key))
    results: list[str] = []
    for entry in items:
        if isinstance(entry, str):
            results.append(entry)
        elif isinstance(entry, dict):
            name = entry.get("name") or entry.get("path") or entry.get("file")
            if name:
                results.append(str(name))
    return results


def _stem_from_path(path: str) -> str:
    name = path.replace("\\", "/").split("/")[-1]
    if "." in name:
        name = name.rsplit(".", 1)[0]
    if name.startswith("yt_"):
        name = name[3:]
    return name


def normalize_run_for_display(run_json: dict[str, Any] | None, files_json: dict[str, Any] | None = None) -> dict[str, Any]:
    run_json = dict(run_json or {})
    files_json = dict(files_json or {})

    selected_youtube_raw = _extract_selected_list(
        run_json,
        [
            "selected_youtube",
            "selected_youtube_videos",
            "selected_video_candidates",
            "selected_videos",
        ],
    )
    selected_webdocs_raw = _extract_selected_list(
        run_json,
        [
            "selected_webdocs",
            "selected_webdoc_candidates",
            "selected_docs",
        ],
    )

    selected_youtube = _normalize_selected_youtube(selected_youtube_raw)
    selected_webdocs = _normalize_selected_webdocs(selected_webdocs_raw)

    report = run_json.get("report") if isinstance(run_json.get("report"), dict) else {}
    artifacts = _safe_list(report.get("artifacts"))

    succeeded_youtube: list[dict[str, Any]] = []
    seen_youtube: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        vid = _coalesce_str(artifact.get("video_id"))
        if not vid or vid in seen_youtube:
            continue
        seen_youtube.add(vid)
        succeeded_youtube.append(
            {
                "video_id": vid,
                "transcript_exists": True,
                "audio_minutes": None,
                "artifact_path": artifact.get("transcript") or artifact.get("extraction"),
            }
        )

    transcript_files = _files_list(files_json, "transcript_files_txt")
    for path in transcript_files:
        vid = _stem_from_path(path)
        if not vid or vid in seen_youtube:
            continue
        seen_youtube.add(vid)
        succeeded_youtube.append(
            {
                "video_id": vid,
                "transcript_exists": True,
                "audio_minutes": None,
                "artifact_path": path,
            }
        )

    doc_text_files = _files_list(files_json, "doc_text_files")
    succeeded_webdocs: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    for path in doc_text_files:
        doc_id = _stem_from_path(path)
        if not doc_id or doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        succeeded_webdocs.append(
            {
                "doc_id": doc_id,
                "text_exists": True,
                "artifact_path": path,
            }
        )

    failed_youtube: list[dict[str, Any]] = []
    failed_webdocs: list[dict[str, Any]] = []
    failures = _safe_list(run_json.get("sample_failures"))
    for failure in failures:
        if not isinstance(failure, dict):
            continue
        item_id = _coalesce_str(failure.get("item") or failure.get("video_id") or failure.get("doc_id") or failure.get("id"))
        if not item_id:
            continue
        stage = _coalesce_str(failure.get("stage") or failure.get("step") or "")
        error_code = _coalesce_str(failure.get("error_code") or failure.get("code") or "")
        error_message = _coalesce_str(failure.get("error") or failure.get("message") or "")
        payload = {
            "stage": stage,
            "error_code": error_code,
            "error_message": error_message,
        }
        if "webdoc" in stage or "doc" in stage:
            failed_webdocs.append({"doc_id": item_id, **payload})
        else:
            failed_youtube.append({"video_id": item_id, **payload})

    if not selected_youtube and transcript_files:
        selected_youtube = _normalize_selected_youtube([_stem_from_path(p) for p in transcript_files])
    if not selected_webdocs and doc_text_files:
        selected_webdocs = _normalize_selected_webdocs([_stem_from_path(p) for p in doc_text_files])

    return {
        "selected_youtube": selected_youtube,
        "selected_webdocs": selected_webdocs,
        "succeeded_youtube": succeeded_youtube,
        "failed_youtube": failed_youtube,
        "succeeded_webdocs": succeeded_webdocs,
        "failed_webdocs": failed_webdocs,
    }
