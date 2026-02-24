from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


@dataclass
class RunStatus:
    run_id: str
    brain_slug: str
    keyword: str
    status: str = "queued"
    step: str = "queued"
    stage_detail: str | None = None
    candidates_found: int = 0
    requested_new: int = 0
    selected_new: int = 0
    transcripts_succeeded: int = 0
    transcripts_failed: int = 0
    total_audio_minutes: float = 0.0
    transcript_failure_reasons: dict[str, int] = field(default_factory=dict)
    sample_failures: list[dict[str, Any]] = field(default_factory=list)
    candidates_found_total: int = 0
    candidates_found_youtube: int = 0
    candidates_found_webdocs: int = 0
    requested_new_total: int = 0
    requested_new_youtube: int = 0
    requested_new_webdocs: int = 0
    selected_new_total: int = 0
    selected_new_youtube: int = 0
    selected_new_webdocs: int = 0
    selected_youtube: list[dict[str, Any]] = field(default_factory=list)
    selected_webdocs: list[dict[str, Any]] = field(default_factory=list)
    items_succeeded_total: int = 0
    items_succeeded_youtube: int = 0
    items_succeeded_webdocs: int = 0
    items_failed_total: int = 0
    items_failed_youtube: int = 0
    items_failed_webdocs: int = 0
    webdocs_failure_reasons: dict[str, int] = field(default_factory=dict)
    webdocs_discovery_provider_used: str | None = None
    webdocs_fallback_reason: str | None = None
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "brain_slug": self.brain_slug,
            "keyword": self.keyword,
            "status": self.status,
            "step": self.step,
            "stage_detail": self.stage_detail,
            "candidates_found": self.candidates_found,
            "requested_new": self.requested_new,
            "selected_new": self.selected_new,
            "transcripts_succeeded": self.transcripts_succeeded,
            "transcripts_failed": self.transcripts_failed,
            "total_audio_minutes": round(self.total_audio_minutes, 3),
            "transcript_failure_reasons": self.transcript_failure_reasons,
            "sample_failures": self.sample_failures,
            "candidates_found_total": self.candidates_found_total,
            "candidates_found_youtube": self.candidates_found_youtube,
            "candidates_found_webdocs": self.candidates_found_webdocs,
            "requested_new_total": self.requested_new_total,
            "requested_new_youtube": self.requested_new_youtube,
            "requested_new_webdocs": self.requested_new_webdocs,
            "selected_new_total": self.selected_new_total,
            "selected_new_youtube": self.selected_new_youtube,
            "selected_new_webdocs": self.selected_new_webdocs,
            "selected_youtube": self.selected_youtube,
            "selected_webdocs": self.selected_webdocs,
            "items_succeeded_total": self.items_succeeded_total,
            "items_succeeded_youtube": self.items_succeeded_youtube,
            "items_succeeded_webdocs": self.items_succeeded_webdocs,
            "items_failed_total": self.items_failed_total,
            "items_failed_youtube": self.items_failed_youtube,
            "items_failed_webdocs": self.items_failed_webdocs,
            "webdocs_failure_reasons": self.webdocs_failure_reasons,
            "webdocs_discovery_provider_used": self.webdocs_discovery_provider_used,
            "webdocs_fallback_reason": self.webdocs_fallback_reason,
            "updated_at": self.updated_at,
        }


class StatusWriter:
    def __init__(self, status_path: Path, status: RunStatus):
        self.status_path = status_path
        self.status = status

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self.status, key):
                setattr(self.status, key, value)
        self.status.updated_at = utc_now()
        _write_json(self.status_path, self.status.to_dict())

    def bump(self, field: str, amount: int = 1) -> None:
        if hasattr(self.status, field):
            current = getattr(self.status, field)
            if isinstance(current, (int, float)):
                setattr(self.status, field, current + amount)
        self.status.updated_at = utc_now()
        _write_json(self.status_path, self.status.to_dict())

    def add_failure_reason(self, field: str, reason: str) -> None:
        if not hasattr(self.status, field):
            return
        current = getattr(self.status, field)
        if isinstance(current, dict):
            current[reason] = int(current.get(reason) or 0) + 1
        self.status.updated_at = utc_now()
        _write_json(self.status_path, self.status.to_dict())

    def add_sample_failure(self, payload: dict[str, Any]) -> None:
        self.status.sample_failures.append(payload)
        self.status.updated_at = utc_now()
        _write_json(self.status_path, self.status.to_dict())
