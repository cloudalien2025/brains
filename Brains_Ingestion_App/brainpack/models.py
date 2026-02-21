from dataclasses import dataclass
from typing import Any


@dataclass
class SourceRecord:
    source_id: str
    source_type: str
    title: str
    channel: str
    url: str
    published_at: str | None = None
    duration_seconds: int | None = None


@dataclass
class BrainCoreRecord:
    id: str
    brain: str
    version_introduced: str
    topic: str
    type: str
    assertion: str
    confidence: float
    status: str
    evidence: list[dict[str, Any]]
    created_at: str
    notes: str | None = None
    conflicts_with_ids: list[str] | None = None
    tags: list[str] | None = None
