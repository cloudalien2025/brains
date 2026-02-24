from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from typing import Literal


@dataclass
class RunContext:
    run_id: str
    brain_id: str
    keyword: str
    brain_root: Path
    run_dir: Path
    payload: Dict[str, Any]
    proxies: Dict[str, str] = field(default_factory=dict)
    proxy_url: Optional[str] = None
    cookies_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0


@dataclass
class VideoCandidate:
    video_id: str
    title: Optional[str] = None
    uploader: Optional[str] = None
    duration: float = 0.0
    url: Optional[str] = None


@dataclass
class DocCandidate:
    doc_id: str
    title: Optional[str] = None
    url: Optional[str] = None
    domain: Optional[str] = None
    provider: Optional[str] = None


@dataclass
class WebdocCandidate:
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    provider: Literal["ddg", "serpapi"] = "ddg"
    rank: int = 0


@dataclass
class WebdocDiscoveryDiagnostics:
    provider_used: str
    parse_status: str
    http_status: int | str | None = None
    elapsed_ms: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    result_count_raw: int | None = None
    result_count_usable: int | None = None


@dataclass
class WebdocIngestDiagnostics:
    url: str
    http_status: int | str | None = None
    content_type: str | None = None
    elapsed_ms: int | None = None
    extract_status: str = "error"
    extracted_chars: int = 0
    error_code: str | None = None
    error_message: str | None = None


@dataclass
class ItemResult:
    item_id: str
    success: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
