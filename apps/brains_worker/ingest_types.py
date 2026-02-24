from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


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
class ItemResult:
    item_id: str
    success: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
