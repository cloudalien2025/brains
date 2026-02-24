from __future__ import annotations

import os
from typing import Any

from apps.brains_worker.ingest_types import DocCandidate


def is_serpapi_configured() -> bool:
    return bool((os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY") or "").strip())


def discover_webdocs(
    keyword: str,
    max_candidates: int,
    config: dict[str, Any],
    status_writer: Any,
) -> tuple[list[DocCandidate], str, str | None]:
    provider = (config.get("webdoc_primary") or "ddg").strip()
    fallback_reason = "not_implemented"
    return [], provider, fallback_reason
