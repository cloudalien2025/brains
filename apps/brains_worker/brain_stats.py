from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BRAINS_DATA_DIR = Path("/opt/brains-data")
DEFAULT_CAPACITY_ITEMS = 500


def _capacity_items() -> int:
    raw = os.getenv("BRAIN_CAPACITY_ITEMS", "")
    if not raw:
        return DEFAULT_CAPACITY_ITEMS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_CAPACITY_ITEMS
    return value if value > 0 else DEFAULT_CAPACITY_ITEMS


def _collect_files(runs_root: Path, patterns: Iterable[str]) -> set[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        for path in runs_root.glob(pattern):
            if path.is_file():
                files.add(path)
    return files


def _latest_run_id_from_status(runs_root: Path) -> str | None:
    status_files = [path for path in runs_root.glob("*/status.json") if path.is_file()]
    if status_files:
        latest = max(status_files, key=lambda path: path.stat().st_mtime)
        return latest.parent.name
    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    if run_dirs:
        latest_dir = max(run_dirs, key=lambda path: path.stat().st_mtime)
        return latest_dir.name
    return None


def compute_brain_stats(brain_slug: str, base_dir: Path | None = None) -> dict[str, Any]:
    base_dir = Path(base_dir) if base_dir else Path(os.getenv("BRAINS_DATA_DIR", DEFAULT_BRAINS_DATA_DIR))
    runs_root = base_dir / "brains" / brain_slug / "runs"

    youtube_files: set[Path] = set()
    webdocs_files: set[Path] = set()
    if runs_root.exists():
        youtube_files = _collect_files(runs_root, ["*/transcripts/*.txt"])
        webdocs_files = _collect_files(
            runs_root,
            ["*/docs/text/*.txt", "*/webdocs/text/*.txt", "*/docs/*.txt"],
        )

    youtube_items = len(youtube_files)
    webdocs_items = len(webdocs_files)
    total_items = youtube_items + webdocs_items
    capacity_items = _capacity_items()
    fill_pct = min(total_items / capacity_items, 1.0) if capacity_items > 0 else 0.0

    last_run_id = _latest_run_id_from_status(runs_root) if runs_root.exists() else None

    return {
        "brain_slug": brain_slug,
        "capacity_items": capacity_items,
        "total_items": total_items,
        "youtube_items": youtube_items,
        "webdocs_items": webdocs_items,
        "fill_pct": fill_pct,
        "last_run_id": last_run_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
