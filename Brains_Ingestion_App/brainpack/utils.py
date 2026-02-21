import json
import re
from datetime import UTC, datetime
from pathlib import Path


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "keyword"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
