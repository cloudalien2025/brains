from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_sources_csv(path: Path, sources: list[dict]) -> None:
    fieldnames = [
        "source_id",
        "source_type",
        "title",
        "channel",
        "url",
        "published_at",
        "duration_seconds",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source in sources:
            writer.writerow(source)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_pack_zip(files: dict[str, str]) -> bytes:
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return memory.getvalue()
