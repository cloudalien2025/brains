import csv
import json
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
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source in sources:
            writer.writerow(source)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
