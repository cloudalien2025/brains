from __future__ import annotations

from pathlib import Path
from typing import Any

import jsonschema

from brainpack.utils import load_json

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "Schemas"
BRAIN_CORE_SCHEMA = load_json(SCHEMA_DIR / "brain_core.schema.json")
RUN_METADATA_SCHEMA = load_json(SCHEMA_DIR / "run_metadata.schema.json")
SOURCES_SCHEMA = load_json(SCHEMA_DIR / "sources.schema.json")
BRAIN_PACK_SCHEMA = load_json(SCHEMA_DIR / "brain_pack.schema.json")

_RESOLVER = jsonschema.RefResolver.from_schema(
    BRAIN_PACK_SCHEMA,
    store={
        BRAIN_CORE_SCHEMA.get("$id"): BRAIN_CORE_SCHEMA,
        RUN_METADATA_SCHEMA.get("$id"): RUN_METADATA_SCHEMA,
        SOURCES_SCHEMA.get("$id"): SOURCES_SCHEMA,
        "sources.schema.json": SOURCES_SCHEMA,
    },
)


def _collect_errors(schema: dict[str, Any], payload: dict[str, Any], resolver: jsonschema.RefResolver | None = None) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema, resolver=resolver)
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    return [f"{'/'.join([str(p) for p in err.path]) or '$'}: {err.message}" for err in errors]


def validate_record(record: dict[str, Any]) -> list[str]:
    errors = _collect_errors(BRAIN_CORE_SCHEMA, record)
    if record.get("type") == "rule":
        has_timestamp = any(
            isinstance(item, dict) and item.get("url") and item.get("timestamp_start") is not None
            for item in record.get("evidence", [])
        )
        if not has_timestamp:
            errors.append("$: rule entries must include evidence with url + timestamp_start")
    return errors


def validate_source(source: dict[str, Any]) -> list[str]:
    return _collect_errors(SOURCES_SCHEMA, source)


def validate_run_metadata(metadata: dict[str, Any]) -> list[str]:
    return _collect_errors(RUN_METADATA_SCHEMA, metadata)


def validate_pack_manifest(manifest: dict[str, Any]) -> list[str]:
    return _collect_errors(BRAIN_PACK_SCHEMA, manifest, resolver=_RESOLVER)
