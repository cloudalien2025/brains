from __future__ import annotations

from datetime import datetime
from typing import Any


TOPICS = {"API", "SEO", "Leads", "Templates", "Monetization", "URLs", "Guardrails", "Other"}
TYPES = {"rule", "fact", "tactic", "warning", "definition"}
STATUSES = {"active", "deprecated", "disputed", "experimental"}


def _is_iso_datetime(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "id",
        "brain",
        "version_introduced",
        "topic",
        "type",
        "assertion",
        "confidence",
        "status",
        "evidence",
        "created_at",
    }
    allowed = required | {"notes", "conflicts_with_ids", "tags"}

    missing = sorted(required - set(record.keys()))
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")

    extra = sorted(set(record.keys()) - allowed)
    if extra:
        errors.append(f"unexpected keys: {', '.join(extra)}")

    if record.get("topic") not in TOPICS:
        errors.append("topic must be one of API/SEO/Leads/Templates/Monetization/URLs/Guardrails/Other")
    if record.get("type") not in TYPES:
        errors.append("type must be one of rule/fact/tactic/warning/definition")

    assertion = record.get("assertion")
    if not isinstance(assertion, str) or not (10 <= len(assertion) <= 500):
        errors.append("assertion must be a string of length 10-500")

    confidence = record.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0 <= float(confidence) <= 1):
        errors.append("confidence must be a number between 0 and 1")

    status = record.get("status")
    if status not in STATUSES:
        errors.append("status must be one of active/deprecated/disputed/experimental")

    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        errors.append("evidence must be an array")
    else:
        for idx, item in enumerate(evidence):
            if not isinstance(item, dict):
                errors.append(f"evidence[{idx}] must be an object")
                continue
            if "source_id" not in item or "note" not in item:
                errors.append(f"evidence[{idx}] must include source_id and note")

    if record.get("type") == "rule" and isinstance(evidence, list) and len(evidence) < 1:
        errors.append("rule entries require evidence minItems=1")
    if status == "deprecated" and not isinstance(record.get("notes"), str):
        errors.append("deprecated entries require notes")
    if status == "disputed":
        conflicts = record.get("conflicts_with_ids")
        if not isinstance(conflicts, list) or len(conflicts) < 1:
            errors.append("disputed entries require conflicts_with_ids")

    if not _is_iso_datetime(record.get("created_at")):
        errors.append("created_at must be ISO date-time")

    return errors


def validate_source(source: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {"source_id", "source_type", "title", "channel", "url"}
    allowed = required | {"published_at", "duration_seconds"}

    missing = sorted(required - set(source.keys()))
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")
    extra = sorted(set(source.keys()) - allowed)
    if extra:
        errors.append(f"unexpected keys: {', '.join(extra)}")

    if source.get("source_type") != "youtube":
        errors.append("source_type must be youtube")
    for field in ["source_id", "title", "channel", "url"]:
        if not isinstance(source.get(field), str) or not source.get(field):
            errors.append(f"{field} must be a non-empty string")

    if "published_at" in source and source["published_at"] is not None and not _is_iso_datetime(source["published_at"]):
        errors.append("published_at must be ISO date-time when present")
    if "duration_seconds" in source and source["duration_seconds"] is not None:
        val = source["duration_seconds"]
        if not isinstance(val, int) or val < 0:
            errors.append("duration_seconds must be an integer >= 0 when present")

    return errors


def validate_run_metadata(metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {"run_id", "keyword", "started_at", "ended_at", "config", "errors", "env"}
    missing = sorted(required - set(metadata.keys()))
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")

    if not _is_iso_datetime(metadata.get("started_at")):
        errors.append("started_at must be ISO date-time")
    if not _is_iso_datetime(metadata.get("ended_at")):
        errors.append("ended_at must be ISO date-time")
    if not isinstance(metadata.get("config"), dict):
        errors.append("config must be an object")
    if not isinstance(metadata.get("errors"), list):
        errors.append("errors must be an array")
    if not isinstance(metadata.get("env"), dict):
        errors.append("env must be an object")

    return errors


def validate_pack_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {"pack_id", "brain", "keyword", "created_at", "sources", "outputs", "stats"}
    missing = sorted(required - set(manifest.keys()))
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")

    if not _is_iso_datetime(manifest.get("created_at")):
        errors.append("created_at must be ISO date-time")

    sources = manifest.get("sources")
    if not isinstance(sources, list):
        errors.append("sources must be an array")
    else:
        for idx, src in enumerate(sources):
            src_errors = validate_source(src)
            errors.extend([f"sources[{idx}] {err}" for err in src_errors])

    outputs = manifest.get("outputs")
    expected_outputs = {
        "Brain_Additions.md",
        "Brain_Core.jsonl",
        "Brain_Diff.md",
        "Sources.csv",
        "Run_Metadata.json",
    }
    if not isinstance(outputs, dict):
        errors.append("outputs must be an object")
    else:
        missing_outputs = sorted(expected_outputs - set(outputs.keys()))
        if missing_outputs:
            errors.append(f"outputs missing: {', '.join(missing_outputs)}")

    stats = manifest.get("stats")
    if not isinstance(stats, dict):
        errors.append("stats must be an object")
    else:
        for k in ["videos_discovered", "videos_queued", "records_extracted"]:
            if not isinstance(stats.get(k), int) or stats.get(k, -1) < 0:
                errors.append(f"stats.{k} must be integer >= 0")

    return errors
