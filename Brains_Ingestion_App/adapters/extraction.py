from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from typing import Any

TOPICS = ["API", "SEO", "Leads", "Templates", "Monetization", "URLs", "Guardrails", "Other"]
TYPES = ["rule", "fact", "tactic", "warning", "definition"]


def _stable_id(brain: str, topic: str, record_type: str, assertion: str) -> str:
    raw = f"{brain}|{topic}|{record_type}|{assertion.strip().lower()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"brain_{digest}"


def _segment_match(assertion: str, segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not segments:
        return None
    words = {w.strip(".,:;!?()[]{}\"'").lower() for w in assertion.split() if len(w) > 3}
    best = None
    best_score = -1
    for seg in segments:
        text = seg.get("text", "").lower()
        score = sum(1 for w in words if w and w in text)
        if score > best_score:
            best_score = score
            best = seg
    return best or segments[0]


def _build_evidence(source: dict, transcript: dict, assertion: str, quote_hint: str | None = None) -> dict:
    seg = _segment_match(assertion, transcript.get("segments", []))
    start = None
    end = None
    quote = (quote_hint or "").strip()
    if seg:
        start = seg.get("start")
        duration = seg.get("duration")
        end = (start + duration) if isinstance(start, (int, float)) and isinstance(duration, (int, float)) else None
        if not quote:
            quote = seg.get("text", "")[:280]

    if transcript.get("method") == "openai_audio_transcription" and not transcript.get("segments"):
        start = None
        end = None

    return {
        "source_id": source["source_id"],
        "note": quote[:280] if quote else "Extracted from transcript",
        "url": source["url"],
        "timestamp_start": start,
        "timestamp_end": end,
        "quote": quote[:280] if quote else None,
    }


def _coerce_candidate(candidate: dict, brain: str, source: dict, transcript: dict) -> dict | None:
    topic = candidate.get("topic") if candidate.get("topic") in TOPICS else "Other"
    rtype = candidate.get("type") if candidate.get("type") in TYPES else "fact"
    assertion = (candidate.get("assertion") or "").strip()
    if len(assertion) < 10:
        return None

    evidence_item = _build_evidence(source, transcript, assertion, candidate.get("quote"))
    confidence = float(candidate.get("confidence", 0.55))

    if transcript.get("method") == "openai_audio_transcription" and evidence_item.get("timestamp_start") is None:
        confidence = min(confidence, 0.6)
        if rtype == "rule":
            rtype = "fact"

    if rtype == "rule" and evidence_item.get("timestamp_start") is None:
        return None

    confidence = max(0.0, min(confidence, 1.0))

    return {
        "id": _stable_id(brain, topic, rtype, assertion),
        "brain": brain,
        "version_introduced": "v1.0",
        "topic": topic,
        "type": rtype,
        "assertion": assertion[:500],
        "confidence": round(confidence, 2),
        "status": candidate.get("status") if candidate.get("status") in {"active", "experimental"} else "active",
        "evidence": [evidence_item],
        "created_at": datetime.now(UTC).isoformat(),
        "tags": candidate.get("tags", []) if isinstance(candidate.get("tags"), list) else [],
    }


def _extract_with_openai(brain: str, keyword: str, source: dict, transcript: dict) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = {
        "task": "Extract 5-25 candidate assertions from transcript.",
        "constraints": {
            "topic_enum": TOPICS,
            "type_enum": TYPES,
            "status_enum": ["active", "experimental"],
            "assertion": "single sentence, max 500 chars",
            "quote_max_chars": 280,
        },
        "context": {
            "brain": brain,
            "keyword": keyword,
            "source_url": source["url"],
            "transcript_excerpt": transcript.get("full_text", "")[:14000],
        },
        "output_schema": {
            "candidates": [
                {
                    "topic": "API",
                    "type": "rule",
                    "assertion": "...",
                    "confidence": 0.7,
                    "quote": "...",
                    "tags": ["..."],
                    "status": "active",
                }
            ]
        },
    }

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "You produce precise JSON only and follow constraints exactly.",
            },
            {
                "role": "user",
                "content": json.dumps(prompt),
            },
        ],
    )
    text = response.output_text
    payload = json.loads(text)
    return payload.get("candidates", [])


def _extract_lite(keyword: str, transcript: dict) -> list[dict]:
    segments = transcript.get("segments", [])
    candidates: list[dict] = []
    for idx, seg in enumerate(segments[:25]):
        text = seg.get("text", "").strip()
        if len(text) < 40:
            continue
        rtype = "tactic" if any(k in text.lower() for k in ["how", "step", "use", "build", "create"]) else "fact"
        topic = TOPICS[idx % len(TOPICS)]
        candidates.append(
            {
                "topic": topic,
                "type": rtype,
                "assertion": f"{keyword.title()}: {text[:220]}",
                "confidence": 0.45 + ((idx % 3) * 0.05),
                "quote": text[:280],
                "tags": [keyword.lower(), "lite-extractor"],
                "status": "experimental" if idx % 4 == 0 else "active",
            }
        )
        if len(candidates) >= 10:
            break
    return candidates


def extract_brain_records(
    brain: str,
    keyword: str,
    source: dict,
    transcript: dict,
    use_openai: bool | None = None,
) -> tuple[list[dict], str]:
    if use_openai is None:
        use_openai = bool(os.getenv("OPENAI_API_KEY"))
    raw_candidates: list[dict]
    if use_openai:
        try:
            raw_candidates = _extract_with_openai(brain, keyword, source, transcript)
        except Exception:
            raw_candidates = _extract_lite(keyword, transcript)
    else:
        raw_candidates = _extract_lite(keyword, transcript)

    records = []
    for candidate in raw_candidates[:25]:
        record = _coerce_candidate(candidate, brain=brain, source=source, transcript=transcript)
        if record:
            records.append(record)

    additions_lines = [f"## Source: {source['title']}", ""]
    for record in records:
        ev = record["evidence"][0]
        ts = ev.get("timestamp_start")
        ts_label = f" @ {ts:.2f}s" if isinstance(ts, (int, float)) else ""
        additions_lines.append(f"- **{record['topic']} / {record['type']}**: {record['assertion']} ({source['url']}{ts_label})")

    return records, "\n".join(additions_lines).strip() + "\n"


def extract_brain_core_records(keyword: str, transcripts: list[dict]) -> list[dict]:
    records: list[dict] = []
    for transcript in transcripts:
        source = transcript["source"]
        source_records, _ = extract_brain_records("BD_Brain", keyword, source, transcript, use_openai=False)
        records.extend(source_records)
    return records
