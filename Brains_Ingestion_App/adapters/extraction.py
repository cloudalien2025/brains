from datetime import UTC, datetime


TOPICS = ["API", "SEO", "Leads", "Templates", "Monetization", "URLs", "Guardrails", "Other"]
TYPES = ["rule", "fact", "tactic", "warning", "definition"]


def extract_brain_core_records(keyword: str, transcripts: list[dict]) -> list[dict]:
    now = datetime.now(UTC).isoformat()
    count = max(3, min(5, len(transcripts)))
    records: list[dict] = []

    for idx in range(count):
        record_type = TYPES[idx % len(TYPES)]
        status = "experimental" if idx == count - 1 else "active"
        rec = {
            "id": f"brain_{keyword.lower().replace(' ', '_')}_{idx + 1:03d}",
            "brain": "BD_Brain",
            "version_introduced": "v1.0",
            "topic": TOPICS[idx % len(TOPICS)],
            "type": record_type,
            "assertion": (
                f"{keyword.title()} insight {idx + 1}: maintain consistent capture, review, "
                "and ranking signals across validated source material."
            ),
            "confidence": round(0.65 + (idx * 0.05), 2),
            "status": status,
            "evidence": [
                {
                    "source_id": transcripts[idx % len(transcripts)]["source_id"],
                    "note": "Derived from placeholder transcript content.",
                }
            ],
            "created_at": now,
            "tags": [keyword.lower(), "mvp"],
        }
        records.append(rec)

    return records
