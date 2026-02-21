def transcribe(source: dict) -> dict:
    return {
        "source_id": source["source_id"],
        "text": (
            f"Placeholder transcript for {source['title']}. "
            "This transcript is synthetic and intended for MVP testing."
        ),
    }
