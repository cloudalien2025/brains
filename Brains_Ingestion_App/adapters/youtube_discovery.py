import os
from datetime import UTC, datetime, timedelta


def discover_videos(keyword: str, max_videos: int) -> list[dict]:
    api_key = os.getenv("YOUTUBE_API_KEY")
    base_count = max(1, min(max_videos, 5 if not api_key else max_videos))

    now = datetime.now(UTC)
    results: list[dict] = []
    for idx in range(base_count):
        results.append(
            {
                "source_id": f"yt_{idx + 1:03d}",
                "source_type": "youtube",
                "title": f"{keyword.title()} Strategy Breakdown #{idx + 1}",
                "channel": "Mocked Channel" if not api_key else "API-Backed Placeholder",
                "url": f"https://youtube.com/watch?v=mock{idx + 1:03d}",
                "published_at": (now - timedelta(days=idx)).isoformat(),
                "duration_seconds": 600 + (idx * 45),
            }
        )
    return results
