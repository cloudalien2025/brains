from __future__ import annotations

import os
from typing import Any

import requests

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


def _parse_iso8601_duration(duration: str) -> int | None:
    if not duration or not duration.startswith("P"):
        return None

    # Lightweight ISO8601 duration parser for YouTube patterns like PT1H2M3S
    total = 0
    number = ""
    in_time = False
    for ch in duration[1:]:
        if ch == "T":
            in_time = True
            continue
        if ch.isdigit():
            number += ch
            continue
        if not number:
            continue
        value = int(number)
        number = ""
        if ch == "D":
            total += value * 86400
        elif in_time and ch == "H":
            total += value * 3600
        elif in_time and ch == "M":
            total += value * 60
        elif in_time and ch == "S":
            total += value
    return total


def discover_youtube_videos(keyword: str, max_videos: int) -> list[dict[str, Any]]:
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY is required for real YouTube discovery.")

    max_results = max(1, min(int(max_videos), 50))
    search_params = {
        "part": "snippet",
        "q": keyword,
        "type": "video",
        "maxResults": max_results,
        "key": api_key,
    }
    search_resp = requests.get(YOUTUBE_SEARCH_URL, params=search_params, timeout=30)
    search_resp.raise_for_status()
    search_payload = search_resp.json()

    video_ids = [
        item.get("id", {}).get("videoId")
        for item in search_payload.get("items", [])
        if item.get("id", {}).get("videoId")
    ]
    if not video_ids:
        return []

    details_resp = requests.get(
        YOUTUBE_VIDEOS_URL,
        params={
            "part": "contentDetails,snippet",
            "id": ",".join(video_ids),
            "key": api_key,
        },
        timeout=30,
    )
    details_resp.raise_for_status()
    details_payload = details_resp.json()

    by_id = {item.get("id"): item for item in details_payload.get("items", [])}

    results: list[dict[str, Any]] = []
    for vid in video_ids:
        item = by_id.get(vid)
        if not item:
            continue
        snippet = item.get("snippet", {})
        duration = _parse_iso8601_duration(item.get("contentDetails", {}).get("duration", ""))
        results.append(
            {
                "source_id": f"yt:{vid}",
                "source_type": "youtube",
                "title": snippet.get("title", "Untitled Video"),
                "channel": snippet.get("channelTitle", "Unknown Channel"),
                "url": f"https://www.youtube.com/watch?v={vid}",
                "published_at": snippet.get("publishedAt"),
                "duration_seconds": duration,
            }
        )
    return results


def discover_videos(keyword: str, max_videos: int) -> list[dict[str, Any]]:
    return discover_youtube_videos(keyword=keyword, max_videos=max_videos)
