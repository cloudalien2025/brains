from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any

import requests

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


@dataclass
class DiscoveryError(Exception):
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass
class DiscoveryOutcome:
    candidates: list[dict[str, Any]]
    method: str
    youtube_api_http_status: int | None = None


def _parse_iso8601_duration(duration: str) -> int | None:
    if not duration or not duration.startswith("P"):
        return None

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


def _score_candidate(item: dict[str, Any]) -> float:
    stats = item.get("statistics", {})
    view_count = int(stats.get("viewCount", 0) or 0)
    return float(view_count)


def _discover_via_youtube_api(
    keyword: str,
    max_candidates: int,
    published_after: str | None,
    language: str | None,
    order: str,
    api_key: str,
) -> DiscoveryOutcome:
    max_results = max(1, min(int(max_candidates), 50))
    search_params: dict[str, Any] = {
        "part": "snippet",
        "q": keyword,
        "type": "video",
        "maxResults": max_results,
        "order": order,
        "key": api_key,
    }
    if published_after:
        search_params["publishedAfter"] = published_after
    if language:
        search_params["relevanceLanguage"] = language

    try:
        search_resp = requests.get(YOUTUBE_SEARCH_URL, params=search_params, timeout=30)
    except requests.RequestException as exc:
        raise DiscoveryError("DISCOVERY_YT_API_FAILED", f"YouTube search API failed: {exc}") from exc

    if search_resp.status_code == 403 and "quota" in search_resp.text.lower():
        raise DiscoveryError("DISCOVERY_QUOTA_EXCEEDED", "YouTube API quota exceeded")
    if search_resp.status_code >= 400:
        raise DiscoveryError("DISCOVERY_YT_API_FAILED", f"YouTube search API HTTP {search_resp.status_code}")

    search_payload = search_resp.json()
    video_ids = [
        item.get("id", {}).get("videoId")
        for item in search_payload.get("items", [])
        if item.get("id", {}).get("videoId")
    ]
    if not video_ids:
        return DiscoveryOutcome(candidates=[], method="youtube_api", youtube_api_http_status=search_resp.status_code)

    try:
        details_resp = requests.get(
            YOUTUBE_VIDEOS_URL,
            params={
                "part": "snippet,contentDetails,statistics",
                "id": ",".join(video_ids),
                "key": api_key,
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise DiscoveryError("DISCOVERY_YT_API_FAILED", f"YouTube videos API failed: {exc}") from exc

    if details_resp.status_code == 403 and "quota" in details_resp.text.lower():
        raise DiscoveryError("DISCOVERY_QUOTA_EXCEEDED", "YouTube API quota exceeded")
    if details_resp.status_code >= 400:
        raise DiscoveryError("DISCOVERY_YT_API_FAILED", f"YouTube videos API HTTP {details_resp.status_code}")

    by_id = {item.get("id"): item for item in details_resp.json().get("items", [])}

    candidates: list[dict[str, Any]] = []
    for vid in video_ids:
        item = by_id.get(vid)
        if not item:
            continue
        snippet = item.get("snippet", {})
        candidates.append(
            {
                "video_id": vid,
                "title": snippet.get("title", "Untitled Video"),
                "channel_id": snippet.get("channelId"),
                "channel_title": snippet.get("channelTitle", "Unknown Channel"),
                "published_at": snippet.get("publishedAt"),
                "description": snippet.get("description", ""),
                "url": f"https://www.youtube.com/watch?v={vid}",
                "thumbnails": snippet.get("thumbnails", {}),
                "source": "youtube_data_api",
                "duration_seconds": _parse_iso8601_duration(item.get("contentDetails", {}).get("duration", "")),
                "view_count": int(item.get("statistics", {}).get("viewCount", 0) or 0),
                "score": _score_candidate(item),
            }
        )
    return DiscoveryOutcome(candidates=candidates, method="youtube_api", youtube_api_http_status=details_resp.status_code)


def _discover_via_fallback(keyword: str, max_candidates: int) -> DiscoveryOutcome:
    limit = max(1, min(int(max_candidates), 25))
    cmd = [
        "yt-dlp",
        f"ytsearch{limit}:{keyword}",
        "--skip-download",
        "--flat-playlist",
        "--dump-json",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise DiscoveryError("DISCOVERY_FALLBACK_FAILED", f"Fallback discovery failed: {exc}") from exc

    candidates: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        video_id = payload.get("id")
        if not video_id:
            continue
        candidates.append(
            {
                "video_id": video_id,
                "title": payload.get("title", "Untitled Video"),
                "channel_title": payload.get("channel", "Unknown Channel"),
                "published_at": None,
                "description": payload.get("description", ""),
                "url": payload.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
                "source": "fallback_ytdlp",
                "score": 0,
            }
        )
    return DiscoveryOutcome(candidates=candidates, method="fallback", youtube_api_http_status=None)


def discover_youtube_videos(
    keyword: str,
    max_candidates: int,
    published_after: str | None,
    language: str | None,
    order: str = "relevance",
) -> DiscoveryOutcome:
    api_key = (os.getenv("YOUTUBE_API_KEY") or "").strip()
    if api_key:
        return _discover_via_youtube_api(keyword, max_candidates, published_after, language, order, api_key)
    return _discover_via_fallback(keyword, max_candidates)
