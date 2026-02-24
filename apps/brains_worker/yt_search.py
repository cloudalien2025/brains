from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apps.brains_worker.discovery.youtube_api import DiscoveryError, discover_youtube_videos
from apps.brains_worker.ingest_types import VideoCandidate


@dataclass
class YTSearchError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message


def ytsearch(
    keyword: str,
    max_candidates: int,
    proxy_url: str | None,
    cookies_path: str | None,
    force_ipv4: bool,
) -> list[VideoCandidate]:
    try:
        outcome = discover_youtube_videos(
            keyword=keyword,
            max_candidates=max_candidates,
            published_after=None,
            language=None,
            order="relevance",
        )
    except DiscoveryError as exc:
        raise YTSearchError(str(exc)) from exc

    candidates: list[VideoCandidate] = []
    for item in outcome.candidates:
        candidates.append(
            VideoCandidate(
                video_id=str(item.get("video_id") or ""),
                title=item.get("title"),
                uploader=item.get("channel_title") or item.get("uploader"),
                duration=float(item.get("duration_seconds") or item.get("duration") or 0.0),
                url=item.get("url"),
            )
        )
    return candidates
