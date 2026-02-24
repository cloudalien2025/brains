from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


class AudioDownloadError(RuntimeError):
    pass


def _video_url(video_id: str) -> str:
    if re.match(r"^https?://", video_id):
        return video_id
    return f"https://www.youtube.com/watch?v={video_id}"


def download_audio(
    video_id: str,
    out_dir: Path,
    proxy_url: str | None,
    cookies_path: str | None,
    timeout_s: int,
    force_ipv4: bool,
) -> Path:
    if shutil.which("yt-dlp") is None:
        raise AudioDownloadError("yt-dlp not available")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    template = out_dir / f"{video_id}.%(ext)s"
    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--no-playlist",
        "-o",
        str(template),
        _video_url(video_id),
    ]
    if force_ipv4:
        cmd.insert(1, "--force-ipv4")
    if proxy_url:
        cmd.extend(["--proxy", proxy_url])
    if cookies_path:
        cmd.extend(["--cookies", cookies_path])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        raise AudioDownloadError(f"yt-dlp timeout: {exc}") from exc
    if result.returncode != 0:
        raise AudioDownloadError(result.stderr.strip() or "yt-dlp failed")
    matches = sorted(out_dir.glob(f"{video_id}.*"))
    if not matches:
        raise AudioDownloadError("yt-dlp output missing")
    return matches[0]
