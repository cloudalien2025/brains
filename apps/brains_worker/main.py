from __future__ import annotations

import json
import html
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from Brains_Ingestion_App.brains.net.proxy_manager import ProxyManager

app = FastAPI(title="Brains Worker")
logger = logging.getLogger(__name__)

ALLOWED_STT_MODELS = {"gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"}
HTTPStatusValue = int | str

# Hard limit for yt-dlp subtitle extraction so the worker can respond within Streamlit's 30s timeout.
YTDLP_SUBS_TIMEOUT_S = int(os.getenv("BRAINS_YTDLP_SUBS_TIMEOUT_S", "12"))
YT_COOKIES_PATH = Path(os.getenv("BRAINS_YTDLP_COOKIES_PATH", "/opt/brains-worker/cookies/youtube_cookies.txt"))
YT_COOKIE_MAX_AGE_S = int(os.getenv("BRAINS_YTDLP_COOKIES_MAX_AGE_S", str(12 * 60 * 60)))
YT_COOKIE_BOOTSTRAP_TIMEOUT_S = int(os.getenv("BRAINS_YTDLP_COOKIE_BOOTSTRAP_TIMEOUT_S", "12"))
YT_COOKIE_BOOTSTRAP_SCRIPT = Path(os.getenv("BRAINS_YTDLP_COOKIE_BOOTSTRAP_SCRIPT", "tools/bootstrap_youtube_cookies.py"))


class TranscriptRequest(BaseModel):
    video_id: str | None = Field(default=None, examples=["yt:5lQf89-AeFo"])
    source_id: str | None = Field(default=None, examples=["yt:5lQf89-AeFo"])
    url: str | None = None
    preferred_language: str | None = None
    captions_enabled: bool = True
    yt_dlp_subs_enabled: bool = True
    audio_fallback_enabled: bool = False
    allow_audio_fallback: bool | None = None
    stt_provider: str = "openai"
    stt_model: str = "gpt-4o-mini-transcribe"
    stt_language: str | None = None
    stt_prompt: str | None = None
    debug_keep_files: bool = False
    debug_include_artifact_paths: bool = False
    proxy_enabled: bool | None = None
    proxy_country: str | None = None
    proxy_sticky: bool = True
    proxy_url: str | None = None


def _tail(value: str | None, limit: int = 300) -> str:
    return (value or "")[-limit:]


def _sanitize_sniff(value: str | None, limit: int = 800) -> str:
    if not value:
        return ""
    cleaned = "".join(ch if ch.isprintable() or ch in {"\n", "\r", "\t"} else " " for ch in value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:limit]


def _classify_caption_sniff(sniff: str) -> str:
    compact = (sniff or "").strip()
    if not compact:
        return "EMPTY"
    upper = compact[:160].upper()
    if upper.startswith("WEBVTT"):
        return "WEBVTT"
    if upper.startswith("{") or ('"EVENTS"' in upper and "{" in upper):
        return "JSON"
    if upper.startswith("<"):
        if "<HTML" in upper or "<!DOCTYPE HTML" in upper:
            return "HTML"
        if "<TT" in upper or "<TRANSCRIPT" in upper or "<?XML" in upper:
            return "XML"
        return "UNKNOWN"
    if "CONSENT.YOUTUBE.COM" in upper or "<FORM" in upper:
        return "HTML"
    return "UNKNOWN"


def _redact_ytdlp_stderr(stderr: str | None, limit: int = 500) -> str:
    if not stderr:
        return ""
    redacted = re.sub(r"(https?://)([^\s/@:]+):([^\s/@]+)@", r"\1***:***@", stderr)
    redacted = re.sub(r"(?i)(signature|sig|sparams|lsig|key|expire|token)=([^&\s]+)", r"\1=<redacted>", redacted)
    redacted = redacted.replace(str(YT_COOKIES_PATH), "<cookies_path>")
    return _sanitize_sniff(redacted, limit)


def _cookie_file_age_seconds(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return max(0, int(time.time() - path.stat().st_mtime))
    except Exception:
        return None


def _cookie_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        count = 0
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
        return count
    except Exception:
        return 0


def _ensure_youtube_cookies() -> dict[str, Any]:
    diag = {
        "cookies_present": YT_COOKIES_PATH.exists(),
        "cookies_age_seconds": _cookie_file_age_seconds(YT_COOKIES_PATH),
        "cookies_bootstrap_attempted": False,
        "cookies_bootstrap_status": "skipped",
        "cookies_bootstrap_elapsed_ms": 0,
        "cookies_cookie_count": _cookie_count(YT_COOKIES_PATH),
    }

    stale = (diag["cookies_age_seconds"] is None) or (diag["cookies_age_seconds"] > YT_COOKIE_MAX_AGE_S)
    if not stale:
        return diag

    script = YT_COOKIE_BOOTSTRAP_SCRIPT
    if not script.is_absolute():
        script = Path(__file__).resolve().parents[2] / script
    if not script.exists():
        return diag

    diag["cookies_bootstrap_attempted"] = True
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            ["python", str(script), "--output", str(YT_COOKIES_PATH)],
            capture_output=True,
            text=True,
            timeout=YT_COOKIE_BOOTSTRAP_TIMEOUT_S,
        )
        diag["cookies_bootstrap_status"] = "success" if proc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        diag["cookies_bootstrap_status"] = "timeout"
    except Exception:
        diag["cookies_bootstrap_status"] = "failed"

    diag["cookies_bootstrap_elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    diag["cookies_present"] = YT_COOKIES_PATH.exists()
    diag["cookies_age_seconds"] = _cookie_file_age_seconds(YT_COOKIES_PATH)
    diag["cookies_cookie_count"] = _cookie_count(YT_COOKIES_PATH)
    return diag


def _build_caption_url_fmt(track: dict[str, Any], fmt: str = "vtt") -> str:
    lang = track.get("languageCode") or ""
    kind = track.get("kind") or ""
    raw_name = track.get("name")
    name = (raw_name.get("simpleText") if isinstance(raw_name, dict) else raw_name) or ""
    return f"lang={lang}&kind={kind}&name={name}&fmt={fmt}"


def _parse_json3_subtitle_to_text(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except Exception:
        return ""
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return ""

    lines: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        segs = event.get("segs")
        if not isinstance(segs, list):
            continue
        parts: list[str] = []
        for seg in segs:
            if isinstance(seg, dict):
                utf8 = seg.get("utf8")
                if isinstance(utf8, str):
                    cleaned = _strip_caption_markup(utf8)
                    if cleaned:
                        parts.append(cleaned)
        line = " ".join(parts).strip()
        if line:
            lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _safe_ytdlp_available() -> bool:
    path = shutil.which("yt-dlp")
    if path:
        return True
    try:
        return subprocess.run(["yt-dlp", "--version"], capture_output=True, timeout=2).returncode == 0
    except Exception:
        return False


def _safe_ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg"))


def _detect_git_commit() -> str:
    env_commit = (os.getenv("BRAINS_GIT_COMMIT") or os.getenv("GIT_COMMIT") or "").strip()
    if env_commit:
        return env_commit[:12]
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            return (result.stdout or "").strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def _version_payload() -> dict[str, Any]:
    return {
        "service": "brains-worker",
        "git_commit": _detect_git_commit(),
        "build_time_utc": os.getenv("BRAINS_BUILD_TIME_UTC", "unknown"),
        "pipeline": {
            "player_json_captions": True,
            "ytdlp_subtitles": True,
            "audio_openai_stt": True,
        },
        "yt_dlp_available": _safe_ytdlp_available(),
        "ffmpeg_available": _safe_ffmpeg_available(),
        "openai_key_present": bool((os.getenv("OPENAI_API_KEY") or "").strip()),
    }


def _expected_api_key() -> str:
    return (os.getenv("BRAINS_API_KEY") or os.getenv("BRAINS_WORKER_API_KEY") or "").strip()


def _auth_error() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"error_code": "WORKER_AUTH_FAILED", "error": "Missing or invalid x-api-key", "diagnostics": {}},
    )


def _error(status_code: int, code: str, message: str, diagnostics: dict[str, Any]) -> JSONResponse:
    diagnostics["final_error_code"] = code
    diagnostics["final_error"] = message
    return JSONResponse(status_code=status_code, content={"error_code": code, "error": message, "diagnostics": diagnostics})


def _build_proxy_url(payload: TranscriptRequest, video_id: str) -> str | None:
    if payload.proxy_url:
        return payload.proxy_url

    proxy_manager = ProxyManager()
    try:
        managed = proxy_manager.get_proxies(proxy_session_key=video_id)
        managed_url = (managed or {}).get("https") or (managed or {}).get("http")
    except Exception:
        managed_url = None

    if payload.proxy_enabled is False:
        return None

    # If ProxyManager provided a proxy and proxy_enabled is not explicitly True, prefer managed proxy.
    if managed_url and payload.proxy_enabled is not True:
        return managed_url

    enabled_flag = payload.proxy_enabled is True or (os.getenv("BRAINS_PROXY_ENABLED") or "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled_flag:
        return managed_url

    host = (os.getenv("DECODO_HOST") or os.getenv("DECDO_HOST") or "").strip() or "gate.decodo.com"
    port = (os.getenv("DECODO_PORT") or os.getenv("DECDO_PORT") or "").strip() or "7000"
    username = (os.getenv("DECODO_USERNAME") or os.getenv("DECDO_USERNAME") or "").strip()
    password = (os.getenv("DECODO_PASSWORD") or os.getenv("DECDO_PASSWORD") or "").strip()
    if not (username and password):
        return managed_url

    if payload.proxy_country:
        username = f"{username}-country-{payload.proxy_country.lower()}"

    if payload.proxy_sticky:
        safe_session = "".join(ch for ch in video_id if ch.isalnum())[:24] or "sticky"
        username = f"{username}-session-{safe_session}"

    return f"http://{username}:{password}@{host}:{port}"


def _extract_video_id(video_id: str | None, source_id: str | None, url: str | None) -> str:
    candidates = [video_id, source_id]
    for candidate in candidates:
        if not candidate:
            continue
        cleaned = candidate.strip()
        if cleaned.startswith("yt:"):
            cleaned = cleaned.split(":", 1)[1].strip()
        if "youtube.com" in cleaned or "youtu.be" in cleaned:
            url = cleaned
            break
        if re.match(r"^[A-Za-z0-9_-]{6,}$", cleaned):
            return cleaned

    if url:
        parsed = urlparse(url.strip())
        host = parsed.netloc.lower()
        if "youtu.be" in host:
            short_id = parsed.path.strip("/").split("/")[0]
            if re.match(r"^[A-Za-z0-9_-]{6,}$", short_id):
                return short_id
        qv = (parse_qs(parsed.query).get("v") or [""])[0]
        if re.match(r"^[A-Za-z0-9_-]{6,}$", qv):
            return qv

    return ""


def _youtube_watch_player_json(video_id: str, proxy: str | None) -> tuple[HTTPStatusValue, dict[str, Any] | None, str | None]:
    proxies = {"http": proxy, "https": proxy} if proxy else None
    url = f"https://www.youtube.com/watch?v={video_id}&hl=en"
    try:
        # Keep this lower so captions stage can't eat the full Streamlit budget.
        response = requests.get(url, timeout=10, proxies=proxies, headers={"User-Agent": "Mozilla/5.0"})
        status = response.status_code
        if status in {429, 403}:
            return status, None, "PLAYER_JSON_BLOCKED"
        if status != 200:
            return status, None, "PLAYER_JSON_FETCH_FAILED"

        text = response.text or ""
        if "consent.youtube.com" in text:
            return status, None, "PLAYER_JSON_CONSENT_REQUIRED"

        match = re.search(r"ytInitialPlayerResponse\s*=\s*(\{.+?\})\s*;", text)
        if not match:
            return status, None, "PLAYER_JSON_MISSING"

        return status, json.loads(match.group(1)), None
    except Exception:
        return "exception", None, "PLAYER_JSON_EXCEPTION"


def _choose_caption_track(tracks: list[dict[str, Any]], preferred_language: str | None) -> dict[str, Any] | None:
    if not tracks:
        return None
    pref = (preferred_language or "").lower()

    def rank(track: dict[str, Any]) -> tuple[int, int, int]:
        lang = (track.get("languageCode") or "").lower()
        exact = 0 if pref and (lang == pref or lang.startswith(f"{pref}-")) else 1
        english = 0 if lang.startswith("en") else 1
        auto = 1 if (track.get("kind") == "asr") else 0  # prefer human captions over ASR
        return exact, english, auto

    return sorted(tracks, key=rank)[0]


def _strip_caption_markup(text: str) -> str:
    # Remove common WebVTT/YouTube caption tags (best effort).
    text = re.sub(r"</?(?:c|v|lang|ruby|rt|b|i|u)(?:\.[^>\s]+)?[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)

    # Unescape HTML entities (&amp; etc.), including numeric ones.
    text = html.unescape(text.replace("\xa0", " "))

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_subtitle_to_text(raw: str, ext: str) -> str:
    content = raw.strip()
    if not content:
        return ""

    # Auto-detect XML/TTML even if fmt/ext claims vtt (seen in the wild).
    if "<tt" in content or "<transcript" in content:
        ext = "xml"

    lines: list[str] = []

    if ext == "vtt":
        # Skip NOTE/STYLE/REGION blocks properly.
        block_mode: str | None = None
        for line in content.splitlines():
            stripped = line.strip()
            upper = stripped.upper()

            if block_mode:
                if not stripped:
                    block_mode = None
                continue

            if not stripped or upper == "WEBVTT":
                continue

            if upper.startswith(("NOTE", "STYLE", "REGION")):
                block_mode = upper.split(maxsplit=1)[0]
                continue

            # YouTube sometimes includes these in VTT downloads.
            if stripped.startswith(("Kind:", "Language:")):
                continue

            # Skip cue timing and numeric cue identifiers.
            if "-->" in stripped or stripped.isdigit():
                continue

            parsed_line = _strip_caption_markup(stripped)
            if parsed_line:
                lines.append(parsed_line)

        return re.sub(r"\s+", " ", " ".join(lines)).strip()

    if ext == "srt":
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.isdigit() or "-->" in stripped:
                continue
            parsed_line = _strip_caption_markup(stripped)
            if parsed_line:
                lines.append(parsed_line)
        return re.sub(r"\s+", " ", " ".join(lines)).strip()

    if ext in {"ttml", "xml"}:
        try:
            root = ET.fromstring(content)
            for node in root.findall(".//{*}p") + root.findall(".//text"):
                val = _strip_caption_markup("".join(node.itertext()))
                if val:
                    lines.append(val)
            return re.sub(r"\s+", " ", " ".join(lines)).strip()
        except ET.ParseError:
            return ""

    return ""


def _download_text(url: str, proxy: str | None) -> tuple[HTTPStatusValue, str | None, str | None]:
    proxies = {"http": proxy, "https": proxy} if proxy else None
    try:
        # Keep this lower so caption downloads can't eat the full Streamlit budget.
        response = requests.get(url, timeout=10, proxies=proxies, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return response.status_code, None, response.headers.get("Content-Type")
        return response.status_code, response.text, response.headers.get("Content-Type")
    except Exception:
        return "exception", None, None


def _run_ytdlp_subtitles(youtube_url: str, preferred_language: str | None, proxy: str | None, work_dir: Path, cookies_path: Path | None = None) -> tuple[str | None, dict[str, Any]]:
    start = time.perf_counter()
    diag = {
        "ytdlp_subs_status": "failed",
        "ytdlp_subs_error_code": None,
        "ytdlp_subs_elapsed_ms": 0,
        "ytdlp_subs_lang": None,
        "ytdlp_subs_format": None,
        "ytdlp_subs_file_bytes": 0,
        "ytdlp_subs_file_path": None,
        "ytdlp_subs_sniff": None,
        "ytdlp_subs_stderr_sniff": None,
        "ytdlp_used_cookies": False,
    }

    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        diag["ytdlp_subs_error_code"] = "YTDLP_SUBS_FAILED"
        diag["ytdlp_subs_elapsed_ms"] = int((time.perf_counter() - start) * 1000)
        return None, diag

    preferred = (preferred_language or "").strip().lower()
    langs = [part for part in [preferred, "en.*", "en"] if part]
    canonical_id = _extract_video_id("", youtube_url, youtube_url)
    output_template = "/tmp/%(id)s.%(ext)s"

    cmd = [
        yt_dlp,
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        ",".join(langs),
        "--sub-format",
        "vtt",
        "--ignore-no-formats-error",
        "--retries",
        "1",
        "--fragment-retries",
        "1",
        "--extractor-retries",
        "1",
        "--socket-timeout",
        "10",
        "-o",
        output_template,
        youtube_url,
    ]

    if cookies_path and cookies_path.exists():
        cmd.extend(["--cookies", str(cookies_path)])
        diag["ytdlp_used_cookies"] = True

    if proxy:
        cmd.extend(["--proxy", proxy])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=YTDLP_SUBS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        diag["ytdlp_subs_status"] = "timeout"
        diag["ytdlp_subs_error_code"] = "YTDLP_TIMEOUT"
        diag["ytdlp_subs_elapsed_ms"] = int((time.perf_counter() - start) * 1000)
        return None, diag
    except Exception:
        diag["ytdlp_subs_error_code"] = "YTDLP_SUBS_FAILED"
        diag["ytdlp_subs_elapsed_ms"] = int((time.perf_counter() - start) * 1000)
        return None, diag

    diag["ytdlp_subs_elapsed_ms"] = int((time.perf_counter() - start) * 1000)
    stderr_sniff = _redact_ytdlp_stderr(proc.stderr)
    if stderr_sniff:
        diag["ytdlp_subs_stderr_sniff"] = stderr_sniff

    stderr_lc = (proc.stderr or "").lower()
    blocked_markers = ["sign in to confirm", "consent", "429", "too many requests"]

    files = [p for p in Path("/tmp").glob(f"{canonical_id}*.vtt") if p.exists() and p.stat().st_size > 0] if canonical_id else []
    if proc.returncode != 0 and not files:
        if any(marker in stderr_lc for marker in blocked_markers):
            diag["ytdlp_subs_status"] = "blocked"
            diag["ytdlp_subs_error_code"] = "YTDLP_BLOCKED"
        elif "subtitle" in stderr_lc and any(marker in stderr_lc for marker in ["not available", "no subtitles", "there are no subtitles"]):
            diag["ytdlp_subs_status"] = "no_subs"
            diag["ytdlp_subs_error_code"] = "NO_SUBTITLE_FILES"
        else:
            diag["ytdlp_subs_error_code"] = "YTDLP_SUBS_FAILED"
        return None, diag

    if not files:
        diag["ytdlp_subs_status"] = "no_subs"
        diag["ytdlp_subs_error_code"] = "NO_SUBTITLE_FILES"
        return None, diag

    preferred_files: list[Path] = []
    if canonical_id:
        preferred_files.extend(sorted(Path("/tmp").glob(f"{canonical_id}.en-orig.vtt")))
        preferred_files.extend(sorted(Path("/tmp").glob(f"{canonical_id}.en.vtt")))
        preferred_files.extend(sorted(Path("/tmp").glob(f"{canonical_id}.en-*.vtt")))

    selected = next((candidate for candidate in preferred_files if candidate.exists() and candidate.stat().st_size > 0), files[0])
    text = _parse_subtitle_to_text(
        selected.read_text(encoding="utf-8", errors="ignore"),
        selected.suffix.lower().lstrip("."),
    )
    if not text:
        diag["ytdlp_subs_error_code"] = "SUBTITLE_PARSE_FAILED"
        return None, diag

    diag["ytdlp_subs_status"] = "success"
    diag["ytdlp_subs_error_code"] = None
    diag["ytdlp_subs_file_bytes"] = selected.stat().st_size
    diag["ytdlp_subs_lang"] = next((part for part in selected.stem.split(".") if len(part) in {2, 5} and part[:2].isalpha()), None)
    diag["ytdlp_subs_format"] = selected.suffix.lower().lstrip(".")
    diag["ytdlp_subs_file_path"] = str(selected)
    diag["ytdlp_subs_sniff"] = _sanitize_sniff(selected.read_text(encoding="utf-8", errors="ignore"), 800)

    return text, diag


def _download_audio(youtube_url: str, proxy: str | None, work_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    start = time.perf_counter()
    diag = {
        "audio_download_status": "failed",
        "audio_download_error_code": None,
        "audio_download_elapsed_ms": 0,
        "audio_file_ext": None,
        "audio_file_bytes": 0,
        "audio_file_path": None,
    }
    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        diag["audio_download_error_code"] = "YTDLP_AUDIO_DOWNLOAD_FAILED"
        return None, diag

    cmd = [
        yt_dlp,
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "-x",
        "--audio-format",
        "mp3",
        "-o",
        str(work_dir / "%(id)s.%(ext)s"),
        youtube_url,
    ]
    if proxy:
        cmd.extend(["--proxy", proxy])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except Exception:
        diag["audio_download_elapsed_ms"] = int((time.perf_counter() - start) * 1000)
        diag["audio_download_error_code"] = "YTDLP_AUDIO_DOWNLOAD_FAILED"
        return None, diag

    diag["audio_download_elapsed_ms"] = int((time.perf_counter() - start) * 1000)

    if proc.returncode != 0:
        diag["audio_download_error_code"] = "YTDLP_AUDIO_DOWNLOAD_FAILED"
        return None, diag

    candidates = sorted([p for p in work_dir.glob("*.*") if p.suffix.lower().lstrip(".") in {"mp3", "m4a", "webm", "opus"}])
    if not candidates:
        diag["audio_download_error_code"] = "YTDLP_AUDIO_DOWNLOAD_FAILED"
        return None, diag

    audio_path = candidates[0]
    diag["audio_download_status"] = "success"
    diag["audio_file_ext"] = audio_path.suffix.lower().lstrip(".")
    diag["audio_file_bytes"] = audio_path.stat().st_size
    diag["audio_file_path"] = str(audio_path)

    return audio_path, diag


def _transcribe_openai(audio_path: Path, model: str, language: str | None, prompt: str | None) -> tuple[str | None, dict[str, Any], int, str | None, str | None]:
    start = time.perf_counter()
    diag = {
        "stt_provider": "openai",
        "stt_model": model,
        "stt_language": language,
        "stt_status": "failed",
        "stt_error_code": None,
        "stt_elapsed_ms": 0,
        "transcript_chars": 0,
    }

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        diag["stt_error_code"] = "OPENAI_KEY_MISSING"
        return None, diag, 500, "OPENAI_KEY_MISSING", "OpenAI API key is missing"

    if model not in ALLOWED_STT_MODELS:
        model = "gpt-4o-mini-transcribe"
        diag["stt_model"] = model

    headers = {"Authorization": f"Bearer {api_key}"}
    data: dict[str, str] = {"model": model}
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt

    mime = "audio/mpeg" if audio_path.suffix.lower() == ".mp3" else "audio/mp4"
    try:
        with audio_path.open("rb") as handle:
            resp = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=headers,
                files={"file": (audio_path.name, handle, mime)},
                data=data,
                timeout=300,
            )
    except Exception:
        diag["stt_elapsed_ms"] = int((time.perf_counter() - start) * 1000)
        diag["stt_error_code"] = "OPENAI_STT_FAILED"
        return None, diag, 503, "OPENAI_STT_FAILED", "OpenAI transcription failed"

    diag["stt_elapsed_ms"] = int((time.perf_counter() - start) * 1000)

    if resp.status_code == 429:
        diag["stt_error_code"] = "OPENAI_RATE_LIMIT"
        return None, diag, 503, "OPENAI_RATE_LIMIT", "OpenAI rate limit"

    if resp.status_code != 200:
        diag["stt_error_code"] = "OPENAI_STT_FAILED"
        logger.error("openai stt failed: %s", _tail(resp.text))
        return None, diag, 503, "OPENAI_STT_FAILED", "OpenAI transcription failed"

    text = (resp.json().get("text") or "").strip()
    if not text:
        diag["stt_error_code"] = "OPENAI_STT_FAILED"
        return None, diag, 503, "OPENAI_STT_FAILED", "OpenAI transcription returned empty transcript"

    diag["stt_status"] = "success"
    diag["transcript_chars"] = len(text)
    return text, diag, 200, None, None


def _base_diagnostics(video_id: str, payload: TranscriptRequest, proxy_enabled: bool, preferred_language: str | None) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "proxy_enabled": proxy_enabled,
        "preferred_language": preferred_language,
        "pipeline_stage_attempts": [],
        "pipeline_stage_success": None,
        "elapsed_ms_total": 0,
        "player_json_attempted": False,
        "player_json_http_status": "exception",
        "player_json_error_code": None,
        "caption_tracks_found": 0,
        "caption_best_track": None,
        "caption_download_http_status": "exception",
        "caption_download_url_fmt": None,
        "caption_content_type": None,
        "caption_body_chars": 0,
        "caption_sniff": None,
        "caption_sniff_hint": "EMPTY",
        "caption_parse_error": None,
        "caption_parse_status": "skipped",
        "caption_chars": 0,
        "ytdlp_subs_attempted": False,
        "ytdlp_subs_status": "skipped",
        "ytdlp_subs_error_code": None,
        "ytdlp_subs_elapsed_ms": 0,
        "ytdlp_subs_lang": None,
        "ytdlp_subs_format": None,
        "ytdlp_subs_file_bytes": 0,
        "ytdlp_subs_file_path": None,
        "ytdlp_subs_sniff": None,
        "ytdlp_subs_stderr_sniff": None,
        "ytdlp_used_cookies": False,
        "cookies_present": False,
        "cookies_age_seconds": None,
        "cookies_bootstrap_attempted": False,
        "cookies_bootstrap_status": "skipped",
        "cookies_bootstrap_elapsed_ms": 0,
        "cookies_cookie_count": 0,
        "audio_fallback_enabled": payload.audio_fallback_enabled if payload.allow_audio_fallback is None else payload.allow_audio_fallback,
        "audio_fallback_attempted": False,
        "audio_download_status": "skipped",
        "audio_download_error_code": None,
        "audio_download_elapsed_ms": 0,
        "audio_file_ext": None,
        "audio_file_bytes": 0,
        "audio_file_path": None,
        "stt_provider": None,
        "stt_model": payload.stt_model,
        "stt_language": payload.stt_language,
        "stt_status": "skipped",
        "stt_error_code": None,
        "stt_elapsed_ms": 0,
        "transcript_chars": 0,
        "final_error_code": None,
        "final_error": None,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    proxy_manager = ProxyManager()
    return {"ok": True, "version": os.getenv("BRAINS_WORKER_VERSION", "dev"), "time": datetime.now(timezone.utc).isoformat(), "proxy": proxy_manager.safe_diagnostics()}


@app.get("/transcript/version")
def transcript_version() -> dict[str, Any]:
    return _version_payload()


@app.get("/transcript/subs-dry-run")
def transcript_subs_dry_run(video_id: str) -> dict[str, Any]:
    canonical_id = _extract_video_id(video_id, None, None)
    if not canonical_id:
        return {"ok": False, "error_code": "INVALID_VIDEO_ID", "video_id": video_id}

    work_dir = Path(tempfile.mkdtemp(prefix=f"{canonical_id}-subs-", dir="/tmp"))
    youtube_url = f"https://www.youtube.com/watch?v={canonical_id}"
    try:
        text, diag = _run_ytdlp_subtitles(
            youtube_url,
            "en",
            None,
            work_dir,
            YT_COOKIES_PATH if YT_COOKIES_PATH.exists() else None,
        )
        return {
            "ok": bool(text),
            "video_id": canonical_id,
            "transcript_chars": len(text or ""),
            "ytdlp_subs_status": diag.get("ytdlp_subs_status"),
            "ytdlp_subs_file_path": diag.get("ytdlp_subs_file_path"),
            "ytdlp_subs_sniff": diag.get("ytdlp_subs_sniff"),
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/transcript")
def transcript(payload: TranscriptRequest, x_api_key: str | None = Header(default=None, alias="x-api-key"), x_api_key_2: str | None = Header(default=None, alias="X-Api-Key")) -> Any:
    expected_key = _expected_api_key()
    provided = (x_api_key or x_api_key_2 or "").strip()
    if expected_key and provided != expected_key:
        return _auth_error()

    started = time.perf_counter()
    canonical_id = _extract_video_id(payload.video_id, payload.source_id, payload.url)
    preferred_language = (payload.preferred_language or "").strip() or None
    if not canonical_id:
        diagnostics = _base_diagnostics("", payload, False, preferred_language)
        diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
        return _error(400, "INVALID_VIDEO_ID", "Unable to determine YouTube video id", diagnostics)

    fallback_enabled = payload.audio_fallback_enabled if payload.allow_audio_fallback is None else payload.allow_audio_fallback
    payload.audio_fallback_enabled = fallback_enabled
    youtube_url = f"https://www.youtube.com/watch?v={canonical_id}"
    proxy_url = _build_proxy_url(payload, canonical_id)

    diagnostics = _base_diagnostics(canonical_id, payload, bool(proxy_url), preferred_language)

    temp_root = Path("/tmp/brains-worker")
    temp_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{canonical_id}-", dir=str(temp_root)))

    def _attempt_player_json_captions() -> str | None:
        if not payload.captions_enabled:
            diagnostics["caption_parse_status"] = "skipped"
            return None

        diagnostics["pipeline_stage_attempts"].append("captions_player_json")
        diagnostics["player_json_attempted"] = True
        status, player, player_err = _youtube_watch_player_json(canonical_id, proxy_url)
        diagnostics["player_json_http_status"] = status
        diagnostics["player_json_error_code"] = player_err

        tracks = ((player or {}).get("captions") or {}).get("playerCaptionsTracklistRenderer", {}).get("captionTracks", [])
        diagnostics["caption_tracks_found"] = len(tracks)

        if not tracks:
            diagnostics["caption_download_http_status"] = "exception"
            return None

        best = _choose_caption_track(tracks, preferred_language)
        if not best:
            return None

        diagnostics["caption_best_track"] = {
            "lang": best.get("languageCode"),
            "kind": best.get("kind"),
            "name": ((best.get("name") or {}).get("simpleText") if isinstance(best.get("name"), dict) else best.get("name")),
        }
        base_url = best.get("baseUrl")
        if not base_url:
            diagnostics["caption_parse_status"] = "failed"
            return None

        sep = "&" if "?" in base_url else "?"
        caption_url = f"{base_url}{sep}fmt=vtt"
        diagnostics["caption_download_url_fmt"] = _build_caption_url_fmt(best, fmt="vtt")
        cap_status, cap_body, cap_content_type = _download_text(caption_url, proxy_url)
        diagnostics["caption_download_http_status"] = cap_status
        diagnostics["caption_content_type"] = cap_content_type
        diagnostics["caption_body_chars"] = len(cap_body or "")
        diagnostics["caption_sniff"] = _sanitize_sniff(cap_body, 800)
        diagnostics["caption_sniff_hint"] = _classify_caption_sniff(diagnostics["caption_sniff"])

        if not cap_body:
            diagnostics["caption_parse_status"] = "failed"
            return None

        parsed = ""
        diagnostics["caption_parse_error"] = None
        try:
            parsed = _parse_subtitle_to_text(cap_body, "vtt")
        except Exception as exc:
            diagnostics["caption_parse_error"] = f"{type(exc).__name__}: {exc}"

        hint = diagnostics["caption_sniff_hint"]
        if not parsed or hint in {"XML", "JSON", "HTML"}:
            try:
                if hint == "XML":
                    parsed = _parse_subtitle_to_text(cap_body, "xml") or parsed
                elif hint == "JSON":
                    parsed = _parse_json3_subtitle_to_text(cap_body) or parsed
                elif hint == "HTML":
                    diagnostics["caption_parse_status"] = "blocked_html"
            except Exception as exc:
                diagnostics["caption_parse_error"] = f"{type(exc).__name__}: {exc}"

        if parsed:
            diagnostics["caption_parse_status"] = "success"
            diagnostics["caption_chars"] = len(parsed)
            return parsed

        if diagnostics["caption_parse_status"] != "blocked_html":
            diagnostics["caption_parse_status"] = "failed"
        return None

    try:
        cookie_diag = _ensure_youtube_cookies()
        diagnostics.update(cookie_diag)

        if payload.yt_dlp_subs_enabled:
            diagnostics["pipeline_stage_attempts"].append("subs_ytdlp")
            diagnostics["ytdlp_subs_attempted"] = True
            text, diag = _run_ytdlp_subtitles(youtube_url, preferred_language, proxy_url, work_dir, YT_COOKIES_PATH if diagnostics["cookies_present"] else None)
            diagnostics.update(diag)
            if payload.debug_include_artifact_paths is False:
                diagnostics["ytdlp_subs_file_path"] = None

            if text:
                diagnostics["pipeline_stage_success"] = "subs_ytdlp"
                diagnostics["transcript_chars"] = len(text)
                diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
                return {"video_id": canonical_id, "transcript_source": "subs_ytdlp", "transcript_text": text, "diagnostics": diagnostics}

        caption_text = _attempt_player_json_captions()
        if caption_text:
            diagnostics["pipeline_stage_success"] = "captions_player_json"
            diagnostics["transcript_chars"] = len(caption_text)
            diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
            return {"video_id": canonical_id, "transcript_source": "captions_player_json", "transcript_text": caption_text, "diagnostics": diagnostics}

        if not fallback_enabled:
            diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
            return _error(422, "NO_CAPTIONS_AND_FALLBACK_DISABLED", "No captions/subtitles found and audio fallback disabled", diagnostics)

        diagnostics["pipeline_stage_attempts"].append("audio_openai_stt")
        diagnostics["audio_fallback_attempted"] = True

        audio_path, audio_diag = _download_audio(youtube_url, proxy_url, work_dir)
        diagnostics.update(audio_diag)
        if payload.debug_include_artifact_paths is False:
            diagnostics["audio_file_path"] = None

        if not audio_path:
            diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
            return _error(502, "YTDLP_AUDIO_DOWNLOAD_FAILED", "yt-dlp audio download failed", diagnostics)

        text, stt_diag, status_code, err_code, err_message = _transcribe_openai(audio_path, payload.stt_model, payload.stt_language, payload.stt_prompt)
        diagnostics.update(stt_diag)
        if not text:
            diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
            return _error(status_code, err_code or "OPENAI_STT_FAILED", err_message or "OpenAI transcription failed", diagnostics)

        diagnostics["pipeline_stage_success"] = "audio_openai_stt"
        diagnostics["transcript_chars"] = len(text)
        diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
        return {"video_id": canonical_id, "transcript_source": "audio_openai_stt", "transcript_text": text, "diagnostics": diagnostics}

    except Exception:
        logger.exception("unexpected error")
        diagnostics["elapsed_ms_total"] = int((time.perf_counter() - started) * 1000)
        return _error(500, "UNEXPECTED_SERVER_ERROR", "Unexpected server error", diagnostics)

    finally:
        if not payload.debug_keep_files:
            shutil.rmtree(work_dir, ignore_errors=True)
