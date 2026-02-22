from __future__ import annotations

import os
import sys

import requests


WORKER_URL = (os.getenv("WORKER_URL") or "http://127.0.0.1:8081").rstrip("/")
WORKER_API_KEY = (os.getenv("WORKER_API_KEY") or "").strip()
VIDEO_ID = os.getenv("SMOKE_VIDEO_ID", "dQw4w9WgXcQ")


def main() -> int:
    headers = {"X-Api-Key": WORKER_API_KEY} if WORKER_API_KEY else {}
    resp = requests.get(f"{WORKER_URL}/v1/health", timeout=20)
    if resp.status_code != 200:
        print(f"FAIL: worker health {resp.status_code}")
        return 1

    params = {
        "v": VIDEO_ID,
        "type": "list",
    }
    timed = requests.get("https://www.youtube.com/api/timedtext", params=params, timeout=20)
    if timed.status_code != 200:
        print(f"FAIL: {timed.status_code} blocked during caption probe")
        return 1

    if "<track" not in timed.text:
        print("FAIL: no caption tracks")
        return 1

    text = requests.get("https://www.youtube.com/api/timedtext", params={"v": VIDEO_ID, "lang": "en", "fmt": "srv3"}, timeout=20)
    if text.status_code != 200:
        print(f"FAIL: {text.status_code} blocked during timedtext fetch")
        return 1
    chars = len(text.text or "")
    if chars < 20:
        print("FAIL: transcript too short")
        return 1

    print("PASS: transcript chars=%s via method=timedtext proxy=%s cookies=%s" % (chars, bool(os.getenv("DECODO_PROXY_URL")), os.path.exists("/workspace/brains/cookies/youtube_cookies.txt")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
