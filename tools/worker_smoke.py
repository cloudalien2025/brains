#!/usr/bin/env python3
from __future__ import annotations
import os
import requests

WORKER_URL = os.getenv("WORKER_URL", "http://127.0.0.1:8787")
API_KEY = os.getenv("WORKER_API_KEY", "")
RUN_ID = os.getenv("RUN_ID", "")
headers = {"X-Api-Key": API_KEY} if API_KEY else {}

health = requests.get(f"{WORKER_URL}/v1/health", timeout=20)
health.raise_for_status()
h = health.json()
assert "yt_dlp_available" in h and "ffmpeg_available" in h, h
print("health_ok", h["yt_dlp_available"], h["ffmpeg_available"])

if RUN_ID:
    diag = requests.get(f"{WORKER_URL}/v1/runs/{RUN_ID}/diagnostics", headers=headers, timeout=20)
    diag.raise_for_status()
    sample = diag.json().get("sample_entries", [])
    for row in sample:
        if (row.get("error_code") == "blocked_429"):
            assert row.get("http_status") == 429, row
    print("diagnostics_ok")
