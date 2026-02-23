#!/usr/bin/env python3
from __future__ import annotations
import json, os, time
from collections import Counter
from pathlib import Path
import requests

WORKER_URL = os.getenv("WORKER_URL", "http://127.0.0.1:8787")
API_KEY = os.getenv("WORKER_API_KEY", "")
BRAIN_ID = os.getenv("BRAIN_ID", "brilliant_directories")

headers = {"X-Api-Key": API_KEY} if API_KEY else {}
payload = {"keyword": "brilliant directories", "selected_new": 3, "max_candidates": 50, "mode": "audio_first"}

r = requests.post(f"{WORKER_URL}/v1/brains/{BRAIN_ID}/ingest", json=payload, headers=headers, timeout=30)
r.raise_for_status()
run_id = r.json()["run_id"]
print("run_id=", run_id)

report = None
for _ in range(240):
    rr = requests.get(f"{WORKER_URL}/v1/runs/{run_id}/report", headers=headers, timeout=30)
    if rr.status_code == 200:
        report = rr.json()
        if report.get("status") in {"success", "partial_success", "blocked", "failed", "no_captions", "completed"}:
            break
    time.sleep(5)

if not report:
    raise SystemExit("report not ready")
print(json.dumps({k: report.get(k) for k in ["run_id", "status", "requested_new", "selected_new", "transcripts_succeeded", "audio_success", "caption_probe_blocked", "caption_probe_attempted", "total_audio_minutes", "transcript_attempts_jsonl"]}, indent=2))

path = Path(report.get("transcript_attempts_jsonl", ""))
if not path.exists():
    raise SystemExit(f"jsonl missing: {path}")
rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
counts = Counter((row.get("phase"), row.get("error_code") or "none") for row in rows if not row.get("success"))
print("Top failures:")
for (phase, reason), c in counts.most_common(5):
    print(f"- {phase}: {reason} => {c}")
for row in rows:
    if row.get("phase") == "audio_fallback" and row.get("stderr_tail"):
        print("sample_audio_stderr=", row.get("stderr_tail", "")[:300].replace("\n", " "))
        break
print("jsonl_path=", str(path))
