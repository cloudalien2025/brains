from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def _base_url() -> str:
    return os.getenv("BRAINS_E2E_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _brain_slug() -> str:
    return os.getenv("BRAINS_E2E_BRAIN_SLUG", "brilliant_directories")


def _data_root() -> Path:
    return Path(os.getenv("BRAINS_E2E_DATA_ROOT", "/opt/brains-data/brains"))


def _api_key() -> str | None:
    key = os.getenv("BRAINS_E2E_API_KEY")
    if key:
        return key.strip()
    default_path = Path("/etc/default/brains-worker")
    if default_path.exists():
        for line in default_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("WORKER_API_KEY="):
                raw = line.split("=", 1)[1].strip().strip("\"")
                return raw or None
    return None


def _request_json(path: str, api_key: str | None = None) -> tuple[int, dict[str, Any]]:
    url = f"{_base_url()}{path}"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-Api-Key"] = api_key
    req = Request(url, headers=headers, method="GET")
    try:
        with urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return resp.status, payload
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise AssertionError(f"HTTP {exc.code} for {url}: {body[:200]}") from exc


def test_health_endpoint_ok() -> None:
    status, payload = _request_json("/v1/health")
    assert status == 200
    assert payload.get("status") == "ok"


def test_stats_endpoint_ok() -> None:
    api_key = _api_key()
    if not api_key:
        raise AssertionError("BRAINS_E2E_API_KEY not set and /etc/default/brains-worker missing")
    status, payload = _request_json(f"/v1/brains/{_brain_slug()}/stats", api_key=api_key)
    assert status == 200
    for key in ["brain_slug", "total_items", "youtube_items", "webdocs_items", "updated_at"]:
        assert key in payload


def test_stats_changes_after_fixture_run() -> None:
    api_key = _api_key()
    if not api_key:
        raise AssertionError("BRAINS_E2E_API_KEY not set and /etc/default/brains-worker missing")

    _, before = _request_json(f"/v1/brains/{_brain_slug()}/stats", api_key=api_key)
    before_total = int(before.get("total_items") or 0)

    run_id = f"e2e_{int(time.time())}_{random.randint(1000, 9999)}"
    root = _data_root() / _brain_slug() / "runs" / run_id
    transcript_dir = root / "transcripts"
    doc_dir = root / "webdocs" / "text"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = transcript_dir / "yt_e2e.txt"
    doc_path = doc_dir / "doc_e2e.txt"
    transcript_path.write_text("e2e transcript fixture", encoding="utf-8")
    doc_path.write_text("e2e doc fixture", encoding="utf-8")

    run_payload = {
        "run_id": run_id,
        "brain_id": _brain_slug(),
        "status": "completed",
        "stage": "completed",
        "created_at": time.time(),
        "payload": {"keyword": "e2e"},
    }
    (root / "run.json").write_text(json.dumps(run_payload), encoding="utf-8")
    (root / "status.json").write_text(json.dumps({"run_id": run_id, "brain_slug": _brain_slug(), "status": "completed"}), encoding="utf-8")

    time.sleep(0.2)
    _, after = _request_json(f"/v1/brains/{_brain_slug()}/stats", api_key=api_key)
    after_total = int(after.get("total_items") or 0)

    try:
        assert after_total >= before_total + 2
    finally:
        try:
            if root.exists():
                for path in sorted(root.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
        except OSError:
            pass
