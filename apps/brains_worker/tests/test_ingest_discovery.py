from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TERMINAL = {"completed", "completed_with_errors", "failed"}


def _load_module(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("WORKER_API_KEY", "test-worker-key")
    monkeypatch.setenv("BRAINS_DATA_DIR", str(tmp_path / "brains-data"))
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-youtube-key")
    sys.modules.pop("apps.brains_worker.main", None)
    return importlib.import_module("apps.brains_worker.main")


def _wait_for_terminal_run(client: TestClient, run_id: str) -> dict:
    for _ in range(40):
        run = client.get(f"/v1/runs/{run_id}", headers={"X-Api-Key": "test-worker-key"}).json()
        if run.get("status") in TERMINAL:
            return run
        time.sleep(0.1)
    raise AssertionError("run did not reach terminal state")


def test_ingest_zero_discovery_results_marks_error(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "discover_youtube_videos",
        lambda **kwargs: module.DiscoveryOutcome(candidates=[], method="youtube_api", youtube_api_http_status=200),
    )

    with TestClient(module.app) as client:
        create = client.post(
            "/v1/brains",
            headers={"X-Api-Key": "test-worker-key"},
            json={"name": "My Brain", "brain_type": "BD"},
        )
        brain_id = create.json()["brain_id"]

        start = client.post(
            f"/v1/brains/{brain_id}/ingest",
            headers={"X-Api-Key": "test-worker-key"},
            json={"keyword": "Brilliant Directories", "n_new_videos": 5},
        )
        run = _wait_for_terminal_run(client, start.json()["run_id"])

    assert run["status"] == "completed_with_errors"
    assert run["candidates_found"] == 0

    run_json = module.load_json(module.BRAINS_ROOT / brain_id / "runs" / start.json()["run_id"] / "run.json", {})
    assert run_json["final_error_code"] == "DISCOVERY_ZERO_RESULTS"


def test_ingest_all_duplicates_completes_cleanly(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "discover_youtube_videos",
        lambda **kwargs: module.DiscoveryOutcome(
            candidates=[
                {
                    "video_id": "dup1",
                    "title": "Duplicate",
                    "channel_title": "Ch",
                    "published_at": "2024-01-01T00:00:00Z",
                    "url": "https://www.youtube.com/watch?v=dup1",
                }
            ],
            method="youtube_api",
            youtube_api_http_status=200,
        ),
    )

    with TestClient(module.app) as client:
        create = client.post(
            "/v1/brains",
            headers={"X-Api-Key": "test-worker-key"},
            json={"name": "Dup Brain", "brain_type": "BD"},
        )
        brain_id = create.json()["brain_id"]

        root = Path(module.BRAINS_ROOT / brain_id)
        module.write_json(root / "ledger.json", {"ingested_video_ids": ["dup1"], "records": []})

        start = client.post(
            f"/v1/brains/{brain_id}/ingest",
            headers={"X-Api-Key": "test-worker-key"},
            json={"keyword": "Brilliant Directories", "n_new_videos": 1},
        )
        run = _wait_for_terminal_run(client, start.json()["run_id"])

    assert run["status"] == "completed"
    assert run["selected_new"] == 0

    run_json = module.load_json(module.BRAINS_ROOT / brain_id / "runs" / start.json()["run_id"] / "run.json", {})
    assert run_json["message"] == "No new videos; all candidates already ingested"
