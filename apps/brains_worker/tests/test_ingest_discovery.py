from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TERMINAL = {"completed", "completed_with_errors", "failed", "success", "partial_success", "no_captions", "blocked"}


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


def test_run_status_exposes_transcript_diagnostics(monkeypatch, tmp_path):
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
            json={"name": "Diag Brain", "brain_type": "BD"},
        )
        brain_id = create.json()["brain_id"]

        start = client.post(
            f"/v1/brains/{brain_id}/ingest",
            headers={"X-Api-Key": "test-worker-key"},
            json={"keyword": "Brilliant Directories", "n_new_videos": 1},
        )
        run_id = start.json()["run_id"]
        _wait_for_terminal_run(client, run_id)
        status = client.get(f"/v1/runs/{run_id}", headers={"X-Api-Key": "test-worker-key"})

    payload = status.json()
    assert "transcript_failure_reasons" in payload
    assert "sample_failures" in payload


def test_run_diagnostics_endpoint_and_selected_accounting(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)

    candidates = [
        {"video_id": f"v{i}", "title": f"Video {i}", "channel_title": "Ch", "published_at": "2024-01-01T00:00:00Z", "url": f"https://www.youtube.com/watch?v=v{i}"}
        for i in range(1, 11)
    ]

    monkeypatch.setattr(
        module,
        "discover_youtube_videos",
        lambda **kwargs: module.DiscoveryOutcome(candidates=candidates, method="youtube_api", youtube_api_http_status=200),
    )

    class FakeSession:
        pass

    monkeypatch.setattr(module, "_build_http_session", lambda: (FakeSession(), {"proxy_enabled": True, "proxy_provider": "decodo", "proxy_url_redacted": "http://***:***@proxy", "cookies_enabled": True, "cookies_source": "cookies/youtube_cookies.txt"}))

    def fake_caption_tracks(session, video_id):
        return ([{"lang_code": "en", "kind": "asr", "name": "English"}], 200, None)

    monkeypatch.setattr(module, "_caption_tracks", fake_caption_tracks)

    def fake_fetch(session, video_id, track):
        if video_id == "v2":
            raise RuntimeError("HTTP 403")
        return (f"transcript for {video_id}", 200)

    monkeypatch.setattr(module, "_fetch_timedtext_transcript", fake_fetch)

    with TestClient(module.app) as client:
        create = client.post("/v1/brains", headers={"X-Api-Key": "test-worker-key"}, json={"name": "Sel Brain", "brain_type": "BD"})
        brain_id = create.json()["brain_id"]
        start = client.post(
            f"/v1/brains/{brain_id}/ingest",
            headers={"X-Api-Key": "test-worker-key"},
            json={"keyword": "Brilliant Directories", "selected_new": 5},
        )
        run_id = start.json()["run_id"]
        run = _wait_for_terminal_run(client, run_id)
        report = client.get(f"/v1/runs/{run_id}/report", headers={"X-Api-Key": "test-worker-key"}).json()
        diagnostics = client.get(f"/v1/runs/{run_id}/diagnostics", headers={"X-Api-Key": "test-worker-key"}).json()

    assert run["selected_new"] == 5
    assert report["selected_new"] == 5
    assert report["transcripts_failed"] <= report["selected_new"]
    assert diagnostics["transcripts_attempted_selected"] == 5
    assert Path(diagnostics["diagnostics_path"]).exists()


def test_run_brain_pack_get_supported(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "discover_youtube_videos", lambda **kwargs: module.DiscoveryOutcome(candidates=[], method="youtube_api", youtube_api_http_status=200))

    with TestClient(module.app) as client:
        create = client.post("/v1/brains", headers={"X-Api-Key": "test-worker-key"}, json={"name": "Pack Brain", "brain_type": "BD"})
        brain_id = create.json()["brain_id"]
        start = client.post(
            f"/v1/brains/{brain_id}/ingest",
            headers={"X-Api-Key": "test-worker-key"},
            json={"keyword": "Brilliant Directories", "selected_new": 1},
        )
        run_id = start.json()["run_id"]
        _wait_for_terminal_run(client, run_id)
        resp = client.get(f"/v1/runs/{run_id}/brain-pack", headers={"X-Api-Key": "test-worker-key"})

    assert resp.status_code != 405
