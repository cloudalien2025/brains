from __future__ import annotations

from pathlib import Path

from apps.brains_worker.main import app
from fastapi.testclient import TestClient


def test_run_files_endpoint_requires_auth(monkeypatch):
    import apps.brains_worker.main as main

    monkeypatch.setattr(main, "WORKER_API_KEY", "test")
    client = TestClient(app)
    resp = client.get("/v1/runs/run_1/files")
    assert resp.status_code == 401


def test_run_files_endpoint_lists_files(monkeypatch, tmp_path):
    import apps.brains_worker.main as main

    monkeypatch.setattr(main, "WORKER_API_KEY", "test")
    brains_root = tmp_path / "brains"
    monkeypatch.setattr(main, "BRAINS_DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "BRAINS_ROOT", brains_root)

    brain_id = "brain1"
    run_id = "run_1"
    run_root = brains_root / brain_id / "runs" / run_id
    (run_root / "transcripts").mkdir(parents=True, exist_ok=True)
    (run_root / "docs" / "text").mkdir(parents=True, exist_ok=True)
    (run_root / "artifacts").mkdir(parents=True, exist_ok=True)

    (run_root / "run.json").write_text("{}", encoding="utf-8")
    (run_root / "transcripts" / "vid1.txt").write_text("hi", encoding="utf-8")
    (run_root / "docs" / "text" / "doc1.txt").write_text("doc", encoding="utf-8")
    (run_root / "artifacts" / "doc1.json").write_text("{}", encoding="utf-8")

    client = TestClient(app)
    resp = client.get(f"/v1/runs/{run_id}/files", headers={"X-Api-Key": "test"})
    assert resp.status_code == 200
    payload = resp.json()

    transcript_names = {item["name"] for item in payload.get("transcript_files_txt", [])}
    doc_text_names = {item["name"] for item in payload.get("doc_text_files", [])}
    artifact_names = {item["name"] for item in payload.get("artifact_files", [])}

    assert "transcripts/vid1.txt" in transcript_names
    assert "docs/text/doc1.txt" in doc_text_names
    assert "artifacts/doc1.json" in artifact_names
