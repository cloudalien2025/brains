from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_app(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("WORKER_API_KEY", "test-worker-key")
    monkeypatch.setenv("BRAINS_DATA_DIR", str(tmp_path / "brains-data"))
    sys.modules.pop("apps.brains_worker.main", None)
    module = importlib.import_module("apps.brains_worker.main")
    return module.app


def test_v1_health_is_public(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_v1_brains_requires_api_key(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path)
    client = TestClient(app)

    unauthorized = client.get("/v1/brains")
    authorized = client.get("/v1/brains", headers={"X-Api-Key": "test-worker-key"})

    assert unauthorized.status_code == 401
    assert unauthorized.json() == {"detail": "Missing or invalid X-Api-Key"}
    assert authorized.status_code == 200
