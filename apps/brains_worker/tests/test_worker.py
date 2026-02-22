from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

from fastapi.testclient import TestClient

from apps.brains_worker.main import app


def test_transcript_no_caption_tracks_includes_http_statuses(monkeypatch):
    monkeypatch.setenv("BRAINS_API_KEY", "test-key")

    def fake_list_tracks(video_id: str, *, proxy: str | None):
        assert video_id == "5lQf89-AeFo"
        return 200, [], None

    monkeypatch.setattr("apps.brains_worker.main.timedtext_list_tracks", fake_list_tracks)

    client = TestClient(app)
    response = client.post(
        "/transcript",
        headers={"x-api-key": "test-key"},
        json={"source_id": "yt:5lQf89-AeFo", "allow_audio_fallback": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["method"] == "error"
    diagnostics = payload["diagnostics"]
    assert diagnostics["timedtext_list_http_status"] == 200
    assert diagnostics["timedtext_fetch_http_status"] == "not_attempted"


def test_transcript_auth_missing_key_returns_401(monkeypatch):
    monkeypatch.setenv("BRAINS_API_KEY", "test-key")

    client = TestClient(app)
    response = client.post("/transcript", json={"source_id": "yt:5lQf89-AeFo"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API key"}
