from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    assert payload["method"] == "audio_fallback_failed"
    diagnostics = payload["diagnostics"]
    assert diagnostics["timedtext_list_http_status"] == 200
    assert diagnostics["timedtext_fetch_http_status"] is None


def test_transcript_auth_missing_key_returns_401(monkeypatch):
    monkeypatch.setenv("BRAINS_API_KEY", "test-key")

    client = TestClient(app)
    response = client.post("/transcript", json={"source_id": "yt:5lQf89-AeFo"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Missing x-api-key"}


def test_transcript_captures_audio_fallback_diagnostics(monkeypatch):
    monkeypatch.setenv("BRAINS_API_KEY", "test-key")

    def fake_list_tracks(video_id: str, *, proxy: str | None):
        return 200, [], None

    def fake_audio(video_id: str, diagnostics: dict, proxy_url: str | None):
        diagnostics["audio_download_ok"] = True
        diagnostics["audio_file_bytes"] = 130_000
        diagnostics["transcription_engine"] = "openai"
        return "hello world", [{"start": 0.0, "duration": 1.0, "text": "hello world"}]

    monkeypatch.setattr("apps.brains_worker.main.timedtext_list_tracks", fake_list_tracks)
    monkeypatch.setattr("apps.brains_worker.main.transcribe_youtube_audio", fake_audio)

    client = TestClient(app)
    response = client.post(
        "/transcript",
        headers={"x-api-key": "test-key"},
        json={"source_id": "yt:5lQf89-AeFo", "allow_audio_fallback": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["method"] == "audio_fallback"
    assert payload["text"] == "hello world"
    diagnostics = payload["diagnostics"]
    assert diagnostics["audio_download_ok"] is True
    assert diagnostics["audio_file_bytes"] == 130_000
    assert diagnostics["transcription_engine"] == "openai"
