from pathlib import Path
import importlib
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient


def test_worker_import_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    module = importlib.import_module("apps.brains_worker.main")
    assert hasattr(module, "app")


def test_transcript_version_endpoint(monkeypatch):
    monkeypatch.setenv("BRAINS_API_KEY", "test-key")
    from apps.brains_worker.main import app

    client = TestClient(app)
    response = client.get("/transcript/version")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "brains-worker"
    assert "audio_fallback_supported" in payload


def test_transcript_non_200_when_fallback_disabled(monkeypatch):
    monkeypatch.setenv("BRAINS_API_KEY", "test-key")
    from apps.brains_worker.main import app

    def fake_list_tracks(video_id: str, *, proxy: str | None):
        return 200, [], None

    monkeypatch.setattr("apps.brains_worker.main.timedtext_list_tracks", fake_list_tracks)

    client = TestClient(app)
    response = client.post(
        "/transcript",
        headers={"x-api-key": "test-key"},
        json={"source_id": "yt:5lQf89-AeFo", "allow_audio_fallback": False},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error_code"] == "NO_CAPTIONS_FALLBACK_DISABLED"
    assert detail["diagnostics"]["timedtext_list_http_status"] == 200
    assert detail["diagnostics"]["timedtext_fetch_http_status"] == 0


def test_transcript_fallback_attempted_and_openai_missing(monkeypatch):
    monkeypatch.setenv("BRAINS_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from apps.brains_worker.main import app

    def fake_list_tracks(video_id: str, *, proxy: str | None):
        return 200, [], None

    def fake_download(video_id: str, proxy: str | None, work_dir: str, diagnostics: dict):
        sample = Path(work_dir) / "sample.mp3"
        sample.write_bytes(b"fake-audio")
        diagnostics["audio_download_status"] = "success"
        diagnostics["audio_file_bytes"] = sample.stat().st_size
        diagnostics["audio_download_elapsed_ms"] = 1.0
        return sample

    monkeypatch.setattr("apps.brains_worker.main.timedtext_list_tracks", fake_list_tracks)
    monkeypatch.setattr("apps.brains_worker.main._run_ytdlp_download", fake_download)

    client = TestClient(app)
    response = client.post(
        "/transcript",
        headers={"x-api-key": "test-key"},
        json={"source_id": "yt:5lQf89-AeFo", "allow_audio_fallback": True},
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["error_code"] == "STT_FAILED"
    assert detail["diagnostics"]["audio_fallback_attempted"] is True


def test_transcript_fallback_success(monkeypatch):
    monkeypatch.setenv("BRAINS_API_KEY", "test-key")
    from apps.brains_worker.main import app

    def fake_list_tracks(video_id: str, *, proxy: str | None):
        return 200, [], None

    def fake_audio(video_id: str, proxy_url: str | None, diagnostics: dict, debug_keep_files: bool):
        diagnostics["audio_download_status"] = "success"
        diagnostics["audio_file_bytes"] = 130000
        diagnostics["audio_download_elapsed_ms"] = 23.0
        diagnostics["stt_provider"] = "openai"
        diagnostics["stt_model"] = "gpt-4o-mini-transcribe"
        diagnostics["stt_status"] = "success"
        diagnostics["stt_elapsed_ms"] = 111.0
        return "hello world"

    monkeypatch.setattr("apps.brains_worker.main.timedtext_list_tracks", fake_list_tracks)
    monkeypatch.setattr("apps.brains_worker.main._audio_fallback", fake_audio)

    client = TestClient(app)
    response = client.post(
        "/transcript",
        headers={"x-api-key": "test-key"},
        json={"source_id": "yt:5lQf89-AeFo", "allow_audio_fallback": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["transcript_source"] == "audio"
    assert payload["transcript_text"] == "hello world"
    assert payload["diagnostics"]["audio_fallback_attempted"] is True


def test_transcript_ytdlp_failure_returns_502(monkeypatch):
    monkeypatch.setenv("BRAINS_API_KEY", "test-key")
    from apps.brains_worker.main import app, _http_error

    def fake_list_tracks(video_id: str, *, proxy: str | None):
        return 200, [], None

    def fake_audio(video_id: str, proxy_url: str | None, diagnostics: dict, debug_keep_files: bool):
        diagnostics["audio_fallback_attempted"] = True
        diagnostics["audio_download_status"] = "failed"
        raise _http_error(502, "YTDLP_DOWNLOAD_FAILED", "download failed", diagnostics)

    monkeypatch.setattr("apps.brains_worker.main.timedtext_list_tracks", fake_list_tracks)
    monkeypatch.setattr("apps.brains_worker.main._audio_fallback", fake_audio)

    client = TestClient(app)
    response = client.post(
        "/transcript",
        headers={"x-api-key": "test-key"},
        json={"source_id": "yt:5lQf89-AeFo", "allow_audio_fallback": True},
    )

    assert response.status_code == 502
    assert response.json()["detail"]["error_code"] == "YTDLP_DOWNLOAD_FAILED"


def test_transcript_stt_failure_returns_503(monkeypatch):
    monkeypatch.setenv("BRAINS_API_KEY", "test-key")
    from apps.brains_worker.main import app, _http_error

    def fake_list_tracks(video_id: str, *, proxy: str | None):
        return 200, [], None

    def fake_audio(video_id: str, proxy_url: str | None, diagnostics: dict, debug_keep_files: bool):
        diagnostics["audio_fallback_attempted"] = True
        diagnostics["audio_download_status"] = "success"
        diagnostics["stt_status"] = "failed"
        raise _http_error(503, "STT_FAILED", "stt failed", diagnostics)

    monkeypatch.setattr("apps.brains_worker.main.timedtext_list_tracks", fake_list_tracks)
    monkeypatch.setattr("apps.brains_worker.main._audio_fallback", fake_audio)

    client = TestClient(app)
    response = client.post(
        "/transcript",
        headers={"x-api-key": "test-key"},
        json={"source_id": "yt:5lQf89-AeFo", "allow_audio_fallback": True},
    )

    assert response.status_code == 503
    assert response.json()["detail"]["error_code"] == "STT_FAILED"
