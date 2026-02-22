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


def test_stage1_player_json_secondary_success(monkeypatch):
    monkeypatch.setenv("BRAINS_API_KEY", "test-key")
    from apps.brains_worker.main import app

    def fake_player(video_id: str, proxy: str | None):
        return 200, {
            "captions": {
                "playerCaptionsTracklistRenderer": {
                    "captionTracks": [
                        {
                            "languageCode": "en",
                            "kind": "asr",
                            "name": {"simpleText": "English"},
                            "baseUrl": "https://example.com/cap",
                        }
                    ]
                }
            }
        }, None

    def fake_download(url: str, proxy: str | None):
        return 200, "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello stage one", "text/vtt"

    monkeypatch.setattr("apps.brains_worker.main._youtube_watch_player_json", fake_player)
    monkeypatch.setattr("apps.brains_worker.main._download_text", fake_download)


    monkeypatch.setattr(
        "apps.brains_worker.main._run_ytdlp_subtitles",
        lambda *args, **kwargs: (None, {
            "ytdlp_subs_status": "no_subs",
            "ytdlp_subs_error_code": "NO_SUBTITLE_FILES",
            "ytdlp_subs_elapsed_ms": 1,
            "ytdlp_subs_lang": None,
            "ytdlp_subs_format": None,
            "ytdlp_subs_file_bytes": 0,
            "ytdlp_subs_file_path": None,
            "ytdlp_subs_stderr_sniff": None,
            "ytdlp_used_cookies": False,
        }),
    )

    client = TestClient(app)
    response = client.post("/transcript", headers={"x-api-key": "test-key"}, json={"video_id": "5lQf89-AeFo"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["transcript_source"] == "captions_player_json"
    assert "hello stage one" in payload["transcript_text"]
    assert payload["diagnostics"]["ytdlp_subs_attempted"] is True
    assert payload["diagnostics"]["ytdlp_subs_status"] == "no_subs"


def test_caption_json3_retry_parse(monkeypatch):
    monkeypatch.setenv("BRAINS_API_KEY", "test-key")
    from apps.brains_worker.main import app

    def fake_player(video_id: str, proxy: str | None):
        return 200, {
            "captions": {
                "playerCaptionsTracklistRenderer": {
                    "captionTracks": [
                        {
                            "languageCode": "en",
                            "kind": "asr",
                            "name": {"simpleText": "English"},
                            "baseUrl": "https://example.com/cap",
                        }
                    ]
                }
            }
        }, None

    def fake_download(url: str, proxy: str | None):
        return 200, '{"events":[{"segs":[{"utf8":"hello "},{"utf8":"json3"}]}]}', "application/json"

    monkeypatch.setattr("apps.brains_worker.main._youtube_watch_player_json", fake_player)
    monkeypatch.setattr("apps.brains_worker.main._download_text", fake_download)


    monkeypatch.setattr(
        "apps.brains_worker.main._run_ytdlp_subtitles",
        lambda *args, **kwargs: (None, {
            "ytdlp_subs_status": "no_subs",
            "ytdlp_subs_error_code": "NO_SUBTITLE_FILES",
            "ytdlp_subs_elapsed_ms": 1,
            "ytdlp_subs_lang": None,
            "ytdlp_subs_format": None,
            "ytdlp_subs_file_bytes": 0,
            "ytdlp_subs_file_path": None,
            "ytdlp_subs_stderr_sniff": None,
            "ytdlp_used_cookies": False,
        }),
    )

    client = TestClient(app)
    response = client.post("/transcript", headers={"x-api-key": "test-key"}, json={"video_id": "5lQf89-AeFo"})
    assert response.status_code == 200
    payload = response.json()
    assert "hello json3" in payload["transcript_text"]
    assert payload["diagnostics"]["caption_sniff_hint"] == "JSON"


def test_stage2_ytdlp_subs_success(monkeypatch):
    monkeypatch.setenv("BRAINS_API_KEY", "test-key")
    from apps.brains_worker.main import app

    def fake_player(video_id: str, proxy: str | None):
        return 200, {}, None

    def fake_subs(youtube_url: str, preferred_language: str | None, proxy: str | None, work_dir: Path, cookies_path: Path | None = None):
        return "hello stage two", {
            "ytdlp_subs_status": "success",
            "ytdlp_subs_error_code": None,
            "ytdlp_subs_elapsed_ms": 1,
            "ytdlp_subs_lang": "en",
            "ytdlp_subs_format": "vtt",
            "ytdlp_subs_file_bytes": 10,
            "ytdlp_subs_file_path": str(work_dir / "x.en.vtt"),
            "ytdlp_subs_stderr_sniff": None,
            "ytdlp_used_cookies": False,
        }

    monkeypatch.setattr("apps.brains_worker.main._youtube_watch_player_json", fake_player)
    monkeypatch.setattr("apps.brains_worker.main._run_ytdlp_subtitles", fake_subs)

    client = TestClient(app)
    response = client.post("/transcript", headers={"x-api-key": "test-key"}, json={"video_id": "5lQf89-AeFo"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["transcript_source"] == "subs_ytdlp"


def test_stage3_audio_openai_success(monkeypatch):
    monkeypatch.setenv("BRAINS_API_KEY", "test-key")
    from apps.brains_worker.main import app

    def fake_player(video_id: str, proxy: str | None):
        return 200, {}, None

    def fake_subs(youtube_url: str, preferred_language: str | None, proxy: str | None, work_dir: Path, cookies_path: Path | None = None):
        return None, {
            "ytdlp_subs_status": "failed",
            "ytdlp_subs_error_code": "NO_SUBTITLE_FILES",
            "ytdlp_subs_elapsed_ms": 1,
            "ytdlp_subs_lang": None,
            "ytdlp_subs_format": None,
            "ytdlp_subs_file_bytes": 0,
            "ytdlp_subs_file_path": None,
            "ytdlp_subs_stderr_sniff": None,
            "ytdlp_used_cookies": False,
        }

    def fake_audio(youtube_url: str, proxy: str | None, work_dir: Path):
        p = work_dir / "x.mp3"
        p.write_bytes(b"abc")
        return p, {
            "audio_download_status": "success",
            "audio_download_error_code": None,
            "audio_download_elapsed_ms": 1,
            "audio_file_ext": "mp3",
            "audio_file_bytes": 3,
            "audio_file_path": str(p),
        }

    def fake_stt(audio_path: Path, model: str, language: str | None, prompt: str | None):
        return "hello stage three", {
            "stt_provider": "openai",
            "stt_model": model,
            "stt_language": language,
            "stt_status": "success",
            "stt_error_code": None,
            "stt_elapsed_ms": 1,
            "transcript_chars": 17,
        }, 200, None, None

    monkeypatch.setattr("apps.brains_worker.main._youtube_watch_player_json", fake_player)
    monkeypatch.setattr("apps.brains_worker.main._run_ytdlp_subtitles", fake_subs)
    monkeypatch.setattr("apps.brains_worker.main._download_audio", fake_audio)
    monkeypatch.setattr("apps.brains_worker.main._transcribe_openai", fake_stt)

    client = TestClient(app)
    response = client.post(
        "/transcript",
        headers={"x-api-key": "test-key"},
        json={"video_id": "5lQf89-AeFo", "audio_fallback_enabled": True},
    )
    assert response.status_code == 200
    assert response.json()["transcript_source"] == "audio_openai_stt"


def test_fallback_disabled_returns_422(monkeypatch):
    monkeypatch.setenv("BRAINS_API_KEY", "test-key")
    from apps.brains_worker.main import app

    monkeypatch.setattr("apps.brains_worker.main._youtube_watch_player_json", lambda *args, **kwargs: (200, {}, None))
    monkeypatch.setattr(
        "apps.brains_worker.main._run_ytdlp_subtitles",
        lambda *args, **kwargs: (
            None,
            {
                "ytdlp_subs_status": "failed",
                "ytdlp_subs_error_code": "NO_SUBTITLE_FILES",
                "ytdlp_subs_elapsed_ms": 1,
                "ytdlp_subs_lang": None,
                "ytdlp_subs_format": None,
                "ytdlp_subs_file_bytes": 0,
                "ytdlp_subs_file_path": None,
            },
        ),
    )

    client = TestClient(app)
    response = client.post(
        "/transcript",
        headers={"x-api-key": "test-key"},
        json={"video_id": "5lQf89-AeFo", "audio_fallback_enabled": False},
    )
    assert response.status_code == 422
    payload = response.json()
    assert payload["error_code"] == "NO_CAPTIONS_AND_FALLBACK_DISABLED"


def test_missing_openai_key_when_stt_needed(monkeypatch):
    monkeypatch.setenv("BRAINS_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from apps.brains_worker.main import app

    monkeypatch.setattr("apps.brains_worker.main._youtube_watch_player_json", lambda *args, **kwargs: (200, {}, None))
    monkeypatch.setattr(
        "apps.brains_worker.main._run_ytdlp_subtitles",
        lambda *args, **kwargs: (
            None,
            {
                "ytdlp_subs_status": "failed",
                "ytdlp_subs_error_code": "NO_SUBTITLE_FILES",
                "ytdlp_subs_elapsed_ms": 1,
                "ytdlp_subs_lang": None,
                "ytdlp_subs_format": None,
                "ytdlp_subs_file_bytes": 0,
                "ytdlp_subs_file_path": None,
            },
        ),
    )

    def fake_audio(youtube_url: str, proxy: str | None, work_dir: Path):
        p = work_dir / "x.mp3"
        p.write_bytes(b"abc")
        return p, {
            "audio_download_status": "success",
            "audio_download_error_code": None,
            "audio_download_elapsed_ms": 1,
            "audio_file_ext": "mp3",
            "audio_file_bytes": 3,
            "audio_file_path": str(p),
        }

    monkeypatch.setattr("apps.brains_worker.main._download_audio", fake_audio)

    client = TestClient(app)
    response = client.post(
        "/transcript",
        headers={"x-api-key": "test-key"},
        json={"video_id": "5lQf89-AeFo", "audio_fallback_enabled": True},
    )
    assert response.status_code == 500
    assert response.json()["error_code"] == "OPENAI_KEY_MISSING"
