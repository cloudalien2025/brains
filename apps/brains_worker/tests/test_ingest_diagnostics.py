from pathlib import Path
import importlib
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_caption_probe_blocked_empty(monkeypatch):
    main = importlib.import_module("apps.brains_worker.main")

    class Resp:
        status_code = 200
        text = ""

    class Session:
        def get(self, *_args, **_kwargs):
            return Resp()

    tracks, status, probe_status, response_len, retry_diag = main._caption_tracks(Session(), "abc")
    assert tracks == []
    assert status == 200
    assert probe_status == "blocked_empty"
    assert response_len == 0
    assert retry_diag["retry_count"] == 0


def test_write_report_includes_request_and_audio_fields(tmp_path, monkeypatch):
    main = importlib.import_module("apps.brains_worker.main")
    root = tmp_path / "brain"
    (root / "runs" / "run1").mkdir(parents=True)
    attempts_path = root / "runs" / "run1" / "transcript_attempts.jsonl"
    attempts_path.write_text(
        "\n".join(
            [
                '{"phase":"caption_probe","probe_status":"blocked","success":false,"proxy_enabled":true}',
                '{"phase":"audio_fallback","success":true}',
                '{"phase":"audio_fallback","success":false,"error_code":"audio_download_failed","stderr_tail":"x"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_payload = {
        "brain_id": "b",
        "status": "blocked",
        "requested_new": 5,
        "selected_new": 5,
        "eligible_count": 2,
        "eligible_shortfall": 0,
        "ingested_video_ids": [],
        "total_audio_minutes": 1.2,
    }
    main.write_report(root, "run1", run_payload)
    report = main.load_json(root / "runs" / "run1" / "report.json", {})
    assert report["requested_new"] == 5
    assert report["selected_new"] == 5
    assert report["eligible_count"] == 2
    assert report["caption_probe_blocked"] == 1
    assert report["audio_success"] == 1
    assert report["total_audio_minutes"] == 1.2
    assert report["audio_failure_reasons"]["audio_download_failed"] == 1
    assert report["proxy_enabled"] is True


def test_health_reports_tooling(monkeypatch):
    main = importlib.import_module("apps.brains_worker.main")
    monkeypatch.setenv("WORKER_API_KEY", "x")
    monkeypatch.setattr(main, "_yt_dlp_exists", lambda: True)
    monkeypatch.setattr(main, "_ffmpeg_exists", lambda: False)
    client = TestClient(main.app)
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["yt_dlp_available"] is True
    assert body["ffmpeg_available"] is False
