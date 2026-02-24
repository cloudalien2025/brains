from __future__ import annotations

import importlib
import os
import sys
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_module(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("WORKER_API_KEY", "test-key")
    monkeypatch.setenv("BRAINS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAIN_CAPACITY_ITEMS", "10")

    def install_stub(module_name: str, attrs: dict[str, object]) -> None:
        if module_name in sys.modules:
            return
        stub = types.ModuleType(module_name)
        for key, value in attrs.items():
            setattr(stub, key, value)
        sys.modules[module_name] = stub

    install_stub(
        "apps.brains_worker.ingest_types",
        {
            "RunContext": type("RunContext", (), {}),
            "VideoCandidate": type("VideoCandidate", (), {}),
            "DocCandidate": type("DocCandidate", (), {}),
            "ItemResult": type("ItemResult", (), {}),
        },
    )
    install_stub(
        "apps.brains_worker.ingest_multisource",
        {"run_ingest_multisource": lambda *args, **kwargs: None},
    )
    install_stub(
        "apps.brains_worker.webdocs_discovery",
        {"is_serpapi_configured": lambda *args, **kwargs: False},
    )
    install_stub(
        "apps.brains_worker.transcribe",
        {
            "transcribe_audio": lambda *args, **kwargs: (None, None),
            "TranscriptionError": type("TranscriptionError", (Exception,), {}),
            "TranscriptionTimeout": type("TranscriptionTimeout", (Exception,), {}),
        },
    )
    install_stub(
        "apps.brains_worker.yt_audio",
        {
            "download_audio": lambda *args, **kwargs: (None, None),
            "AudioDownloadError": type("AudioDownloadError", (Exception,), {}),
        },
    )
    install_stub(
        "apps.brains_worker.db",
        {
            "db_session": lambda *args, **kwargs: None,
            "get_engine": lambda *args, **kwargs: None,
        },
    )
    install_stub(
        "apps.brains_worker.models",
        {
            "Artifact": type("Artifact", (), {}),
            "Brain": type("Brain", (), {}),
            "BrainSource": type("BrainSource", (), {}),
            "IngestionAttempt": type("IngestionAttempt", (), {}),
            "Source": type("Source", (), {}),
            "TranscriptIndex": type("TranscriptIndex", (), {}),
        },
    )
    install_stub(
        "apps.brains_worker.storage",
        {
            "upload_transcript": lambda *args, **kwargs: None,
            "build_s3_uri": lambda *args, **kwargs: "",
        },
    )
    module = importlib.import_module("apps.brains_worker.main")
    importlib.reload(module)
    return module


def _touch(path: Path, mtime: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_brain_stats_endpoint_counts(monkeypatch, tmp_path: Path):
    module = _build_module(monkeypatch, tmp_path)
    brain_slug = "brainy"
    runs_root = tmp_path / "brains" / brain_slug / "runs"

    run1 = runs_root / "run_1"
    run2 = runs_root / "run_2"

    _touch(run1 / "transcripts" / "yt_a.txt")
    _touch(run2 / "transcripts" / "yt_b.txt")
    _touch(run1 / "docs" / "text" / "doc1.txt")
    _touch(run2 / "webdocs" / "text" / "doc2.txt")
    _touch(run2 / "docs" / "doc3.txt")

    t1 = time.time() - 100
    t2 = time.time()
    _touch(run1 / "status.json", mtime=t1)
    _touch(run2 / "status.json", mtime=t2)

    payload = module.brain_stats(brain_slug, x_api_key="test-key")
    assert payload["brain_slug"] == brain_slug
    assert payload["youtube_items"] == 2
    assert payload["webdocs_items"] == 3
    assert payload["total_items"] == 5
    assert payload["capacity_items"] == 10
    assert abs(payload["fill_pct"] - 0.5) < 1e-6
    assert payload["last_run_id"] == "run_2"
    assert "updated_at" in payload
