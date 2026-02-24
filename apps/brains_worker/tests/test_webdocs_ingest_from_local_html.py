from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.brains_worker.ingest_types import DocCandidate, RunContext
from apps.brains_worker.webdocs_ingest import fetch_and_extract_text, process_doc


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200, content_type: str = "text/html"):
        self.text = text
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}


def test_webdocs_ingest_from_local_html(monkeypatch, tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "webdoc_sample.html"
    html = fixture.read_text(encoding="utf-8")

    def fake_get(*args, **kwargs):
        return _FakeResponse(html)

    monkeypatch.setenv("WEBDOCS_MIN_TEXT_CHARS", "200")
    monkeypatch.setattr("apps.brains_worker.webdocs_ingest.requests.get", fake_get)

    text, diag = fetch_and_extract_text("https://example.com/doc")
    assert diag.extract_status == "ok"
    assert len(text) >= 200

    run_dir = tmp_path / "run"
    run_ctx = RunContext(
        run_id="run1",
        brain_id="brain1",
        keyword="keyword",
        brain_root=tmp_path / "brain",
        run_dir=run_dir,
        payload={},
    )
    doc = DocCandidate(doc_id="doc1", title="Sample", url="https://example.com/doc", domain="example.com", provider="ddg")
    result = process_doc(doc, run_ctx, {})
    assert result.success
    assert (run_dir / "webdocs" / "text" / "doc1.txt").exists()
    assert (run_dir / "webdocs" / "meta" / "doc1.json").exists()
    assert (run_dir / "artifacts" / "webdocs_doc1.json").exists()
