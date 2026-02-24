from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.brains_worker.webdocs_discovery import _extract_ddg_results


def test_ddg_parser_from_fixture():
    fixture = Path(__file__).parent / "fixtures" / "ddg_sample.html"
    html = fixture.read_text(encoding="utf-8")
    candidates, raw_count = _extract_ddg_results(html)
    assert raw_count == 2
    assert len(candidates) == 2
    assert candidates[0].title == "Example One"
    assert candidates[0].url == "https://example.com/page1"
