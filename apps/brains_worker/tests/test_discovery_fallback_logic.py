from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.brains_worker.ingest_types import WebdocDiscoveryDiagnostics, WebdocCandidate
from apps.brains_worker.webdocs_discovery import discover_webdocs


def test_discovery_fallback_logic(monkeypatch):
    def fake_ddg(*args, **kwargs):
        diag = WebdocDiscoveryDiagnostics(
            provider_used="ddg",
            parse_status="no_results",
            http_status=200,
            result_count_raw=0,
            result_count_usable=0,
        )
        return [], diag

    def fake_serpapi(*args, **kwargs):
        diag = WebdocDiscoveryDiagnostics(
            provider_used="serpapi",
            parse_status="ok",
            http_status=200,
            result_count_raw=1,
            result_count_usable=1,
        )
        return [
            WebdocCandidate(url="https://example.com", title="Example", snippet="test", provider="serpapi", rank=1)
        ], diag

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr("apps.brains_worker.webdocs_discovery.discover_ddg", fake_ddg)
    monkeypatch.setattr("apps.brains_worker.webdocs_discovery.discover_serpapi", fake_serpapi)

    candidates, diagnostics = discover_webdocs("query", 5, {}, None)

    assert diagnostics["webdocs_discovery_provider_used"] == "serpapi"
    assert diagnostics["webdocs_fallback_reason"] == "ddg_no_results"
    assert len(candidates) == 1
