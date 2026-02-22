from __future__ import annotations

import os
import re
import time

import pytest
import requests
from playwright.sync_api import sync_playwright


@pytest.mark.e2e
def test_streamlit_selected_new_contract_and_diagnostics():
    app_url = os.getenv("BRAINS_STREAMLIT_URL")
    worker_url = os.getenv("BRAINS_WORKER_URL")
    api_key = os.getenv("BRAINS_WORKER_API_KEY")
    if not app_url or not worker_url or not api_key:
        pytest.skip("BRAINS_STREAMLIT_URL/BRAINS_WORKER_URL/BRAINS_WORKER_API_KEY required")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(app_url, wait_until="domcontentloaded")
        page.get_by_label("Keyword").fill("brilliant directories")
        slider = page.get_by_label("N new videos")
        slider.fill("5")
        expect_text = page.locator("text=N new videos")
        expect_text.wait_for(timeout=10000)
        page.get_by_role("button", name="Discover + Ingest New Videos").click()
        page.wait_for_selector("text=Ingest started. run_id=", timeout=30000)
        run_line = page.locator("text=Ingest started. run_id=").first.inner_text()
        run_id = re.search(r"run_id=([^\s]+)", run_line).group(1)
        browser.close()

    headers = {"X-Api-Key": api_key}
    report = None
    for _ in range(120):
        resp = requests.get(f"{worker_url.rstrip('/')}/v1/runs/{run_id}/report", headers=headers, timeout=30)
        if resp.status_code == 200:
            report = resp.json()
            if report.get("status") in {"completed", "success", "partial_success", "no_captions", "blocked", "failed", "completed_with_errors"}:
                break
        time.sleep(2)

    assert report is not None
    assert report["selected_new"] <= 5
    assert report["transcripts_failed"] <= report["selected_new"]

    diagnostics = requests.get(f"{worker_url.rstrip('/')}/v1/runs/{run_id}/diagnostics", headers=headers, timeout=30)
    assert diagnostics.status_code == 200
    payload = diagnostics.json()
    assert "diagnostics_path" in payload
