from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from apps.brains_worker.ingest_types import DocCandidate, ItemResult, RunContext, WebdocIngestDiagnostics


DEFAULT_USER_AGENT = os.getenv(
    "WEBDOCS_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _doc_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _extract_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "footer", "header", "nav", "form", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    return _clean_text(main.get_text(" ", strip=True))


def fetch_and_extract_text(url: str) -> tuple[str, WebdocIngestDiagnostics]:
    started = time.perf_counter()
    diag = WebdocIngestDiagnostics(url=url)
    timeout_s = int(os.getenv("WEBDOCS_TIMEOUT_S", "15"))
    min_chars = int(os.getenv("WEBDOCS_MIN_TEXT_CHARS", "400"))

    try:
        resp = requests.get(url, headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=timeout_s)
    except requests.RequestException as exc:
        diag.error_code = "request_failed"
        diag.error_message = str(exc)
        diag.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return "", diag

    diag.http_status = resp.status_code
    diag.content_type = resp.headers.get("Content-Type")
    if resp.status_code != 200:
        diag.error_code = "http_error"
        diag.error_message = f"HTTP {resp.status_code}"
        diag.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return "", diag

    content_type = (diag.content_type or "").lower()
    text = ""
    if "text/plain" in content_type:
        text = resp.text
    elif "text/html" in content_type or "application/xhtml+xml" in content_type or content_type == "":
        text = _extract_html_text(resp.text)
    else:
        diag.error_code = "unsupported_content_type"
        diag.error_message = f"Unsupported content type: {content_type}"
        diag.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return "", diag

    text = _clean_text(text)
    diag.extracted_chars = len(text)
    if diag.extracted_chars < min_chars:
        diag.extract_status = "too_short"
        diag.error_code = "too_short"
        diag.error_message = f"extracted_chars<{min_chars}"
        diag.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return text, diag

    diag.extract_status = "ok"
    diag.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return text, diag


def ingest_webdocs(run_dir: Path, candidates: list[DocCandidate], max_ingest: int) -> dict[str, Any]:
    run_dir = Path(run_dir)
    text_dir = run_dir / "webdocs" / "text"
    meta_dir = run_dir / "webdocs" / "meta"
    artifact_dir = run_dir / "artifacts"
    text_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rate_limit_ms = int(os.getenv("WEBDOCS_RATE_LIMIT_MS", "350"))
    successes = 0
    failures = 0
    failure_reasons: dict[str, int] = {}

    for cand in candidates[:max_ingest]:
        if rate_limit_ms > 0:
            time.sleep(rate_limit_ms / 1000.0)
        text, diag = fetch_and_extract_text(cand.url or "")
        doc_id = cand.doc_id or _doc_id(cand.url or "")
        meta_payload = {
            "doc_id": doc_id,
            "url": cand.url,
            "title": cand.title,
            "provider": cand.provider,
            "discovered_at": _now_iso(),
        }

        artifact_path = artifact_dir / f"webdocs_{doc_id}.json"
        artifact_path.write_text(json.dumps(asdict(diag), indent=2), encoding="utf-8")

        if diag.extract_status != "ok":
            failures += 1
            reason = diag.error_code or "extract_failed"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            continue

        (text_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        (meta_dir / f"{doc_id}.json").write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
        successes += 1

    return {
        "items_succeeded_webdocs": successes,
        "items_failed_webdocs": failures,
        "webdocs_failure_reasons": failure_reasons,
    }


def process_doc(doc: DocCandidate, run_ctx: RunContext, config: dict[str, Any]) -> ItemResult:
    run_dir = run_ctx.run_dir
    run_dir = Path(run_dir)
    run_dir.joinpath("webdocs", "text").mkdir(parents=True, exist_ok=True)
    run_dir.joinpath("webdocs", "meta").mkdir(parents=True, exist_ok=True)
    run_dir.joinpath("artifacts").mkdir(parents=True, exist_ok=True)

    rate_limit_ms = int(os.getenv("WEBDOCS_RATE_LIMIT_MS", "350"))
    if rate_limit_ms > 0:
        time.sleep(rate_limit_ms / 1000.0)

    if not doc.url:
        return ItemResult(item_id=doc.doc_id, success=False, error_code="missing_url", error_message="doc url missing")

    text, diag = fetch_and_extract_text(doc.url)
    doc_id = doc.doc_id or _doc_id(doc.url)

    artifact_path = run_dir / "artifacts" / f"webdocs_{doc_id}.json"
    artifact_path.write_text(json.dumps(asdict(diag), indent=2), encoding="utf-8")

    if diag.extract_status != "ok":
        return ItemResult(
            item_id=doc_id,
            success=False,
            error_code=diag.error_code or "extract_failed",
            error_message=diag.error_message or "webdocs extraction failed",
        )

    meta_payload = {
        "doc_id": doc_id,
        "url": doc.url,
        "title": doc.title,
        "provider": doc.provider,
        "discovered_at": _now_iso(),
    }
    (run_dir / "webdocs" / "text" / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    (run_dir / "webdocs" / "meta" / f"{doc_id}.json").write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    return ItemResult(item_id=doc_id, success=True, error_code=None, error_message=None)
