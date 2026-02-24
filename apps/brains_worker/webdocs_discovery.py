from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import asdict
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

from apps.brains_worker.ingest_types import DocCandidate, WebdocCandidate, WebdocDiscoveryDiagnostics


DEFAULT_USER_AGENT = os.getenv(
    "WEBDOCS_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
)


def is_serpapi_configured() -> bool:
    return bool((os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY") or "").strip())


def _blocked_domains() -> set[str]:
    raw = os.getenv(
        "WEBDOCS_BLOCKED_DOMAINS",
        "youtube.com,youtu.be,facebook.com,instagram.com,tiktok.com,linkedin.com",
    )
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _allowed_domains() -> set[str]:
    raw = os.getenv("WEBDOCS_ALLOWED_DOMAINS", "")
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _domain_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        return ""
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _normalize_url(url: str) -> str:
    try:
        parts = urlsplit(url)
    except Exception:
        return url
    scheme = (parts.scheme or "http").lower()
    netloc = (parts.netloc or "").lower()
    path = parts.path or ""
    query = parts.query or ""
    return urlunsplit((scheme, netloc, path, query, ""))


def _unwrap_ddg_redirect(url: str) -> str:
    try:
        parts = urlsplit(url)
    except Exception:
        return url
    if "duckduckgo.com" not in (parts.netloc or ""):
        return url
    qs = parse_qs(parts.query)
    target = qs.get("uddg", [None])[0]
    if not target:
        return url
    try:
        return unquote(target)
    except Exception:
        return target


def _filter_candidate(url: str, blocked: set[str], allowed: set[str]) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    domain = _domain_from_url(url)
    if not domain:
        return False
    if domain in blocked or any(domain.endswith(f".{b}") for b in blocked):
        return False
    if allowed and not (domain in allowed or any(domain.endswith(f".{a}") for a in allowed)):
        return False
    return True


def _extract_ddg_results(html: str) -> tuple[list[WebdocCandidate], int]:
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.select("a.result__a")
    if not anchors:
        anchors = soup.select("a.result__url")
    candidates: list[WebdocCandidate] = []
    rank = 0
    for anchor in anchors:
        href = anchor.get("href") or ""
        if not href:
            continue
        title = anchor.get_text(strip=True) or None
        snippet = None
        snippet_node = anchor.find_parent(class_="result")
        if snippet_node:
            snippet_el = snippet_node.select_one(".result__snippet")
            if snippet_el:
                snippet = snippet_el.get_text(" ", strip=True) or None
        rank += 1
        candidates.append(WebdocCandidate(url=href, title=title, snippet=snippet, provider="ddg", rank=rank))
    return candidates, len(anchors)


def discover_ddg(
    query: str,
    max_results: int,
    timeout_s: int,
    blocked_domains: set[str],
    allowed_domains: set[str],
) -> tuple[list[WebdocCandidate], WebdocDiscoveryDiagnostics]:
    started = time.perf_counter()
    diag = WebdocDiscoveryDiagnostics(provider_used="ddg", parse_status="error")
    url = "https://duckduckgo.com/html/"
    try:
        resp = requests.get(
            url,
            params={"q": query},
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=timeout_s,
        )
    except requests.RequestException as exc:
        diag.error_code = "request_failed"
        diag.error_message = str(exc)
        diag.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return [], diag

    diag.http_status = resp.status_code
    if resp.status_code in {403, 429}:
        diag.parse_status = "blocked"
        diag.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return [], diag
    if resp.status_code != 200:
        diag.parse_status = "error"
        diag.error_code = "http_error"
        diag.error_message = f"HTTP {resp.status_code}"
        diag.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return [], diag

    try:
        raw_candidates, raw_count = _extract_ddg_results(resp.text)
    except Exception as exc:
        diag.parse_status = "error"
        diag.error_code = "parse_failed"
        diag.error_message = str(exc)
        diag.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return [], diag

    diag.result_count_raw = raw_count

    seen: set[str] = set()
    filtered: list[WebdocCandidate] = []
    for cand in raw_candidates:
        url = _unwrap_ddg_redirect(cand.url)
        if not _filter_candidate(url, blocked_domains, allowed_domains):
            continue
        normalized = _normalize_url(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        cand.url = url
        filtered.append(cand)
        if len(filtered) >= max_results:
            break

    diag.result_count_usable = len(filtered)
    if not filtered:
        diag.parse_status = "no_results"
    else:
        diag.parse_status = "ok"
    diag.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return filtered, diag


def discover_serpapi(
    query: str,
    max_results: int,
    timeout_s: int,
    blocked_domains: set[str],
    allowed_domains: set[str],
) -> tuple[list[WebdocCandidate], WebdocDiscoveryDiagnostics]:
    started = time.perf_counter()
    diag = WebdocDiscoveryDiagnostics(provider_used="serpapi", parse_status="error")
    api_key = (os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY") or "").strip()
    if not api_key:
        diag.parse_status = "error"
        diag.error_code = "serpapi_key_missing"
        diag.error_message = "SERPAPI_API_KEY missing"
        diag.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return [], diag

    try:
        resp = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": query, "num": max_results, "api_key": api_key},
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=timeout_s,
        )
    except requests.RequestException as exc:
        diag.error_code = "request_failed"
        diag.error_message = str(exc)
        diag.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return [], diag

    diag.http_status = resp.status_code
    if resp.status_code in {403, 429}:
        diag.parse_status = "blocked"
        diag.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return [], diag
    if resp.status_code != 200:
        diag.parse_status = "error"
        diag.error_code = "http_error"
        diag.error_message = f"HTTP {resp.status_code}"
        diag.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return [], diag

    try:
        payload = resp.json()
    except json.JSONDecodeError as exc:
        diag.parse_status = "error"
        diag.error_code = "json_parse_failed"
        diag.error_message = str(exc)
        diag.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return [], diag

    organic = payload.get("organic_results") or []
    diag.result_count_raw = len(organic) if isinstance(organic, list) else 0
    seen: set[str] = set()
    filtered: list[WebdocCandidate] = []
    if isinstance(organic, list):
        for rank, item in enumerate(organic, start=1):
            if not isinstance(item, dict):
                continue
            url = item.get("link") or item.get("url") or ""
            if not url:
                continue
            if not _filter_candidate(url, blocked_domains, allowed_domains):
                continue
            normalized = _normalize_url(url)
            if normalized in seen:
                continue
            seen.add(normalized)
            filtered.append(
                WebdocCandidate(
                    url=url,
                    title=item.get("title"),
                    snippet=item.get("snippet"),
                    provider="serpapi",
                    rank=rank,
                )
            )
            if len(filtered) >= max_results:
                break

    diag.result_count_usable = len(filtered)
    if not filtered:
        diag.parse_status = "no_results"
    else:
        diag.parse_status = "ok"
    diag.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return filtered, diag


def _build_doc_candidates(candidates: list[WebdocCandidate]) -> list[DocCandidate]:
    docs: list[DocCandidate] = []
    for cand in candidates:
        if not cand.url:
            continue
        doc_id = hashlib.sha1(cand.url.encode("utf-8")).hexdigest()[:12]
        docs.append(
            DocCandidate(
                doc_id=doc_id,
                title=cand.title,
                url=cand.url,
                domain=_domain_from_url(cand.url),
                provider=cand.provider,
            )
        )
    return docs


def discover_webdocs(
    keyword: str,
    max_candidates: int,
    config: dict[str, Any],
    status_writer: Any,
) -> tuple[list[DocCandidate], dict[str, Any]]:
    timeout_s = int(os.getenv("WEBDOCS_TIMEOUT_S", "15"))
    max_results = int(os.getenv("WEBDOCS_MAX_RESULTS", "20"))
    max_results = min(max_candidates, max_results)
    blocked = _blocked_domains()
    allowed = _allowed_domains()

    ddg_candidates, ddg_diag = discover_ddg(keyword, max_results, timeout_s, blocked, allowed)
    provider_used = "ddg"
    fallback_reason: str | None = None
    serp_candidates: list[WebdocCandidate] = []
    serp_diag: WebdocDiscoveryDiagnostics | None = None

    ddg_usable = ddg_diag.result_count_usable or 0
    if ddg_diag.parse_status in {"blocked", "error"}:
        if ddg_diag.parse_status == "blocked":
            fallback_reason = "ddg_blocked"
        else:
            fallback_reason = "ddg_parse_failed"
    elif ddg_usable == 0:
        fallback_reason = "ddg_no_results"

    if fallback_reason:
        if is_serpapi_configured():
            serp_candidates, serp_diag = discover_serpapi(keyword, max_results, timeout_s, blocked, allowed)
            provider_used = "serpapi"
        else:
            fallback_reason = "serpapi_key_missing"

    final_candidates = serp_candidates if provider_used == "serpapi" else ddg_candidates
    doc_candidates = _build_doc_candidates(final_candidates)

    diagnostics_bundle = {
        "webdocs_discovery_provider_used": provider_used,
        "webdocs_fallback_reason": fallback_reason,
        "ddg": asdict(ddg_diag),
        "serpapi": asdict(serp_diag) if serp_diag else None,
    }

    if status_writer is not None:
        status_writer.update(
            webdocs_discovery_provider_used=provider_used,
            webdocs_fallback_reason=fallback_reason,
            webdocs_discovery_diagnostics=diagnostics_bundle,
        )

    return doc_candidates, diagnostics_bundle
