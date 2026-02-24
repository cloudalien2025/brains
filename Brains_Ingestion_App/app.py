from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

try:
    from brains.status_adapter import (
        normalize_run_status,
        normalize_report,
        normalize_run_for_display,
        report_counters,
        report_not_ready,
    )
except ModuleNotFoundError:
    from Brains_Ingestion_App.brains.status_adapter import (
        normalize_run_status,
        normalize_report,
        normalize_run_for_display,
        report_counters,
        report_not_ready,
    )

TERMINAL_RUN_STATES = {"completed", "completed_with_errors", "failed", "success", "partial_success", "no_captions", "blocked"}
WORKER_DEFAULT_TIMEOUT = 90


def _secret(name: str) -> str:
    try:
        value = st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        value = None
    return str(value or "").strip()


def _snippet(text: str, limit: int = 500) -> str:
    return (text or "")[:limit]


def _format_duration(seconds: Any) -> str:
    try:
        value = int(seconds)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    mins = value // 60
    secs = value % 60
    return f"{mins}:{secs:02d}"


def compute_brain_fill(brain_slug: str) -> dict[str, Any]:
    capacity_fallback = int(os.getenv("BRAIN_CAPACITY_ITEMS", "500"))
    fallback = {
        "brain_slug": brain_slug,
        "capacity_items": capacity_fallback,
        "total_items": 0,
        "youtube_items": 0,
        "webdocs_items": 0,
        "fill_pct": 0.0,
        "available": False,
    }
    if not worker_url or not worker_api_key:
        return fallback
    try:
        response = worker_request("GET", f"/v1/brains/{brain_slug}/stats")
        if not response.ok:
            return fallback
        payload = response.json()
        if not isinstance(payload, dict):
            return fallback
    except requests.RequestException:
        return fallback

    fill_pct = float(payload.get("fill_pct") or 0.0)
    fill_pct = max(0.0, min(fill_pct, 1.0))

    return {
        "brain_slug": brain_slug,
        "capacity_items": int(payload.get("capacity_items") or capacity_fallback),
        "total_items": int(payload.get("total_items") or 0),
        "youtube_items": int(payload.get("youtube_items") or 0),
        "webdocs_items": int(payload.get("webdocs_items") or 0),
        "fill_pct": fill_pct,
        "available": True,
    }


def render_brain_fill(fill: dict[str, Any]) -> None:
    pct = float(fill.get("fill_pct") or 0.0) * 100.0
    pct_display = f"{pct:.1f}%"
    total = int(fill.get("total_items") or 0)
    capacity = int(fill.get("capacity_items") or 0)
    yt = int(fill.get("youtube_items") or 0)
    docs = int(fill.get("webdocs_items") or 0)

    st.markdown(
        f"""
        <div style="display:flex; gap:24px; align-items:flex-end;">
          <div style="
            width:140px;
            height:260px;
            border:2px solid #28323a;
            border-radius:70px;
            background:linear-gradient(180deg, #f2f5f7 0%, #e7ecef 100%);
            position:relative;
            overflow:hidden;">
            <div style="
              position:absolute;
              bottom:0;
              left:0;
              width:100%;
              height:{pct:.3f}%;
              background:linear-gradient(180deg, #32c26b 0%, #1f8a52 100%);
              transition:height 0.6s ease;">
            </div>
            <div style="
              position:absolute;
              inset:0;
              box-shadow:inset 0 0 18px rgba(0,0,0,0.12);">
            </div>
          </div>
          <div style="min-width:220px;">
            <div style="font-size:20px; font-weight:600;">{total} items ingested</div>
            <div style="color:#4b5964; margin-top:4px;">{total} / {capacity} capacity ({pct_display})</div>
            <div style="margin-top:12px; color:#4b5964;">YouTube: {yt}</div>
            <div style="color:#4b5964;">WebDocs: {docs}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not fill.get("available"):
        st.info("Stats unavailable")
    else:
        st.caption("TEST_HOOK:STATS_OK")


def _render_worker_http_error(response: requests.Response) -> None:
    status = response.status_code
    body_snippet = _snippet(response.text)

    if status == 404:
        st.error(
            "Worker is reachable but /v1/health not found. This usually means the droplet is still "
            "running the old worker. Deploy/restart the new Brains Worker v1 service and confirm "
            "https://worker.aiohut.com/v1/health returns 200."
        )
    elif status == 401:
        st.error("Invalid worker API key. Check Streamlit secrets worker_api_key.")
    else:
        st.error(f"Worker request failed with HTTP {status}. Response: {body_snippet}")


st.set_page_config(page_title="Brains ingestion", layout="wide")
st.title("Brains ingestion")
st.caption("Select a Brain, ingest new videos by keyword, generate a Brain Pack.")
st.caption("Discovery powered by YouTube Data API (server-side).")
st.caption("TEST_HOOK:APP_LOADED")

for key, default in {
    "brains": [],
    "selected_brain_id": None,
    "run_id": None,
    "last_report": None,
    "last_error": None,
    "last_run_data": None,
    "brain_pack_id": None,
}.items():
    st.session_state.setdefault(key, default)

worker_url = _secret("worker_url")
worker_api_key = _secret("worker_api_key")


def worker_request(method: str, path: str, json: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> requests.Response:
    url = worker_url.rstrip("/") + path
    headers = {"X-Api-Key": worker_api_key}
    return requests.request(method, url, headers=headers, json=json, params=params, timeout=WORKER_DEFAULT_TIMEOUT)


def _auto_refresh_report(run_id: str, placeholder: st.delta_generator.DeltaGenerator, interval_s: float = 2.0, max_wait_s: float = 30.0) -> None:
    report_snapshot = st.session_state.get("last_report")
    if isinstance(report_snapshot, dict) and not report_not_ready(report_snapshot):
        return

    state = st.session_state.get("report_poll_state") or {}
    if state.get("run_id") != run_id:
        state = {"run_id": run_id, "started_at": time.time(), "timed_out": False}

    elapsed = time.time() - float(state.get("started_at") or time.time())
    if elapsed >= max_wait_s:
        state["timed_out"] = True
        st.session_state["report_poll_state"] = state
        placeholder.info("Report not ready yet—check back in a moment.")
        return

    try:
        response = worker_request("GET", f"/v1/runs/{run_id}/report")
        if response.ok:
            payload = response.json() if response.text else {}
            st.session_state["last_report"] = payload
            if report_not_ready(payload):
                placeholder.info("Report still building… refreshing")
                st.session_state["report_poll_state"] = state
                time.sleep(interval_s)
                st.rerun()
            else:
                placeholder.empty()
            return
        if response.status_code == 202:
            placeholder.info("Report still building… refreshing")
            st.session_state["report_poll_state"] = state
            time.sleep(interval_s)
            st.rerun()
            return
        _render_worker_http_error(response)
    except requests.RequestException as exc:
        st.error(f"Failed to fetch report: {exc}")


def fetch_run_files(run_id: str) -> dict[str, Any] | None:
    try:
        response = worker_request("GET", f"/v1/runs/{run_id}/files")
        if response.ok:
            payload = response.json()
            return payload if isinstance(payload, dict) else None
        if response.status_code in {404, 202}:
            return None
        _render_worker_http_error(response)
        return None
    except requests.RequestException:
        return None


def poll_run_status_with_retry(run_id: str, retries: int = 2) -> tuple[dict[str, Any] | None, str | None]:
    for attempt in range(retries + 1):
        try:
            response = worker_request("GET", f"/v1/runs/{run_id}")
            if response.ok:
                return response.json(), None
            _render_worker_http_error(response)
            return None, "http_error"
        except requests.ReadTimeout:
            if attempt < retries:
                time.sleep(attempt + 1)
                continue
            return None, "timeout"
        except requests.RequestException as exc:
            return None, str(exc)
    return None, "timeout"


def normalize_brain(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalize worker brain objects to a consistent shape for the UI."""
    brain_id = obj.get("brain_id") or obj.get("id") or obj.get("uuid")
    brain_name = obj.get("brain_name") or obj.get("name") or obj.get("title")
    brain_type = obj.get("brain_type") or obj.get("type") or obj.get("kind")
    default_keyword = obj.get("default_keyword") or obj.get("keyword") or obj.get("defaultKey")
    return {
        "brain_id": brain_id,
        "brain_name": brain_name,
        "brain_type": brain_type,
        "default_keyword": default_keyword,
        "_raw": obj,
    }


def build_create_brain_payload(brain_name: str, brain_type: str, default_keyword: str | None) -> dict[str, Any]:
    """Build a payload compatible with worker schema variations."""
    kw = (default_keyword or "").strip() or None
    name = brain_name.strip()

    payload = {
        "name": name,
        "brain_type": brain_type,
        "default_keyword": kw,
        "brain_name": name,
        "type": brain_type,
    }
    return payload


def fetch_brains() -> None:
    if not worker_url or not worker_api_key:
        return
    try:
        response = worker_request("GET", "/v1/brains")
        if response.ok:
            payload = response.json()
            raw_brains = payload if isinstance(payload, list) else payload.get("items", [])
            st.session_state["brains"] = [normalize_brain(b) for b in raw_brains if isinstance(b, dict)]
        else:
            st.session_state["last_error"] = f"HTTP {response.status_code}: {_snippet(response.text)}"
    except requests.RequestException as exc:
        st.session_state["last_error"] = str(exc)


st.subheader("Worker Status")
worker_healthy = False

if not worker_url:
    st.error("Missing Streamlit secret: worker_url")
elif not worker_api_key:
    st.error("Missing Streamlit secret: worker_api_key")
else:
    try:
        health_response = worker_request("GET", "/v1/health")
        if health_response.ok:
            health_json = health_response.json() if health_response.text else {}
            if isinstance(health_json, dict) and str(health_json.get("status", "")).lower() == "ok":
                worker_healthy = True
                st.success("✅ Worker ACTIVE")
            else:
                st.error(f"❌ Worker ERROR: unexpected health payload: {_snippet(health_response.text)}")
        else:
            st.error("❌ Worker ERROR")
            _render_worker_http_error(health_response)
    except requests.RequestException as exc:
        st.error(f"❌ Worker ERROR: {exc}")

if worker_healthy:
    fetch_brains()

st.subheader("Brain Fill Level")
selected_brain_id_for_fill = st.session_state.get("selected_brain_id")
if selected_brain_id_for_fill:
    fill = compute_brain_fill(str(selected_brain_id_for_fill))
    render_brain_fill(fill)
else:
    st.info("Select a Brain to see the fill level.")

st.subheader("Brain")
brains = st.session_state.get("brains", [])

if brains:
    options = {
        f"{b.get('brain_name', 'Unnamed')} ({b.get('brain_type', 'Unknown')})": b.get("brain_id")
        for b in brains
    }
    labels = list(options.keys())

    selected_id = st.session_state.get("selected_brain_id")
    default_index = 0
    if selected_id:
        for idx, label in enumerate(labels):
            if options[label] == selected_id:
                default_index = idx
                break

    selected_label = st.selectbox("Brain", labels, index=default_index)
    st.session_state["selected_brain_id"] = options[selected_label]
else:
    st.info("No brains found yet. Create one below.")

with st.expander("Create Brain", expanded=not bool(brains)):
    new_name = st.text_input("brain_name", key="new_brain_name")
    new_type = st.selectbox("brain_type", ["BD", "UAP"], key="new_brain_type")
    new_default_keyword = st.text_input("default_keyword (optional)", key="new_brain_keyword")

    if st.button("Create Brain"):
        if not worker_healthy:
            st.error("Worker must be healthy before creating a Brain.")
        elif not new_name.strip():
            st.error("brain_name is required.")
        else:
            payload = build_create_brain_payload(new_name, new_type, new_default_keyword)
            st.session_state["last_create_brain_payload"] = payload
            try:
                response = worker_request("POST", "/v1/brains", json=payload)
                if response.ok:
                    created_norm = normalize_brain(response.json())
                    created_id = created_norm.get("brain_id")
                    st.success("Brain created.")
                    fetch_brains()
                    if created_id:
                        st.session_state["selected_brain_id"] = created_id
                    else:
                        st.error(f"Brain created but brain_id missing in response: {_snippet(response.text)}")
                    st.rerun()
                else:
                    _render_worker_http_error(response)
            except requests.RequestException as exc:
                st.error(f"Failed to create brain: {exc}")

with st.expander("Create Brain debug", expanded=False):
    st.write({"last_create_brain_payload": st.session_state.get("last_create_brain_payload")})

with st.expander("Ingest debug", expanded=False):
    st.write({"last_ingest_payload": st.session_state.get("last_ingest_payload")})

selected_brain = next((b for b in brains if b.get("brain_id") == st.session_state.get("selected_brain_id")), None)
default_keyword = (selected_brain or {}).get("default_keyword") or ""

st.subheader("Ingest Controls")
keyword = st.text_input("Keyword", value=default_keyword, key="ingest_keyword")
n_new = st.number_input("N new videos", min_value=1, max_value=500, value=20, step=1)

with st.expander("Longform settings (advanced)"):
    chunk_seconds = st.number_input("chunk_seconds", min_value=60, max_value=3600, value=600, step=30)
    overlap_seconds = st.number_input("overlap_seconds", min_value=0, max_value=300, value=15, step=1)

if st.button("Discover + Ingest New Videos", type="primary", disabled=not worker_healthy):
    brain_id = st.session_state.get("selected_brain_id")
    if not brain_id:
        st.error("Please select or create a Brain first.")
    elif not keyword.strip():
        st.error("Keyword is required.")
    else:
        ingest_payload = {
            "keyword": keyword.strip(),
            "selected_new": int(n_new),
            "n_new_videos": int(n_new),
            "max_candidates": 50,
            "mode": "audio_first",
            "longform": {
                "chunk_seconds": int(chunk_seconds),
                "overlap_seconds": int(overlap_seconds),
            },
        }
        st.session_state["last_ingest_payload"] = ingest_payload
        st.write({"ingest_payload": ingest_payload})
        try:
            response = worker_request("POST", f"/v1/brains/{brain_id}/ingest", json=ingest_payload)
            if response.ok:
                ingest_json = response.json()
                run_id = ingest_json.get("run_id")
                if run_id:
                    st.session_state["run_id"] = run_id
                    st.session_state["last_report"] = None
                    st.session_state["brain_pack_id"] = None
                    st.success(f"Ingest started. run_id={run_id}")
                else:
                    st.error(f"Ingest started but run_id missing. Response: {_snippet(response.text)}")
            else:
                _render_worker_http_error(response)
        except requests.RequestException as exc:
            st.error(f"Failed to start ingest: {exc}")

run_id = st.session_state.get("run_id")
if run_id:
    st.subheader("Run Progress")
    polling_enabled = st.checkbox("Polling", value=True)

    run_data: dict[str, Any] | None = None
    run_data, run_error = poll_run_status_with_retry(run_id)
    if run_error == "timeout":
        st.warning("Worker reachable but run status timed out; retrying…")
    elif run_error and run_error != "http_error":
        st.warning(f"Run status fetch failed temporarily; retrying. Details: {run_error}")

    if run_data:
        run_data = normalize_run_status(run_data)
        st.session_state["last_run_data"] = run_data
        status = str(run_data.get("status", "unknown"))
        selected_new = int(run_data.get("selected_new") or 0)
        completed = int(run_data.get("completed") or 0)

        col1, col2, col3 = st.columns(3)
        col1.metric("status", status)
        col2.metric("candidates_found", int(run_data.get("candidates_found") or 0))
        col3.metric("selected_new", selected_new)

        col4, col5 = st.columns(2)
        col4.metric("skipped_duplicates", int(run_data.get("skipped_duplicates") or 0))
        col5.metric("failed", int(run_data.get("failed") or 0))

        st.metric("completed", completed)

        progress_total = selected_new if selected_new > 0 else 1
        st.progress(min(completed / progress_total, 1.0))

        if status in TERMINAL_RUN_STATES:
            report_placeholder = st.empty()
            _auto_refresh_report(run_id, report_placeholder)

        files_json = fetch_run_files(run_id)
        run_display_payload = dict(run_data)
        report_snapshot = st.session_state.get("last_report")
        if isinstance(report_snapshot, dict):
            run_display_payload["report"] = report_snapshot
        display = normalize_run_for_display(run_display_payload, files_json)

        summary_tab, selected_tab, raw_tab = st.tabs(["Summary", "Selected & Ingested", "Raw JSON"])

        with summary_tab:
            st.subheader("Counters")
            c1, c2, c3 = st.columns(3)
            c1.metric("candidates_found_youtube", int(run_data.get("candidates_found_youtube") or 0))
            c2.metric("candidates_found_webdocs", int(run_data.get("candidates_found_webdocs") or 0))
            c3.metric("items_succeeded_total", int(run_data.get("items_succeeded_total") or 0))

            c4, c5, c6 = st.columns(3)
            c4.metric("items_succeeded_youtube", int(run_data.get("items_succeeded_youtube") or 0))
            c5.metric("items_succeeded_webdocs", int(run_data.get("items_succeeded_webdocs") or 0))
            c6.metric("items_failed_total", int(run_data.get("items_failed_total") or 0))

            c7, c8, c9 = st.columns(3)
            c7.metric("items_failed_youtube", int(run_data.get("items_failed_youtube") or 0))
            c8.metric("items_failed_webdocs", int(run_data.get("items_failed_webdocs") or 0))
            c9.metric("selected_new_webdocs", int(run_data.get("selected_new_webdocs") or 0))

            st.write({
                "webdocs_discovery_provider_used": run_data.get("webdocs_discovery_provider_used"),
                "webdocs_fallback_reason": run_data.get("webdocs_fallback_reason"),
            })

            with st.expander("Webdocs failure reasons"):
                st.json(run_data.get("webdocs_failure_reasons") or {})

            current = run_data.get("current") or {}
            if isinstance(current, dict) and current:
                st.write(
                    {
                        "current.source_id": current.get("source_id"),
                        "current.stage": current.get("stage"),
                        "current.detail": current.get("detail"),
                    }
                )

        with selected_tab:
            st.subheader("YouTube (Selected)")
            selected_youtube_rows = []
            for item in display.get("selected_youtube", []):
                selected_youtube_rows.append(
                    {
                        "video_id": item.get("video_id"),
                        "title": item.get("title"),
                        "uploader": item.get("uploader"),
                        "duration": _format_duration(item.get("duration_s")),
                        "url": item.get("url"),
                    }
                )
            if selected_youtube_rows:
                st.dataframe(selected_youtube_rows, use_container_width=True)
            else:
                st.info("No selected YouTube items available yet.")

            st.subheader("WebDocs (Selected)")
            selected_webdocs_rows = []
            for item in display.get("selected_webdocs", []):
                selected_webdocs_rows.append(
                    {
                        "doc_id": item.get("doc_id"),
                        "title": item.get("title"),
                        "domain": item.get("domain"),
                        "provider": item.get("provider"),
                        "url": item.get("url"),
                    }
                )
            if selected_webdocs_rows:
                st.dataframe(selected_webdocs_rows, use_container_width=True)
            else:
                st.info("No selected WebDocs available yet.")

            st.subheader("Succeeded")
            yt_success = display.get("succeeded_youtube", [])
            doc_success = display.get("succeeded_webdocs", [])
            st.write(
                {
                    "youtube_succeeded": len(yt_success),
                    "webdocs_succeeded": len(doc_success),
                }
            )

            yt_success_rows = [
                {
                    "video_id": item.get("video_id"),
                    "transcript": "✅" if item.get("transcript_exists") else "—",
                    "artifact_path": item.get("artifact_path") or "",
                }
                for item in yt_success
            ]
            if yt_success_rows:
                st.dataframe(yt_success_rows, use_container_width=True)
            else:
                st.info("No succeeded YouTube items yet.")

            doc_success_rows = [
                {
                    "doc_id": item.get("doc_id"),
                    "text": "✅" if item.get("text_exists") else "—",
                    "artifact_path": item.get("artifact_path") or "",
                }
                for item in doc_success
            ]
            if doc_success_rows:
                st.dataframe(doc_success_rows, use_container_width=True)
            else:
                st.info("No succeeded WebDocs yet.")

            st.subheader("Failed")
            failures = []
            for item in display.get("failed_youtube", []):
                failures.append(
                    {
                        "type": "youtube",
                        "item_id": item.get("video_id"),
                        "stage": item.get("stage"),
                        "error_code": item.get("error_code"),
                        "error_message": item.get("error_message"),
                    }
                )
            for item in display.get("failed_webdocs", []):
                failures.append(
                    {
                        "type": "webdocs",
                        "item_id": item.get("doc_id"),
                        "stage": item.get("stage"),
                        "error_code": item.get("error_code"),
                        "error_message": item.get("error_message"),
                    }
                )
            if failures:
                st.dataframe(failures, use_container_width=True)
            else:
                st.info("No failed items recorded yet.")

        with raw_tab:
            with st.expander("Raw run JSON"):
                st.json(run_data)

        if status not in TERMINAL_RUN_STATES and polling_enabled:
            time.sleep(1.5)
            st.rerun()

if run_id:
    st.subheader("Run Report")
    report_snapshot = st.session_state.get("last_report")
    run_data = st.session_state.get("last_run_data")

    report_ready = isinstance(report_snapshot, dict) and not report_not_ready(report_snapshot)
    report_payload = normalize_report(report_snapshot, run_data) if report_ready else {}
    counters = report_counters(report_payload if report_ready else None, run_data)

    report_state = st.session_state.get("report_poll_state") or {}
    if report_not_ready(report_snapshot):
        st.info("Report still building… refreshing")
    elif report_state.get("timed_out"):
        st.info("Report not ready yet—check back in a moment.")

    st.write(
        {
            "items_succeeded_total": counters.get("items_succeeded_total"),
            "items_failed_total": counters.get("items_failed_total"),
            "total_audio_minutes": counters.get("total_audio_minutes"),
        }
    )

    with st.expander("Raw report JSON"):
        st.json(report_snapshot or {})

    existing_pack_id = report_payload.get("brain_pack_id") if report_ready else None
    if existing_pack_id:
        download_url = f"{worker_url.rstrip('/')}/v1/brain-packs/{existing_pack_id}/download"
        st.link_button("Download Brain Pack", download_url)
    elif report_ready:
        st.info("No brain_pack_id in report yet.")
        if st.button("Build Brain Pack"):
            run_id_for_pack = st.session_state.get("run_id")
            if not run_id_for_pack:
                st.error("Missing run_id.")
            else:
                try:
                    build_response = worker_request("POST", f"/v1/runs/{run_id_for_pack}/brain-pack")
                    if build_response.ok:
                        build_json = build_response.json()
                        st.session_state["brain_pack_id"] = build_json.get("brain_pack_id")
                    else:
                        _render_worker_http_error(build_response)
                except requests.RequestException as exc:
                    st.error(f"Failed to start brain pack build: {exc}")

pack_id = st.session_state.get("brain_pack_id")
if pack_id:
    st.subheader("Brain Pack Status")
    poll_pack = st.checkbox("Polling brain pack status", value=True)
    try:
        pack_response = worker_request("GET", f"/v1/brain-packs/{pack_id}")
        if pack_response.ok:
            pack = pack_response.json()
            pack_status = str(pack.get("status", "unknown"))
            st.write({"brain_pack_id": pack_id, "status": pack_status})
            if pack_status == "completed":
                st.link_button("Download Brain Pack", f"{worker_url.rstrip('/')}/v1/brain-packs/{pack_id}/download")
            elif poll_pack:
                time.sleep(1.5)
                st.rerun()
        else:
            _render_worker_http_error(pack_response)
    except requests.RequestException as exc:
        st.error(f"Failed to fetch brain pack status: {exc}")
