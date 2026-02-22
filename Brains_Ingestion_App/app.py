from __future__ import annotations

import time
from typing import Any

import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


TERMINAL_RUN_STATES = {"completed", "completed_with_errors", "failed"}


def _secret(name: str) -> str:
    try:
        value = st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        value = None
    return str(value or "").strip()


def _snippet(text: str, limit: int = 500) -> str:
    return (text or "")[:limit]


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

for key, default in {
    "brains": [],
    "selected_brain_id": None,
    "run_id": None,
    "last_report": None,
    "last_error": None,
    "brain_pack_id": None,
}.items():
    st.session_state.setdefault(key, default)

worker_url = _secret("worker_url")
worker_api_key = _secret("worker_api_key")


def worker_request(method: str, path: str, json: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> requests.Response:
    url = worker_url.rstrip("/") + path
    headers = {"X-Api-Key": worker_api_key}
    return requests.request(method, url, headers=headers, json=json, params=params, timeout=30)


def fetch_brains() -> None:
    if not worker_url or not worker_api_key:
        return
    try:
        response = worker_request("GET", "/v1/brains")
        if response.ok:
            payload = response.json()
            st.session_state["brains"] = payload if isinstance(payload, list) else payload.get("items", [])
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

st.subheader("Brain")
brains = st.session_state.get("brains", [])

if brains:
    options = {f"{b.get('brain_name', 'Unnamed')} ({b.get('brain_type', 'Unknown')})": b.get("brain_id") for b in brains}
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
            payload = {
                "name": new_name.strip(),
                "type": new_type,
                "default_keyword": new_default_keyword.strip() or None,
            }
            try:
                response = worker_request("POST", "/v1/brains", json=payload)
                if response.ok:
                    created = response.json()
                    created_id = created.get("brain_id")
                    st.success("Brain created.")
                    fetch_brains()
                    if created_id:
                        st.session_state["selected_brain_id"] = created_id
                    st.rerun()
                else:
                    _render_worker_http_error(response)
            except requests.RequestException as exc:
                st.error(f"Failed to create brain: {exc}")

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
            "n_new": int(n_new),
            "mode": "audio_first",
            "longform": {
                "chunk_seconds": int(chunk_seconds),
                "overlap_seconds": int(overlap_seconds),
            },
        }
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
    try:
        run_response = worker_request("GET", f"/v1/runs/{run_id}")
        if run_response.ok:
            run_data = run_response.json()
        else:
            _render_worker_http_error(run_response)
    except requests.RequestException as exc:
        st.error(f"Failed to fetch run status: {exc}")

    if run_data:
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

        current = run_data.get("current") or {}
        if isinstance(current, dict) and current:
            st.write(
                {
                    "current.source_id": current.get("source_id"),
                    "current.stage": current.get("stage"),
                    "current.detail": current.get("detail"),
                }
            )

        if status in TERMINAL_RUN_STATES:
            try:
                report_response = worker_request("GET", f"/v1/runs/{run_id}/report")
                if report_response.ok:
                    st.session_state["last_report"] = report_response.json()
                else:
                    _render_worker_http_error(report_response)
            except requests.RequestException as exc:
                st.error(f"Failed to fetch report: {exc}")
        elif polling_enabled:
            time.sleep(1.5)
            st.rerun()

report = st.session_state.get("last_report")
if report:
    st.subheader("Run Report")
    st.write(
        {
            "ingested_new": report.get("ingested_new"),
            "transcripts_succeeded": report.get("transcripts_succeeded"),
            "transcripts_failed": report.get("transcripts_failed"),
            "total_audio_minutes": report.get("total_audio_minutes"),
        }
    )

    existing_pack_id = report.get("brain_pack_id")
    if existing_pack_id:
        download_url = f"{worker_url.rstrip('/')}/v1/brain-packs/{existing_pack_id}/download"
        st.link_button("Download Brain Pack", download_url)
    else:
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
