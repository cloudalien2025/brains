from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from uuid import uuid4

import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from adapters.extraction import extract_brain_records
from adapters.transcription import (
    TranscriptionError,
    get_transcript_for_source,
    get_transcript_from_worker,
    get_yta_runtime_info,
)
from adapters.youtube_discovery import discover_youtube_videos
from brainpack.exporters import build_pack_zip, write_json, write_jsonl, write_sources_csv
from brainpack.utils import now_iso, slugify
from brainpack.validators import validate_pack_manifest, validate_record, validate_run_metadata, validate_source
from brains.net.proxy_manager import ProxyConfig, ProxyManager




def _bool_env(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() == "true"


def _secret_or_env(name: str) -> str:
    try:
        secret_value = st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        secret_value = None
    if secret_value is not None:
        return str(secret_value).strip()
    return (os.getenv(name) or "").strip()


st.set_page_config(page_title="Brains Ingestion", layout="wide")
st.title("Brains Ingestion")

keyword = st.text_input("Keyword", value="Brilliant Directories")
max_videos = st.number_input("Max videos", min_value=1, max_value=50, value=5, step=1)
discovery_only = st.toggle("Discovery only", value=False)
use_openai_extraction = st.toggle(
    "Use OpenAI for extraction (recommended)",
    value=bool(os.getenv("OPENAI_API_KEY")),
)
allow_audio_fallback = st.toggle("Allow audio transcription fallback", value=False)

proxy_enabled = st.checkbox("Use Decodo proxy", value=_bool_env("DECODO_ENABLED", False))
proxy_sticky = st.checkbox(
    "Sticky per video",
    value=os.getenv("DECODO_STICKY_MODE", "per_video") == "per_video",
    disabled=not proxy_enabled,
)
proxy_country = st.text_input("Country (optional)", value=os.getenv("DECODO_COUNTRY", ""), disabled=not proxy_enabled)

worker_url = _secret_or_env("BRAINS_WORKER_URL")
worker_api_key = _secret_or_env("BRAINS_WORKER_API_KEY")

st.subheader("Worker Status")
if worker_url and worker_api_key:
    st.success("Worker ACTIVE (URL + API key present)")
elif worker_url and not worker_api_key:
    st.error("Worker URL set but API key missing — running locally")
else:
    st.warning("Worker disabled — running locally")

proxy_manager = ProxyManager(
    ProxyConfig(
        enabled=proxy_enabled,
        gateway_host=os.getenv("DECODO_GATEWAY_HOST", "gate.decodo.com"),
        gateway_port=int(os.getenv("DECODO_GATEWAY_PORT", "7000")),
        user=os.getenv("DECODO_USER"),
        password=os.getenv("DECODO_PASS"),
        country=proxy_country.strip() or None,
        sticky_mode="per_video" if proxy_sticky else "off",
        timeout_seconds=int(os.getenv("DECODO_TIMEOUT_SECONDS", "30")),
        max_retries=int(os.getenv("DECODO_MAX_RETRIES", "3")),
    )
)

st.caption("Default ingestion is transcript-first and does not download video/audio streams.")
if allow_audio_fallback:
    st.warning("Audio fallback uses yt-dlp/ffmpeg + OpenAI and can fail on Streamlit Cloud; it is recommended for local runs.")
if worker_url:
    st.info(f"Worker routing enabled: {worker_url}")

with st.expander("Worker diagnostics", expanded=False):
    st.write(f"worker_url present: {bool(worker_url)}")
    st.write(f"worker_api_key present: {bool(worker_api_key)}")
    st.write(f"worker_api_key length: {len(worker_api_key)}")

with st.expander("Proxy diagnostics", expanded=False):
    st.json(proxy_manager.safe_diagnostics())
    if st.button("Run proxy IP check"):
        if worker_url and worker_api_key:
            try:
                response = requests.get(
                    f"{worker_url}/proxy/health",
                    headers={"x-api-key": worker_api_key.strip()},
                    timeout=15,
                )
                st.json(response.json())
            except Exception as exc:
                st.error(f"Worker proxy check failed: {exc}")
        else:
            st.warning("Worker not configured — cannot test proxy.")

if "advanced_logs" not in st.session_state:
    st.session_state.advanced_logs = []


def _log(video_id: str, message: str) -> None:
    st.session_state.advanced_logs.append({"video": video_id, "message": message})


if st.button("Generate Brain Pack", type="primary"):
    st.session_state.advanced_logs = []
    os.environ["DECODO_ENABLED"] = "true" if proxy_enabled else "false"
    os.environ["DECODO_COUNTRY"] = proxy_country.strip()
    os.environ["DECODO_STICKY_MODE"] = "per_video" if proxy_sticky else "off"
    runtime_errors: list[str] = []
    per_video_errors: list[str] = []
    validation_errors: list[str] = []
    started_at = now_iso()
    openai_api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    yta_runtime = get_yta_runtime_info()

    if not os.getenv("YOUTUBE_API_KEY"):
        st.error("YOUTUBE_API_KEY is required in real ingestion mode.")
        st.stop()

    try:
        videos = discover_youtube_videos(keyword=keyword, max_videos=int(max_videos))
    except Exception as exc:
        st.error(f"Discovery failed: {exc}")
        st.stop()

    for src in videos:
        source_errors = validate_source(src)
        if source_errors:
            validation_errors.extend([f"source:{src.get('source_id', 'unknown')} {err}" for err in source_errors])

    st.subheader("Discovery results")
    st.dataframe(videos, use_container_width=True)

    csv_preview = "\n".join(
        ["source_id,source_type,title,channel,url,published_at,duration_seconds"]
        + [
            f"{v['source_id']},{v['source_type']},{v['title']},{v['channel']},{v['url']},{v.get('published_at','')},{v.get('duration_seconds','')}"
            for v in videos
        ]
    )
    st.download_button("Download Sources.csv", data=csv_preview.encode("utf-8"), file_name="Sources.csv", mime="text/csv")

    if discovery_only:
        st.info("Discovery only is enabled; ingestion stopped after discovery.")
        st.stop()

    queue = videos[: int(max_videos)]
    records: list[dict] = []
    additions_blocks: list[str] = []
    successful_sources = 0
    failed_sources = 0
    video_diagnostics: list[dict] = []

    for source in queue:
        video_key = source.get("source_id", "unknown")
        _log(video_key, "Discovery status: queued")

        try:
            if worker_url:
                if not worker_api_key:
                    raise RuntimeError("BRAINS_WORKER_API_KEY missing — refusing local transcript execution")
                transcript = get_transcript_from_worker(
                    source,
                    worker_url=worker_url,
                    worker_api_key=worker_api_key,
                    allow_audio_fallback=allow_audio_fallback,
                )
                _log(video_key, f"Worker transcript method used: {transcript.get('method')}")
            else:
                transcript = get_transcript_for_source(
                    source,
                    allow_audio_fallback=allow_audio_fallback,
                    openai_api_key_present=openai_api_key_present,
                )
            transcript["source"] = source
            diagnostics = transcript.get("diagnostics", {})
            video_diagnostics.append({"video_id": video_key, **diagnostics})
            _log(video_key, f"Transcript method used: {transcript.get('method')}")
            _log(video_key, f"Transcript diagnostics: {json.dumps(diagnostics, ensure_ascii=False)}")
        except TranscriptionError as exc:
            failed_sources += 1
            diagnostics = getattr(exc, "diagnostics", {}) or {}
            video_diagnostics.append({"video_id": video_key, **diagnostics})
            if exc.code == "transcript_unavailable_disabled":
                label = "transcript_unavailable_yta; transcript_unavailable_timedtext (audio fallback disabled)"
            elif exc.code in {"audio_fallback_failed", "audio_fallback_requires_openai_key"}:
                label = str(exc)
            else:
                label = f"{exc.code}: {exc}"
            err = f"transcription:{video_key} {label}"
            runtime_errors.append(err)
            per_video_errors.append(f"{video_key}: {label}")
            _log(video_key, f"Transcript failed: {label}")
            _log(video_key, f"Transcript diagnostics: {json.dumps(diagnostics, ensure_ascii=False)}")
            continue
        except Exception as exc:
            failed_sources += 1
            err = f"transcription:{video_key} {exc}"
            runtime_errors.append(err)
            per_video_errors.append(f"{video_key}: {exc}")
            _log(video_key, f"Transcript failed: {exc}")
            continue

        try:
            source_records, additions_md = extract_brain_records(
                brain="BD_Brain",
                keyword=keyword,
                source=source,
                transcript=transcript,
                use_openai=use_openai_extraction and openai_api_key_present,
            )
            additions_blocks.append(additions_md)
            _log(video_key, f"Extraction counts: {len(source_records)}")
            successful_sources += 1
        except Exception as exc:
            failed_sources += 1
            err = f"extraction:{video_key} {exc}"
            runtime_errors.append(err)
            per_video_errors.append(f"{video_key}: {exc}")
            _log(video_key, f"Extraction failed: {exc}")
            continue

        for rec in source_records:
            record_errors = validate_record(rec)
            if record_errors:
                validation_errors.extend([f"record:{rec.get('id', 'unknown')} {err}" for err in record_errors])
        records.extend(source_records)

    run_metadata = {
        "run_id": f"run_{uuid4().hex[:8]}",
        "keyword": keyword,
        "started_at": started_at,
        "ended_at": now_iso(),
        "config": {
            "max_videos": int(max_videos),
            "discovery_only": discovery_only,
            "use_openai_extraction": use_openai_extraction,
            "allow_audio_fallback": allow_audio_fallback,
            "use_decodo_proxy": proxy_enabled,
            "decodo_sticky_per_video": proxy_sticky,
            "decodo_country": proxy_country.strip() or None,
            "worker_url": worker_url or None,
        },
        "errors": runtime_errors,
        "yta_runtime": yta_runtime,
        "video_diagnostics": video_diagnostics,
        "env": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "has_youtube_api_key": bool(os.getenv("YOUTUBE_API_KEY")),
        },
    }
    validation_errors.extend([f"run_metadata {err}" for err in validate_run_metadata(run_metadata)])

    pack_slug = slugify(keyword)
    pack_name = f"{started_at[:10]}__{pack_slug}__pack_v1"
    pack_dir = Path(__file__).resolve().parents[1] / "BD_Brain" / "Packs" / pack_name

    outputs = {
        "Brain_Additions.md": "Brain_Additions.md",
        "Brain_Core.jsonl": "Brain_Core.jsonl",
        "Brain_Diff.md": "Brain_Diff.md",
        "Sources.csv": "Sources.csv",
        "Run_Metadata.json": "Run_Metadata.json",
    }

    manifest = {
        "pack_id": pack_name,
        "brain": "BD_Brain",
        "keyword": keyword,
        "created_at": now_iso(),
        "sources": queue,
        "outputs": outputs,
        "stats": {
            "videos_discovered": len(videos),
            "videos_queued": len(queue),
            "records_extracted": len(records),
        },
    }
    validation_errors.extend([f"pack_manifest {err}" for err in validate_pack_manifest(manifest)])

    validation_status = "passed" if not validation_errors else "failed"
    for source in queue:
        _log(source.get("source_id", "unknown"), f"Validation status: {validation_status}")

    st.info(f"Sources succeeded: {successful_sources} | Sources failed: {failed_sources}")
    if per_video_errors:
        st.subheader("Per-video errors")
        for err in per_video_errors:
            st.write(f"- {err}")

    with st.expander("Transcript diagnostics", expanded=False):
        st.json({"yta_runtime": yta_runtime, "video_diagnostics": video_diagnostics})

    if validation_errors:
        st.error("Validation failed. Brain Pack was not written.")
        for err in validation_errors:
            st.write(f"- {err}")
        st.stop()

    if not records:
        st.warning("No transcripts/extractions succeeded; pack not created.")
        st.stop()

    pack_dir.mkdir(parents=True, exist_ok=True)
    additions_text = "# Brain Additions\n\n" + "\n\n".join(additions_blocks)
    diff_text = (
        "# Brain Diff\n\n"
        f"- Sources: {len(queue)}\n"
        f"- Records extracted: {len(records)}\n"
        f"- Keyword: {keyword}\n"
    )

    write_jsonl(pack_dir / "Brain_Core.jsonl", records)
    write_sources_csv(pack_dir / "Sources.csv", queue)
    write_json(pack_dir / "Run_Metadata.json", run_metadata)
    write_json(pack_dir / "Pack_Manifest.json", manifest)
    write_json(pack_dir / "pack_manifest.json", manifest)
    (pack_dir / "Brain_Additions.md").write_text(additions_text, encoding="utf-8")
    (pack_dir / "Brain_Diff.md").write_text(diff_text, encoding="utf-8")

    pack_files = {
        "Brain_Additions.md": additions_text,
        "Brain_Core.jsonl": "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        "Brain_Diff.md": diff_text,
        "Sources.csv": (pack_dir / "Sources.csv").read_text(encoding="utf-8"),
        "Run_Metadata.json": (pack_dir / "Run_Metadata.json").read_text(encoding="utf-8"),
        "pack_manifest.json": (pack_dir / "pack_manifest.json").read_text(encoding="utf-8"),
    }
    zip_bytes = build_pack_zip(pack_files)

    st.success(f"Generated Brain Pack at: {pack_dir}")
    st.download_button(
        "Download Brain Pack (.zip)",
        data=zip_bytes,
        file_name=f"{pack_name}.zip",
        mime="application/zip",
    )

    with st.expander("Advanced logs", expanded=False):
        for entry in st.session_state.advanced_logs:
            st.write(f"[{entry['video']}] {entry['message']}")
