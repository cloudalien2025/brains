# Brains Ingestion App

Streamlit app for generating validated Brain Packs from real YouTube discovery and transcript-driven extraction.

## Features

- Real YouTube discovery using YouTube Data API v3.
- Generic proxy-backed HTTP layer (Decodo-ready) used by timedtext/transcript fetches.
- Optional DigitalOcean worker routing for transcript extraction.
- Structured record extraction with evidence (URL + timestamps when available).
- Strict JSON Schema validation before writing any Brain Pack files.
- Local pack write + in-app ZIP download for ephemeral deployments.

## Required secrets

- `YOUTUBE_API_KEY` (required)
- `OPENAI_API_KEY` (optional but recommended for richer extraction and audio fallback)

## Optional Decodo proxy secrets

- `DECODO_ENABLED=true|false` (default `false`)
- `DECODO_GATEWAY_HOST=gate.decodo.com`
- `DECODO_GATEWAY_PORT=7000`
- `DECODO_USER=...`
- `DECODO_PASS=...`
- `DECODO_COUNTRY=us` (optional)
- `DECODO_STICKY_MODE=per_video|off` (default `per_video`)
- `DECODO_TIMEOUT_SECONDS=30`
- `DECODO_MAX_RETRIES=3`

## Optional worker settings

- `BRAINS_WORKER_URL=https://<domain-or-ip>`
- `BRAINS_WORKER_API_KEY=<secret>`

If `BRAINS_WORKER_URL` is set, Streamlit routes transcript extraction to the worker first and falls back to local extraction if worker calls fail.

## Run

```bash
cd Brains_Ingestion_App
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Worker API (FastAPI)

Run locally:

```bash
uvicorn apps.brains_worker.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` returns `{"ok": true, "version": "...", "time": "..."}`.
- `POST /transcript` with `x-api-key` (or `X-Api-Key`) header.
  - Missing key returns `401 {"detail": "Missing x-api-key"}`.
  - Wrong key returns `401 {"detail": "Invalid x-api-key"}`.
- Worker transcript diagnostics now always include timedtext HTTP statuses and audio fallback telemetry (`audio_download_ok`, `audio_file_bytes`, `transcription_engine`).

## Deployment Notes (DigitalOcean Ubuntu 22.04)

1. Install runtime deps: `python3-venv python3-pip ffmpeg`.
2. Clone repo to `/opt/brains/brains` and create `.venv`.
3. `pip install -r requirements.txt` and ensure worker deps are in the same runtime venv, e.g. `sudo -u brains /opt/brains-worker/.venv/bin/pip install -U yt-dlp faster-whisper ctranslate2`.
4. Create `/opt/brains/brains/.env.worker` with Decodo + `BRAINS_WORKER_API_KEY`.
5. Add systemd service:
   - `ExecStart=/opt/brains-worker/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8787 --proxy-headers`
6. `sudo systemctl daemon-reload && sudo systemctl enable --now brains-worker`.
7. Put nginx/TLS in front or restrict firewall ingress.
