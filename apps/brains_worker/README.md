# Brains Worker v1

FastAPI worker that handles heavy ingestion for Brains with API-key auth (`X-Api-Key`).

## Features

- Multi-brain registry with persistent storage on droplet.
- Incremental ingest queue (`queued -> processing -> completed/completed_with_errors/failed`).
- YouTube discovery + dedupe against per-brain ledger.
- Audio-first transcript acquisition (`yt-dlp` + `ffmpeg` chunking + OpenAI STT).
- Profile extraction/synthesis for `BD` and `UAP` brain types.
- Brain Pack zip builder + metadata/download endpoints.

## Local run

```bash
pip install -r apps/brains_worker/requirements.txt
uvicorn apps.brains_worker.main:app --reload --port 8081
```

Required env vars:

- `WORKER_API_KEY`
- `OPENAI_API_KEY`

Optional env vars:

- `BRAINS_DATA_DIR` (default `/opt/brains-data`)
- `MAX_CONCURRENT_DOWNLOADS` (default `3`)
- `MAX_CONCURRENT_STT` (default `1`)
- `MAX_CONCURRENT_SYNTHESIS` (default `1`)
- `CHUNK_SECONDS` (default `600`)
- `OVERLAP_SECONDS` (default `15`)
- `ARCHIVE_AUDIO` (default `false`)

## API

All `/v1/*` endpoints require header:

```http
X-Api-Key: <WORKER_API_KEY>
```

- `GET /v1/health`
- `GET /v1/brains`
- `POST /v1/brains`
- `GET /v1/brains/{brain_id}`
- `POST /v1/brains/{brain_id}/ingest`
- `GET /v1/runs/{run_id}`
- `GET /v1/runs/{run_id}/report`
- `POST /v1/runs/{run_id}/brain-pack`
- `GET /v1/brain-packs/{brain_pack_id}`
- `GET /v1/brain-packs/{brain_pack_id}/download`

See `apps/brains_worker/DEPLOYMENT.md` for systemd/nginx notes.
