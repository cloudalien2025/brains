# Brains Ingestion App

Streamlit app for generating validated Brain Packs from real YouTube discovery and transcript-driven extraction.

## Features

- Real YouTube discovery using YouTube Data API v3.
- Transcript-first ingestion (`youtube-transcript-api`) with optional audio fallback (`yt-dlp` + OpenAI transcription).
- Structured record extraction with evidence (URL + timestamps when available).
- Strict JSON Schema validation before writing any Brain Pack files.
- Local pack write + in-app ZIP download for ephemeral deployments.

## Required secrets

- `YOUTUBE_API_KEY` (required)
- `OPENAI_API_KEY` (optional but recommended for richer extraction and audio fallback)

## Run

```bash
cd Brains_Ingestion_App
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deployment Notes (Streamlit Cloud)

- Transcript-first mode is the most reliable cloud path.
- Audio fallback can require system `ffmpeg` depending on source media/container behavior.
- Keep `Allow audio transcription fallback` off unless you have runtime support for `yt-dlp` workflows and accept additional API costs/latency.
