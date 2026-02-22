# Brains Worker

## Deploy on droplet

1. SSH to droplet and pull the latest commit in `/opt/brains-worker`.
2. Install/update deps in venv:
   - `/opt/brains-worker/.venv/bin/pip install -r apps/brains_worker/requirements.txt`
3. One-time Playwright browser install (required for cookie bootstrap):
   - `/opt/brains-worker/.venv/bin/python -m playwright install chromium`
4. Ensure worker env has:
   - `BRAINS_API_KEY`
   - `OPENAI_API_KEY` (required for audio fallback)
   - `BRAINS_GIT_COMMIT` and `BRAINS_BUILD_TIME_UTC` (optional but recommended for `/transcript/version`)
5. Ensure cookie directory exists:
   - `mkdir -p /opt/brains-worker/cookies`
6. Optional prewarm cookie jar manually:
   - `/opt/brains-worker/.venv/bin/python tools/bootstrap_youtube_cookies.py --output /opt/brains-worker/cookies/youtube_cookies.txt`
7. Reload and restart systemd:
   - `sudo systemctl daemon-reload`
   - `sudo systemctl restart brains-worker`
8. Verify service:
   - `curl -s https://worker.aiohut.com/transcript/version | jq`
   - `sudo systemctl status brains-worker --no-pager`

## Runtime notes

- Endpoint `GET /transcript/version` reports commit/build and feature availability.
- Endpoint `POST /transcript` is yt-dlp-subtitles first, then player JSON captions fallback, then OpenAI STT fallback when enabled.
- Worker attempts to refresh `/opt/brains-worker/cookies/youtube_cookies.txt` when missing/stale before yt-dlp subtitle calls.
- Failures return non-200 with `error_code`, `error`, and `diagnostics`.
