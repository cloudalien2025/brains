# Brains Worker

## Deploy on droplet

1. SSH to droplet and pull the latest commit in `/opt/brains-worker`.
2. Install/update deps in venv:
   - `/opt/brains-worker/.venv/bin/pip install -r apps/brains_worker/requirements.txt`
3. Ensure worker env has:
   - `BRAINS_API_KEY`
   - `OPENAI_API_KEY` (required for audio fallback)
   - `BRAINS_GIT_COMMIT` and `BRAINS_BUILD_TIME_UTC` (optional but recommended for `/transcript/version`)
4. Reload and restart systemd:
   - `sudo systemctl daemon-reload`
   - `sudo systemctl restart brains-worker`
5. Verify service:
   - `curl -s https://worker.aiohut.com/transcript/version | jq`
   - `sudo systemctl status brains-worker --no-pager`

## Runtime notes

- Endpoint `GET /transcript/version` reports commit/build and feature availability.
- Endpoint `POST /transcript` is captions-first and uses yt-dlp + OpenAI STT fallback when captions are unavailable and `allow_audio_fallback=true`.
- Failures return non-200 with `detail.error_code`, `detail.error`, and `detail.diagnostics`.
