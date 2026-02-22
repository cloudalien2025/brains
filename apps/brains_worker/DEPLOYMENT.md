# Brains Worker Deployment Notes (worker.aiohut.com)

## systemd service

Create `/etc/systemd/system/brains-worker.service`:

```ini
[Unit]
Description=Brains Worker API
After=network.target

[Service]
User=root
WorkingDirectory=/opt/brains-worker
Environment=PYTHONUNBUFFERED=1
Environment=WORKER_API_KEY=<set-me>
Environment=OPENAI_API_KEY=<set-me>
Environment=BRAINS_DATA_DIR=/opt/brains-data
Environment=MAX_CONCURRENT_DOWNLOADS=3
Environment=MAX_CONCURRENT_STT=1
Environment=MAX_CONCURRENT_SYNTHESIS=1
Environment=CHUNK_SECONDS=600
Environment=OVERLAP_SECONDS=15
ExecStart=/opt/brains-worker/.venv/bin/uvicorn apps.brains_worker.main:app --host 0.0.0.0 --port 8081
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Commands:

```bash
sudo systemctl daemon-reload
sudo systemctl enable brains-worker
sudo systemctl restart brains-worker
sudo systemctl status brains-worker
```

## nginx route for worker.aiohut.com

Use a dedicated server block so existing domains/routes stay untouched:

```nginx
server {
    listen 80;
    server_name worker.aiohut.com;

    location / {
        proxy_pass http://127.0.0.1:8081;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Then run:

```bash
sudo nginx -t
sudo systemctl reload nginx
```
