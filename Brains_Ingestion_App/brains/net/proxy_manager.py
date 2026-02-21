from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from urllib.parse import quote

import requests


@dataclass(frozen=True)
class ProxyConfig:
    enabled: bool = False
    gateway_host: str = "gate.decodo.com"
    gateway_port: int = 7000
    user: str | None = None
    password: str | None = None
    country: str | None = None
    sticky_mode: str = "per_video"
    timeout_seconds: int = 30
    max_retries: int = 3


class ProxyManager:
    def __init__(self, config: ProxyConfig | None = None):
        self.config = config or self.from_env()

    @classmethod
    def from_env(cls) -> ProxyConfig:
        enabled = os.getenv("DECODO_ENABLED", "false").lower() == "true"
        return ProxyConfig(
            enabled=enabled,
            gateway_host=os.getenv("DECODO_GATEWAY_HOST", "gate.decodo.com"),
            gateway_port=int(os.getenv("DECODO_GATEWAY_PORT", "7000")),
            user=os.getenv("DECODO_USER"),
            password=os.getenv("DECODO_PASS"),
            country=(os.getenv("DECODO_COUNTRY") or "").strip() or None,
            sticky_mode=os.getenv("DECODO_STICKY_MODE", "per_video"),
            timeout_seconds=int(os.getenv("DECODO_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("DECODO_MAX_RETRIES", "3")),
        )

    def is_enabled(self) -> bool:
        return bool(self.config.enabled and self.config.user and self.config.password)

    def _build_username(self, proxy_session_key: str | None = None) -> str:
        username = self.config.user or ""
        country = (self.config.country or "").strip().lower()
        if country:
            username = f"{username}-country-{country}"
        if self.config.sticky_mode == "per_video" and proxy_session_key:
            digest = hashlib.sha1(proxy_session_key.encode("utf-8")).hexdigest()[:12]
            username = f"{username}-session-{digest}"
        return username

    def get_proxies(self, proxy_session_key: str | None = None) -> dict[str, str] | None:
        if not self.is_enabled():
            return None
        username = quote(self._build_username(proxy_session_key), safe="")
        password = quote(self.config.password or "", safe="")
        endpoint = f"http://{username}:{password}@{self.config.gateway_host}:{self.config.gateway_port}"
        return {"http": endpoint, "https": endpoint}

    def safe_diagnostics(self) -> dict:
        return {
            "enabled": self.is_enabled(),
            "gateway_host": self.config.gateway_host,
            "gateway_port": self.config.gateway_port,
            "country": self.config.country,
            "sticky_mode": self.config.sticky_mode,
            "timeout_seconds": self.config.timeout_seconds,
            "max_retries": self.config.max_retries,
        }

    def health_check(self, proxy_session_key: str | None = None) -> dict:
        started = time.perf_counter()
        diagnostics = {"ok": False, "latency_ms": None, "error": None, "exit_ip": None, "raw": None}
        try:
            response = requests.get(
                "http://ip.decodo.com/json",
                timeout=self.config.timeout_seconds,
                proxies=self.get_proxies(proxy_session_key=proxy_session_key),
            )
            diagnostics["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            diagnostics["raw"] = response.text[:500]
            payload = response.json()
            diagnostics["exit_ip"] = payload.get("ip")
            diagnostics["ok"] = response.status_code == 200 and bool(payload.get("ip"))
        except Exception as exc:
            diagnostics["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            diagnostics["error"] = str(exc)
        return diagnostics
