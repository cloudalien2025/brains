from __future__ import annotations

import random
import time
from dataclasses import dataclass

import requests

from .proxy_manager import ProxyManager

BLOCK_INDICATORS = ("unusual traffic", "detected unusual", "verify you are human")


@dataclass
class HttpResponse:
    status_code: int
    text: str
    url: str
    headers: dict
    attempts: int


class HttpClient:
    def __init__(self, proxy_manager: ProxyManager | None = None):
        self.proxy_manager = proxy_manager or ProxyManager()

    def _is_blocked(self, status_code: int, body: str) -> bool:
        lowered = (body or "").lower()
        return status_code in {403, 429} or any(marker in lowered for marker in BLOCK_INDICATORS)

    def get(
        self,
        url: str,
        *,
        headers: dict | None = None,
        timeout_seconds: int | None = None,
        proxy_session_key: str | None = None,
        treat_empty_as_block: bool = False,
    ) -> HttpResponse:
        max_retries = max(1, self.proxy_manager.config.max_retries)
        timeout = timeout_seconds or self.proxy_manager.config.timeout_seconds
        last_response: requests.Response | None = None
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=timeout,
                    proxies=self.proxy_manager.get_proxies(proxy_session_key=proxy_session_key),
                )
                body = response.text or ""
                blocked = self._is_blocked(response.status_code, body)
                empty_block = treat_empty_as_block and response.status_code == 200 and not body.strip()
                if not blocked and not empty_block:
                    return HttpResponse(
                        status_code=response.status_code,
                        text=body,
                        url=response.url,
                        headers=dict(response.headers),
                        attempts=attempt,
                    )
                last_response = response
            except Exception as exc:
                last_exc = exc

            if attempt < max_retries:
                time.sleep(min(4.0, (2 ** (attempt - 1)) + random.uniform(0, 0.4)))

        if last_response is not None:
            return HttpResponse(
                status_code=last_response.status_code,
                text=last_response.text or "",
                url=last_response.url,
                headers=dict(last_response.headers),
                attempts=max_retries,
            )
        raise RuntimeError(f"HTTP GET failed for {url}: {last_exc}")
