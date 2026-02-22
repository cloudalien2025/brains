from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


def _to_netscape_cookie_line(cookie: dict) -> str:
    domain = cookie.get("domain", "")
    include_subdomains = "TRUE" if domain.startswith(".") else "FALSE"
    path = cookie.get("path", "/")
    secure = "TRUE" if cookie.get("secure") else "FALSE"
    expires = int(cookie.get("expires", 0) or 0)
    name = cookie.get("name", "")
    value = cookie.get("value", "")
    return "\t".join([domain, include_subdomains, path, secure, str(expires), name, value])


def _accept_consent(page) -> bool:
    selectors = [
        "button:has-text('Accept all')",
        "button:has-text('I agree')",
        "button:has-text('Agree')",
        "button:has-text('Accept the use of cookies')",
        "text=Accept all",
    ]
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.is_visible(timeout=1200):
                locator.click(timeout=1200)
                page.wait_for_timeout(500)
                return True
        except Exception:
            continue
    return False


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        handle.write(content)
        tmp = Path(handle.name)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/opt/brains-worker/cookies/youtube_cookies.txt")
    parser.add_argument("--state", default="/opt/brains-worker/cookies/youtube_storage_state.json")
    args = parser.parse_args()

    output_path = Path(args.output)
    state_path = Path(args.state)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        try:
            page.goto("https://www.youtube.com/", wait_until="domcontentloaded", timeout=8000)
            _accept_consent(page)
            page.wait_for_load_state("networkidle", timeout=5000)
        except PlaywrightTimeoutError:
            pass

        cookies = context.cookies(["https://www.youtube.com/"])
        state = context.storage_state()
        browser.close()

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    netscape_lines = ["# Netscape HTTP Cookie File"]
    netscape_lines.extend(_to_netscape_cookie_line(cookie) for cookie in cookies)
    _atomic_write(output_path, "\n".join(netscape_lines) + "\n")

    print(f"youtube cookie bootstrap complete; cookies={len(cookies)} output={output_path} state={state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
