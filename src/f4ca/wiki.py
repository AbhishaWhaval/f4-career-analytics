"""Fetch rendered Wikipedia pages through the MediaWiki API, with an on-disk cache.

The cache makes the pipeline reproducible: once data/raw/ is filled, every later step
runs offline. Delete a file (or pass refresh=True) to re-download it.
"""
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import requests

API = "https://en.wikipedia.org/w/api.php"
log = logging.getLogger(__name__)


def cache_path(raw_dir: Path, title: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:80]
    digest = hashlib.md5(title.encode("utf-8")).hexdigest()[:8]
    return raw_dir / f"{slug}__{digest}.html"


def missing_marker(raw_dir: Path, title: str) -> Path:
    return cache_path(raw_dir, title).with_suffix(".missing")


class WikiClient:
    def __init__(self, raw_dir: Path, user_agent: str, pause: float = 0.6, refresh: bool = False):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.pause = pause
        self.refresh = refresh
        self._last = 0.0

    def get_html(self, title: str) -> Optional[str]:
        """Return the page's rendered HTML, or None if the page does not exist."""
        path, miss = cache_path(self.raw_dir, title), missing_marker(self.raw_dir, title)
        if not self.refresh:
            if path.exists():
                return path.read_text(encoding="utf-8")
            if miss.exists():
                return None

        wait = self.pause - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        params = {"action": "parse", "page": title, "prop": "text", "redirects": 1,
                  "format": "json", "formatversion": 2}
        for attempt in range(4):
            try:
                resp = self.session.get(API, params=params, timeout=30)
                self._last = time.time()
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                data = resp.json()
                break
            except (requests.RequestException, json.JSONDecodeError) as exc:
                if attempt == 3:
                    log.error("Giving up on %r: %s", title, exc)
                    return None
                time.sleep(2 ** attempt)

        if "error" in data:
            if data["error"].get("code") == "missingtitle":
                log.warning("No Wikipedia page: %s", title)
                miss.write_text("missing", encoding="utf-8")
            else:
                log.error("API error for %r: %s", title, data["error"])
            return None

        html = data["parse"]["text"]
        path.write_text(html, encoding="utf-8")
        return html
