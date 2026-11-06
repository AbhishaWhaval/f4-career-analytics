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


# ---------------------------------------------------------------- title search fallback
def title_matches(title: str, season: int, keywords) -> bool:
    """Accept a search result only if it names the right season and the right series."""
    split_year = f"{season - 1}\u2013{season % 100:02d}"
    has_year = title.startswith(f"{season} ") or title.startswith(split_year)
    has_series = any(k.lower() in title.lower() for k in keywords)
    is_season_page = any(w in title.lower() for w in ("championship", "series"))
    return has_year and has_series and is_season_page


def _search(self, query: str, limit: int = 8):
    params = {"action": "query", "list": "search", "srsearch": query, "srlimit": limit,
              "srnamespace": 0, "format": "json", "formatversion": 2}
    try:
        resp = self.session.get(API, params=params, timeout=30)
        return [r["title"] for r in resp.json().get("query", {}).get("search", [])]
    except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
        log.error("Search failed for %r: %s", query, exc)
        return []


def find_season_title(self, series: str, season: int, keywords) -> Optional[str]:
    """Search Wikipedia for a season page when the configured title does not exist."""
    for query in (f"{season} {series} championship", f"{season} {' '.join(keywords)} Formula 4"):
        for title in self._search(query):
            if title_matches(title, season, keywords):
                return title
    return None


WikiClient._search = _search
WikiClient.find_season_title = find_season_title


# ---------------------------------------------------------------- birth dates
BIRTH_TEMPLATE = re.compile(r"\{\{\s*birth[ _]date(?:[ _]and[ _]age)?\s*\|([^{}]*)\}\}", re.I)
BIRTH_YEAR_TEMPLATE = re.compile(r"\{\{\s*birth[ _]year(?:[ _]and[ _]age)?\s*\|\s*(\d{4})", re.I)
BIRTH_FIELD = re.compile(r"\|\s*birth_date\s*=\s*([^\n]*)", re.I)
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}


def parse_birth_date(wikitext: str) -> str:
    """Return 'YYYY-MM-DD', 'YYYY' or '' from an article's wikitext infobox."""
    if not wikitext:
        return ""
    m = BIRTH_TEMPLATE.search(wikitext)
    if m:
        nums = [p.strip() for p in m.group(1).split("|") if p.strip().isdigit()]
        if len(nums) >= 3 and len(nums[0]) == 4:
            return f"{int(nums[0]):04d}-{int(nums[1]):02d}-{int(nums[2]):02d}"
        if nums and len(nums[0]) == 4:
            return nums[0]
    m = BIRTH_YEAR_TEMPLATE.search(wikitext)
    if m:
        return m.group(1)
    m = BIRTH_FIELD.search(wikitext)
    if m:
        text = m.group(1)
        d = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+((?:19|20)\d{2})", text) or \
            re.search(r"([A-Za-z]+)\s+(\d{1,2}),?\s+((?:19|20)\d{2})", text)
        if d:
            a, b, y = d.groups()
            day, month = (a, b) if a.isdigit() else (b, a)
            if month.lower() in MONTHS:
                return f"{y}-{MONTHS[month.lower()]:02d}-{int(day):02d}"
        y = re.search(r"\b((?:19|20)\d{2})\b", text)
        if y:
            return y.group(1)
    return ""


def fetch_birth_dates(self, titles, batch: int = 50) -> dict:
    """{title: birth date string} for article titles, 50 per API call, cached on disk."""
    cache_file = self.raw_dir / "driver_birthdates.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    todo = [t for t in dict.fromkeys(titles) if self.refresh or t not in cache]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        params = {"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
                  "titles": "|".join(chunk), "redirects": 1, "format": "json", "formatversion": 2}
        wait = self.pause - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            data = self.session.get(API, params=params, timeout=60).json()
            self._last = time.time()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            log.error("Birth-date batch failed: %s", exc)
            continue
        q = data.get("query", {})
        # follow normalisation and redirects back to the titles we asked for
        back = {t: t for t in chunk}
        for step in ("normalized", "redirects"):
            for r in q.get(step, []):
                for orig, cur in list(back.items()):
                    if cur == r["from"]:
                        back[orig] = r["to"]
        text_by_title = {}
        for page in q.get("pages", []):
            revs = page.get("revisions") or []
            text_by_title[page["title"]] = revs[0]["slots"]["main"]["content"] if revs else ""
        for orig, final in back.items():
            cache[orig] = parse_birth_date(text_by_title.get(final, ""))
        log.info("Birth dates: %d/%d articles checked", min(i + batch, len(todo)), len(todo))
        cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    return {t: cache.get(t, "") for t in titles}


WikiClient.fetch_birth_dates = fetch_birth_dates
