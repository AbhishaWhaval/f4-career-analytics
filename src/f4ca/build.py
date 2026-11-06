"""Stages 1-2: download season pages and turn them into tidy tables.

data/processed/f4_results.csv   one row per driver x F4 championship x season
data/processed/ladder.csv       one row per driver x higher championship x season
data/processed/coverage.csv     which pages were found and parsed (data audit)
data/processed/birthdates.csv   birth date per driver article (for age at debut)
"""
import json
import logging
from typing import Dict, Iterable, Tuple
from urllib.parse import unquote

import pandas as pd

from .config import SERIES_KEYWORDS, Config
from .parse import normalise_name, parse_standings
from .wiki import WikiClient, cache_path

log = logging.getLogger(__name__)
OVERRIDES = "title_overrides.json"


def _pages(cfg: Config) -> Iterable[Tuple[str, str, str, int, str]]:
    """Yield (kind, tier, series, season, title) for every page the project uses."""
    for series, seasons in cfg.f4_series.items():
        for season, title in seasons.items():
            if season <= cfg.last_complete_season:
                yield "f4", "f4", series, season, title
    for tier, series_map in cfg.tiers.items():
        for series, seasons in series_map.items():
            for season, title in seasons.items():
                if season <= cfg.last_complete_season:
                    yield "ladder", tier, series, season, title


def _load_overrides(cfg: Config) -> Dict[str, str]:
    f = cfg.raw_dir / OVERRIDES
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def fetch(cfg: Config, refresh: bool = False) -> None:
    """Download every season page. If a configured title does not exist, search Wikipedia
    for the right one and remember the match in data/raw/title_overrides.json."""
    cfg.ensure_dirs()
    client = WikiClient(cfg.raw_dir, cfg.user_agent, cfg.request_pause, refresh)
    overrides = _load_overrides(cfg)
    pages = list(_pages(cfg))
    for i, (kind, _, series, season, title) in enumerate(pages, 1):
        log.info("[%d/%d] %s %s", i, len(pages), series, season)
        if title in overrides:
            client.get_html(overrides[title])
            continue
        if client.get_html(title) is None and kind == "f4" and series in SERIES_KEYWORDS:
            alt = client.find_season_title(series, season, SERIES_KEYWORDS[series])
            if alt and alt != title and client.get_html(alt) is not None:
                log.info("   using %r instead", alt)
                overrides[title] = alt
                (cfg.raw_dir / OVERRIDES).write_text(json.dumps(overrides, indent=1, ensure_ascii=False),
                                                     encoding="utf-8")


def _read_cached(cfg: Config, title: str):
    p = cache_path(cfg.raw_dir, title)
    return p.read_text(encoding="utf-8") if p.exists() else None


def build(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Parse cached pages (no network) into processed CSVs."""
    cfg.ensure_dirs()
    overrides = _load_overrides(cfg)
    f4_rows, ladder_rows, coverage = [], [], []

    for kind, tier, series, season, title in _pages(cfg):
        used = overrides.get(title, title)
        html = _read_cached(cfg, used)
        status, n = "missing", 0
        if html is not None:
            df = parse_standings(html)
            n = len(df)
            status = "parsed" if n else "no standings table"
            if n:
                df["series"], df["season"], df["tier"] = series, season, tier
                (f4_rows if kind == "f4" else ladder_rows).append(df)
        coverage.append(dict(kind=kind, tier=tier, series=series, season=season,
                             title=used, status=status, drivers=n))

    cov = pd.DataFrame(coverage)
    cov.to_csv(cfg.processed_dir / "coverage.csv", index=False)
    if not f4_rows:
        raise RuntimeError("No F4 standings parsed. Run `python -m f4ca fetch` first.")

    f4 = pd.concat(f4_rows, ignore_index=True)
    f4["n_drivers"] = f4.groupby(["series", "season"])["pos"].transform("max")
    f4["season_races"] = f4.groupby(["series", "season"])["entries"].transform("max")
    f4["name_key"] = f4["driver"].map(normalise_name)

    ladder = (pd.concat(ladder_rows, ignore_index=True) if ladder_rows else
              pd.DataFrame(columns=["driver_id", "driver", "tier", "series", "season"]))
    ladder["name_key"] = ladder["driver"].map(normalise_name)

    f4.to_csv(cfg.processed_dir / "f4_results.csv", index=False)
    ladder.to_csv(cfg.processed_dir / "ladder.csv", index=False)
    parsed = cov[cov.status == "parsed"]
    log.info("Parsed %d of %d pages: %d F4 driver-seasons, %d ladder driver-seasons",
             len(parsed), len(cov), len(f4), len(ladder))
    build_bios(cfg, quiet=True)
    return {"f4": f4, "ladder": ladder, "coverage": cov}


# ---------------------------------------------------------------- birth dates
def article_title(driver_id: str) -> str:
    """'Andrea_Kimi_Antonelli' or URL-encoded ids -> 'Andrea Kimi Antonelli'."""
    return unquote(driver_id).replace("_", " ")


def fetch_bios(cfg: Config, refresh: bool = False) -> None:
    """Look up birth dates for every F4 driver who has a Wikipedia article (network)."""
    f4 = pd.read_csv(cfg.processed_dir / "f4_results.csv")
    ids = sorted({d for d in f4["driver_id"] if not str(d).startswith("name:")})
    client = WikiClient(cfg.raw_dir, cfg.user_agent, cfg.request_pause, refresh)
    log.info("Looking up birth dates for %d driver articles", len(ids))
    client.fetch_birth_dates([article_title(d) for d in ids])
    build_bios(cfg)


def build_bios(cfg: Config, quiet: bool = False) -> pd.DataFrame:
    """Turn the cached birth dates into data/processed/birthdates.csv (offline)."""
    cache_file = cfg.raw_dir / "driver_birthdates.json"
    if not cache_file.exists():
        if not quiet:
            log.warning("No birth-date cache yet; run `python -m f4ca bios`.")
        return pd.DataFrame()
    cache = json.loads(cache_file.read_text(encoding="utf-8"))
    f4 = pd.read_csv(cfg.processed_dir / "f4_results.csv", usecols=["driver_id"])
    rows = []
    for d in f4["driver_id"].drop_duplicates():
        raw = cache.get(article_title(d), "") if not str(d).startswith("name:") else ""
        rows.append({"driver_id": d, "birth_date": raw})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.processed_dir / "birthdates.csv", index=False)
    known = (out["birth_date"].fillna("") != "").sum()
    log.info("Birth dates known for %d of %d drivers", known, len(out))
    return out
