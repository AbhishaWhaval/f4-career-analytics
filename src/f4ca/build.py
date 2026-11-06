"""Stage 1-2: download season pages and turn them into two tidy tables.

data/processed/f4_results.csv   one row per driver x F4 championship x season
data/processed/ladder.csv       one row per driver x higher championship x season
data/processed/coverage.csv     which pages were found and parsed (data audit)
"""
import logging
from typing import Dict, Iterable, Tuple

import pandas as pd

from .config import Config
from .parse import normalise_name, parse_standings
from .wiki import WikiClient

log = logging.getLogger(__name__)


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


def fetch(cfg: Config, refresh: bool = False) -> None:
    cfg.ensure_dirs()
    client = WikiClient(cfg.raw_dir, cfg.user_agent, cfg.request_pause, refresh)
    pages = list(_pages(cfg))
    for i, (_, _, series, season, title) in enumerate(pages, 1):
        log.info("[%d/%d] %s %s", i, len(pages), series, season)
        client.get_html(title)


def build(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Parse cached pages (no network) into processed CSVs."""
    cfg.ensure_dirs()
    client = WikiClient(cfg.raw_dir, cfg.user_agent)
    client.get_html = _offline(client)           # build never touches the network
    f4_rows, ladder_rows, coverage = [], [], []

    for kind, tier, series, season, title in _pages(cfg):
        html = client.get_html(title)
        status, n = "missing", 0
        if html is not None:
            df = parse_standings(html)
            n = len(df)
            status = "parsed" if n else "no standings table"
            if n:
                df["series"], df["season"], df["tier"] = series, season, tier
                (f4_rows if kind == "f4" else ladder_rows).append(df)
        coverage.append(dict(kind=kind, tier=tier, series=series, season=season,
                             title=title, status=status, drivers=n))

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
    return {"f4": f4, "ladder": ladder, "coverage": cov}


def _offline(client: WikiClient):
    from .wiki import cache_path

    def get(title: str):
        p = cache_path(client.raw_dir, title)
        return p.read_text(encoding="utf-8") if p.exists() else None
    return get
