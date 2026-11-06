"""Project configuration: which championships to scrape and how outcomes are defined.

Every championship is a mapping of season -> Wikipedia page title. Pages that do not
exist are skipped with a warning, so year ranges can be generous.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


def _titles(template: str, years) -> Dict[int, str]:
    return {y: template.format(y=y) for y in years}


def _uae_titles() -> Dict[int, str]:
    # Winter series. The first two seasons use split-year titles (season = the year it ends);
    # from 2019 Wikipedia uses single years, and 2025 ran as the F4 Middle East Championship.
    t = {2017: "2016\u201317 Formula 4 UAE Championship", 2018: "2017\u201318 Formula 4 UAE Championship"}
    t.update({y: f"{y} Formula 4 UAE Championship" for y in range(2019, 2025)})
    t[2025] = "2025 F4 Middle East Championship"
    return t


def _danish_titles() -> Dict[int, str]:
    # Renamed Nordic 4 from 2024
    t = {y: f"{y} F4 Danish Championship" for y in range(2017, 2024)}
    t.update({y: f"{y} Nordic 4 Championship" for y in range(2024, 2026)})
    return t


# ---- FIA-certified Formula 4 championships (the population) ---------------------------
F4_SERIES: Dict[str, Dict[int, str]] = {
    "Italian F4": _titles("{y} Italian F4 Championship", range(2014, 2026)),
    "ADAC F4": _titles("{y} ADAC Formula 4 Championship", range(2015, 2026)),
    "British F4": _titles("{y} F4 British Championship", range(2015, 2026)),
    "Spanish F4": _titles("{y} F4 Spanish Championship", range(2016, 2026)),
    "French F4": _titles("{y} French F4 Championship", range(2015, 2026)),
    "Japanese F4": _titles("{y} F4 Japanese Championship", range(2015, 2026)),
    "US F4": _titles("{y} Formula 4 United States Championship", range(2016, 2026)),
    "Australian F4": _titles("{y} Australian Formula 4 Championship", range(2015, 2020)),
    "Chinese F4": _titles("{y} F4 Chinese Championship", range(2015, 2026)),
    "SEA F4": _titles("{y} F4 South East Asia Championship", range(2016, 2026)),
    "Danish F4": _danish_titles(),
    "NEZ F4": _titles("{y} SMP F4 Championship", range(2015, 2020)),   # F4 NEZ ran as SMP F4
    "Indian F4": _titles("{y} F4 Indian Championship", range(2023, 2026)),
    "F4 UAE": _uae_titles(),
}

# Words a search result must contain to be accepted as a season page of each series
# (used when a configured title does not exist; see wiki.WikiClient.find_season_title)
SERIES_KEYWORDS: Dict[str, List[str]] = {
    "Italian F4": ["Italian"], "ADAC F4": ["ADAC"], "British F4": ["British"],
    "Spanish F4": ["Spanish"], "French F4": ["French"], "Japanese F4": ["Japanese"],
    "US F4": ["United States"], "Australian F4": ["Australian"], "Chinese F4": ["Chinese"],
    "SEA F4": ["South East Asia", "Southeast Asia", "SEA"], "Danish F4": ["Danish", "Nordic"],
    "NEZ F4": ["SMP", "NEZ", "North European"], "Indian F4": ["Indian"],
    "F4 UAE": ["UAE", "Middle East"],
}

# ---- Higher rungs of the ladder (the outcomes) -----------------------------------------
TIERS: Dict[str, Dict[str, Dict[int, str]]] = {
    "regional": {
        "FRECA": _titles("{y} Formula Regional European Championship", range(2019, 2026)),
        "Euroformula Open": _titles("{y} Euroformula Open Championship", range(2014, 2026)),
    },
    "tier3": {
        "FIA F3": _titles("{y} FIA Formula 3 Championship", range(2019, 2026)),
        "GP3": _titles("{y} GP3 Series", range(2014, 2019)),
        "European F3": _titles("{y} FIA Formula 3 European Championship", range(2014, 2019)),
    },
    "f2": {
        "FIA F2": _titles("{y} FIA Formula 2 Championship", range(2017, 2026)),
        "GP2": _titles("{y} GP2 Series", range(2014, 2017)),
    },
    "f1": {
        "Formula One": _titles("{y} Formula One World Championship", range(2015, 2026)),
    },
}

TIER_ORDER: List[str] = ["regional", "tier3", "f2", "f1"]


@dataclass
class Config:
    data_dir: Path = ROOT / "data"
    reports_dir: Path = ROOT / "reports"
    app_data_dir: Path = ROOT / "app" / "data"
    f4_series: Dict[str, Dict[int, str]] = field(default_factory=lambda: F4_SERIES)
    tiers: Dict[str, Dict[str, Dict[int, str]]] = field(default_factory=lambda: TIERS)
    last_complete_season: int = 2025      # newest season whose results are final
    target: str = "tier3"                 # main prediction target
    horizon: int = 4                      # years after F4 debut allowed to reach the target
    test_from: int = 2019                 # cohorts debuting from this year onward = holdout
    n_bootstrap: int = 1000
    random_state: int = 42
    user_agent: str = "f4-career-analytics/1.0 (portfolio project; https://github.com/AbhishaWhaval)"
    request_pause: float = 0.6            # seconds between Wikipedia API calls

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.processed_dir, self.figures_dir):
            p.mkdir(parents=True, exist_ok=True)
