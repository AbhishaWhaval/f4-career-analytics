"""Stage 3: one row per driver, described only by their FIRST F4 season.

Prediction setting: it is the end of a driver's rookie F4 year. Using only what is known
at that moment, how likely are they to reach the target tier within `horizon` years?

Nothing from later seasons enters the features, which keeps the model free of leakage.
"""
import numpy as np
import pandas as pd

from .config import TIER_ORDER, Config
from .strength import estimate_strength, finish_percentile

FEATURES = [
    "best_pct", "mean_pct", "best_points_share", "win_rate", "podium_rate", "pole_rate",
    "fastest_lap_rate", "finish_rate", "participation", "n_series", "champion",
    "series_strength", "strength_adj_pct", "age_at_debut",
]


def age_at_debut(birth_date: pd.Series, debut_season: pd.Series) -> pd.Series:
    """Age in years at the middle of the debut season (1 July). 'YYYY' alone counts as 1 July.
    Values outside 13-30 are treated as parsing errors and set to missing."""
    s = birth_date.fillna("").astype(str)
    year = pd.to_numeric(s.str[:4], errors="coerce")
    month = pd.to_numeric(s.str[5:7], errors="coerce").fillna(7)
    day = pd.to_numeric(s.str[8:10], errors="coerce").fillna(1)
    born = year + (month - 1) / 12 + (day - 1) / 365
    age = debut_season + 0.5 - born
    return age.where(age.between(13, 30))


def _match_ladder(drivers: pd.DataFrame, ladder: pd.DataFrame) -> pd.DataFrame:
    """First season each driver appears in each tier. Matches on the Wikipedia article
    first, then on the normalised name (catches redirects and unlinked entries)."""
    out = drivers[["driver_id", "name_key"]].copy()
    for tier in TIER_ORDER:
        lt = ladder[ladder["tier"] == tier]
        by_id = lt.groupby("driver_id")["season"].min()
        by_name = lt.groupby("name_key")["season"].min()
        first = out["driver_id"].map(by_id)
        out[f"first_{tier}"] = first.fillna(out["name_key"].map(by_name))
    return out.drop(columns=["name_key"])


def build_features(f4: pd.DataFrame, ladder: pd.DataFrame, cfg: Config, births: pd.DataFrame = None):
    f4 = f4.copy()
    f4["pct"] = finish_percentile(f4)
    champ_pts = f4.groupby(["series", "season"])["points"].transform("max").replace(0, np.nan)
    f4["points_share"] = (f4["points"] / champ_pts).fillna(0)

    strength = estimate_strength(f4, max_season=cfg.test_from - 1)
    f4["series_strength"] = f4["series"].map(strength.set_index("series")["strength"]).fillna(0)
    # Strength-adjusted percentile: a mid-field finish in a strong series counts for more
    f4["adj_pct"] = f4["pct"] - f4["series_strength"]

    debut = f4.groupby("driver_id")["season"].transform("min")
    rookie = f4[f4["season"] == debut]

    g = rookie.groupby("driver_id")
    entries = g["entries"].sum().clip(lower=1)
    best_row = rookie.loc[rookie.groupby("driver_id")["adj_pct"].idxmin()].set_index("driver_id")

    feats = pd.DataFrame({
        "driver": g["driver"].first(),
        "name_key": g["name_key"].first(),
        "debut_season": g["season"].first(),
        "debut_series": best_row["series"],
        "best_pos": g["pos"].min(),
        "best_pct": g["pct"].min(),
        "mean_pct": g["pct"].mean(),
        "best_points_share": g["points_share"].max(),
        "win_rate": g["wins"].sum() / entries,
        "podium_rate": g["podiums"].sum() / entries,
        "pole_rate": g["poles"].sum() / entries,
        "fastest_lap_rate": g["fastest_laps"].sum() / entries,
        "finish_rate": g["finishes"].sum() / entries,
        "participation": (g["entries"].sum() / g["season_races"].sum().clip(lower=1)).clip(upper=1),
        "n_series": g["series"].nunique(),
        "champion": (g["pos"].min() == 1).astype(int),
        "series_strength": best_row["series_strength"],
        "strength_adj_pct": best_row["adj_pct"],
        "f4_seasons_total": f4.groupby("driver_id")["season"].nunique(),
    }).reset_index()

    if births is not None and len(births):
        feats = feats.merge(births[["driver_id", "birth_date"]], on="driver_id", how="left")
    else:
        feats["birth_date"] = np.nan
    feats["age_at_debut"] = age_at_debut(feats["birth_date"], feats["debut_season"])

    labels = _match_ladder(feats, ladder)
    feats = feats.merge(labels, on="driver_id", how="left")
    for tier in TIER_ORDER:
        first = feats[f"first_{tier}"]
        feats[f"reached_{tier}"] = (first.notna() & (first >= feats["debut_season"])).astype(int)
        feats[f"reached_{tier}_in_horizon"] = (
            first.notna() & (first >= feats["debut_season"]) &
            (first <= feats["debut_season"] + cfg.horizon)).astype(int)

    # A cohort is "labelled" only if its full horizon has been observed
    feats["labelled"] = feats["debut_season"] + cfg.horizon <= cfg.last_complete_season
    feats["split"] = np.where(feats["debut_season"] >= cfg.test_from, "test", "train")
    feats.loc[~feats["labelled"], "split"] = "unlabelled"
    return feats, strength


def run(cfg: Config):
    f4 = pd.read_csv(cfg.processed_dir / "f4_results.csv")
    ladder = pd.read_csv(cfg.processed_dir / "ladder.csv")
    bfile = cfg.processed_dir / "birthdates.csv"
    births = pd.read_csv(bfile, dtype={"birth_date": str}) if bfile.exists() else None
    feats, strength = build_features(f4, ladder, cfg, births)
    feats.to_csv(cfg.processed_dir / "drivers.csv", index=False)
    strength.to_csv(cfg.processed_dir / "series_strength.csv", index=False)
    return feats, strength
