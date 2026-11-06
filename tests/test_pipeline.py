import json

import pandas as pd

from f4ca import build
from f4ca.cli import analyse
from f4ca.config import Config
from f4ca.features import FEATURES
from synthetic import make_world


def small_config(tmp_path):
    years = range(2014, 2026)
    cfg = Config(data_dir=tmp_path / "data", reports_dir=tmp_path / "reports",
                 app_data_dir=tmp_path / "app_data",
                 f4_series={s: {y: f"{y} {s} Championship" for y in years} for s in ["Alpha F4", "Beta F4", "Gamma F4"]},
                 tiers={"regional": {"Reg": {y: f"{y} Reg" for y in years}},
                        "tier3": {"F3": {y: f"{y} F3" for y in years}},
                        "f2": {"F2": {y: f"{y} F2" for y in years}},
                        "f1": {"F1": {y: f"{y} F1" for y in years}}},
                 n_bootstrap=50)
    cfg.ensure_dirs()
    return cfg


def test_end_to_end(tmp_path):
    cfg = small_config(tmp_path)
    make_world(cfg, drivers_per_season=30)
    out = build.build(cfg)
    assert (out["coverage"]["status"] == "parsed").all()
    analyse(cfg)

    drivers = pd.read_csv(cfg.processed_dir / "drivers.csv")
    assert drivers[[f for f in FEATURES if f != "age_at_debut"]].notna().all().all()  # age may be missing
    assert set(drivers["split"]) == {"train", "test", "unlabelled"}
    # no driver can be labelled as progressing before their debut
    assert ((drivers["first_tier3"].isna()) | (drivers["first_tier3"] >= drivers["debut_season"]) |
            (drivers["reached_tier3"] == 0)).all()

    metrics = json.loads((cfg.reports_dir / "metrics.json").read_text())
    gb = metrics["models"]["Gradient boosting"]["test"]
    assert 0.5 < gb["roc_auc"] <= 1.0          # skill drives progression in the fake world
    assert (cfg.reports_dir / "results.md").exists()
    for f in ["funnel", "roc_pr", "calibration", "importance", "series_strength",
              "progression_by_finish", "age_vs_progression"]:
        assert (cfg.figures_dir / f"{f}.png").exists()

    # age feature: present for ~70% of drivers, plausible values, ablation computed
    assert 0.5 < drivers["age_at_debut"].notna().mean() < 0.9
    assert drivers["age_at_debut"].dropna().between(13, 30).all()
    assert "ablation" in metrics["age"]
    # app data exported for the Streamlit app
    for f in ["drivers.csv", "f4_results.csv", "ladder.csv", "metrics.json"]:
        assert (cfg.app_data_dir / f).exists()
