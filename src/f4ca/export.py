"""Copy the small processed tables the Streamlit app needs into app/data/.

app/data/ is committed to git so the app can run on Streamlit Community Cloud without
re-scraping Wikipedia. Everything in it comes from public Wikipedia standings (CC BY-SA 4.0).
"""
import logging
import shutil

from .config import Config

log = logging.getLogger(__name__)
APP_FILES = ["drivers.csv", "f4_results.csv", "ladder.csv", "series_strength.csv"]


def export_app_data(cfg: Config, dest=None) -> None:
    dest = dest or cfg.app_data_dir
    dest.mkdir(parents=True, exist_ok=True)
    for name in APP_FILES:
        src = cfg.processed_dir / name
        if src.exists():
            shutil.copy(src, dest / name)
    metrics = cfg.reports_dir / "metrics.json"
    if metrics.exists():
        shutil.copy(metrics, dest / "metrics.json")
    log.info("App data written to %s", dest)
