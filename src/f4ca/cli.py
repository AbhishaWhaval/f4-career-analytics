"""Command line: python -m f4ca {fetch,build,features,train,all}"""
import argparse
import logging

import pandas as pd

from . import build as build_mod
from . import features as feat_mod
from .config import Config
from .model import plot_descriptive, train_and_evaluate
from .report import write_report


def analyse(cfg: Config) -> None:
    feats, strength = feat_mod.run(cfg)
    coverage = pd.read_csv(cfg.processed_dir / "coverage.csv")
    desc = plot_descriptive(feats, strength, cfg)
    results, best, importance, coefs, watchlist = train_and_evaluate(feats, cfg)
    write_report(cfg, feats, coverage, desc, results, best, importance, coefs, watchlist)
    logging.info("Report written to %s", cfg.reports_dir / "results.md")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="f4ca", description=__doc__)
    p.add_argument("command", choices=["fetch", "build", "train", "all"],
                   help="fetch = download pages; build = parse to CSV; train = features, models, report; all = everything")
    p.add_argument("--refresh", action="store_true", help="re-download pages already in data/raw")
    p.add_argument("--target", choices=["regional", "tier3", "f2", "f1"], default=None)
    p.add_argument("--horizon", type=int, default=None, help="years after F4 debut to reach the target")
    p.add_argument("--test-from", type=int, default=None, help="first debut year in the holdout set")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = Config()
    if a.target:
        cfg.target = a.target
    if a.horizon:
        cfg.horizon = a.horizon
    if a.test_from:
        cfg.test_from = a.test_from

    if a.command in ("fetch", "all"):
        build_mod.fetch(cfg, refresh=a.refresh)
    if a.command in ("build", "all"):
        build_mod.build(cfg)
    if a.command in ("train", "all"):
        analyse(cfg)
    return 0
