"""Stage 4: train, validate and explain the progression models.

Validation is temporal: models are fit on earlier F4 cohorts and scored on later ones,
which mirrors how a scout would actually use them. Test metrics carry bootstrap 95% CIs
because positive cases are rare and point estimates alone would overstate certainty.
"""
import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss,  # noqa: E402
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .config import TIER_ORDER, Config
from .features import FEATURES

log = logging.getLogger(__name__)
PALETTE = {"Baseline: finish only": "#9aa4ad", "Logistic regression": "#1f6fd1",
           "Gradient boosting": "#d7262e"}
TIER_LABELS = {"regional": "Regional F3", "tier3": "FIA F3 / GP3 / Euro F3",
               "f2": "F2 / GP2", "f1": "Formula 1"}


def _impute():
    # Median imputation WITHOUT a missing-indicator: whether a driver has a Wikipedia article
    # (and so a known birth date) partly depends on later success, so "age is missing" would
    # leak the outcome. Imputing hides that signal from both models.
    return SimpleImputer(strategy="median", keep_empty_features=True)


def models(seed: int):
    return {
        "Baseline: finish only": (["best_pct"], make_pipeline(StandardScaler(), LogisticRegression())),
        "Logistic regression": (FEATURES, make_pipeline(_impute(), StandardScaler(),
                                                        LogisticRegression(C=0.5, max_iter=2000))),
        "Gradient boosting": (FEATURES, make_pipeline(_impute(), HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=250, l2_regularization=1.0,
            min_samples_leaf=20, random_state=seed))),
    }


def age_analysis(feats: pd.DataFrame, cfg: Config) -> dict:
    """How much does age at debut add? Compared on drivers whose age is known, so the
    comparison is not distorted by imputed values."""
    target = f"reached_{cfg.target}_in_horizon"
    lab = feats[feats["labelled"]]
    known = lab["age_at_debut"].notna()
    out = {
        "coverage_all": float(feats["age_at_debut"].notna().mean()),
        "coverage_progressed": float(known[lab[target] == 1].mean()) if (lab[target] == 1).any() else None,
        "coverage_not_progressed": float(known[lab[target] == 0].mean()) if (lab[target] == 0).any() else None,
    }
    tr = feats[(feats["split"] == "train") & feats["age_at_debut"].notna()]
    te = feats[(feats["split"] == "test") & feats["age_at_debut"].notna()]
    if min(tr[target].sum(), te[target].sum()) >= 5 and min((tr[target] == 0).sum(), (te[target] == 0).sum()) >= 5:
        without = [f for f in FEATURES if f != "age_at_debut"]
        res = {}
        for name, cols in [("with_age", FEATURES), ("without_age", without)]:
            m = make_pipeline(_impute(), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
            p = m.fit(tr[cols], tr[target]).predict_proba(te[cols])[:, 1]
            res[name] = {"pr_auc": float(average_precision_score(te[target], p)),
                         "roc_auc": float(roc_auc_score(te[target], p))}
        out["ablation"] = {"n_test": int(len(te)), "positives": int(te[target].sum()), **res}

    # Progression by age group (labelled drivers with a known age)
    la = lab[known].copy()
    if len(la):
        bins = [0, 15.5, 16.5, 17.5, 19, 99]
        names = ["Under 15.5", "15.5\u201316.5", "16.5\u201317.5", "17.5\u201319", "19+"]
        la["age_group"] = pd.cut(la["age_at_debut"], bins=bins, labels=names)
        g = la.groupby("age_group", observed=False).agg(drivers=(target, "size"), progressed=(target, "sum"))
        g["rate"] = g["progressed"] / g["drivers"].clip(lower=1)
        out["by_age"] = g.reset_index().to_dict("records")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(g.index.astype(str), g["rate"], color="#e8742a")
        for i, (r, n) in enumerate(zip(g["rate"], g["drivers"])):
            ax.text(i, r, f"{r:.0%}\nn={n}", ha="center", va="bottom", fontsize=8)
        _style(ax, f"Age at F4 debut vs reaching {TIER_LABELS[cfg.target]} in {cfg.horizon} yrs")
        ax.set_ylabel("Share who progressed"); ax.set_xlabel("Age at debut (years)")
        ax.set_ylim(0, max(0.1, g["rate"].max() * 1.3))
        fig.tight_layout(); fig.savefig(cfg.figures_dir / "age_vs_progression.png", dpi=150); plt.close(fig)
    return out


def bootstrap_ci(y, p, metric, n, seed):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if y[s].min() == y[s].max():
            continue
        vals.append(metric(y[s], p[s]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def evaluate(y, p, n_boot, seed):
    out = {}
    for name, fn in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score),
                     ("brier", brier_score_loss)]:
        out[name] = float(fn(y, p))
        out[name + "_ci"] = bootstrap_ci(y, p, fn, n_boot, seed)
    out["base_rate"] = float(y.mean())
    out["n"] = int(len(y))
    out["positives"] = int(y.sum())
    # Of the 10% highest-scored drivers, how many actually made it?
    k = max(1, int(round(0.1 * len(y))))
    top = np.argsort(-p)[:k]
    out["precision_at_top10pct"] = float(y[top].mean())
    out["recall_at_top10pct"] = float(y[top].sum() / max(1, y.sum()))
    return out


def train_and_evaluate(feats: pd.DataFrame, cfg: Config):
    cfg.ensure_dirs()
    target = f"reached_{cfg.target}_in_horizon"
    train = feats[feats["split"] == "train"]
    test = feats[feats["split"] == "test"]
    if train[target].nunique() < 2 or test[target].nunique() < 2:
        raise RuntimeError("Train or test set has only one class; widen the data or change test_from.")

    ytr, yte = train[target].to_numpy(), test[target].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=cfg.random_state)
    results, fitted, test_preds = {}, {}, {}
    for name, (cols, est) in models(cfg.random_state).items():
        cv_p = cross_val_predict(clone(est), train[cols], ytr, cv=cv, method="predict_proba")[:, 1]
        m = clone(est).fit(train[cols], ytr)
        p = m.predict_proba(test[cols])[:, 1]
        results[name] = {"cv_train": {"roc_auc": float(roc_auc_score(ytr, cv_p)),
                                      "pr_auc": float(average_precision_score(ytr, cv_p))},
                         "test": evaluate(yte, p, cfg.n_bootstrap, cfg.random_state)}
        fitted[name], test_preds[name] = (cols, m), p
        log.info("%-24s test ROC-AUC %.3f  PR-AUC %.3f", name,
                 results[name]["test"]["roc_auc"], results[name]["test"]["pr_auc"])

    best = max(["Logistic regression", "Gradient boosting"], key=lambda k: results[k]["cv_train"]["pr_auc"])
    cols, best_model = fitted[best]
    imp = permutation_importance(best_model, test[cols], yte, scoring="average_precision",
                                 n_repeats=30, random_state=cfg.random_state)
    importance = (pd.DataFrame({"feature": cols, "importance": imp.importances_mean, "std": imp.importances_std})
                  .sort_values("importance", ascending=False))

    coefs = None
    lr_cols, lr = fitted["Logistic regression"]
    coefs = (pd.DataFrame({"feature": lr_cols, "coef_per_sd": lr[-1].coef_[0]})
             .assign(odds_ratio=lambda d: np.exp(d["coef_per_sd"]))
             .sort_values("coef_per_sd", key=abs, ascending=False))

    # Final model on every labelled cohort, then score drivers whose outcome is still open
    labelled = feats[feats["labelled"]]
    final = clone(models(cfg.random_state)[best][1]).fit(labelled[cols], labelled[target])
    open_ = feats[(feats["split"] == "unlabelled") & (feats[f"reached_{cfg.target}"] == 0)].copy()
    open_["probability"] = final.predict_proba(open_[cols])[:, 1] if len(open_) else []
    watchlist = open_.sort_values("probability", ascending=False)[
        ["driver", "debut_season", "debut_series", "best_pos", "n_series", "probability"]]

    preds = test[["driver", "debut_season", "debut_series", target]].copy()
    for name, p in test_preds.items():
        preds[name] = p

    out = cfg.reports_dir
    with open(out / "metrics.json", "w") as fh:
        json.dump({"target": cfg.target, "horizon": cfg.horizon, "best_model": best,
                   "train_cohorts": [int(train.debut_season.min()), int(train.debut_season.max())],
                   "test_cohorts": [int(test.debut_season.min()), int(test.debut_season.max())],
                   "models": results}, fh, indent=2)
    importance.to_csv(out / "permutation_importance.csv", index=False)
    coefs.to_csv(out / "logit_coefficients.csv", index=False)
    preds.to_csv(out / "test_predictions.csv", index=False)
    watchlist.to_csv(out / "watchlist.csv", index=False)

    age = age_analysis(feats, cfg)
    with open(out / "metrics.json") as fh:
        m = json.load(fh)
    m["age"] = age
    with open(out / "metrics.json", "w") as fh:
        json.dump(m, fh, indent=2)

    plot_curves(yte, test_preds, cfg)
    plot_calibration(yte, test_preds, cfg)
    plot_importance(importance, best, cfg)
    return results, best, importance, coefs, watchlist, age


# ---------------------------------------------------------------------- figures
def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def plot_curves(y, preds, cfg):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        a.plot(fpr, tpr, color=PALETTE[name], lw=2, label=f"{name} ({roc_auc_score(y, p):.2f})")
        pr, rc, _ = precision_recall_curve(y, p)
        b.plot(rc, pr, color=PALETTE[name], lw=2, label=f"{name} ({average_precision_score(y, p):.2f})")
    a.plot([0, 1], [0, 1], ls="--", color="#bbb")
    b.axhline(y.mean(), ls="--", color="#bbb", label=f"Base rate ({y.mean():.2f})")
    _style(a, "ROC curve (holdout cohorts)"); a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate")
    _style(b, "Precision-recall (holdout cohorts)"); b.set_xlabel("Recall"); b.set_ylabel("Precision")
    a.legend(fontsize=8, loc="lower right"); b.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(cfg.figures_dir / "roc_pr.png", dpi=150); plt.close(fig)


def plot_calibration(y, preds, cfg):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in preds.items():
        frac, mean = calibration_curve(y, p, n_bins=6, strategy="quantile")
        ax.plot(mean, frac, "o-", color=PALETTE[name], label=name)
    ax.plot([0, 1], [0, 1], ls="--", color="#bbb")
    _style(ax, "Calibration (holdout cohorts)")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed share who progressed")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(cfg.figures_dir / "calibration.png", dpi=150); plt.close(fig)


def plot_importance(imp, model_name, cfg):
    imp = imp.sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp["feature"], imp["importance"], xerr=imp["std"], color="#1f6fd1", alpha=0.85)
    _style(ax, f"Permutation importance: {model_name}")
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    fig.tight_layout(); fig.savefig(cfg.figures_dir / "importance.png", dpi=150); plt.close(fig)


def plot_descriptive(feats: pd.DataFrame, strength: pd.DataFrame, cfg: Config) -> dict:
    """Funnel, progression by rookie finish, and series strength. Returns summary tables."""
    cfg.ensure_dirs()
    lab = feats[feats["labelled"]]

    funnel = pd.DataFrame({"stage": ["F4 debut"] + [TIER_LABELS[t] for t in TIER_ORDER],
                           "drivers": [len(lab)] + [int(lab[f"reached_{t}"].sum()) for t in TIER_ORDER]})
    funnel["share"] = funnel["drivers"] / max(1, len(lab))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(funnel["stage"][::-1], funnel["drivers"][::-1], color="#d7262e")
    for i, (n, s) in enumerate(zip(funnel["drivers"][::-1], funnel["share"][::-1])):
        ax.text(n, i, f"  {n} ({s:.1%})", va="center", fontsize=9)
    _style(ax, f"The ladder: F4 debutants {int(lab.debut_season.min())}\u2013{int(lab.debut_season.max())}")
    ax.set_xlabel("Drivers (ever reached, any time)")
    fig.tight_layout(); fig.savefig(cfg.figures_dir / "funnel.png", dpi=150); plt.close(fig)

    bins = [0, 1, 3, 5, 10, 999]
    names = ["Champion", "P2\u2013P3", "P4\u2013P5", "P6\u2013P10", "P11+"]
    lab = lab.assign(finish=pd.cut(lab["best_pos"], bins=bins, labels=names, right=True))
    target = f"reached_{cfg.target}_in_horizon"
    by_finish = lab.groupby("finish", observed=False).agg(drivers=(target, "size"), progressed=(target, "sum"))
    by_finish["rate"] = by_finish["progressed"] / by_finish["drivers"].clip(lower=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_finish.index.astype(str), by_finish["rate"], color="#1f6fd1")
    for i, (r, n) in enumerate(zip(by_finish["rate"], by_finish["drivers"])):
        ax.text(i, r, f"{r:.0%}\nn={n}", ha="center", va="bottom", fontsize=8)
    _style(ax, f"Rookie-season finish vs reaching {TIER_LABELS[cfg.target]} in {cfg.horizon} yrs")
    ax.set_ylabel("Share who progressed"); ax.set_ylim(0, max(0.1, by_finish["rate"].max() * 1.3))
    fig.tight_layout(); fig.savefig(cfg.figures_dir / "progression_by_finish.png", dpi=150); plt.close(fig)

    s = strength[strength["estimated"]].sort_values("strength")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(s["series"], s["strength"], color=["#d7262e" if v > 0 else "#9aa4ad" for v in s["strength"]])
    ax.axvline(0, color="#555", lw=1)
    _style(ax, "Estimated field strength by F4 series")
    ax.set_xlabel("Percentile penalty for the same driver (higher = tougher field)")
    fig.tight_layout(); fig.savefig(cfg.figures_dir / "series_strength.png", dpi=150); plt.close(fig)

    by_series = (lab.groupby("debut_series")
                 .agg(drivers=(target, "size"), progressed=(target, "sum"))
                 .assign(rate=lambda d: d["progressed"] / d["drivers"])
                 .sort_values("rate", ascending=False))
    return {"funnel": funnel, "by_finish": by_finish, "by_series": by_series}
