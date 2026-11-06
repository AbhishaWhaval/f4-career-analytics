"""Stage 5: write reports/results.md from the pipeline outputs."""
from datetime import date

import pandas as pd

from .config import Config
from .model import TIER_LABELS


def _fmt_ci(m, key, digits=3):
    lo, hi = m[key + "_ci"]
    return f"{m[key]:.{digits}f} ({lo:.{digits}f}\u2013{hi:.{digits}f})"


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(cfg: Config, feats, coverage, desc, results, best, importance, coefs, watchlist, age=None) -> str:
    lab = feats[feats["labelled"]]
    tgt = TIER_LABELS[cfg.target]
    parsed = coverage[coverage["status"] == "parsed"]
    f4_parsed = parsed[parsed["kind"] == "f4"]

    metrics = pd.DataFrame([{
        "Model": name,
        "CV PR-AUC (train)": f"{r['cv_train']['pr_auc']:.3f}",
        "Test ROC-AUC (95% CI)": _fmt_ci(r["test"], "roc_auc"),
        "Test PR-AUC (95% CI)": _fmt_ci(r["test"], "pr_auc"),
        "Test Brier": f"{r['test']['brier']:.3f}",
        "Precision @ top 10%": f"{r['test']['precision_at_top10pct']:.0%}",
    } for name, r in results.items()])
    t = results[best]["test"]

    funnel = desc["funnel"].assign(share=lambda d: d["share"].map("{:.1%}".format))
    by_finish = desc["by_finish"].reset_index().assign(rate=lambda d: d["rate"].map("{:.0%}".format))
    by_series = (desc["by_series"][desc["by_series"]["drivers"] >= 10].reset_index()
                 .assign(rate=lambda d: d["rate"].map("{:.0%}".format)))
    imp = importance.head(8).assign(importance=lambda d: d["importance"].map("{:.3f}".format),
                                    std=lambda d: d["std"].map("{:.3f}".format))
    co = coefs.head(8).assign(coef_per_sd=lambda d: d["coef_per_sd"].map("{:+.2f}".format),
                              odds_ratio=lambda d: d["odds_ratio"].map("{:.2f}".format))
    wl = watchlist.head(15).assign(probability=lambda d: d["probability"].map("{:.0%}".format))

    age_md = _age_section(age or {}, tgt, cfg)
    md = f"""# Results: who climbs from Formula 4?

_Generated {date.today():%d %b %Y} by `python -m f4ca all`. Numbers change if the data is re-scraped._

## Data
- **{len(f4_parsed)}** F4 championship seasons parsed across **{f4_parsed['series'].nunique()}** series, plus **{len(parsed) - len(f4_parsed)}** seasons of higher categories.
- **{len(feats):,}** distinct F4 drivers; **{len(lab):,}** have a fully observed {cfg.horizon}-year outcome window (debut {int(lab.debut_season.min())}\u2013{int(lab.debut_season.max())}).
- Page-level audit: `data/processed/coverage.csv`.

## The ladder
{_table(funnel)}

![Funnel](figures/funnel.png)

## Rookie-season finish vs reaching {tgt} within {cfg.horizon} years
{_table(by_finish)}

![Progression by finish](figures/progression_by_finish.png)

## Progression rate by debut series (series with 10+ debutants)
{_table(by_series)}

![Series strength](figures/series_strength.png)

## Model performance
Target: reached **{tgt}** within **{cfg.horizon}** years of F4 debut, predicted from the rookie season only.
Train cohorts {int(feats[feats.split=='train'].debut_season.min())}\u2013{int(feats[feats.split=='train'].debut_season.max())}, holdout cohorts {int(feats[feats.split=='test'].debut_season.min())}\u2013{int(feats[feats.split=='test'].debut_season.max())}.
Holdout: {t['n']} drivers, {t['positives']} progressed (base rate {t['base_rate']:.1%}).

{_table(metrics)}

Selected model (best cross-validated PR-AUC on train): **{best}**.

![ROC and PR](figures/roc_pr.png)
![Calibration](figures/calibration.png)

## What drives the prediction
Permutation importance on the holdout set ({best}):

{_table(imp)}

![Importance](figures/importance.png)

Logistic regression coefficients (per standard deviation; odds ratio > 1 raises the odds):

{_table(co)}

{age_md}

## Watchlist: open cases, highest predicted probability
Drivers whose outcome window has not closed yet and who have not reached {tgt}.

{_table(wl)}

## Caveats
- Wikipedia coverage is uneven; missing seasons are listed in `coverage.csv`.
- Money, age, team and junior-academy membership drive careers but are not in the data. The model measures on-track signal only.
- Drivers are matched across pages by Wikipedia article, then by normalised name; rare mismatches are possible.
- Series strength is estimated from drivers who raced in several F4 championships in the same year, using training seasons only.
"""
    (cfg.reports_dir / "results.md").write_text(md, encoding="utf-8")
    return md


def _age_section(age: dict, tgt: str, cfg: Config) -> str:
    if not age or not age.get("coverage_all"):
        return "## Age at debut\nNo birth dates available. Run `python -m f4ca bios` to add them.\n"
    lines = ["## Age at debut",
             f"Birth dates were found for **{age['coverage_all']:.0%}** of drivers (those with a Wikipedia article)."]
    cp, cn = age.get("coverage_progressed"), age.get("coverage_not_progressed")
    if cp is not None and cn is not None:
        if cp - cn > 0.05:
            lines.append(f"Coverage is uneven: {cp:.0%} of drivers who progressed have a known age versus "
                         f"{cn:.0%} of those who did not, because successful drivers are more likely to have "
                         "an article. A \"missing age\" flag would therefore leak the outcome, so missing ages "
                         "are imputed with the median and no missing-flag is given to the model.")
        else:
            lines.append(f"Coverage is similar for drivers who progressed ({cp:.0%}) and those who did not "
                         f"({cn:.0%}). Missing ages are imputed with the median, with no missing-flag, "
                         "so article existence cannot leak the outcome.")
    if age.get("by_age"):
        t = pd.DataFrame(age["by_age"]).assign(rate=lambda d: d["rate"].map("{:.0%}".format))
        lines += ["", f"Progression to {tgt} within {cfg.horizon} years, by age at F4 debut (known ages only):",
                  "", _table(t), "", "![Age](figures/age_vs_progression.png)"]
    ab = age.get("ablation")
    if ab:
        w, wo = ab["with_age"], ab["without_age"]
        lines += ["", f"**Ablation** (logistic regression, holdout drivers with a known age: n={ab['n_test']}, "
                  f"{ab['positives']} progressed):", "",
                  "| Model | ROC-AUC | PR-AUC |", "|---|---|---|",
                  f"| Without age | {wo['roc_auc']:.3f} | {wo['pr_auc']:.3f} |",
                  f"| With age | {w['roc_auc']:.3f} | {w['pr_auc']:.3f} |"]
    return "\n".join(lines)
