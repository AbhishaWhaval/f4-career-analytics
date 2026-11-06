"""F4 → F1 career explorer (Streamlit).

Run locally:   streamlit run app/streamlit_app.py
Data:          app/data/ (written by `python -m f4ca train`)
"""
import json
import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f4ca.features import FEATURES  # noqa: E402

DATA = Path(os.environ.get("F4CA_APP_DATA", Path(__file__).parent / "data"))
RUNGS = ["F4", "Regional", "F3 level", "F2", "Formula 1"]
TIER_TO_RUNG = {"f4": "F4", "regional": "Regional", "tier3": "F3 level", "f2": "F2", "f1": "Formula 1"}

st.set_page_config(page_title="F4 → F1 career explorer", page_icon="🏎️", layout="wide")


@st.cache_data
def load():
    drivers = pd.read_csv(DATA / "drivers.csv")
    f4 = pd.read_csv(DATA / "f4_results.csv")
    ladder = pd.read_csv(DATA / "ladder.csv")
    metrics = json.loads((DATA / "metrics.json").read_text()) if (DATA / "metrics.json").exists() else {}
    return drivers, f4, ladder, metrics


@st.cache_data
def score(drivers: pd.DataFrame, target: str) -> pd.Series:
    """Probability of reaching the target rung, as the model would have seen it.
    Labelled drivers get out-of-fold predictions (the model never saw their outcome);
    open cases are scored by a model trained on every labelled driver."""
    def model():
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    lab = drivers[drivers["labelled"]]
    p = pd.Series(np.nan, index=drivers.index)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    p[lab.index] = cross_val_predict(model(), lab[FEATURES], lab[target], cv=cv, method="predict_proba")[:, 1]
    open_ = drivers[~drivers["labelled"]]
    if len(open_):
        p[open_.index] = model().fit(lab[FEATURES], lab[target]).predict_proba(open_[FEATURES])[:, 1]
    return p


if not (DATA / "drivers.csv").exists():
    st.error("No data found in app/data/. Run `python -m f4ca train` first, then reload.")
    st.stop()

drivers, f4, ladder, metrics = load()
target_tier = metrics.get("target", "tier3")
horizon = metrics.get("horizon", 4)
target = f"reached_{target_tier}_in_horizon"
drivers["probability"] = score(drivers, target)

st.title("From F4 to F1")
st.caption("Who climbs motorsport's junior ladder? Built from Wikipedia standings for every FIA Formula 4 "
           "championship since 2014 and every series above it.")

tab_driver, tab_ladder, tab_about = st.tabs(["Driver lookup", "The ladder", "About the model"])

# ---------------------------------------------------------------- driver lookup
with tab_driver:
    options = drivers.sort_values("driver")
    labels = dict(zip(options["driver_id"],
                      options["driver"] + " (" + options["debut_series"] + ", " +
                      options["debut_season"].astype(str) + ")"))
    f1 = options[options["reached_f1"] == 1]
    default = f1["driver_id"].iloc[0] if len(f1) else options["driver_id"].iloc[0]
    pick = st.selectbox("Find a driver (type to search)", options["driver_id"].tolist(),
                        index=options["driver_id"].tolist().index(default), format_func=labels.get)
    d = drivers[drivers["driver_id"] == pick].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F4 debut", int(d["debut_season"]))
    c2.metric("Rookie series", d["debut_series"])
    c3.metric("Best rookie finish", f"P{int(d['best_pos'])}")
    c4.metric("Age at debut", f"{d['age_at_debut']:.1f}" if pd.notna(d.get("age_at_debut")) else "unknown")

    reached = [r for r in ["f1", "f2", "tier3", "regional"] if d.get(f"reached_{r}") == 1]
    highest = TIER_TO_RUNG[reached[0]] if reached else "F4"
    prob = d["probability"]
    if d["labelled"]:
        outcome = "Yes" if d[target] == 1 else "No"
        st.info(f"**Highest rung reached:** {highest}.  \n"
                f"**Model estimate after the rookie season** of reaching F3 level within {horizon} years: "
                f"**{prob:.0%}** (out-of-sample). Actually did: **{outcome}**.")
    else:
        st.info(f"**Highest rung reached so far:** {highest}.  \n"
                f"**Model estimate** of reaching F3 level within {horizon} years of debut: **{prob:.0%}**. "
                "The outcome window is still open.")

    career = pd.concat([
        f4[f4["driver_id"] == pick].assign(rung="F4"),
        ladder[ladder["driver_id"] == pick].assign(rung=lambda x: x["tier"].map(TIER_TO_RUNG)),
    ], ignore_index=True)
    if len(career):
        chart = (alt.Chart(career).mark_line(point=alt.OverlayMarkDef(size=90), color="#9aa4ad")
                 .encode(x=alt.X("season:O", title="Season"),
                         y=alt.Y("rung:N", sort=RUNGS[::-1], scale=alt.Scale(domain=RUNGS[::-1]), title=None),
                         tooltip=["season", "series", "pos", "points", "wins"])
                 + alt.Chart(career).mark_point(size=120, filled=True)
                 .encode(x="season:O", y=alt.Y("rung:N", sort=RUNGS[::-1]),
                         color=alt.Color("series:N", title="Series"),
                         tooltip=["season", "series", "pos", "points", "wins"]))
        st.altair_chart(chart.properties(height=280), width="stretch")
        st.dataframe(career.sort_values(["season", "rung"])[["season", "series", "pos", "points", "wins", "podiums"]]
                     .rename(columns=str.capitalize), hide_index=True, width="stretch")
    st.caption("Estimates use public race results only. They are not a judgement of any driver: "
               "budget, age, team and backing shape careers too.")

# ---------------------------------------------------------------- the ladder
with tab_ladder:
    lab = drivers[drivers["labelled"]]
    st.subheader(f"F4 debutants {int(lab.debut_season.min())}–{int(lab.debut_season.max())}: who reached each rung")
    funnel = pd.DataFrame({"Rung": RUNGS,
                           "Drivers": [len(lab)] + [int(lab[f"reached_{t}"].sum())
                                                    for t in ["regional", "tier3", "f2", "f1"]]})
    funnel["Share"] = funnel["Drivers"] / max(1, len(lab))
    st.altair_chart(alt.Chart(funnel).mark_bar(color="#d7262e").encode(
        x=alt.X("Drivers:Q"), y=alt.Y("Rung:N", sort=RUNGS),
        tooltip=["Rung", "Drivers", alt.Tooltip("Share:Q", format=".1%")]).properties(height=220),
        width="stretch")

    st.subheader(f"Rookie finish vs reaching F3 level within {horizon} years")
    bins, names = [0, 1, 3, 5, 10, 999], ["Champion", "P2–P3", "P4–P5", "P6–P10", "P11+"]
    g = (lab.assign(Finish=pd.cut(lab["best_pos"], bins=bins, labels=names))
         .groupby("Finish", observed=False)[target].agg(["mean", "size"]).reset_index())
    g.columns = ["Finish", "Rate", "Drivers"]
    st.altair_chart(alt.Chart(g).mark_bar(color="#1f6fd1").encode(
        x=alt.X("Finish:N", sort=names), y=alt.Y("Rate:Q", axis=alt.Axis(format="%")),
        tooltip=["Finish", "Drivers", alt.Tooltip("Rate:Q", format=".0%")]).properties(height=260),
        width="stretch")

    if metrics.get("models"):
        st.subheader("Model performance on unseen cohorts")
        rows = [{"Model": k, "ROC-AUC": v["test"]["roc_auc"], "PR-AUC": v["test"]["pr_auc"],
                 "Precision @ top 10%": v["test"]["precision_at_top10pct"]} for k, v in metrics["models"].items()]
        st.dataframe(pd.DataFrame(rows).style.format({"ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}",
                                                      "Precision @ top 10%": "{:.0%}"}),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown(f"""
**Question.** Using only a driver's first Formula 4 season, how likely are they to reach FIA F3 level
(FIA F3, GP3 or FIA European F3) within {horizon} years?

**Features.** Rookie-season finishing percentile, points relative to the champion, win, podium, pole and
fastest-lap rates, finishing rate, how many F4 series they entered, estimated field strength of the series,
and age at debut.

**Validation.** Trained on earlier cohorts and tested on later ones (temporal split), so scores reflect how
the model would have done on drivers it had never seen. Probabilities shown for past drivers are
out-of-fold, so the model did not see that driver's outcome.

**Limitations.** Budget, team and junior-academy backing drive careers but are not in the data. Coverage of
Wikipedia varies by series and year.

**Source.** English Wikipedia season articles (CC BY-SA 4.0). Code:
[github.com/AbhishaWhaval/f4-career-analytics](https://github.com/AbhishaWhaval/f4-career-analytics)
""")
