"""Estimate how competitive each F4 championship is.

Finishing P3 in Italian F4 is not the same as P3 in a six-car national series. Drivers
who race two or more F4 championships in the same year link the series together:

    finish_percentile[driver, series, year] = ability[driver, year] + difficulty[series] + noise

Solving this by least squares (a two-way fixed-effects model) gives each series a
difficulty score. Positive = the same drivers finish further down = a stronger field.
Only seasons in the training window are used, so the holdout years stay unseen.
"""
import numpy as np
import pandas as pd


def finish_percentile(df: pd.DataFrame) -> pd.Series:
    """0 = champion, 1 = last classified."""
    return (df["pos"] - 1) / (df["n_drivers"] - 1).clip(lower=1)


def estimate_strength(f4: pd.DataFrame, max_season: int, min_shared: int = 3) -> pd.DataFrame:
    df = f4[(f4["season"] <= max_season) & (f4["entries"] >= 2)].copy()
    df["pct"] = finish_percentile(df)
    df["dy"] = df["driver_id"] + "|" + df["season"].astype(str)
    multi = df[df.groupby("dy")["series"].transform("nunique") >= 2]

    all_series = sorted(f4["series"].unique())
    shared = multi.groupby("series")["dy"].nunique().reindex(all_series, fill_value=0)
    linked = [s for s in all_series if shared[s] >= min_shared]

    effects = pd.Series(0.0, index=all_series)
    if len(linked) >= 2:
        m = multi[multi["series"].isin(linked)]
        m = m[m.groupby("dy")["series"].transform("nunique") >= 2]
        dys = m["dy"].unique()
        dy_idx = {k: i for i, k in enumerate(dys)}
        s_idx = {k: i for i, k in enumerate(linked)}
        X = np.zeros((len(m) + 1, len(dys) + len(linked)))
        for r, (dy, s) in enumerate(zip(m["dy"], m["series"])):
            X[r, dy_idx[dy]] = 1
            X[r, len(dys) + s_idx[s]] = 1
        X[-1, len(dys):] = 1                       # constraint: series effects sum to zero
        y = np.append(m["pct"].to_numpy(), 0.0)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        effects.loc[linked] = coef[len(dys):]

    out = pd.DataFrame({"series": all_series, "strength": effects.values,
                        "shared_driver_seasons": shared.values})
    out["estimated"] = out["series"].isin(linked)
    return out.sort_values("strength", ascending=False).reset_index(drop=True)
