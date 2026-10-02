"""Run values.

1. Linear weights per PA result, estimated by regressing runs scored in each half-inning on the counts
   of each result type in it. Trackman has no base/runner state, so the classic RE24 table isn't
   available; half-inning regression gets the same weights from data we do have. With too little
   data (tests, early setup) baseball-style defaults are used and flagged.
2. Weights are centered so the average plate appearance is worth 0.
3. Count values: V(balls, strikes) = average final-PA weight of every PA that passed through that count.
4. Pitch run value (batter perspective, + = good for the batter):
     PA-ending pitch: weight(result) - V(count before)
     otherwise:       V(count after) - V(count before)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EVENTS = ["1B", "2B", "3B", "HR", "BB", "HBP", "ROE", "FC", "SF", "SH", "K", "OUT"]

DEFAULT_WEIGHTS = {
    "1B": 0.47, "2B": 0.78, "3B": 1.06, "HR": 1.40, "BB": 0.31, "HBP": 0.33, "ROE": 0.48,
    "FC": -0.20, "SF": -0.05, "SH": -0.12, "K": -0.28, "OUT": -0.26,
}
MIN_HALF_INNINGS = 300
HALF_INNING = ["game_uid", "inning", "top_bottom"]
PA_KEY = ["game_uid", "inning", "top_bottom", "pa_of_inning"]


def fit_linear_weights(p: pd.DataFrame, ridge: float = 1.0) -> tuple[dict[str, float], dict]:
    ends = p[p["pa_ending"]]
    counts = ends.pivot_table(index=HALF_INNING, columns="pa_result", values="pitch_uid", aggfunc="count", fill_value=0)
    counts = counts.reindex(columns=EVENTS, fill_value=0)
    runs = p.groupby(HALF_INNING)["runs_scored"].sum().reindex(counts.index).fillna(0)
    n = len(counts)
    info = {"half_innings": n, "source": "regression"}
    if n < MIN_HALF_INNINGS:
        info["source"] = "default"
        return dict(DEFAULT_WEIGHTS), info
    X = counts.to_numpy(float)
    y = runs.to_numpy(float)
    # Ridge toward the defaults, so rare events (3B, HBP, SH) don't swing wildly.
    prior = np.array([DEFAULT_WEIGHTS[e] for e in EVENTS])
    A = X.T @ X + ridge * np.eye(len(EVENTS))
    b = X.T @ y + ridge * prior
    beta = np.linalg.solve(A, b)
    info["r2"] = float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return dict(zip(EVENTS, map(float, beta))), info


def center_weights(weights: dict[str, float], p: pd.DataFrame) -> dict[str, float]:
    freq = p.loc[p["pa_ending"], "pa_result"].value_counts()
    if freq.empty:
        return weights
    mean = sum(weights.get(e, 0.0) * c for e, c in freq.items()) / freq.sum()
    return {e: w - mean for e, w in weights.items()}


def count_values(p: pd.DataFrame, weights: dict[str, float]) -> dict[str, float]:
    final = p[p["pa_ending"]].set_index(PA_KEY)["pa_result"].map(weights)
    final = final[~final.index.duplicated()]
    seen = p[PA_KEY + ["balls", "strikes"]].dropna().drop_duplicates()
    seen = seen[(seen.balls <= 3) & (seen.strikes <= 2)]
    seen = seen.join(final.rename("w"), on=PA_KEY).dropna(subset=["w"])
    cv = seen.groupby(["balls", "strikes"])["w"].mean()
    out = {}
    for b in range(4):
        for s in range(3):
            out[f"{b}-{s}"] = float(cv.get((b, s), np.nan))
    return out


def pitch_run_values(p: pd.DataFrame, weights: dict[str, float], cvals: dict[str, float]) -> pd.Series:
    p = p.sort_values(PA_KEY + ["pitch_no"])
    key = p["balls"].astype("Int64").astype(str) + "-" + p["strikes"].astype("Int64").astype(str)
    before = key.map(cvals)
    nxt = key.groupby([p[k] for k in PA_KEY]).shift(-1).map(cvals)
    end_val = p["pa_result"].map(weights)
    rv = np.where(p["pa_ending"], end_val - before, nxt - before)
    return pd.Series(rv, index=p.index).reindex(p.index)


def compute(p: pd.DataFrame) -> tuple[pd.Series, dict]:
    raw, info = fit_linear_weights(p)
    weights = center_weights(raw, p)
    cvals = count_values(p, weights)
    # Fill counts never observed (tiny datasets) from the centered defaults' neutral value.
    cvals = {k: (0.0 if np.isnan(v) else v) for k, v in cvals.items()}
    rv = pitch_run_values(p, weights, cvals)
    return rv, {"linear_weights": weights, "count_values": cvals, "fit": info}
