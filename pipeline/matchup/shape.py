"""Pitch shape features shared by arsenal clustering and batter similarity.

Vertical approach angle depends heavily on where the pitch crosses the plate (a low pitch arrives
steeper), so raw VAA mixes shape with location. vaa_adj removes the location part with a league-wide
linear fit on plate height, leaving the part that belongs to the pitch itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ZONE_MID_HEIGHT = 2.25

# Arm-relative: describes a pitcher's arsenal independent of which side the batter stands on.
CLUSTER_FEATURES = ["rel_speed", "induced_vert_break", "hb_arm", "vaa_adj"]
# Batter-relative: what the batter actually sees. Location is deliberately excluded (shape only).
SIMILARITY_FEATURES = ["rel_speed", "induced_vert_break", "hb_in", "rel_height", "rel_side_in", "vaa_adj"]
# Release position differs a lot between pitchers and mostly matters through the approach angle it creates
# (already a feature), so it gets a small weight; at 0.5 it excluded nearly every other pitcher's pitches.
SIMILARITY_WEIGHTS = {"rel_speed": 1.0, "induced_vert_break": 1.0, "hb_in": 1.0,
                      "rel_height": 0.1, "rel_side_in": 0.1, "vaa_adj": 0.75}


def fit_vaa_slope(p: pd.DataFrame) -> float:
    t = p[p["pitch_tracked"]]
    if len(t) < 20:
        return 0.0
    x = t["plate_loc_height"].to_numpy(float)
    y = t["vert_appr_angle"].to_numpy(float)
    return float(np.polyfit(x, y, 1)[0])


def add_shape(p: pd.DataFrame, vaa_slope: float) -> pd.DataFrame:
    p = p.copy()
    p["vaa_adj"] = p["vert_appr_angle"] - vaa_slope * (p["plate_loc_height"] - ZONE_MID_HEIGHT)
    b_sign = np.where(p["b_side"] == "R", 1.0, -1.0)
    p["rel_side_in"] = p["rel_side"] * b_sign
    return p


def league_scale(p: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Mean/std per feature over tracked pitches. Horizontal features are standardized with
    mean 0 because their sign carries meaning (arm side / toward the batter)."""
    t = p[p["pitch_tracked"]]
    cols = sorted(set(CLUSTER_FEATURES + SIMILARITY_FEATURES))
    out = {}
    for c in cols:
        sd = float(t[c].std()) if len(t) > 1 else 1.0
        out[c] = {"mean": 0.0 if c in ("hb_arm", "hb_in", "rel_side_in") else float(t[c].mean()),
                  "std": sd if sd and sd == sd and sd > 1e-6 else 1.0}
    return out


def standardize(df: pd.DataFrame, cols: list[str], scale: dict) -> np.ndarray:
    return np.column_stack([(df[c].to_numpy(float) - scale[c]["mean"]) / scale[c]["std"] for c in cols])


def hard_hit_threshold(p: pd.DataFrame, quantile: float = 0.75) -> float | None:
    ev = p.loc[p["bip_ev_valid"], "exit_speed"].dropna()
    return float(ev.quantile(quantile)) if len(ev) >= 20 else None
