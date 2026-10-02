"""Handedness-aware derived features and outcome flags.

Raw horizontal values are pitcher's view, positive = toward the RHH box. Two re-orientations:
  * arm-relative  (sign by pitcher hand): + = arm side. Used to compare arsenals across RHP/LHP.
  * batter-relative (sign by batter side): + = toward the batter / inside. Used for batter history lookups.
A RHH stands on the positive side, so for a RHH positive already means "toward the batter"; for a LHH flip it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _sign(hand: pd.Series) -> pd.Series:
    return np.where(hand == "R", 1.0, -1.0)


def pa_result(row) -> str | None:
    if row.kor_bb == "Strikeout":
        return "K"
    if row.kor_bb == "Walk":
        return "BB"
    if row.pitch_call == "HitByPitch":
        return "HBP"
    pr = row.play_result
    if pr and pr != "Undefined":
        if pr == "Sacrifice":
            return "SH" if row.tagged_hit_type == "Bunt" else "SF"
        return C.PLAY_RESULT_MAP.get(pr, "OUT")
    return None


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    p_sign = _sign(df["p_throws"])
    b_sign = _sign(df["b_side"])

    df["hb_arm"] = df["horz_break"] * p_sign
    df["rel_side_arm"] = df["rel_side"] * p_sign
    df["hb_in"] = df["horz_break"] * b_sign
    df["loc_in"] = df["plate_loc_side"] * b_sign
    df["haa_in"] = df["horz_appr_angle"] * b_sign
    df["same_side"] = df["p_throws"] == df["b_side"]

    has_loc = df["plate_loc_side"].notna() & df["plate_loc_height"].notna()
    in_zone = (
        (df["plate_loc_side"].abs() <= C.ZONE_HALF_WIDTH)
        & df["plate_loc_height"].between(C.ZONE_BOTTOM, C.ZONE_TOP)
    )
    df["in_zone"] = in_zone.where(has_loc, other=pd.NA).astype("boolean")

    call = df["pitch_call"]
    df["is_swing"] = call.isin(C.SWING_CALLS)
    df["is_whiff"] = call.isin(C.WHIFF_CALLS)
    df["is_called_strike"] = call.isin(C.CALLED_STRIKE_CALLS)
    df["is_bip"] = call == "InPlay"
    df["is_two_strike"] = df["strikes"].fillna(0) >= 2
    df["bip_ev_valid"] = df["is_bip"] & df["exit_speed"].notna() & df["hit_launch_conf"].isin(C.GOOD_LAUNCH_CONF)

    df["pa_result"] = df.apply(pa_result, axis=1)
    df["pa_ending"] = df["pa_result"].notna()
    return df
