"""League calibration: every constant the matchup engine needs, derived from the data.

  * run values (linear weights per PA result, count values, run value per pitch)
  * hard-hit exit-velo threshold (top quartile of tracked balls in play)
  * VAA-vs-height slope (removes location from approach angle) and feature scales
  * baseline rates by pitcher hand x batter side, all counts and two strikes
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import runvalues, shape

HARD_HIT_QUANTILE = 0.75


@dataclass
class League:
    linear_weights: dict[str, float]
    count_values: dict[str, float]
    rv_fit: dict
    hard_hit_mph: float
    vaa_slope: float
    scale: dict[str, dict[str, float]]
    baselines: dict[str, dict[str, float]] = field(default_factory=dict)
    n_pitches: int = 0
    n_games: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=float)


def rates(df: pd.DataFrame, hard_hit: float, w: np.ndarray | None = None) -> dict[str, float]:
    """Weighted outcome rates. w defaults to 1 per pitch."""
    w = np.ones(len(df)) if w is None else np.asarray(w, float)
    oz = (df["in_zone"] == False).fillna(False).to_numpy()  # noqa: E712
    swing = df["is_swing"].to_numpy()
    bipv = df["bip_ev_valid"].to_numpy()
    hard = bipv & (df["exit_speed"].fillna(0).to_numpy() >= hard_hit)
    s = lambda m: float((w * m).sum())  # noqa: E731
    n, sw, ozn, bip = s(np.ones(len(df))), s(swing), s(oz), s(bipv)
    return {
        "pitches": n,
        "whiff": s(df["is_whiff"].to_numpy()) / sw if sw else np.nan,
        "chase": s(swing & oz) / ozn if ozn else np.nan,
        "called_strike": s(df["is_called_strike"].to_numpy()) / n if n else np.nan,
        "hard_hit": s(hard) / bip if bip else np.nan,
        "rv100": 100 * s(df["rv"].fillna(0).to_numpy()) / n if n and "rv" in df else np.nan,
    }


def calibrate(df: pd.DataFrame) -> tuple[League, pd.DataFrame]:
    """Returns the League constants and the frame with rv and vaa_adj/rel_side_in added."""
    df = df.copy()
    rv, info = runvalues.compute(df)
    df["rv"] = rv.reindex(df.index).fillna(0.0)
    slope = shape.fit_vaa_slope(df)
    df = shape.add_shape(df, slope)
    hh = shape.hard_hit_threshold(df, HARD_HIT_QUANTILE) or 60.0
    lg = League(
        linear_weights=info["linear_weights"], count_values=info["count_values"], rv_fit=info["fit"],
        hard_hit_mph=hh, vaa_slope=slope, scale=shape.league_scale(df),
        n_pitches=len(df), n_games=int(df["game_uid"].nunique()),
    )
    for (ph, bs), g in df.groupby(["p_throws", "b_side"]):
        lg.baselines[f"{ph}HP vs {bs}HH"] = rates(g, hh)
        lg.baselines[f"{ph}HP vs {bs}HH, 2 strikes"] = rates(g[g["is_two_strike"]], hh)
    lg.baselines["all"] = rates(df, hh)
    return lg, df


def describe(lg: League) -> str:
    lines = [
        f"League: {lg.n_games:,} games, {lg.n_pitches:,} pitches",
        f"Run values ({lg.rv_fit.get('source')} fit on {lg.rv_fit.get('half_innings', 0):,} half-innings"
        + (f", R^2 {lg.rv_fit['r2']:.2f}" if "r2" in lg.rv_fit else "") + "), runs relative to an average PA:",
        "  " + "  ".join(f"{k} {v:+.2f}" for k, v in sorted(lg.linear_weights.items(), key=lambda kv: -kv[1])),
        "Count values (runs, batter's view):",
    ]
    for b in range(4):
        lines.append("  " + "  ".join(f"{b}-{s} {lg.count_values[f'{b}-{s}']:+.3f}" for s in range(3)))
    lines += [
        f"Hard-hit threshold: {lg.hard_hit_mph:.1f} mph exit velo (top {100 * (1 - HARD_HIT_QUANTILE):.0f}% of tracked balls in play)",
        f"VAA location slope: {lg.vaa_slope:.2f} deg per ft of plate height",
        "Baselines            pitches  whiff%  chase%  called-K%  hard-hit%  RV/100",
    ]
    for k, r in lg.baselines.items():
        lines.append(f"  {k:<26}{r['pitches']:>8,.0f}  {100 * r['whiff']:5.1f}   {100 * r['chase']:5.1f}   "
                     f"{100 * r['called_strike']:6.1f}     {100 * r['hard_hit']:6.1f}   {r['rv100']:+5.2f}")
    return "\n".join(lines)
