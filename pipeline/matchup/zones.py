"""5x5 location grids of OPS vs pitches shaped like a pitcher's (Gameday card, back page).

Grid (pitcher's view, the project's frame: positive PlateLocSide = toward the RHH box = pitcher's right):
  columns: < -0.71 | -0.71..-0.237 | -0.237..0.237 | 0.237..0.71 | > 0.71 ft
  rows:    > 3.0   | 3.0..2.5      | 2.5..2.0      | 2.0..1.5    | < 1.5 ft     (row 0 = top)
The inner 3x3 is the strike zone; the outer ring is the chase band. Cell index = row * 5 + col.

Each cell's OPS uses only plate appearances that ENDED on a pitch in that cell, weighted by how similar the
pitch was to the pitcher's shapes (same kernel, recency and batter-relative frame as engine.py), so:
  * "all": similarity to her arsenal, mixed by how often she throws each pitch to that batter side
  * "ch":  similarity to her changeup cluster(s) only (clusters named Changeup... or Change-type)

A hitter has only a handful of PAs per cell, so each cell is shrunk toward a prior: how all same-side hitters
did in that cell vs those shapes, shifted by the hitter's overall OBP/SLG skill vs that pitcher hand (the same
prior design as every other number). `pa` keeps the hitter's own weighted PAs per cell so the card can show
where her own results carry weight. OPS by location has NOT been backtested.
"""
from __future__ import annotations

import re

import numpy as np

from . import engine
from .engine import IDX, Context, _frame, _X, kernel

X_EDGES = np.array([-0.71, -0.71 / 3, 0.71 / 3, 0.71])
Y_EDGES = np.array([1.5, 2.0, 2.5, 3.0])
CHANGEUP = re.compile(r"^change", re.I)
_COLS = [IDX[c] for c in ("obp_num", "obp_den", "tb", "ab")]


def cells(side_ft: np.ndarray, height_ft: np.ndarray) -> np.ndarray:
    """Cell index 0..24 (row 0 = top, col 0 = pitcher's left), -1 where location is missing."""
    col = np.digitize(side_ft, X_EDGES)            # 0..4 left -> right
    row = 4 - np.digitize(height_ft, Y_EDGES)      # 0 = highest
    out = row * 5 + col
    out[~(np.isfinite(side_ft) & np.isfinite(height_ft))] = -1
    return out


def _side_cells(sd) -> np.ndarray:
    if getattr(sd, "_cells", None) is None:
        sd._cells = cells(sd.pool["plate_loc_side"].to_numpy(float), sd.pool["plate_loc_height"].to_numpy(float))
    return sd._cells


def _grid(sd, w: np.ndarray, want: np.ndarray):
    """Shrunk OPS per (wanted batter, cell) and the batters' own weighted PAs per cell."""
    c = _side_cells(sd)
    ind = sd.ind[:, _COLS]
    ok = (c >= 0) & ((ind[:, 1] > 0) | (ind[:, 3] > 0)) & (w > 0)
    nb = len(sd.batters)
    key = sd.codes[ok] * 25 + c[ok]
    S = np.stack([np.bincount(key, weights=w[ok] * ind[ok, j], minlength=nb * 25) for j in range(4)], axis=-1)
    S = S.reshape(nb, 25, 4)
    pop = S.sum(0)                                               # (25, 4)
    with np.errstate(invalid="ignore", divide="ignore"):
        pop_obp = np.where(pop[:, 1] > 0, pop[:, 0] / pop[:, 1], np.nan)
        pop_slg = np.where(pop[:, 3] > 0, pop[:, 2] / pop[:, 3], np.nan)
    # cells nobody reached: fall back to the overall population rate for these shapes
    tot = pop.sum(0)
    pop_obp = np.where(np.isfinite(pop_obp), pop_obp, tot[0] / tot[1] if tot[1] else np.nan)
    pop_slg = np.where(np.isfinite(pop_slg), pop_slg, tot[2] / tot[3] if tot[3] else np.nan)
    sub = S[want]                                                # (k, 25, 4)
    p_obp = np.clip(pop_obp[None, :] + sd.skill["obp"][want][:, None], 0.005, 0.995)
    p_slg = np.clip(pop_slg[None, :] + sd.skill["slg"][want][:, None], 0.0, 4.0)
    k_obp, k_slg = engine.prior_strength("obp"), engine.prior_strength("slg")
    obp = (sub[..., 0] + k_obp * p_obp) / (sub[..., 1] + k_obp)
    slg = (sub[..., 2] + k_slg * p_slg) / (sub[..., 3] + k_slg)
    return obp + slg, sub[..., 1]


def zone_ops(ctx: Context, arsenal, lg, batter_sides: dict[str, str | None]) -> dict[str, dict]:
    """batter id -> {"all": [25], "pa": [25], "ch": [25] | None, "pa_ch": [25] | None} for the side she bats
    against this pitcher's hand (from the engine's summary)."""
    out: dict[str, dict] = {}
    hand = arsenal.throws
    ch_ids = [c.cid for c in arsenal.clusters if CHANGEUP.match(c.label or "")]
    for side in ("L", "R"):
        bats = [b for b, s in batter_sides.items() if s == side]
        sd = ctx.side(hand, side)
        bats = [b for b in bats if b in sd.code_of]
        if not bats:
            continue
        want = np.array([sd.code_of[b] for b in bats])
        cl = _frame(arsenal.pitches, side)
        usage = arsenal.usage(side)
        w_all = np.zeros(len(sd.codes))
        w_ch = np.zeros(len(sd.codes))
        u_ch = sum(usage.get(c, 0.0) for c in ch_ids)
        for c in arsenal.clusters:
            u = usage.get(c.cid, 0.0)
            if u <= 0 and c.cid not in ch_ids:
                continue
            k = kernel(sd.X, _X(cl[cl["cluster"] == c.cid], lg)) * sd.r
            w_all += u * k
            if c.cid in ch_ids:
                w_ch += (u / u_ch if u_ch > 0 else 1.0 / len(ch_ids)) * k
        ops_all, pa_all = _grid(sd, w_all, want)
        ops_ch, pa_ch = _grid(sd, w_ch, want) if ch_ids else (None, None)
        for i, b in enumerate(bats):
            out[b] = {"all": ops_all[i], "pa": pa_all[i],
                      "ch": None if ops_ch is None else ops_ch[i], "pa_ch": None if pa_ch is None else pa_ch[i]}
    return out
