"""Matchup engine: how each batter has done against pitches shaped like this pitcher's.

For each of the pitcher's movement clusters and each batter side:
  1. Express the cluster in the batter's frame (break and release side relative to the batter).
  2. Weight every league pitch thrown by a same-handed pitcher to a same-side batter by how similar it
     is (Mahalanobis distance to the cluster on standardized shape features) x how recent it is.
  3. Sum outcomes per batter -> raw rates against 'pitches like this'.
  4. Shrink toward a prior = how all same-side batters did against this shape, shifted by the batter's own
     overall skill vs that pitcher hand. Small samples stay near the prior; big samples speak for themselves.
Then combine clusters by how often the pitcher throws each one to that side (and in two-strike counts).

Overall score = expected run value per 100 pitches vs this arsenal, as a percentile among all qualified
D1 batters: 50 = average hitter, higher = better for the pitcher.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from . import shape
from .arsenal import Arsenal
from .recency import recency_weights

BANDWIDTH = 0.3          # added to the cluster covariance (standardized units): how far 'similar' reaches.
                         # Chosen on a synthetic league with realistic pitcher-to-pitcher variation (purity vs
                         # sample size); the backtest tunes it on real data.
MIN_WEIGHT = 0.01        # pitches less typical than this (chi-square tail) get no weight
REFERENCE_MIN_PITCHES = 150

# metric: (numerator, denominator, prior strength in denominator units, scale for display)
METRICS = {
    "whiff": ("wh", "sw", 50.0),
    "chase": ("ch", "oz", 80.0),
    "called_strike": ("cs", "n", 120.0),
    "hard_hit": ("hard", "bip", 40.0),
    "rv": ("rv", "n", 200.0),
    "obp": ("obp_num", "obp_den", 40.0),
    "slg": ("tb", "ab", 40.0),
}
SUM_COLS = ["n", "sw", "wh", "oz", "ch", "cs", "bip", "hard", "rv", "obp_num", "obp_den", "tb", "ab"]
_HITS = {"1B": 1, "2B": 2, "3B": 3, "HR": 4}


def indicators(df: pd.DataFrame, hard_hit: float) -> pd.DataFrame:
    oz = (df["in_zone"] == False).fillna(False).to_numpy()  # noqa: E712
    swing = df["is_swing"].to_numpy()
    res = df["pa_result"].where(df["pa_ending"], None)
    hit = res.isin(list(_HITS)).to_numpy()
    on = (hit | res.isin(["BB", "HBP"]).to_numpy())
    ab = res.isin(list(_HITS) + ["OUT", "K", "FC", "ROE"]).to_numpy()
    obp_den = ab | res.isin(["BB", "HBP", "SF"]).to_numpy()
    bipv = df["bip_ev_valid"].to_numpy()
    return pd.DataFrame({
        "n": 1.0, "sw": swing, "wh": df["is_whiff"].to_numpy(), "oz": oz, "ch": swing & oz,
        "cs": df["is_called_strike"].to_numpy(), "bip": bipv,
        "hard": bipv & (df["exit_speed"].fillna(0).to_numpy() >= hard_hit),
        "rv": df["rv"].fillna(0).to_numpy(), "obp_num": on, "obp_den": obp_den,
        "tb": res.map(_HITS).fillna(0).to_numpy(), "ab": ab,
    }, index=df.index).astype(float)


def _frame(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Batter-relative features for a batter standing on `side` (whatever side is recorded on the row)."""
    s = 1.0 if side == "R" else -1.0
    out = df.copy()
    out["hb_in"] = out["horz_break"] * s
    out["rel_side_in"] = out["rel_side"] * s
    return out


def _X(df: pd.DataFrame, lg) -> np.ndarray:
    X = shape.standardize(df, shape.SIMILARITY_FEATURES, lg.scale)
    return X * np.sqrt([shape.SIMILARITY_WEIGHTS[f] for f in shape.SIMILARITY_FEATURES])


def kernel(Xpool: np.ndarray, Xc: np.ndarray) -> np.ndarray:
    """Similarity of each pool pitch to a cluster: the chi-square tail probability of its Mahalanobis
    distance, i.e. 'how typical would this pitch be of the cluster'. A pitch from the cluster itself scores
    0.5 on average regardless of how many features there are; an outlier scores ~0."""
    from scipy.stats import chi2

    mu = Xc.mean(axis=0)
    cov = np.cov(Xc, rowvar=False) if len(Xc) > 2 else np.eye(Xc.shape[1])
    cov = cov + BANDWIDTH * np.eye(Xc.shape[1])
    inv = np.linalg.inv(cov)
    d = Xpool - mu
    d2 = np.einsum("ij,jk,ik->i", d, inv, d)
    w = chi2.sf(d2, df=Xc.shape[1])
    w[(w < MIN_WEIGHT) | ~np.isfinite(d2)] = 0.0
    return w


def _rates(sums: pd.DataFrame, prior: pd.DataFrame | None = None) -> pd.DataFrame:
    """Shrunk rates. prior: same index as sums, one column per metric (None -> raw rates)."""
    out = {}
    for m, (num, den, k) in METRICS.items():
        if prior is None:
            out[m] = sums[num] / sums[den].replace(0, np.nan)
        else:
            out[m] = (sums[num] + k * prior[m]) / (sums[den] + k)
    r = pd.DataFrame(out, index=sums.index)
    r["ops"] = r["obp"] + r["slg"]
    return r


def _clip_prior(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    for m in ("whiff", "chase", "called_strike", "hard_hit", "obp"):
        p[m] = p[m].clip(0.005, 0.995)
    p["slg"] = p["slg"].clip(0.0, 4.0)
    return p


def batter_side_vs(hist: pd.DataFrame, batter_id: str, p_hand: str) -> str | None:
    h = hist[hist["batter_tm_id"] == batter_id]
    v = h[h["p_throws"] == p_hand]["b_side"]
    v = v if len(v) else h["b_side"]
    return v.mode().iloc[0] if len(v) else None


@dataclass
class Result:
    arsenal: Arsenal
    batters: pd.DataFrame      # one row per batter, usage-weighted across clusters
    detail: pd.DataFrame       # batter x cluster x split
    population: pd.DataFrame   # all same-side batters vs each cluster (the priors' base)


def evaluate(hist: pd.DataFrame, arsenal: Arsenal, lg, batter_ids: list[str],
             as_of: date | None = None, season: str | None = None) -> Result:
    """hist: calibrated regular-season pitches (with rv, vaa_adj). batter_ids: batters to report on."""
    p_hand = arsenal.throws
    as_of = as_of or max(hist["game_date"])
    season = season or hist.loc[hist["game_date"] == as_of, "season"].iloc[0]
    rec = recency_weights(hist["game_date"], hist["season"], as_of, season)
    rec = rec / rec.max() if rec.max() > 0 else rec
    hist = hist.assign(_r=rec)
    vs_hand = hist[hist["p_throws"] == p_hand]
    ind_all = indicators(vs_hand, lg.hard_hit_mph)

    detail_rows, pop_rows, xrv_ref = [], [], {}
    for side in ("L", "R"):
        side_mask = vs_hand["b_side"] == side
        overall_sums = (ind_all[side_mask].mul(vs_hand.loc[side_mask, "_r"], axis=0)
                        .groupby(vs_hand.loc[side_mask, "batter_tm_id"]).sum())
        pop_overall = _rates(pd.DataFrame([overall_sums.sum()], index=["pop"]))
        batter_overall = _rates(overall_sums, prior=pd.DataFrame(
            np.repeat(pop_overall.values, len(overall_sums), axis=0), index=overall_sums.index, columns=pop_overall.columns))
        skill = batter_overall - pop_overall.iloc[0]

        pool = _frame(vs_hand[side_mask & vs_hand["pitch_tracked"]].dropna(subset=["horz_break", "rel_side", "vaa_adj",
                                                                                   "rel_speed", "induced_vert_break",
                                                                                   "rel_height"]), side)
        Xpool = _X(pool, lg)
        ind = ind_all.loc[pool.index]
        cl_pitches = _frame(arsenal.pitches, side)
        for split in ("all", "2k"):
            smask = np.ones(len(pool), bool) if split == "all" else pool["is_two_strike"].to_numpy()
            usage = arsenal.usage(side, two_strike=(split == "2k"))
            for c in arsenal.clusters:
                Xc = _X(cl_pitches[cl_pitches["cluster"] == c.cid], lg)
                w = kernel(Xpool, Xc) * pool["_r"].to_numpy() * smask
                sums = ind.mul(w, axis=0).groupby(pool["batter_tm_id"]).sum()
                pop_s = sums.sum()
                pop_r = _rates(pd.DataFrame([pop_s], index=["pop"])).iloc[0]
                pop_rows.append({"side": side, "split": split, "cluster": c.cid, "label": c.label,
                                 "sim_pitches": pop_s["n"], **pop_r.to_dict()})
                sk = skill.reindex(sums.index).fillna(0.0)
                prior = _clip_prior(pd.DataFrame({m: pop_r[m] + sk[m] for m in METRICS}, index=sums.index))
                shrunk = _rates(sums, prior)
                raw = _rates(sums)
                for bid in sums.index:
                    xrv_ref.setdefault((bid, side, split), 0.0)
                    xrv_ref[(bid, side, split)] += usage.get(c.cid, 0.0) * shrunk.at[bid, "rv"] * 100
                want = [b for b in batter_ids if b in sums.index]
                for bid in want:
                    row = {"batter_tm_id": bid, "side": side, "split": split, "cluster": c.cid, "label": c.label,
                           "usage": usage.get(c.cid, 0.0), "sim_pitches": float(sums.at[bid, "n"]),
                           "sim_swings": float(sums.at[bid, "sw"]), "sim_bip": float(sums.at[bid, "bip"]),
                           "sim_pa": float(sums.at[bid, "obp_den"])}
                    row.update({m: float(shrunk.at[bid, m]) for m in list(METRICS) + ["ops"]})
                    row.update({f"raw_{m}": float(raw.at[bid, m]) for m in list(METRICS) + ["ops"]})
                    row.update({f"pop_{m}": float(pop_r[m]) for m in list(METRICS) + ["ops"]})
                    detail_rows.append(row)

    detail = pd.DataFrame(detail_rows)
    population = pd.DataFrame(pop_rows)

    # Reference distribution for the percentile score: every batter with enough pitches vs this hand.
    counts = vs_hand.groupby(["batter_tm_id", "b_side"]).size()
    ref = pd.Series({k: v for k, v in xrv_ref.items() if k[2] == "all" and counts.get((k[0], k[1]), 0) >= REFERENCE_MIN_PITCHES})

    rows = []
    for bid in batter_ids:
        side = batter_side_vs(hist, bid, p_hand)
        h = hist[hist["batter_tm_id"] == bid]
        info = {"batter_tm_id": bid, "batter_name": h["batter_name"].iloc[-1] if len(h) else None,
                "batter_team": h["batter_team"].iloc[-1] if len(h) else None, "side": side,
                "pitches_vs_hand": int(((h["p_throws"] == p_hand)).sum())}
        direct = h[(h["pitcher_tm_id"] == arsenal.pitcher_id) & h["pa_ending"]]
        info["direct_pa"] = len(direct)
        vc = direct["pa_result"].value_counts()
        info["direct_line"] = (f"{int(vc[vc.index.isin(list(_HITS))].sum())} H, {int(vc.get('K', 0))} K, "
                               f"{int(vc.get('BB', 0) + vc.get('HBP', 0))} BB/HBP") if len(direct) else ""
        d = detail[(detail["batter_tm_id"] == bid) & (detail["side"] == side)] if side and len(detail) else pd.DataFrame()
        for split, suffix in (("all", ""), ("2k", "_2k")):
            ds = d[d["split"] == split] if len(d) else d
            if not len(ds) or ds["usage"].sum() == 0:
                for m in ("whiff", "chase", "called_strike", "hard_hit", "ops", "rv"):
                    info[m + suffix] = info["pop_" + m + suffix] = np.nan
                info["sim_pitches" + suffix] = 0.0
                continue
            u = ds["usage"] / ds["usage"].sum()
            for m in ("whiff", "chase", "called_strike", "hard_hit", "ops", "rv"):
                info[m + suffix] = float((u * ds[m]).sum())
                info["pop_" + m + suffix] = float((u * ds["pop_" + m]).sum())
            info["sim_pitches" + suffix] = float((u * ds["sim_pitches"]).sum())
        info["xrv100"] = xrv_ref.get((bid, side, "all"), np.nan)
        info["xrv100_2k"] = xrv_ref.get((bid, side, "2k"), np.nan)
        if len(ref) and np.isfinite(info["xrv100"]):
            # share of qualified hitters who would do better against this arsenal: 50 = average hitter,
            # higher = better matchup for the pitcher
            info["score"] = float(100 * (ref > info["xrv100"]).mean())
        else:
            info["score"] = np.nan
        n = info["sim_pitches"]
        info["confidence"] = "High" if n >= 100 else "Medium" if n >= 30 else "Low"
        rows.append(info)
    return Result(arsenal, pd.DataFrame(rows), detail, population)
