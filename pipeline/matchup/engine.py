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

Everything that doesn't depend on the pitcher (the league pool per hand x side, its shape features,
outcome indicators, recency weights, batter totals) is built once in a Context and reused, so a backtest
over dozens of pitchers costs one similarity pass per pitch cluster.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
PRIOR_SCALE: dict[str, float] | float = 1.0   # multiplies prior strengths; per metric after backtest tuning

# metric: (numerator, denominator, prior strength in denominator units)
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
IDX = {c: i for i, c in enumerate(SUM_COLS)}
_HITS = {"1B": 1, "2B": 2, "3B": 3, "HR": 4}
SHAPE_REQUIRED = ["horz_break", "rel_side", "vaa_adj", "rel_speed", "induced_vert_break", "rel_height"]


def prior_strength(m: str) -> float:
    scale = PRIOR_SCALE.get(m, 1.0) if isinstance(PRIOR_SCALE, dict) else PRIOR_SCALE
    return METRICS[m][2] * scale


def indicators(df: pd.DataFrame, hard_hit: float) -> np.ndarray:
    oz = (df["in_zone"] == False).fillna(False).to_numpy()  # noqa: E712
    swing = df["is_swing"].to_numpy(bool)
    res = df["pa_result"].where(df["pa_ending"], None)
    hit = res.isin(list(_HITS)).to_numpy()
    on = hit | res.isin(["BB", "HBP"]).to_numpy()
    ab = res.isin(list(_HITS) + ["OUT", "K", "FC", "ROE"]).to_numpy()
    obp_den = ab | res.isin(["BB", "HBP", "SF"]).to_numpy()
    bipv = df["bip_ev_valid"].to_numpy(bool)
    cols = {
        "n": np.ones(len(df)), "sw": swing, "wh": df["is_whiff"].to_numpy(bool), "oz": oz, "ch": swing & oz,
        "cs": df["is_called_strike"].to_numpy(bool), "bip": bipv,
        "hard": bipv & (df["exit_speed"].fillna(0).to_numpy() >= hard_hit),
        "rv": df["rv"].fillna(0).to_numpy(float), "obp_num": on, "obp_den": obp_den,
        "tb": res.map(_HITS).fillna(0).to_numpy(float), "ab": ab,
    }
    return np.column_stack([np.asarray(cols[c], float) for c in SUM_COLS])


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


def _rates(S: np.ndarray, prior: dict | None = None) -> dict[str, np.ndarray]:
    """Rates from summed indicators S (rows x SUM_COLS). With prior: shrunk toward it."""
    S = np.atleast_2d(S)
    out = {}
    for m, (num, den, _) in METRICS.items():
        a, b = S[:, IDX[num]], S[:, IDX[den]]
        if prior is None:
            out[m] = np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)
        else:
            k = prior_strength(m)
            out[m] = (a + k * prior[m]) / (b + k)
    out["ops"] = out["obp"] + out["slg"]
    return out


def _clip_prior(p: dict) -> dict:
    p = dict(p)
    for m in ("whiff", "chase", "called_strike", "hard_hit", "obp"):
        p[m] = np.clip(p[m], 0.005, 0.995)
    p["slg"] = np.clip(p["slg"], 0.0, 4.0)
    return p


@dataclass
class SideData:
    batters: np.ndarray        # batter ids (index = code)
    code_of: dict              # batter id -> code
    counts: np.ndarray         # pitches seen vs this hand from this side, per batter
    skill: dict                # metric -> per-batter (overall shrunk rate - population rate)
    overall: dict              # metric -> per-batter overall shrunk rate vs this hand
    pool: pd.DataFrame         # tracked pitches with shape (batter-relative frame)
    X: np.ndarray
    codes: np.ndarray
    ind: np.ndarray
    r: np.ndarray
    two: np.ndarray


@dataclass
class Context:
    hist: pd.DataFrame
    lg: object
    as_of: date | None = None
    season: str | None = None
    _sides: dict = field(default_factory=dict)

    def __post_init__(self):
        h = self.hist
        self.as_of = self.as_of or max(h["game_date"])
        self.season = self.season or h.loc[h["game_date"] == self.as_of, "season"].iloc[0]
        rec = recency_weights(h["game_date"], h["season"], self.as_of, self.season)
        self.hist = h.assign(_r=rec / rec.max() if rec.max() > 0 else rec)

    def side(self, hand: str, side: str) -> SideData:
        key = (hand, side)
        if key in self._sides:
            return self._sides[key]
        h = self.hist[(self.hist["p_throws"] == hand) & (self.hist["b_side"] == side)]
        codes, batters = pd.factorize(h["batter_tm_id"])
        ind = indicators(h, self.lg.hard_hit_mph)
        r = h["_r"].to_numpy(float)
        nb = len(batters)
        S = np.column_stack([np.bincount(codes, weights=r * ind[:, j], minlength=nb) for j in range(ind.shape[1])])
        pop = _rates(S.sum(0))
        pop1 = {m: v[0] for m, v in pop.items()}
        overall = _rates(S, {m: np.full(nb, pop1[m]) for m in METRICS})
        skill = {m: overall[m] - pop1[m] for m in METRICS}
        ok = (h["pitch_tracked"].to_numpy(bool) & h[SHAPE_REQUIRED].notna().all(axis=1).to_numpy())
        pool = _frame(h[ok], side)
        sd = SideData(np.asarray(batters), {b: i for i, b in enumerate(batters)}, np.bincount(codes, minlength=nb),
                      skill, overall, pool, _X(pool, self.lg), codes[ok], ind[ok], r[ok],
                      h["is_two_strike"].to_numpy(bool)[ok])
        self._sides[key] = sd
        return sd


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
             as_of: date | None = None, season: str | None = None, ctx: Context | None = None) -> Result:
    """hist: calibrated regular-season pitches (with rv, vaa_adj). batter_ids: batters to report on."""
    ctx = ctx or Context(hist, lg, as_of, season)
    p_hand = arsenal.throws
    want = set(batter_ids)
    detail_rows, pop_rows, ref_xrv, xrv_of = [], [], [], {}
    mnames = list(METRICS) + ["ops"]
    for side in ("L", "R"):
        sd = ctx.side(p_hand, side)
        nb = len(sd.batters)
        cl_pitches = _frame(arsenal.pitches, side)
        wanted_codes = [sd.code_of[b] for b in sd.batters if b in want]
        xrv = {"all": np.zeros(nb), "2k": np.zeros(nb)}
        for split in ("all", "2k"):
            smask = np.ones(len(sd.codes), bool) if split == "all" else sd.two
            usage = arsenal.usage(side, two_strike=(split == "2k"))
            for c in arsenal.clusters:
                Xc = _X(cl_pitches[cl_pitches["cluster"] == c.cid], lg)
                w = kernel(sd.X, Xc) * sd.r * smask
                S = np.column_stack([np.bincount(sd.codes, weights=w * sd.ind[:, j], minlength=nb)
                                     for j in range(sd.ind.shape[1])])
                pop_r = {m: v[0] for m, v in _rates(S.sum(0)).items()}
                pop_rows.append({"side": side, "split": split, "cluster": c.cid, "label": c.label,
                                 "sim_pitches": float(S[:, 0].sum()), **pop_r})
                prior = _clip_prior({m: pop_r[m] + sd.skill[m] for m in METRICS})
                shrunk = _rates(S, prior)
                raw = _rates(S)
                xrv[split] += usage.get(c.cid, 0.0) * shrunk["rv"] * 100
                for i in wanted_codes:
                    row = {"batter_tm_id": sd.batters[i], "side": side, "split": split, "cluster": c.cid,
                           "label": c.label, "usage": usage.get(c.cid, 0.0), "sim_pitches": float(S[i, IDX["n"]]),
                           "sim_swings": float(S[i, IDX["sw"]]), "sim_bip": float(S[i, IDX["bip"]]),
                           "sim_pa": float(S[i, IDX["obp_den"]])}
                    row.update({m: float(shrunk[m][i]) for m in mnames})
                    row.update({f"raw_{m}": float(raw[m][i]) for m in mnames})
                    row.update({f"pop_{m}": float(pop_r[m]) for m in mnames})
                    row["base_rv"] = float(prior["rv"][i])  # expected rv vs this shape from overall skill alone
                    row.update({f"prior_{m}": float(prior[m][i]) for m in METRICS})
                    row.update({f"bat_{m}": float(sd.overall[m][i]) for m in METRICS})
                    detail_rows.append(row)
        qualified = sd.counts >= REFERENCE_MIN_PITCHES
        ref_xrv.append(xrv["all"][qualified])
        for i in wanted_codes:
            xrv_of[(sd.batters[i], side)] = (xrv["all"][i], xrv["2k"][i])

    detail = pd.DataFrame(detail_rows)
    population = pd.DataFrame(pop_rows)
    ref = np.concatenate(ref_xrv) if ref_xrv else np.array([])

    rows = []
    hist = ctx.hist
    for bid in batter_ids:
        side = batter_side_vs(hist, bid, p_hand)
        h = hist[hist["batter_tm_id"] == bid]
        info = {"batter_tm_id": bid, "batter_name": h["batter_name"].iloc[-1] if len(h) else None,
                "batter_team": h["batter_team"].iloc[-1] if len(h) else None, "side": side,
                "pitches_vs_hand": int((h["p_throws"] == p_hand).sum())}
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
                info["fit100" + suffix] = np.nan
                continue
            u = ds["usage"] / ds["usage"].sum()
            for m in ("whiff", "chase", "called_strike", "hard_hit", "ops", "rv"):
                info[m + suffix] = float((u * ds[m]).sum())
                info["pop_" + m + suffix] = float((u * ds["pop_" + m]).sum())
            info["sim_pitches" + suffix] = float((u * ds["sim_pitches"]).sum())
            # Pitch-shape fit: how she does vs pitches shaped like these, relative to what her overall level
            # vs this hand predicts. Separates 'good hitter' from 'good matchup'. Runs per 100 pitches.
            info["fit100" + suffix] = float((u * (ds["rv"] - ds["base_rv"])).sum() * 100)
        x = xrv_of.get((bid, side), (np.nan, np.nan))
        info["xrv100"], info["xrv100_2k"] = float(x[0]), float(x[1])
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
