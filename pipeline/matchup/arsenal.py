"""A pitcher's arsenal as movement clusters.

Clusters are fitted on arm-relative shape (velo, induced vertical break, arm-side break, location-adjusted
VAA), so a LHP and a RHP with mirror-image pitches get the same clusters. Pitch-type tags are only used
to name a cluster when they agree with each other; otherwise the name comes from the movement.
Usage is recency-weighted and split by batter side and by two-strike counts, because pitchers use their
arsenal differently against LHH and RHH and when ahead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from . import shape

MIN_PITCHES = 60
MAX_CLUSTERS = 6
MIN_SHARE = 0.04
IGNORE_TAGS = {None, "", "Undefined", "Other", "Fastball", "Offspeed"}


@dataclass
class Cluster:
    cid: int
    label: str
    n: int
    share: float
    usage: dict[str, float]          # 'L', 'R', 'L2K', 'R2K' -> share of pitches to that batter side (count)
    means: dict[str, float]          # display metrics
    tag_mix: dict[str, float]


@dataclass
class Arsenal:
    pitcher_id: str
    pitcher_name: str
    team: str | None
    throws: str
    n_pitches: int
    clusters: list[Cluster]
    pitches: pd.DataFrame = field(repr=False)   # tracked pitches with a 'cluster' column
    model: object = field(default=None, repr=False)   # (GaussianMixture, remap) to assign new pitches
    scale: dict = field(default=None, repr=False)

    def assign(self, df: pd.DataFrame) -> pd.Series:
        """Cluster for new pitches by this pitcher (e.g. later games). NaN where shape is missing."""
        gm, remap = self.model
        ok = df[shape.CLUSTER_FEATURES].notna().all(axis=1) & df["pitch_tracked"]
        out = pd.Series(np.nan, index=df.index)
        if ok.any():
            X = shape.standardize(df[ok], shape.CLUSTER_FEATURES, self.scale)
            out[ok] = pd.Series(gm.predict(X), index=df.index[ok]).map(remap).astype(float)
        return out

    def usage(self, side: str, two_strike: bool = False) -> dict[int, float]:
        key = side + ("2K" if two_strike else "")
        return {c.cid: c.usage.get(key, 0.0) for c in self.clusters}


def _label(m: dict, max_velo: float, tags: dict[str, float]) -> str:
    top = max(tags.items(), key=lambda kv: kv[1]) if tags else (None, 0)
    if top[0] and top[1] >= 0.5:
        return str(top[0])
    if m["induced_vert_break"] >= 3.5:
        return "Rise-type"
    if m["rel_speed"] <= max_velo - 6:
        return "Change-type"
    if m["hb_arm"] <= -4:
        return "Curve-type (glove side)"
    if m["induced_vert_break"] <= -2:
        return "Drop-type"
    if m["hb_arm"] >= 4:
        return "Screw-type (arm side)"
    return "Fastball-type"


_DESCRIBE = [  # feature, word when this cluster has more of it, word when less
    ("hb_arm", "more arm-side run", "more glove-side"),
    ("induced_vert_break", "more rise", "more drop"),
    ("rel_speed", "harder", "softer"),
    ("vaa_adj", "flatter", "steeper"),
]


def _disambiguate(clusters: list, lg) -> None:
    """Two clusters with the same name (e.g. two riseballs) get the feature that separates them most:
    'Riseball (more arm-side run)' / 'Riseball (more glove-side)'."""
    by_label: dict[str, list] = {}
    for c in clusters:
        by_label.setdefault(c.label, []).append(c)
    for label, group in by_label.items():
        if len(group) < 2:
            continue
        def spread(f):
            vals = [c.means[f] for c in group]
            sd = lg.scale.get(f, {}).get("std", 1.0) if f != "hb_arm" else lg.scale.get("hb_arm", {}).get("std", 1.0)
            return (max(vals) - min(vals)) / (sd or 1.0)
        feat, more, less = max(_DESCRIBE, key=lambda d: spread(d[0]))
        vals = [c.means[feat] for c in group]
        mid = np.mean(vals)
        # Sign-aware words: two arm-side pitches are 'more run' vs 'straighter', two glove-side pitches
        # 'straighter' vs 'more glove-side break'; same idea for rise vs drop.
        if feat == "hb_arm":
            more, less = (("more arm-side run", "straighter") if min(vals) >= -1 else
                          ("straighter", "more glove-side break") if max(vals) <= 1 else
                          ("arm-side run", "glove-side break"))
        if feat == "induced_vert_break":
            more, less = (("more rise", "less rise") if min(vals) >= 0 else
                          ("less drop", "more drop") if max(vals) <= 0 else ("rise", "drop"))
        for c in group:
            word = more if c.means[feat] >= mid else less
            c.label = f"{label} ({word})"
        names = [c.label for c in group]
        for i, c in enumerate(group):  # 3+ of a kind can still collide
            if names.count(c.label) > 1:
                c.label = f"{c.label} {i + 1}"


def fit_arsenal(pitches: pd.DataFrame, lg, weights: np.ndarray | None = None, seed: int = 0) -> Arsenal:
    """pitches: every pitch by one pitcher (any game type); lg: calibrate.League."""
    if pitches["pitcher_tm_id"].nunique() != 1:
        raise ValueError("fit_arsenal expects one pitcher")
    pitches = pitches.copy()
    pitches["w"] = 1.0 if weights is None else weights
    t = pitches[pitches["pitch_tracked"]].dropna(subset=shape.CLUSTER_FEATURES)
    if len(t) < MIN_PITCHES:
        raise ValueError(f"only {len(t)} tracked pitches; need {MIN_PITCHES} to describe an arsenal")
    X = shape.standardize(t, shape.CLUSTER_FEATURES, lg.scale)

    best, best_bic = None, np.inf
    for k in range(1, min(MAX_CLUSTERS, len(t) // 25) + 1):
        gm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=seed, reg_covar=1e-3).fit(X)
        labels = gm.predict(X)
        if k > 1 and np.bincount(labels, minlength=k).min() < MIN_SHARE * len(t):
            continue
        bic = gm.bic(X)
        if bic < best_bic:
            best, best_bic = gm, bic
    labels = best.predict(X)
    t = t.assign(cluster=labels)

    max_velo = t.groupby("cluster")["rel_speed"].mean().max()
    clusters = []
    order = t.groupby("cluster")["rel_speed"].mean().sort_values(ascending=False).index
    remap = {old: new for new, old in enumerate(order)}
    t["cluster"] = t["cluster"].map(remap)
    for cid, g in t.groupby("cluster"):
        usage = {}
        for side in ("L", "R"):
            for two in (False, True):
                sel = t[(t["b_side"] == side) & (t["is_two_strike"] if two else True)]
                tot = sel["w"].sum()
                usage[side + ("2K" if two else "")] = float(g.loc[g.index.intersection(sel.index), "w"].sum() / tot) if tot else 0.0
        m = {c: float(g[c].mean()) for c in ("rel_speed", "induced_vert_break", "hb_arm", "spin_rate",
                                                "vert_appr_angle", "vaa_adj", "rel_height", "rel_side_arm")}
        tags = g["tagged_pitch_type"].where(~g["tagged_pitch_type"].isin(IGNORE_TAGS)).value_counts(normalize=True)
        clusters.append(Cluster(int(cid), _label(m, max_velo, tags.to_dict()), len(g), len(g) / len(t), usage, m,
                                {str(k): round(float(v), 3) for k, v in tags.head(3).items()}))
    _disambiguate(clusters, lg)
    first = pitches.iloc[0]
    return Arsenal(str(first["pitcher_tm_id"]), str(first["pitcher_name"]), first.get("pitcher_team"),
                   str(first["p_throws"]), len(t), clusters, t, model=(best, remap), scale=lg.scale)
