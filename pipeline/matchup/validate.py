"""Backtest: does the matchup model predict future outcomes better than simpler baselines?

Time split: everything before the split date is 'known'; every later pitch thrown by a sample of pitchers
is predicted from known data only, then scored against what actually happened.

Each test pitch is predicted six ways, from simplest to the full model:
  league   league rate for this pitcher hand x batter side (and count split)
  batter   the hitter's own (shrunk) rate vs this pitcher hand: 'how good is she'
  log5     the classic matchup formula (odds ratio): hitter's rate x pitcher's rate / league rate, using
           each one's overall numbers and nothing about pitch shape. The field's standard reference point.
  shape    all same-side hitters vs pitches shaped like this pitch's cluster: 'how good is the pitch'
  prior    shape + the hitter's overall skill (the model's starting point)
  model    prior + the hitter's own history against similar pitches (what the report shows)
If 'model' doesn't beat 'prior', the hitter-specific pitch-shape evidence (Shape fit) adds nothing yet.

Metrics: whiff (per swing), chase (swing per out-of-zone pitch), called strike (per pitch),
hard-hit (per tracked ball in play), run value (per pitch). Binary metrics are scored with the Brier score,
run value with squared error; 'skill' = % improvement over the league baseline. Uncertainty comes from a
bootstrap over hitters, so one hot or cold hitter can't drive the verdict.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from . import shape
from .arsenal import fit_arsenal
from .calibrate import calibrate, rates
from .engine import METRICS, SUM_COLS, Context, _rates, evaluate, indicators

PREDICTORS = ["league", "batter", "log5", "shape", "prior", "model"]
BINARY = {  # metric: (eligible-pitch mask, outcome)
    "whiff": (lambda d: d["is_swing"], lambda d: d["is_whiff"]),
    "chase": (lambda d: (d["in_zone"] == False).fillna(False), lambda d: d["is_swing"]),  # noqa: E712
    "called_strike": (lambda d: pd.Series(True, index=d.index), lambda d: d["is_called_strike"]),
    "hard_hit": (lambda d: d["bip_ev_valid"], None),  # outcome needs the hard-hit line
}
METRIC_ORDER = ["whiff", "chase", "called_strike", "hard_hit", "rv"]


@dataclass
class Config:
    split: date | None = None        # default: the date with 60% of regular-season pitches before it
    n_pitchers: int = 60
    min_train: int = 300             # tracked pitches before the split, to describe the arsenal
    min_test: int = 100              # pitches after the split, to be worth scoring
    boot: int = 300
    seed: int = 0


@dataclass
class Prepared:
    cfg: Config
    league: object
    train: pd.DataFrame
    arsenals: dict
    tests: dict                      # pitcher id -> test pitches with cluster assigned


def split_date(regular: pd.DataFrame, frac: float = 0.6) -> date:
    d = pd.Series(sorted(regular["game_date"]))
    return d.iloc[int(frac * (len(d) - 1))]


def prepare(df_all: pd.DataFrame, cfg: Config, verbose: bool = True) -> Prepared:
    regular = df_all[df_all["game_type"].fillna("regular") == "regular"]
    cfg.split = cfg.split or split_date(regular)
    train_raw = regular[regular["game_date"] < cfg.split]
    test_raw = regular[regular["game_date"] >= cfg.split]
    lg, train = calibrate(train_raw)
    test = shape.add_shape(test_raw, lg.vaa_slope)
    from .runvalues import pitch_run_values
    test = test.assign(rv=pitch_run_values(test, lg.linear_weights, lg.count_values).reindex(test.index).fillna(0.0))

    tr_n = train[train["pitch_tracked"]].groupby("pitcher_tm_id").size()
    te_n = test.groupby("pitcher_tm_id").size()
    ok = te_n[(te_n >= cfg.min_test) & te_n.index.isin(tr_n[tr_n >= cfg.min_train].index)]
    chosen = ok.sort_values(ascending=False).index[: cfg.n_pitchers]
    if verbose:
        print(f"backtest: train < {cfg.split} ({len(train):,} pitches), test >= {cfg.split} ({len(test):,} pitches); "
              f"{len(ok)} eligible pitchers, using {len(chosen)}")
    arsenals, tests = {}, {}
    for pid in chosen:
        try:
            a = fit_arsenal(train[train["pitcher_tm_id"] == pid], lg)
        except ValueError:
            continue
        t = test[test["pitcher_tm_id"] == pid].copy()
        t["cluster"] = a.assign(t)
        arsenals[pid], tests[pid] = a, t[t["cluster"].notna()]
    return Prepared(cfg, lg, train, arsenals, tests)


def _league_rates(lg, train: pd.DataFrame) -> dict:
    out = {}
    for (ph, bs), g in train.groupby(["p_throws", "b_side"]):
        for split, gg in (("all", g), ("2k", g[g["is_two_strike"]])):
            r = rates(gg, lg.hard_hit_mph)
            out[(ph, bs, split)] = {"whiff": r["whiff"], "chase": r["chase"], "called_strike": r["called_strike"],
                                    "hard_hit": r["hard_hit"], "rv": r["rv100"] / 100}
    return out


def _pitcher_rates(lg, train: pd.DataFrame, pid: str) -> pd.DataFrame:
    """Her own rates before the split, per batter side and count split, shrunk toward the league for her
    hand (same prior strengths as everything else). Used by the log5 baseline."""
    mine = train[train["pitcher_tm_id"] == pid]
    hand = mine["p_throws"].iloc[0]
    out = []
    for side in ("L", "R"):
        for split in ("all", "2k"):
            sel = lambda d: d[(d["b_side"] == side) & (d["is_two_strike"] if split == "2k" else True)]  # noqa: E731
            lgS = indicators(sel(train[train["p_throws"] == hand]), lg.hard_hit_mph).sum(0)
            base = {m: v[0] for m, v in _rates(lgS).items()}
            S = indicators(sel(mine), lg.hard_hit_mph).sum(0)
            r = _rates(S, {m: np.array([base[m] if np.isfinite(base[m]) else 0.0]) for m in METRICS})
            out.append({"b_side": side, "split": split, **{f"_pit_{m}": float(r[m][0]) for m in METRIC_ORDER}})
    return pd.DataFrame(out)


def _log5(b, p, l, metric):
    """Odds-ratio combination of hitter (b), pitcher (p) and league (l) rates; additive for run value."""
    if metric == "rv":
        return b + p - l
    c = lambda x: np.clip(x, 0.001, 0.999)  # noqa: E731
    o = c(b) / (1 - c(b)) * c(p) / (1 - c(p)) / (c(l) / (1 - c(l)))
    return o / (1 + o)


def predict(prep: Prepared, verbose: bool = True) -> pd.DataFrame:
    """One row per test pitch: outcomes and every predictor's prediction for every metric."""
    lg, train = prep.league, prep.train
    lrates = _league_rates(lg, train)
    ctx = Context(train, lg)
    rows = []
    t0 = time.time()
    for i, (pid, a) in enumerate(prep.arsenals.items(), 1):
        t = prep.tests[pid]
        res = evaluate(train, a, lg, sorted(t["batter_tm_id"].unique()), ctx=ctx)
        t = t.assign(split=np.where(t["is_two_strike"], "2k", "all"), cluster=t["cluster"].astype(int))
        det = res.detail.rename(columns={"side": "b_side"})
        cols = {m: f"p_model_{m}" for m in METRIC_ORDER}
        cols.update({f"prior_{m}": f"p_prior_{m}" for m in METRIC_ORDER})
        cols.update({f"bat_{m}": f"p_batter_{m}" for m in METRIC_ORDER})
        det = det[["batter_tm_id", "b_side", "split", "cluster"] + list(cols)].rename(columns=cols)
        det["has_history"] = True
        t = t.merge(det, on=["batter_tm_id", "b_side", "split", "cluster"], how="left")
        pop = res.population.rename(columns={"side": "b_side", **{m: f"p_shape_{m}" for m in METRIC_ORDER}})
        t = t.merge(pop[["b_side", "split", "cluster"] + [f"p_shape_{m}" for m in METRIC_ORDER]],
                    on=["b_side", "split", "cluster"], how="left")
        lr = pd.DataFrame([{"b_side": s, "split": sp, **{f"p_league_{m}": v[m] for m in METRIC_ORDER}}
                           for (ph, s, sp), v in lrates.items() if ph == a.throws])
        t = t.merge(lr, on=["b_side", "split"], how="left")
        la = lr[lr["split"] == "all"].drop(columns="split").rename(columns={f"p_league_{m}": f"_lall_{m}" for m in METRIC_ORDER})
        t = t.merge(la, on="b_side", how="left")
        t = t.merge(_pitcher_rates(lg, train, pid), on=["b_side", "split"], how="left")
        t["has_history"] = t["has_history"].fillna(False).astype(bool)
        for m in METRIC_ORDER:
            # Hitters with no history vs this hand fall back the way the report does.
            t[f"p_batter_{m}"] = t[f"p_batter_{m}"].fillna(t[f"_lall_{m}"])
            # The hitter's own rate is over all counts; shift it by the league's two-strike effect so this
            # baseline knows the count as well as the others do (otherwise two-strike chase alone sinks it).
            t[f"p_batter_{m}"] = t[f"p_batter_{m}"] + (t[f"p_league_{m}"] - t[f"_lall_{m}"])
            if m != "rv":
                t[f"p_batter_{m}"] = t[f"p_batter_{m}"].clip(0.001, 0.999)
            for name in ("prior", "model"):
                t[f"p_{name}_{m}"] = t[f"p_{name}_{m}"].fillna(t[f"p_shape_{m}"])
            t[f"p_log5_{m}"] = _log5(t[f"p_batter_{m}"], t[f"_pit_{m}"].fillna(t[f"p_league_{m}"]), t[f"p_league_{m}"], m)
        rows.append(t)
        if verbose:
            print(f"\r  predicted {i}/{len(prep.arsenals)} pitchers ({time.time() - t0:.0f}s)", end="", flush=True)
    if verbose:
        print()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _targets(p: pd.DataFrame, metric: str, hard_hit: float) -> tuple[pd.Series, pd.Series]:
    if metric == "rv":
        return pd.Series(True, index=p.index), p["rv"]
    mask_f, y_f = BINARY[metric]
    mask = mask_f(p).astype(bool)
    y = (p["exit_speed"].fillna(0) >= hard_hit) if metric == "hard_hit" else y_f(p)
    return mask, y.astype(float)


def score(pred: pd.DataFrame, hard_hit: float, boot: int = 300, seed: int = 0, history_only: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    p = pred[pred["has_history"]] if history_only else pred
    for m in METRIC_ORDER:
        mask, y = _targets(p, m, hard_hit)
        d = p[mask]
        y = y[mask]
        losses = {}
        for name in PREDICTORS:
            yhat = d[f"p_{name}_{m}"].clip(0, 1) if m != "rv" else d[f"p_{name}_{m}"]
            losses[name] = (yhat - y) ** 2
        L = pd.DataFrame(losses).dropna()
        if not len(L):
            continue
        per_b = L.groupby(d.loc[L.index, "batter_tm_id"]).agg(["sum", "count"])
        sums = {n: per_b[(n, "sum")].to_numpy() for n in PREDICTORS}
        cnt = per_b[(PREDICTORS[0], "count")].to_numpy()
        base = sums["league"].sum()
        row = {"metric": m, "n": int(len(L)), "hitters": int(len(cnt))}
        for n in PREDICTORS:
            row[f"skill_{n}"] = 100 * (1 - sums[n].sum() / base)
        # bootstrap over hitters: model vs prior (hitter-specific shape evidence) and model vs batter
        idx = rng.integers(0, len(cnt), size=(boot, len(cnt)))
        for a, b in (("model", "prior"), ("model", "batter"), ("prior", "batter"), ("model", "log5")):
            diff = (sums[b][idx].sum(1) - sums[a][idx].sum(1)) / sums["league"][idx].sum(1) * 100
            row[f"{a}_vs_{b}"] = 100 * (1 - sums[a].sum() / base) - 100 * (1 - sums[b].sum() / base)
            row[f"{a}_vs_{b}_lo"], row[f"{a}_vs_{b}_hi"] = np.percentile(diff, [5, 95])
        out.append(row)
    return pd.DataFrame(out)


def verdict(row) -> str:
    """Plain-language answer to 'can coaches trust this column?'."""
    best = max(PREDICTORS, key=lambda n: row[f"skill_{n}"])
    if row[f"skill_{best}"] <= 0.05:
        return "mostly noise: nothing beats the league rate"
    # 'validated' = the full model beats BOTH simpler alternatives with confidence
    if row["model_vs_prior_lo"] > 0 and row["model_vs_batter_lo"] > 0:
        return "validated: hitter-vs-shape history adds value"
    if row["model_vs_prior_hi"] < 0:
        return "hitter-specific part HURTS: needs heavier shrinkage (run backtest --tune)"
    if row["model_vs_batter_hi"] < 0:
        return "hitter's overall rate is the best guide"
    if row["prior_vs_batter_lo"] > 0 or row["model_vs_batter_lo"] > 0:
        return "pitch shape adds value; hitter-specific part unproven"
    return "no proven edge over the hitter's overall rate"


def describe(sc: pd.DataFrame, title: str) -> str:
    lines = [title, f"{'metric':<14}{'n':>9}{'hitters':>8}  " + "".join(f"{n:>8}" for n in PREDICTORS)
             + "   model-prior (90% CI)   model-log5   verdict"]
    for _, r in sc.iterrows():
        lines.append(f"{r['metric']:<14}{r['n']:>9,}{r['hitters']:>8}  " + "".join(f"{r[f'skill_{n}']:>8.2f}" for n in PREDICTORS)
                     + f"   {r['model_vs_prior']:+.2f} ({r['model_vs_prior_lo']:+.2f}, {r['model_vs_prior_hi']:+.2f})"
                     + f"   {r['model_vs_log5']:+.2f}{'*' if r['model_vs_log5_lo'] > 0 else ' '}   {verdict(r)}")
    lines.append("skill = % lower error than the league baseline (higher is better; 0 = no better than league average)")
    lines.append("model-log5 = how much better the model is than the classic log5 formula (* = with 90% confidence)")
    return "\n".join(lines)


def objective(sc: pd.DataFrame) -> float:
    """Average model skill across metrics: what tuning maximizes."""
    return float(sc["skill_model"].mean()) if len(sc) else -np.inf


GRID_BANDWIDTH = (0.15, 0.3, 0.6, 1.0)
GRID_HALF_LIFE = (30.0, 60.0, 120.0, float("inf"))
GRID_PRIOR = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
GRID_EXTENSION = (0.0, 0.25, 0.5)
GRID_COMBINE = ("add", "odds")
GRID_BORROW = (0.0, 0.5, 1.0)
TUNED_METRICS = ["whiff", "chase", "called_strike", "hard_hit", "rv"]


def run_with(prep: Prepared, values: dict, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    from . import settings
    settings.apply(values)
    pred = predict(prep, verbose=verbose)
    return pred, score(pred, prep.league.hard_hit_mph, boot=prep.cfg.boot, seed=prep.cfg.seed)


def tune(prep: Prepared, start: dict, log=print) -> tuple[dict, list[dict]]:
    """Coordinate search: bandwidth, then recency half-life, then prior strength per metric.
    Each step keeps the best value found so far. Returns (best settings, every trial)."""
    best = dict(start)
    trials = []

    def trial(values, label):
        _, sc = run_with(prep, values)
        obj = objective(sc)
        row = {"step": label, **{k: values.get(k) for k in ("bandwidth", "half_life_days", "extension_weight",
                                                             "prior_combine", "borrow")},
               "prior_scale": json_safe(values["prior_scale"]), "objective": obj,
               **{f"skill_{r.metric}": r.skill_model for r in sc.itertuples()}}
        trials.append(row)
        log(f"  {label:<38} objective {obj:+.3f}")
        return obj, sc

    log("tuning similarity width...")
    scores = {bw: trial({**best, "bandwidth": bw}, f"bandwidth {bw}")[0] for bw in GRID_BANDWIDTH}
    best["bandwidth"] = max(scores, key=scores.get)
    log("tuning recency half-life...")
    scores = {hl: trial({**best, "half_life_days": hl}, f"half-life {hl:g} days")[0] for hl in GRID_HALF_LIFE}
    best["half_life_days"] = max(scores, key=scores.get)
    log("tuning release extension in similarity...")
    scores = {x: trial({**best, "extension_weight": x}, f"extension weight {x:g}")[0] for x in GRID_EXTENSION}
    best["extension_weight"] = _keep_default(scores, 0.0)
    log("tuning how hitter level and pitch shape combine...")
    scores = {c: trial({**best, "prior_combine": c}, f"combine: {c}")[0] for c in GRID_COMBINE}
    best["prior_combine"] = _keep_default(scores, "add")
    log("tuning similar-hitter borrowing...")
    scores = {b: trial({**best, "borrow": b}, f"similar-hitter borrowing {b:g}")[0] for b in GRID_BORROW}
    best["borrow"] = _keep_default(scores, 0.0)
    log("tuning shrinkage strength per metric...")
    per_metric = {m: {} for m in TUNED_METRICS}
    for ps in GRID_PRIOR:
        _, sc = trial({**best, "prior_scale": ps}, f"prior strength x{ps:g}")
        for r in sc.itertuples():
            if r.metric in per_metric:
                per_metric[r.metric][ps] = r.skill_model
    best["prior_scale"] = {m: max(v, key=v.get) for m, v in per_metric.items() if v}
    return best, trials


MIN_GAIN = 0.02   # an optional feature must beat 'off' by this much objective (%-points of skill) to be kept


def _keep_default(scores: dict, default):
    """Best option, but stay with the default (simpler) one unless the gain is more than noise-level."""
    best = max(scores, key=scores.get)
    return best if scores[best] > scores.get(default, -np.inf) + MIN_GAIN else default


def json_safe(v):
    if isinstance(v, dict):
        return {k: json_safe(x) for k, x in v.items()}
    if isinstance(v, float) and not np.isfinite(v):
        return "inf"
    return v
