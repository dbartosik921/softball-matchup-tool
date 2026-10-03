"""Precompute matchups and write them to Neon for the dashboard (web/).

Coverage (both directions for the home team):
  * every home-team pitcher vs every hitter on every roster in the latest season
  * every other pitcher with enough tracked pitches vs the home team's hitters

One evaluate() per pitcher covers all of her batters at once, and the league context (similarity pools,
batter totals) is built once and shared, so a full publish is minutes, not hours.

Writes go to a new run_id; the run is marked complete only after every row is in, then older runs are
deleted. The site reads the latest complete run, so it never shows a half-written publish.
"""
from __future__ import annotations

import difflib
import json
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from . import engine, lineup as L, settings, zones
from .calibrate import League
from .engine import Context, evaluate
from .run import build_arsenal

BATCH_ROWS = 400
SUMMARY_KEYS = [
    "score", "xrv100", "xrv100_2k", "fit100", "fit100_2k", "confidence", "sim_pitches", "sim_pitches_2k",
    "pitches_vs_hand", "direct_pa", "direct_line",
    *[f"{p}{m}{s}" for m in ("whiff", "chase", "called_strike", "hard_hit", "ops", "rv")
      for p in ("", "pop_") for s in ("", "_2k")],
]
DETAIL_METRICS = ("whiff", "chase", "called_strike", "hard_hit", "ops")
DETAIL_KEYS = ["split", "cluster", "label", "usage", "sim_pitches", "fit100", "rv",
               *[f"{p}{m}" for m in DETAIL_METRICS for p in ("", "pop_", "raw_")]]
MEANS = ("rel_speed", "induced_vert_break", "hb_arm", "spin_rate", "vaa_adj", "rel_height")


def _num(v, nd: int = 4):
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else round(float(v), nd)
    return v


def _pack(d: dict, keys: Iterable[str]) -> list:
    """Values in a fixed key order (keys are stored once in pub_runs.meta): ~3x smaller than dicts."""
    return [_num(d.get(k), 3 if isinstance(d.get(k), float) and k not in ("score",) else 4) for k in keys]


def league_ops(hist: pd.DataFrame) -> float | None:
    e = hist[hist["pa_ending"].fillna(False).astype(bool)]
    r = e["pa_result"]
    hits = {"1B": 1, "2B": 2, "3B": 3, "HR": 4}
    ab = r.isin(AB_RESULTS).sum()
    on = r.isin(list(hits) + ["BB", "HBP"]).sum()
    den = ab + r.isin(["BB", "HBP", "SF"]).sum()
    tb = r.map(hits).fillna(0).sum()
    return _num(on / den + tb / ab, 3) if ab and den else None


def find_team(regular: pd.DataFrame, team: str) -> str:
    t = team.strip().upper()
    teams = sorted(set(regular["batter_team"].dropna()) | set(regular["pitcher_team"].dropna()))
    if t in teams:
        return t
    hits = [x for x in teams if t in x] or difflib.get_close_matches(t, teams, n=6, cutoff=0.5)
    if len(hits) == 1 and t in hits[0]:
        return hits[0]
    raise SystemExit(f"No team '{team}' in the data." + (f" Did you mean: {', '.join(hits[:8])}?" if hits else ""))


def arsenal_json(a) -> list[dict]:
    return [{"cid": int(c.cid), "label": c.label, "share": _num(c.share), "n": int(c.n),
             "usage": {k: _num(v) for k, v in c.usage.items()},
             "means": {k: _num(c.means.get(k), 2) for k in MEANS},
             "tag_mix": {k: _num(v, 3) for k, v in c.tag_mix.items()}} for c in a.clusters]


def _zones_json(z: dict | None):
    """[ops_all(25), pa_all(25), ops_ch(25) | null, pa_ch(25) | null] for the back-page strike zones."""
    if not z:
        return None
    r = lambda a, nd: None if a is None else [_num(float(x), nd) for x in a]  # noqa: E731
    return [r(z["all"], 3), r(z["pa"], 1), r(z["ch"], 3), r(z["pa_ch"], 1)]


def matchup_rows(res, zones: dict | None = None) -> list[dict]:
    det = res.detail
    by = det.groupby("batter_tm_id").indices if len(det) else {}
    out = []
    for r in res.batters.to_dict("records"):
        bid = r["batter_tm_id"]
        d = det.iloc[by[bid]] if bid in by else det.iloc[:0]
        d = d[d["side"] == r["side"]] if len(d) else d
        out.append({
            "pitcher_tm_id": res.arsenal.pitcher_id, "batter_tm_id": bid, "batter_name": r.get("batter_name"),
            "batter_team": r.get("batter_team"), "side": r.get("side"),
            "summary": _pack(r, SUMMARY_KEYS),
            "detail": [_pack(x, DETAIL_KEYS) for x in d.to_dict("records")],
            "zones": _zones_json((zones or {}).get(bid)),
        })
    return out


@dataclass
class Plan:
    home: str
    season: str
    teams: dict[str, pd.DataFrame]          # team -> that team's batting rows in the latest season
    home_pitchers: list[str]
    opp_pitchers: list[str]
    all_batters: list[str]
    home_batters: list[str]
    skipped: list[str] = field(default_factory=list)


def plan(df_all: pd.DataFrame, hist: pd.DataFrame, home: str, min_pitches: int, opponents: bool = True) -> Plan:
    season = hist["season"].max()
    hs = hist[hist["season"] == season]
    teams = {t: g for t, g in hs.groupby("batter_team") if t}
    # pitchers: season counts of tracked regular-season pitches; home staff needs far less to be useful
    cur = df_all[(df_all["season"] == season) & df_all["pitch_tracked"].fillna(False).astype(bool)]
    counts = cur.groupby(["pitcher_tm_id", "pitcher_team"]).size().reset_index(name="n")
    counts = counts.sort_values("n").drop_duplicates("pitcher_tm_id", keep="last")
    home_p = counts[(counts["pitcher_team"] == home) & (counts["n"] >= 50)]
    opp_p = counts[(counts["pitcher_team"] != home) & (counts["n"] >= min_pitches)] if opponents else counts.iloc[:0]
    home_bat = list(dict.fromkeys(teams[home]["batter_tm_id"])) if home in teams else []
    all_bat = list(dict.fromkeys(hs["batter_tm_id"].dropna()))
    return Plan(home, season, teams, list(home_p.sort_values("n", ascending=False)["pitcher_tm_id"]),
                list(opp_p.sort_values("n", ascending=False)["pitcher_tm_id"]), all_bat, home_bat)


AB_RESULTS = {"1B", "2B", "3B", "HR", "OUT", "K", "FC", "ROE"}
OUT_RESULTS = {"OUT", "SF", "SH", "FC"}


def at_bats(g: pd.DataFrame) -> pd.Series:
    e = g[g["pa_ending"].fillna(False).astype(bool)]
    return e[e["pa_result"].isin(AB_RESULTS)].groupby("batter_tm_id").size()


def innings_pitched(df: pd.DataFrame) -> pd.Series:
    """Innings pitched per pitcher (outs / 3) from plate-appearance endings: Trackman's OutsOnPlay where
    recorded (catches double plays), otherwise one out per out-type result; plus strikeouts."""
    e = df[df["pa_ending"].fillna(False).astype(bool)]
    oop = pd.to_numeric(e["outs_on_play"], errors="coerce") if "outs_on_play" in e else pd.Series(np.nan, index=e.index)
    fallback = e["pa_result"].isin(OUT_RESULTS).astype(float)
    outs = oop.where(oop.notna(), fallback) + (e["pa_result"] == "K").astype(float)
    return outs.groupby(e["pitcher_tm_id"]).sum() / 3.0


def team_rows(p: Plan) -> list[dict]:
    rows = []
    for t, g in p.teams.items():
        r = L.roster(g, t)
        ab = at_bats(g)
        order, last = L.latest_lineup(g, t)
        rows.append({
            "team": t, "season": p.season, "last_game": str(last) if last is not None else None,
            "lineup": order,
            "roster": [{"id": x.batter_tm_id, "name": x.batter_name, "pa": int(x.pa), "ab": int(ab.get(x.batter_tm_id, 0)),
                        "games": int(x.games), "last_game": str(x.last_game)} for x in r.itertuples()],
        })
    return rows


# --- database writes (one statement per batch; works over both transports) -------------------------

_INS = {
    "pub_pitchers": ("pitcher_tm_id text, pitcher_name text, team text, throws text, n_pitches int, "
                     "is_home boolean, arsenal jsonb, ip real"),
    "pub_teams": "team text, season text, last_game date, lineup jsonb, roster jsonb",
    "pub_matchups": ("pitcher_tm_id text, batter_tm_id text, batter_name text, batter_team text, side text, "
                     "summary jsonb, detail jsonb, zones jsonb"),
}


def insert_statement(table: str, run_id: int, rows: list[dict]):
    cols = _INS[table]
    names = ", ".join(c.split()[0] for c in cols.split(", "))
    sql = (f"insert into {table} (run_id, {names}) select %s::bigint, {names} "
           f"from jsonb_to_recordset(%s::jsonb) as x({cols})")
    return sql, (run_id, json.dumps(rows, default=str))


def write_rows(conn, table: str, run_id: int, rows: list[dict]) -> None:
    for i in range(0, len(rows), BATCH_ROWS):
        conn.query(*insert_statement(table, run_id, rows[i:i + BATCH_ROWS]))


def start_run(conn, home: str, data_through, meta: dict) -> int:
    return int(conn.query("insert into pub_runs (home_team, data_through, meta) values (%s, %s::date, %s::jsonb) "
                          "returning run_id", (home, str(data_through), json.dumps(meta, default=str)))[0][0])


def finish_run(conn, run_id: int) -> None:
    conn.batch([
        ("update pub_runs set complete = true where run_id = %s", (run_id,)),
        ("delete from pub_runs where run_id <> %s", (run_id,)),   # cascades to the run's rows
    ])


def publish(conn, df_all: pd.DataFrame, lg: League, hist: pd.DataFrame, home: str, min_pitches: int = 150,
            opponents: bool = True, log: Callable[[str], None] = print) -> dict:
    regular = df_all[df_all["game_type"].fillna("regular") == "regular"]
    home = find_team(regular, home)
    p = plan(df_all, hist, home, min_pitches, opponents)
    if not p.home_pitchers and not p.opp_pitchers:
        raise SystemExit(f"No pitchers to publish for {home} in {p.season}.")
    log(f"{home} {p.season}: {len(p.home_pitchers)} home pitchers x {len(p.all_batters):,} hitters, "
        f"{len(p.opp_pitchers)} opponent pitchers x {len(p.home_batters)} {home} hitters")

    meta = {
        "n_games": lg.n_games, "hard_hit_mph": lg.hard_hit_mph, "season": p.season,
        "slot_pa": [float(x) for x in L.expected_pa_by_slot(hist)],
        "fit_components": list(engine.FIT_COMPONENTS), "validation": settings.validation(),
        "settings": {"bandwidth": engine.BANDWIDTH, "prior_scale": engine.PRIOR_SCALE},
        "summary_keys": SUMMARY_KEYS, "detail_keys": DETAIL_KEYS,
        "league_ops": league_ops(hist), "zone_edges": {"x": list(zones.X_EDGES), "y": list(zones.Y_EDGES)},
    }
    run_id = start_run(conn, home, max(hist["game_date"]), meta)
    write_rows(conn, "pub_teams", run_id, team_rows(p))

    ctx = Context(hist, lg)
    season_rows = df_all[(df_all["season"] == p.season) & (df_all["game_type"].fillna("regular") == "regular")]
    ip = innings_pitched(season_rows)
    by_pitcher = df_all.groupby("pitcher_tm_id").indices
    jobs = [(pid, True) for pid in p.home_pitchers] + [(pid, False) for pid in p.opp_pitchers]
    n_rows = 0
    for k, (pid, is_home) in enumerate(jobs, 1):
        mine = df_all.iloc[by_pitcher[pid]]
        try:
            a = build_arsenal(df_all, pid, lg, mine=mine)
            res = evaluate(hist, a, lg, p.all_batters if is_home else p.home_batters, ctx=ctx)
            z = zones.zone_ops(ctx, a, lg, dict(zip(res.batters["batter_tm_id"], res.batters["side"])))
        except Exception as e:  # one odd pitcher (e.g. almost no tracked pitches) must not stop the publish
            p.skipped.append(f"{mine['pitcher_name'].iloc[-1]}: {e}")
            continue
        write_rows(conn, "pub_pitchers", run_id, [{
            "pitcher_tm_id": pid, "pitcher_name": a.pitcher_name, "team": a.team, "throws": a.throws,
            "n_pitches": int(a.n_pitches), "is_home": is_home, "arsenal": arsenal_json(a),
            "ip": _num(float(ip.get(pid, 0.0)), 2)}])
        rows = matchup_rows(res, z)
        write_rows(conn, "pub_matchups", run_id, rows)
        n_rows += len(rows)
        if is_home or k % 25 == 0 or k == len(jobs):
            log(f"  {k}/{len(jobs)} {a.pitcher_name} ({a.team}): {len(rows):,} hitters")
    finish_run(conn, run_id)
    out = {"run_id": run_id, "pitchers": len(jobs) - len(p.skipped), "matchups": n_rows, "skipped": p.skipped}
    log(f"published run {run_id}: {out['pitchers']} pitchers, {n_rows:,} pitcher-hitter matchups")
    for s in p.skipped[:10]:
        log(f"  skipped {s}")
    return out
