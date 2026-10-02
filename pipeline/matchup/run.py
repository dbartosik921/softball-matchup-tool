"""End-to-end matchup run on an in-memory frame (the CLI loads the frame from the database)."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import lineup as L
from .arsenal import fit_arsenal
from .calibrate import League, calibrate
from .engine import Result, evaluate
from .recency import recency_weights
from .report import render

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


@dataclass
class MatchupRun:
    result: Result
    team: str
    lineup: list[str]
    slot_pa: object
    league: League
    html: str
    data_through: object


def find_pitcher(df: pd.DataFrame, query: str) -> tuple[str, str]:
    """Pitcher by Trackman ID or name ('Burnham, Payton' / 'Payton Burnham'). Returns (id, name)."""
    q = query.strip()
    if q in set(df["pitcher_tm_id"]):
        return q, df.loc[df["pitcher_tm_id"] == q, "pitcher_name"].iloc[-1]
    from .ids import name_key
    want = name_key(q)
    names = df[["pitcher_tm_id", "pitcher_name"]].dropna().drop_duplicates()
    names = names.assign(k=names["pitcher_name"].map(name_key))
    hit = names[names["k"] == want]
    if not len(hit):
        close = difflib.get_close_matches(want, names["k"].unique().tolist(), n=5, cutoff=0.6)
        raise SystemExit(f"No pitcher named '{query}'." + (f" Did you mean: {', '.join(close)}?" if close else ""))
    counts = df[df["pitcher_tm_id"].isin(hit["pitcher_tm_id"])].groupby("pitcher_tm_id").size()
    pid = counts.idxmax()
    return pid, hit.loc[hit["pitcher_tm_id"] == pid, "pitcher_name"].iloc[0]


def check_team(df: pd.DataFrame, team: str) -> str:
    t = team.strip().upper()
    teams = set(df["batter_team"].dropna())
    if t in teams:
        return t
    close = difflib.get_close_matches(t, sorted(teams), n=5, cutoff=0.5)
    raise SystemExit(f"No team '{team}' in the data." + (f" Did you mean: {', '.join(close)}?" if close else ""))


def run(df_all: pd.DataFrame, pitcher: str, team: str, league: tuple[League, pd.DataFrame] | None = None) -> MatchupRun:
    regular = df_all[df_all["game_type"].fillna("regular") == "regular"]
    lg, hist = league or calibrate(regular)
    pid, pname = find_pitcher(df_all, pitcher)
    team = check_team(regular, team)

    # Her arsenal: every tracked pitch she has thrown (fall included: it's her most current arsenal),
    # with recency-weighted usage. Shape features use the league calibration.
    from . import shape
    mine = shape.add_shape(df_all[df_all["pitcher_tm_id"] == pid], lg.vaa_slope)
    as_of = max(mine["game_date"])
    season = mine.loc[mine["game_date"] == as_of, "season"].iloc[0]
    w = recency_weights(mine["game_date"], mine["season"], as_of, season)
    arsenal = fit_arsenal(mine, lg, weights=w / w.max() if w.max() > 0 else w)

    roster = L.roster(hist, team)
    lineup, _ = L.latest_lineup(hist[hist["season"] == L.team_season(hist, team)], team)
    slot_pa = L.expected_pa_by_slot(hist)
    res = evaluate(hist, arsenal, lg, list(roster["batter_tm_id"]))
    through = max(hist["game_date"])
    page = render(res, team, lineup, slot_pa, lg, through, lg.n_games)
    return MatchupRun(res, team, lineup, slot_pa, lg, page, through)


def write(run_: MatchupRun, out_dir: Path = REPORTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", f"{run_.result.arsenal.pitcher_name}_vs_{run_.team}").strip("_")
    path = out_dir / f"{stem}.html"
    path.write_text(run_.html, encoding="utf-8")
    run_.result.batters.to_csv(out_dir / f"{stem}_batters.csv", index=False)
    run_.result.detail.to_csv(out_dir / f"{stem}_by_pitch.csv", index=False)
    return path
