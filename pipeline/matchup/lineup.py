"""Opponent roster, inferred lineup and expected plate appearances by lineup slot."""
from __future__ import annotations

import numpy as np
import pandas as pd

PA_KEY = ["game_uid", "inning", "top_bottom", "pa_of_inning"]


def _pas(hist: pd.DataFrame) -> pd.DataFrame:
    """One row per plate appearance (its first pitch), in game order."""
    p = hist.sort_values(["game_uid", "inning", "top_bottom", "pa_of_inning", "pitch_of_pa"])
    return p.drop_duplicates(PA_KEY)


def team_season(hist: pd.DataFrame, team: str) -> str | None:
    s = hist.loc[hist["batter_team"] == team, "season"]
    return s.max() if len(s) else None


def roster(hist: pd.DataFrame, team: str) -> pd.DataFrame:
    """Batters who batted for `team` in its latest season, with plate appearances and games."""
    season = team_season(hist, team)
    h = hist[(hist["batter_team"] == team) & (hist["season"] == season)]
    pas = _pas(h)
    r = (pas.groupby("batter_tm_id")
         .agg(batter_name=("batter_name", "last"), pa=("pitch_uid", "size"), games=("game_uid", "nunique"),
              last_game=("game_date", "max"))
         .reset_index().sort_values("pa", ascending=False))
    return r


def latest_lineup(hist: pd.DataFrame, team: str) -> tuple[list[str], object]:
    """Batting order from the team's most recent game: first nine batters by first plate appearance."""
    h = hist[hist["batter_team"] == team]
    if not len(h):
        return [], None
    last = h["game_date"].max()
    g = h[h["game_date"] == last]
    g = g[g["game_uid"] == g["game_uid"].iloc[-1]]
    pas = _pas(g)
    order = list(dict.fromkeys(pas["batter_tm_id"]))
    return order[:9], last


def expected_pa_by_slot(hist: pd.DataFrame) -> np.ndarray:
    """Average plate appearances per game for lineup slots 1-9 across all team-games.
    The n-th plate appearance of a team's game belongs to slot ((n - 1) mod 9) + 1."""
    pas = _pas(hist)
    pas = pas.assign(team_game=pas["game_uid"] + "|" + pas["batter_team"].fillna(""))
    n = pas.groupby("team_game").cumcount()
    slot = (n % 9).to_numpy()
    games = pas["team_game"].nunique()
    if games == 0:
        return np.full(9, 3.5)
    return np.bincount(slot, minlength=9)[:9] / games
