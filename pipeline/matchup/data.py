"""Load pitches from the database into a DataFrame, with a local cache.

Neon over HTTPS returns text, so values are converted here to the same dtypes the parser produces.
The cache (pipeline/.cache/pitches.pkl) is reused until the database's pitch count or latest ingest
time changes, so repeated report runs don't re-download ~250k pitches.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
PAGE = 5_000

TEXT = ["pitch_uid", "game_uid", "season", "game_type", "top_bottom", "pitcher_tm_id", "pitcher_name",
        "pitcher_team", "p_throws", "batter_tm_id", "batter_name", "batter_team", "b_side", "tagged_pitch_type",
        "pitch_call", "kor_bb", "tagged_hit_type", "play_result", "hit_launch_conf", "pa_result",
        "home_team", "away_team"]
INT = ["inning", "pa_of_inning", "pitch_of_pa", "pitch_no", "balls", "strikes", "outs", "runs_scored"]
FLOAT = ["rel_speed", "spin_rate", "rel_height", "rel_side", "extension", "induced_vert_break", "horz_break",
         "plate_loc_height", "plate_loc_side", "vert_appr_angle", "horz_appr_angle", "exit_speed", "launch_angle",
         "hb_arm", "rel_side_arm", "hb_in", "loc_in", "haa_in"]
BOOL = ["pitch_tracked", "same_side", "is_swing", "is_whiff", "is_called_strike", "is_bip", "is_two_strike",
        "bip_ev_valid", "pa_ending"]
NULLABLE_BOOL = ["in_zone"]
COLUMNS = TEXT[:-2] + INT + FLOAT + BOOL + NULLABLE_BOOL


def _bool(s: pd.Series, nullable=False) -> pd.Series:
    def conv(v):
        if v is None or (isinstance(v, float) and v != v) or v is pd.NA:
            return None
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        return str(v).lower() in ("t", "true", "1")

    m = s.astype(object).map(conv)
    return m.astype("boolean") if nullable else m.fillna(False).astype(bool)


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    for c in INT:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in FLOAT:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in BOOL:
        if c in df:
            df[c] = _bool(df[c])
    for c in NULLABLE_BOOL:
        if c in df:
            df[c] = _bool(df[c], nullable=True)
    if "game_date" in df:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def _select_cols() -> str:
    cols = [f"p.{c}" for c in COLUMNS] + ["p.game_date::text as game_date", "g.game_type", "g.home_team", "g.away_team"]
    cols = [c for c in cols if c not in ("p.game_type",)]
    return ", ".join(cols)


def load_pitches(conn, use_cache: bool = True, verbose: bool = True) -> pd.DataFrame:
    stamp = conn.query("select count(*), max(ingested_at)::text from pitches")[0]
    stamp = [str(stamp[0]), str(stamp[1])]
    CACHE_DIR.mkdir(exist_ok=True)
    cache, meta = CACHE_DIR / "pitches.pkl", CACHE_DIR / "pitches.json"
    if use_cache and cache.exists() and meta.exists() and json.loads(meta.read_text()) == stamp:
        if verbose:
            print(f"data: {int(stamp[0]):,} pitches (cached)")
        return pd.read_pickle(cache)

    names = [c.split(" as ")[-1].split(".")[-1] for c in _select_cols().split(", ")]
    rows, last = [], ""
    while True:
        page = conn.query(
            f"select {_select_cols()} from pitches p join games g using (game_uid) "
            f"where p.pitch_uid > %s order by p.pitch_uid limit {PAGE}",
            (last,),
        )
        if not page:
            break
        rows += page
        last = page[-1][0]
        if verbose:
            print(f"\rdata: downloading {len(rows):,} / {int(stamp[0]):,} pitches", end="", flush=True)
        if len(page) < PAGE:
            break
    if verbose:
        print()
    df = coerce(pd.DataFrame(rows, columns=names))
    df.to_pickle(cache)
    meta.write_text(json.dumps(stamp))
    return df


def season_weight_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for opponent-matchup history: regular-season games only."""
    return df[df["game_type"].fillna("regular") == "regular"]


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make a parser/synthetic frame look like a loaded one (tests)."""
    df = df.copy()
    for c in ("home_team", "away_team"):
        if c not in df:
            df[c] = None
    if "game_type" not in df:
        df["game_type"] = "regular"
    for c in INT + FLOAT:
        if c not in df:
            df[c] = np.nan
    return df
