"""Database access: migrations and idempotent loads into Neon (or any Postgres)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import psycopg

from .ingest import ParsedFile

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "db" / "migrations"

PITCH_COLUMNS = [
    "pitch_uid", "game_uid", "game_date", "season", "pitch_no", "inning", "top_bottom", "pa_of_inning",
    "pitch_of_pa", "outs", "balls", "strikes",
    "pitcher_tm_id", "pitcher_name", "pitcher_team", "p_throws",
    "batter_tm_id", "batter_name", "batter_team", "b_side", "catcher_tm_id",
    "tagged_pitch_type", "pitch_call", "kor_bb", "tagged_hit_type", "play_result", "outs_on_play", "runs_scored",
    "rel_speed", "spin_rate", "spin_axis", "rel_height", "rel_side", "extension", "vert_rel_angle",
    "horz_rel_angle", "induced_vert_break", "vert_break", "horz_break", "plate_loc_height", "plate_loc_side",
    "zone_speed", "vert_appr_angle", "horz_appr_angle", "zone_time", "pitch_tracked",
    "exit_speed", "launch_angle", "direction", "distance", "hit_launch_conf",
    "hb_arm", "rel_side_arm", "hb_in", "loc_in", "haa_in", "same_side",
    "in_zone", "is_swing", "is_whiff", "is_called_strike", "is_bip", "is_two_strike", "bip_ev_valid",
    "pa_ending", "pa_result",
]


def connect(url: str | None = None) -> psycopg.Connection:
    url = url or os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set (put it in pipeline/.env or your shell).")
    # autocommit: each conn.transaction() block is a real transaction, committed per game file.
    return psycopg.connect(url, autocommit=True)


def migrate(conn: psycopg.Connection) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("create table if not exists schema_migrations (version text primary key, "
                    "applied_at timestamptz not null default now())")
        cur.execute("select version from schema_migrations")
        done = {r[0] for r in cur.fetchall()}
    applied = []
    for f in sorted(MIGRATIONS_DIR.glob("*.sql")):
        version = f.name.split("_", 1)[0]
        if version in done:
            continue
        with conn.transaction():
            conn.execute(f.read_text())
            conn.execute("insert into schema_migrations(version) values (%s) on conflict do nothing", (version,))
        applied.append(f.name)
    return applied


def known_hashes(conn: psycopg.Connection) -> set[str]:
    return {r[0] for r in conn.execute("select sha256 from ingest_files").fetchall()}


def _clean(v):
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float) and v != v:
        return None
    if hasattr(v, "item"):  # numpy scalar
        return v.item()
    return v


def load(conn: psycopg.Connection, pf: ParsedFile, source: str = "folder") -> int:
    """Insert one game. Re-loading the same file or the same pitches is a no-op."""
    g = pf.game
    rows = [tuple(_clean(v) for v in r) for r in pf.pitches[PITCH_COLUMNS].itertuples(index=False, name=None)]
    with conn.transaction():
        conn.execute(
            """insert into games (game_uid, game_id, game_date, season, home_team, away_team, stadium, level, league)
               values (%(game_uid)s, %(game_id)s, %(game_date)s, %(season)s, %(home_team)s, %(away_team)s,
                       %(stadium)s, %(level)s, %(league)s)
               on conflict (game_uid) do nothing""",
            g,
        )
        with conn.cursor() as cur:
            cur.execute("drop table if exists _p")
            cur.execute("create temp table _p (like pitches including defaults)")
            with cur.copy(f"copy _p ({', '.join(PITCH_COLUMNS)}) from stdin") as cp:
                for r in rows:
                    cp.write_row(r)
            cur.execute(f"""insert into pitches ({', '.join(PITCH_COLUMNS)})
                            select {', '.join(PITCH_COLUMNS)} from _p
                            on conflict (pitch_uid) do nothing""")
            loaded = cur.rowcount
            cur.execute("drop table _p")
        conn.execute(
            """insert into ingest_files (sha256, file_name, game_uid, rows_read, rows_loaded, warnings, source)
               values (%s, %s, %s, %s, %s, %s, %s) on conflict (sha256) do nothing""",
            (pf.sha256, pf.file_name, g["game_uid"], pf.rows_read, loaded, json.dumps(pf.warnings), source),
        )
    return loaded
