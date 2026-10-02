"""Database access: migrations and idempotent loads into Neon (or any Postgres).

Two transports behind one small interface:
  * tcp  - normal Postgres connection on port 5432 (psycopg). Used for local Postgres and CI.
  * http - Neon's SQL-over-HTTPS endpoint (port 443), the same protocol Neon's serverless driver uses.
           Needed on networks that block port 5432 (campus networks often do).

Default: http for *.neon.tech hosts, tcp for everything else. Override with DB_TRANSPORT=tcp|http.

Every write is a single SQL statement (data-modifying CTEs) or an explicit batch, so both transports
are atomic per game file without holding a session open.
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

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
    "pa_ending", "pa_result", "pitch_key",
]

Statement = tuple[str, tuple]


class DBError(RuntimeError):
    pass


def _to_dollar(sql: str) -> str:
    """psycopg-style %s placeholders -> Postgres $1, $2 ... (HTTP endpoint uses server-side binding)."""
    n = 0

    def rep(_):
        nonlocal n
        n += 1
        return f"${n}"

    return re.sub(r"%s", rep, sql)


def _param(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return str(v)


class TcpConn:
    transport = "tcp"

    def __init__(self, url: str):
        import psycopg

        self._c = psycopg.connect(url, autocommit=True, connect_timeout=15)

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        cur = self._c.execute(sql, params or None)
        return cur.fetchall() if cur.description else []

    def batch(self, statements: list[Statement]) -> list[list[tuple]]:
        out = []
        with self._c.transaction():
            for sql, params in statements:
                out.append(self.query(sql, params))
        return out

    def close(self):
        self._c.close()


class HttpConn:
    """Neon SQL-over-HTTP. Protocol mirrors @neondatabase/serverless:
    POST https://api.<region host>/sql, header Neon-Connection-String, body {query, params}
    or {queries: [...]} for a transaction."""

    transport = "http"

    def __init__(self, url: str):
        import requests

        try:  # use the operating system's certificates (works behind campus TLS inspection)
            import truststore

            truststore.inject_into_ssl()
        except ImportError:
            pass
        host = urlparse(url).hostname or ""
        self._endpoint = "https://" + re.sub(r"^[^.]+\.", "api.", host) + "/sql"
        self._s = requests.Session()
        self._s.headers.update({
            "Neon-Connection-String": url,
            "Neon-Raw-Text-Output": "true",
            "Neon-Array-Mode": "true",
            "Content-Type": "application/json",
        })

    def _post(self, body: dict, extra_headers: dict | None = None) -> dict:
        import requests

        try:
            r = self._s.post(self._endpoint, data=json.dumps(body), headers=extra_headers, timeout=120)
        except requests.RequestException as e:
            raise DBError(f"Could not reach Neon over HTTPS ({self._endpoint}): {e}") from e
        if r.status_code != 200:
            try:
                msg = r.json().get("message", r.text)
            except ValueError:
                msg = r.text
            raise DBError(f"Neon HTTP {r.status_code}: {msg}")
        return r.json()

    @staticmethod
    def _q(sql: str, params: tuple) -> dict:
        return {"query": _to_dollar(sql), "params": [_param(p) for p in params]}

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        res = self._post(self._q(sql, params))
        return [tuple(r) for r in res.get("rows", [])]

    def batch(self, statements: list[Statement]) -> list[list[tuple]]:
        res = self._post(
            {"queries": [self._q(s, p) for s, p in statements]},
            {"Neon-Batch-Isolation-Level": "ReadCommitted", "Neon-Batch-Read-Only": "false"},
        )
        return [[tuple(r) for r in x.get("rows", [])] for x in res["results"]]

    def close(self):
        self._s.close()


def connect(url: str | None = None, transport: str | None = None):
    url = url or os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set (put it in pipeline/.env or your shell).")
    host = urlparse(url).hostname or ""
    if host == "host":
        raise SystemExit("DATABASE_URL still has the example value. Put your Neon connection string in pipeline/.env.")
    transport = transport or os.environ.get("DB_TRANSPORT") or ("http" if host.endswith(".neon.tech") else "tcp")
    return HttpConn(url) if transport == "http" else TcpConn(url)


def split_sql(text: str) -> list[str]:
    """Split a migration file into statements (HTTP runs one statement per query).
    Handles -- comments and '...' strings; migrations must not use $$-quoted bodies."""
    text = re.sub(r"--[^\n]*", "", text)
    out, buf, in_str = [], [], False
    for ch in text:
        if ch == "'":
            in_str = not in_str
        if ch == ";" and not in_str:
            s = "".join(buf).strip()
            if s:
                out.append(s)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def migrate(conn) -> list[str]:
    conn.query("create table if not exists schema_migrations (version text primary key, "
               "applied_at timestamptz not null default now())")
    done = {r[0] for r in conn.query("select version from schema_migrations")}
    applied = []
    for f in sorted(MIGRATIONS_DIR.glob("*.sql")):
        version = f.name.split("_", 1)[0]
        if version in done:
            continue
        stmts = [(s, ()) for s in split_sql(f.read_text())]
        stmts.append(("insert into schema_migrations(version) values (%s) on conflict do nothing", (version,)))
        conn.batch(stmts)  # one transaction: a failed migration leaves nothing half-applied
        applied.append(f.name)
    return applied


def known_hashes(conn) -> set[str]:
    return {r[0] for r in conn.query("select sha256 from ingest_files")}


def _json_value(v):
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if hasattr(v, "item"):  # numpy scalar
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return v


def _pitches_json(pf: ParsedFile) -> str:
    recs = [
        {c: _json_value(v) for c, v in zip(PITCH_COLUMNS, row)}
        for row in pf.pitches[PITCH_COLUMNS].itertuples(index=False, name=None)
    ]
    return json.dumps(recs, separators=(",", ":"))


_COLS = ", ".join(PITCH_COLUMNS)
LOAD_SQL = f"""
with g as (
  insert into games (game_uid, game_id, game_date, season, home_team, away_team, stadium, level, league, game_type)
  values (%s, %s, %s::date, %s, %s, %s, %s, %s, %s, %s)
  on conflict (game_uid) do nothing
  returning 1
), p as (
  insert into pitches ({_COLS})
  select {_COLS} from jsonb_populate_recordset(null::pitches, %s::jsonb)
  on conflict (pitch_uid) do nothing
  returning 1
), f as (
  insert into ingest_files (sha256, file_name, game_uid, rows_read, rows_loaded, warnings, source)
  select %s, %s, %s, %s::int, (select count(*) from p), %s::jsonb, %s
  on conflict (sha256) do nothing
  returning 1
)
select (select count(*) from p), (select count(*) from g), (select count(*) from f)
"""


def load_statement(pf: ParsedFile, source: str = "folder") -> Statement:
    g = pf.game
    params = (
        g["game_uid"], g["game_id"], g["game_date"], g["season"], g["home_team"], g["away_team"],
        g["stadium"], g["level"], g["league"], g["game_type"],
        _pitches_json(pf),
        pf.sha256, pf.file_name, g["game_uid"], pf.rows_read, json.dumps(pf.warnings), source,
    )
    return LOAD_SQL, params


def load(conn, pf: ParsedFile, source: str = "folder") -> int:
    """Insert one game atomically. Re-loading the same file or the same pitches is a no-op."""
    rows = conn.query(*load_statement(pf, source))
    return int(rows[0][0])


def load_many(conn, files: list[ParsedFile], source: str = "folder") -> list[int]:
    """Insert several games in one round trip (one transaction)."""
    results = conn.batch([load_statement(pf, source) for pf in files])
    return [int(r[0][0]) for r in results]


def duplicate_statement(pf: ParsedFile, kept_game_uid: str, source: str = "folder") -> Statement:
    """Record a file skipped as a duplicate, so later syncs don't re-read it."""
    return (
        """insert into ingest_files (sha256, file_name, game_uid, rows_read, rows_loaded, warnings, source,
                                     status, duplicate_of)
           select %s, %s, %s, %s::int, 0, %s::jsonb, %s, 'duplicate', %s
           where exists (select 1 from games where game_uid = %s)
           on conflict (sha256) do nothing""",
        (pf.sha256, pf.file_name, kept_game_uid, pf.rows_read, json.dumps(pf.warnings), source,
         pf.game["game_uid"], kept_game_uid),
    )


def replace_game(conn, old_game_uid: str, pf: ParsedFile, source: str = "folder") -> int:
    """Swap a stored game for a more complete copy, atomically."""
    res = conn.batch([("delete from games where game_uid = %s", (old_game_uid,)), load_statement(pf, source)])
    return int(res[1][0][0])
