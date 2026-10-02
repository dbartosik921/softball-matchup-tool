"""Load into a real Postgres over both transports. Set TEST_DATABASE_URL to run (skipped otherwise).

The http transport is exercised against tests/neon_mock.py, which speaks Neon's SQL-over-HTTP
protocol and forwards to the same Postgres.
"""
import os
from pathlib import Path

import pytest

from matchup import db
from matchup.ingest import parse_file

FIXTURE = Path(__file__).parent / "fixtures" / "20260424-BoglePark-1_SB.csv"
URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture(params=["tcp", "http"])
def conn(request):
    reset = db.TcpConn(URL)
    reset.query("drop schema public cascade")
    reset.query("create schema public")
    reset.close()
    if request.param == "tcp":
        c = db.TcpConn(URL)
    else:
        from tests import neon_mock

        srv = neon_mock.start(URL)
        c = db.HttpConn("postgresql://u:p@ep-fake-123.local/db")
        c._endpoint = f"http://127.0.0.1:{srv.server_port}/sql"
        request.addfinalizer(srv.shutdown)
    request.addfinalizer(c.close)
    assert db.migrate(c) == ["001_init.sql"]
    return c


def test_load_is_idempotent(conn):
    pf = parse_file(FIXTURE)
    assert db.load(conn, pf) == 219
    assert db.load(conn, pf) == 0
    assert conn.query("select count(*) from pitches")[0][0] in (219, "219")
    assert conn.query("select count(*) from ingest_files")[0][0] in (1, "1")
    assert db.migrate(conn) == []


def test_load_many(conn, tmp_path):
    import pandas as pd

    pf = parse_file(FIXTURE)
    # a second, different game file
    raw = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    raw["GameUID"] = "test-game-2"
    raw["PitchUID"] = raw["PitchUID"] + "-2"
    other = tmp_path / "other.csv"
    raw.to_csv(other, index=False)
    assert db.load_many(conn, [pf, parse_file(other)]) == [219, 219]
    assert db.load_many(conn, [pf]) == [0]


def test_stored_values(conn):
    db.load(conn, parse_file(FIXTURE))
    row = conn.query(
        "select batter_tm_id, season, p_throws, game_date::text from pitches where batter_tm_id = %s limit 1",
        ("1000000000570",),
    )[0]
    assert row == ("1000000000570", "2025-26", "R", "2026-04-24")
    lhh = conn.query("select bool_and(loc_in = -plate_loc_side) from pitches where b_side = 'L' and pitch_tracked")
    assert lhh[0][0] in (True, "t")
    untracked = conn.query("select count(*) from pitches where not pitch_tracked")[0][0]
    assert int(untracked) == 7
    ingest = conn.query("select rows_loaded, jsonb_array_length(warnings) from ingest_files")[0]
    assert [int(x) for x in ingest] == [219, 1]


def test_failed_batch_rolls_back(conn):
    pf = parse_file(FIXTURE)
    with pytest.raises(Exception):
        conn.batch([db.load_statement(pf), ("select * from no_such_table", ())])
    assert int(conn.query("select count(*) from pitches")[0][0]) == 0


def test_placeholder_url_rejected():
    with pytest.raises(SystemExit, match="example value"):
        db.connect("postgresql://user:password@host/matchup?sslmode=require")


def test_neon_hosts_default_to_http():
    c = db.connect("postgresql://u:p@ep-super-field-b49e0om9.c-6.us-east-2.aws.neon.tech/matchup?sslmode=require")
    assert c.transport == "http"
    assert c._endpoint == "https://api.c-6.us-east-2.aws.neon.tech/sql"
    c.close()
