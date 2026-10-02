"""Load into a real Postgres. Set TEST_DATABASE_URL to run (skipped otherwise)."""
import os
from pathlib import Path

import pytest

from matchup import db
from matchup.ingest import parse_file

FIXTURE = Path(__file__).parent / "fixtures" / "20260424-BoglePark-1_SB.csv"
URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture()
def conn():
    with db.connect(URL) as c:
        c.execute("drop schema public cascade; create schema public;")
        db.migrate(c)
        yield c


def test_load_is_idempotent(conn):
    pf = parse_file(FIXTURE)
    assert db.load(conn, pf) == 219
    assert db.load(conn, pf) == 0
    assert conn.execute("select count(*) from pitches").fetchone()[0] == 219
    assert conn.execute("select count(*) from ingest_files").fetchone()[0] == 1
    assert db.migrate(conn) == []


def test_stored_values(conn):
    db.load(conn, parse_file(FIXTURE))
    row = conn.execute(
        "select batter_tm_id, season, p_throws from pitches where batter_tm_id = '1000000000570' limit 1"
    ).fetchone()
    assert row == ("1000000000570", "2025-26", "R")
    lhh = conn.execute("select bool_and(loc_in = -plate_loc_side) from pitches where b_side='L' and pitch_tracked").fetchone()[0]
    assert lhh is True
