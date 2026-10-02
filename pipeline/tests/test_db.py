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
    assert db.migrate(c) == ["001_init.sql", "002_dedup.sql", "003_game_type.sql"]
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


def test_duplicate_of_stored_game_skipped_or_replaced(conn, tmp_path):
    from matchup.dedup import match_existing
    from tests.test_dedup import reexport

    partial = parse_file(reexport(tmp_path, "partial", drop=50))
    db.load(conn, partial)
    assert int(conn.query("select count(*) from pitches")[0][0]) == 169

    # A same-size copy: detected as a duplicate of the stored game.
    copy = parse_file(reexport(tmp_path, "copy2", drop=50))
    hit = match_existing(conn, [copy])
    assert hit == {0: ("uid-partial", 169)}

    # The complete copy replaces the partial one; nothing is double counted.
    full = parse_file(FIXTURE)
    hit = match_existing(conn, [full])
    assert hit == {0: ("uid-partial", 169)}
    assert db.replace_game(conn, "uid-partial", full) == 219
    assert int(conn.query("select count(*) from pitches")[0][0]) == 219
    assert int(conn.query("select count(*) from games")[0][0]) == 1

    conn.batch([db.duplicate_statement(copy, full.game["game_uid"])])
    rows = conn.query("select status, duplicate_of from ingest_files where sha256 = %s", (copy.sha256,))
    assert rows == [("duplicate", "uid-copy2")]


def test_sync_end_to_end(conn, tmp_path, capsys, monkeypatch):
    """Folder with an original, a re-processed duplicate and a partial copy: one game, 219 pitches."""
    import shutil

    from matchup import cli
    from tests.test_dedup import reexport

    folder = tmp_path / "logs"
    folder.mkdir()
    shutil.copy(FIXTURE, folder / "original.csv")
    shutil.move(reexport(tmp_path, "dup"), folder / "dup.csv")
    shutil.move(reexport(tmp_path, "part", drop=30), folder / "part.csv")
    monkeypatch.delenv("REGISTRY_DATABASE_URL", raising=False)

    class Args:
        paths = [str(folder)]

    assert cli.cmd_sync(Args, conn) == 0
    assert int(conn.query("select count(*) from pitches")[0][0]) == 219
    assert int(conn.query("select count(*) from ingest_files where status = 'duplicate'")[0][0]) == 2
    # Re-running is a no-op: all three files are known.
    assert cli.cmd_sync(Args, conn) == 0
    out = capsys.readouterr().out
    assert "3 already loaded" in out


def test_game_type_stored(conn, tmp_path):
    import pandas as pd

    db.load(conn, parse_file(FIXTURE))
    raw = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    raw["GameUID"] = "fall-game"
    raw["PitchUID"] = raw["PitchUID"] + "-f"
    raw["Date"] = "2026-09-11"
    raw["UTCDateTime"] = raw["UTCDateTime"].str.replace("2026-04-24", "2026-09-11")
    f = tmp_path / "Ark Live ABs 9.11.26.csv"
    raw.to_csv(f, index=False)
    db.load(conn, parse_file(f))
    rows = sorted(tuple(r) for r in conn.query("select game_uid, game_type, season from games"))
    assert rows == [("5ca95647-9a33-4467-b411-ccad8e2db357", "regular", "2025-26"), ("fall-game", "fall", "2026-27")]


def test_sync_compilation_file_with_overlap(conn, tmp_path, monkeypatch):
    """A season compilation file overlapping single-game files: every game once, nothing lost."""
    import pandas as pd

    from matchup import cli
    from tests.test_dedup import reexport

    folder = tmp_path / "logs"
    folder.mkdir()
    a = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    b = pd.read_csv(reexport(tmp_path, "g2", shift_seconds=4 * 3600), dtype=str, keep_default_na=False)
    a.to_csv(folder / "game1.csv", index=False)                       # game 1 alone
    pd.concat([a, b]).to_csv(folder / "Season as of 3.9.26.csv", index=False)  # game 1 + game 2
    monkeypatch.delenv("REGISTRY_DATABASE_URL", raising=False)

    class Args:
        paths = [str(folder)]

    assert cli.cmd_sync(Args, conn) == 0
    assert int(conn.query("select count(*) from games")[0][0]) == 2
    assert int(conn.query("select count(*) from pitches")[0][0]) == 438
    assert cli.cmd_sync(Args, conn) == 0  # second run: nothing new, nothing doubled
    assert int(conn.query("select count(*) from pitches")[0][0]) == 438


def test_load_pitches_roundtrip(conn, tmp_path, monkeypatch):
    """Download path used by `league` / `matchup`: paging, dtypes, cache."""
    from matchup import data

    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(data, "PAGE", 50)             # force several pages
    pf = parse_file(FIXTURE)
    db.load(conn, pf)
    df = data.load_pitches(conn, verbose=False)
    assert len(df) == 219 and df.pitch_uid.is_unique
    assert df.game_type.unique().tolist() == ["regular"]
    assert df.rel_speed.dtype == float and df.is_swing.dtype == bool
    assert int(df.is_whiff.sum()) == int(pf.pitches.is_whiff.sum())
    assert df.in_zone.isna().sum() == 7
    assert str(df.game_date.iloc[0]) == "2026-04-24"
    again = data.load_pitches(conn, verbose=False)       # served from cache
    assert len(again) == 219
