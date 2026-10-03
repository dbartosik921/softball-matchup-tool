"""Publish to a real Postgres (both transports) and read back what the dashboard will read."""
import json
import os

import pytest

from matchup import db
from matchup.calibrate import calibrate
from matchup.data import ensure_columns
from matchup.publish import find_team, publish
from tests.synth import make_league

URL = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture(scope="module")
def league():
    df = ensure_columns(make_league())
    lg, hist = calibrate(df)
    return df, lg, hist


def test_find_team(league):
    df, _, _ = league
    assert find_team(df, "uni_ark_sb") == "UNI_ARK_SB"
    assert find_team(df, "ARK") == "UNI_ARK_SB"          # unique substring
    with pytest.raises(SystemExit, match="Did you mean"):
        find_team(df, "TEAM")                             # ambiguous


@pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")
@pytest.mark.parametrize("transport", ["tcp", "http"])
def test_publish_round_trip(league, transport, request):
    df, lg, hist = league
    reset = db.TcpConn(URL)
    reset.query("drop schema public cascade")
    reset.query("create schema public")
    reset.close()
    if transport == "tcp":
        conn = db.TcpConn(URL)
    else:
        from tests import neon_mock
        srv = neon_mock.start(URL)
        conn = db.HttpConn("postgresql://u:p@ep-fake-123.local/db")
        conn._endpoint = f"http://127.0.0.1:{srv.server_port}/sql"
        request.addfinalizer(srv.shutdown)
    request.addfinalizer(conn.close)
    db.migrate(conn)

    out = publish(conn, df, lg, hist, "UNI_ARK_SB", min_pitches=100, log=lambda m: None)
    assert out["pitchers"] == 17 and not out["skipped"]
    n_hitters = hist["batter_tm_id"].nunique()
    ark_hitters = hist.loc[hist.batter_team == "UNI_ARK_SB", "batter_tm_id"].nunique()
    assert out["matchups"] == 3 * n_hitters + 14 * ark_hitters

    # second publish replaces the first: only one run, and it is complete
    publish(conn, df, lg, hist, "UNI_ARK_SB", min_pitches=100, opponents=False, log=lambda m: None)
    runs = conn.query("select run_id, complete, home_team from pub_runs")
    assert len(runs) == 1 and str(runs[0][1]).lower() in ("true", "t") and runs[0][2] == "UNI_ARK_SB"
    rid = runs[0][0]
    assert int(conn.query("select count(*) from pub_matchups where run_id = %s", (rid,))[0][0]) == 3 * n_hitters

    # what the dashboard reads: Burnham isn't home here, so check a home pitcher vs AUB's lineup
    team = conn.query("select lineup, roster from pub_teams where run_id = %s and team = 'AUB_TIG_SB'", (rid,))[0]
    lineup = team[0] if isinstance(team[0], list) else json.loads(team[0])
    assert len(lineup) == 9
    pid = conn.query("select pitcher_tm_id from pub_pitchers where run_id = %s and is_home limit 1", (rid,))[0][0]
    s, d = conn.query("select summary, detail from pub_matchups where run_id = %s and pitcher_tm_id = %s "
                      "and batter_tm_id = %s", (rid, pid, lineup[0]))[0]
    s = s if isinstance(s, list) else json.loads(s)
    d = d if isinstance(d, list) else json.loads(d)
    meta = conn.query("select meta from pub_runs where run_id = %s", (rid,))[0][0]
    meta = meta if isinstance(meta, dict) else json.loads(meta)
    s = dict(zip(meta["summary_keys"], s))
    d = [dict(zip(meta["detail_keys"], x)) for x in d]
    assert 0 <= s["score"] <= 100 and "fit100" in s and "whiff_2k" in s
    assert {x["split"] for x in d} == {"all", "2k"} and all("pop_whiff" in x for x in d)
    z, ip = conn.query("select m.zones, p.ip from pub_matchups m join pub_pitchers p using (run_id, pitcher_tm_id) "
                       "where m.run_id = %s and m.pitcher_tm_id = %s and m.batter_tm_id = %s", (rid, pid, lineup[0]))[0]
    z = z if isinstance(z, list) else json.loads(z)
    assert len(z) == 4 and len(z[0]) == 25 and len(z[1]) == 25 and float(ip) > 0
    roster = team[1] if isinstance(team[1], list) else json.loads(team[1])
    assert all("ab" in r for r in roster) and any(r["ab"] > 0 for r in roster)


@pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")
def test_plan_pitches_and_append(league):
    df, lg, hist = league
    reset = db.TcpConn(URL)
    reset.query("drop schema public cascade")
    reset.query("create schema public")
    reset.close()
    conn = db.TcpConn(URL)
    db.migrate(conn)
    # full publish with a high bar: opponent pitchers are left out ...
    out = publish(conn, df, lg, hist, "UNI_ARK_SB", min_pitches=100_000, log=lambda m: None)
    assert out["pitchers"] == 3
    rid = out["run_id"]
    # ... then --opponent AUB --append adds AUB's staff to the same run, nobody else
    add = publish(conn, df, lg, hist, "UNI_ARK_SB", min_pitches=100_000, focus=["AUB"], append=True, log=lambda m: None)
    assert add["run_id"] == rid and add["pitchers"] == 3
    teams = {r[0] for r in conn.query("select team from pub_pitchers where run_id = %s and not is_home", (rid,))}
    assert teams == {"AUB_TIG_SB"}
    # appending again adds nothing
    assert publish(conn, df, lg, hist, "UNI_ARK_SB", min_pitches=100_000, focus=["AUB"], append=True,
                   log=lambda m: None)["pitchers"] == 0
    # plan pitches are her cluster ids
    meta = conn.query("select meta from pub_runs where run_id = %s", (rid,))[0][0]
    meta = meta if isinstance(meta, dict) else json.loads(meta)
    k = meta["summary_keys"]
    rows = conn.query("select m.summary, p.arsenal from pub_matchups m join pub_pitchers p using (run_id, pitcher_tm_id) "
                      "where m.run_id = %s and p.is_home", (rid,))
    for s, ars in rows[:20]:
        s = s if isinstance(s, list) else json.loads(s)
        ars = ars if isinstance(ars, list) else json.loads(ars)
        cids = {c["cid"] for c in ars}
        assert s[k.index("attack")] in cids and s[k.index("putaway")] in cids
        assert all({"LFP", "RBH", "L2K"} <= set(c["usage"]) for c in ars)
    conn.close()
