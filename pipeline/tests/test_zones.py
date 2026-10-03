"""5x5 OPS zones: cell mapping (pitcher's view) and a planted location effect."""
import numpy as np
import pytest

from matchup.calibrate import calibrate
from matchup.data import ensure_columns
from matchup.engine import Context
from matchup.publish import innings_pitched
from matchup.run import build_arsenal
from matchup.zones import cells, zone_ops
from tests.synth import make_league


def test_cells_pitchers_view():
    # middle-middle, top-left (pitcher's view: negative side = toward the LHH box = pitcher's left), low chase
    x = np.array([0.0, -0.6, 0.6, 0.0, 1.5, np.nan])
    y = np.array([2.25, 2.9, 1.6, 0.8, 4.0, 2.0])
    assert cells(x, y).tolist() == [12, 6, 18, 22, 4, -1]


@pytest.fixture(scope="module")
def planted():
    df = ensure_columns(make_league())
    lg, hist = calibrate(df)
    bid = hist.loc[hist.batter_name == "Masher, Drop", "batter_tm_id"].iloc[0]
    end = hist["pa_ending"] & (hist["batter_tm_id"] == bid)
    hits = end & hist["pa_result"].isin(["1B", "2B", "3B", "HR"])
    outs = end & hist["pa_result"].isin(["OUT", "K"])
    hist.loc[hits, ["plate_loc_side", "plate_loc_height"]] = [0.5, 2.75]    # cell 8: up, pitcher's right
    hist.loc[outs, ["plate_loc_side", "plate_loc_height"]] = [-0.5, 1.75]   # cell 16: down, pitcher's left
    a = build_arsenal(df, hist.loc[hist.pitcher_name == "Burnham, Payton", "pitcher_tm_id"].iloc[0], lg)
    return df, lg, hist, a, bid


def test_planted_location_effect(planted):
    df, lg, hist, a, bid = planted
    side = hist.loc[(hist.batter_tm_id == bid) & (hist.p_throws == a.throws), "b_side"].mode().iloc[0]
    other = hist.loc[(hist.batter_team == "AUB_TIG_SB") & (hist.batter_tm_id != bid), "batter_tm_id"].iloc[0]
    z = zone_ops(Context(hist, lg), a, lg, {bid: side, other: None})
    assert set(z) == {bid}                                   # side None -> skipped
    g = z[bid]
    assert len(g["all"]) == 25 and np.isfinite(g["all"]).all() and (g["all"] > 0).all()
    assert g["all"][8] > g["all"][16] + 0.2                  # her hits up-right, outs down-left
    assert g["pa"][8] > 0 and g["pa"][16] > 0 and g["pa"][0] == 0
    # Burnham has a changeup cluster: a separate grid, weighted to changeup-like pitches
    assert g["ch"] is not None and len(g["ch"]) == 25
    assert any(c.label.lower().startswith("change") for c in a.clusters)


def test_no_changeup_gives_none(planted):
    df, lg, hist, a, bid = planted
    for c in a.clusters:
        if c.label.lower().startswith("change"):
            c.label = "Offspeed"
    side = hist.loc[(hist.batter_tm_id == bid) & (hist.p_throws == a.throws), "b_side"].mode().iloc[0]
    z = zone_ops(Context(hist, lg), a, lg, {bid: side})
    assert z[bid]["ch"] is None and z[bid]["pa_ch"] is None


def test_innings_pitched():
    import pandas as pd
    d = pd.DataFrame({"pitcher_tm_id": ["a"] * 5 + ["b"],
                      "pa_ending": [True, True, True, True, False, True],
                      "pa_result": ["K", "OUT", "OUT", "1B", None, "FC"],
                      "outs_on_play": [0, 1, 2, 0, None, None]})        # a: 1 K + 1 + DP = 4 outs; b: FC fallback
    ip = innings_pitched(d)
    assert ip["a"] == pytest.approx(4 / 3) and ip["b"] == pytest.approx(1 / 3)
