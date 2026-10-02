"""Matchup engine on a synthetic league with known answers."""
import numpy as np
import pandas as pd
import pytest

from matchup.arsenal import fit_arsenal
from matchup.calibrate import calibrate
from matchup.data import coerce, ensure_columns
from matchup.engine import _X, _frame, evaluate, kernel
from matchup.features import add_features
from matchup.lineup import expected_pa_by_slot, latest_lineup, roster
from matchup.run import run
from tests.synth import make_league


@pytest.fixture(scope="module")
def league():
    df = ensure_columns(make_league())
    lg, hist = calibrate(df)
    return df, lg, hist


@pytest.fixture(scope="module")
def burnham(league):
    df, lg, hist = league
    a = fit_arsenal(hist[hist.pitcher_name == "Burnham, Payton"], lg)
    aub = sorted(hist[hist.batter_team == "AUB_TIG_SB"].batter_tm_id.unique())
    return a, evaluate(hist, a, lg, aub)


def _name_ids(hist):
    return dict(zip(hist.batter_name, hist.batter_tm_id))


def test_calibration(league):
    _, lg, _ = league
    assert lg.vaa_slope == pytest.approx(1.1, abs=0.1)        # planted in the generator
    assert lg.linear_weights["HR"] > lg.linear_weights["1B"] > lg.linear_weights["K"]
    assert lg.count_values["3-0"] > lg.count_values["0-0"] > lg.count_values["0-2"]
    assert 55 < lg.hard_hit_mph < 70


def test_arsenal_recovers_pitch_types(league, burnham):
    df, lg, hist = league
    a, _ = burnham
    assert sorted(c.label for c in a.clusters) == ["Changeup", "Dropball", "Riseball"]
    for c in a.clusters:
        assert max(c.tag_mix.values()) > 0.9
        assert sum(c.usage[k] for k in ("L",) for c in a.clusters) == pytest.approx(1.0)
    lefties = hist[(hist.p_throws == "L") & (hist.tagged_pitch_type == "Curveball")].pitcher_name.unique()
    assert len(lefties)
    lefty = fit_arsenal(hist[hist.pitcher_name == lefties[0]], lg)
    assert lefty.throws == "L"
    curve = [c for c in lefty.clusters if c.label == "Curveball"]
    assert curve and curve[0].means["hb_arm"] < -4      # glove side, arm-relative, for a lefty too


def test_similarity_is_pitch_type_pure(league, burnham):
    _, lg, hist = league
    a, _ = burnham
    pool = _frame(hist[(hist.p_throws == "R") & (hist.b_side == "L") & hist.pitch_tracked].dropna(subset=["vaa_adj"]), "L")
    cl = _frame(a.pitches, "L")
    # Tags aren't a perfect answer key: a soft dropball genuinely resembles other pitchers' changeups.
    # The rise (distinct shape) must stay pure, and every cluster must be mostly its own type.
    for c in a.clusters:
        w = kernel(_X(pool, lg), _X(cl[cl.cluster == c.cid], lg))
        by_type = pd.Series(w, index=pool.index).groupby(pool.tagged_pitch_type).sum()
        purity = by_type[c.label] / by_type.sum()
        assert purity > (0.95 if c.label == "Riseball" else 0.6)
        assert w.sum() > 3 * (cl.cluster == c.cid).sum() * 0.5   # reaches well beyond her own pitches


def test_planted_tendencies_found(league, burnham):
    _, _, hist = league
    _, res = burnham
    ids = _name_ids(hist)
    d = res.detail[res.detail.split == "all"].set_index(["batter_tm_id", "label"])

    rise = d.loc[(ids["Weak, Rise"], "Riseball")]
    assert rise.raw_whiff > rise.pop_whiff + 0.2
    assert rise.pop_whiff < rise.whiff < rise.raw_whiff            # shrunk, but still clearly high
    assert rise.whiff > d.loc[(ids["Weak, Rise"], "Dropball")].whiff

    change = d.loc[(ids["Weak, Change"], "Changeup")]
    assert change.whiff > change.pop_whiff + 0.05

    drop = d.loc[(ids["Masher, Drop"], "Dropball")]
    assert drop.hard_hit > drop.pop_hard_hit + 0.03

    # The weakness shows up as run value too: her riseball outcomes are worse than similar hitters'.
    assert rise.rv < rise.pop_rv
    assert drop.rv > drop.pop_rv
    # Shape fit separates matchup from talent: negative for the rise-weak hitter, positive for the drop masher.
    b = res.batters.set_index("batter_name")
    assert b.loc["Weak, Rise", "fit100"] < 0 < b.loc["Masher, Drop", "fit100"]


def test_handedness_mirror_gives_same_answers(league, burnham):
    """Flip every hand and every horizontal coordinate: a RHP becomes a LHP, LHH become RHH.
    Every matchup number must be unchanged, which is only true if handedness is handled everywhere."""
    df, _, _ = league
    _, res = burnham
    m = df.copy()
    flip = {"L": "R", "R": "L"}
    m["p_throws"] = m.p_throws.map(flip)
    m["b_side"] = m.b_side.map(flip)
    for c in ("horz_break", "rel_side", "plate_loc_side", "horz_appr_angle"):
        m[c] = -m[c]
    m = ensure_columns(add_features(m))
    lg2, hist2 = calibrate(m)
    a2 = fit_arsenal(hist2[hist2.pitcher_name == "Burnham, Payton"], lg2)
    assert a2.throws == "L"
    aub = sorted(hist2[hist2.batter_team == "AUB_TIG_SB"].batter_tm_id.unique())
    res2 = evaluate(hist2, a2, lg2, aub)
    one = res.batters.set_index("batter_tm_id").sort_index()
    two = res2.batters.set_index("batter_tm_id").sort_index()
    assert (one.side.map(flip) == two.side).all()
    for col in ("whiff", "chase", "called_strike", "hard_hit", "xrv100", "whiff_2k"):
        np.testing.assert_allclose(one[col], two[col], rtol=1e-6, atol=1e-9)


def test_switch_hitter_side_is_opposite_the_pitcher(league):
    _, _, hist = league
    from matchup.engine import batter_side_vs
    sides = hist.groupby("batter_tm_id").b_side.nunique()
    sw = sides[sides == 2].index[0]  # the generator's switch hitters bat from both sides
    assert batter_side_vs(hist, sw, "R") == "L"
    assert batter_side_vs(hist, sw, "L") == "R"


def test_lineup_and_expected_pa(league):
    _, _, hist = league
    slot = expected_pa_by_slot(hist)
    assert len(slot) == 9 and slot[0] > slot[8] and np.all(np.diff(slot) <= 1e-9)
    order, _ = latest_lineup(hist, "AUB_TIG_SB")
    assert len(order) == 9
    assert set(order) <= set(roster(hist, "AUB_TIG_SB").batter_tm_id)


def test_end_to_end_report(league):
    df, _, _ = league
    r = run(df, "Payton Burnham", "aub_tig_sb")
    assert "Burnham, Payton (RHP) vs AUB_TIG_SB" in r.html
    assert "Weak, Rise" in r.html and "Riseball" in r.html
    assert len(r.lineup) == 9
    with pytest.raises(SystemExit, match="Did you mean"):
        run(df, "Payton Burnam", "AUB_TIG_SB")


def test_coerce_http_text():
    raw = pd.DataFrame({"balls": ["1"], "rel_speed": ["64.5"], "is_swing": ["t"], "in_zone": [None],
                        "pitch_tracked": ["f"], "game_date": ["2026-04-24"]})
    c = coerce(raw)
    assert c.balls.iloc[0] == 1 and c.rel_speed.iloc[0] == 64.5
    assert c.is_swing.iloc[0] is True or c.is_swing.iloc[0] == True  # noqa: E712
    assert pd.isna(c.in_zone.iloc[0]) and c.pitch_tracked.iloc[0] == False  # noqa: E712
