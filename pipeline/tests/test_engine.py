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


def test_shape_fit_uses_only_validated_components(league, burnham, tmp_path, monkeypatch):
    import json
    from matchup import engine, report, settings
    df, lg, hist = league
    a, _ = burnham
    aub = sorted(hist[hist.batter_team == "AUB_TIG_SB"].batter_tm_id.unique())
    path = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "PATH", path)
    from matchup import recency
    keep = (engine.FIT_COMPONENTS, engine.BANDWIDTH, engine.PRIOR_SCALE, recency.HALF_LIFE_DAYS)
    try:
        verdicts = {"whiff": "validated: hitter-vs-shape history adds value",
                    "called_strike": "validated: hitter-vs-shape history adds value",
                    "chase": "no proven edge over the hitter's overall rate",
                    "hard_hit": "pitch shape adds value; hitter-specific part unproven"}
        path.write_text(json.dumps({**settings.DEFAULTS, "validation": verdicts}))
        settings.load()
        assert engine.FIT_COMPONENTS == ("whiff", "called_strike")
        assert "whiff + called strike" in report._fit_cell(-0.5)
        whiff_cs = evaluate(hist, a, lg, aub).batters.set_index("batter_name")["fit100"]

        verdicts["hard_hit"] = verdicts["whiff"]
        path.write_text(json.dumps({**settings.DEFAULTS, "validation": verdicts}))
        settings.load()
        assert engine.FIT_COMPONENTS == ("whiff", "called_strike", "hard_hit")
        with_hh = evaluate(hist, a, lg, aub).batters.set_index("batter_name")["fit100"]
        # The drop masher's edge is hard contact: adding hard-hit makes her fit more hitter-friendly.
        assert with_hh["Masher, Drop"] > whiff_cs["Masher, Drop"]

        path.write_text(json.dumps({**settings.DEFAULTS, "validation": {k: "mostly noise" for k in verdicts}}))
        settings.load()
        assert engine.FIT_COMPONENTS == ()
        assert evaluate(hist, a, lg, aub).batters["fit100"].isna().all()
        assert "No outcome passed" in report._fit_cell(np.nan)
    finally:
        engine.FIT_COMPONENTS, engine.BANDWIDTH, engine.PRIOR_SCALE, recency.HALF_LIFE_DAYS = keep


def test_prior_options(league, burnham):
    """Defaults reproduce the original additive prior; odds-ratio and similar-hitter borrowing change the
    starting point but keep it a valid rate, and the planted tendencies survive both."""
    from matchup import engine, settings, shape
    _, lg, hist = league
    a, base = burnham
    aub = sorted(hist[hist.batter_team == "AUB_TIG_SB"].batter_tm_id.unique())
    keys = ["batter_tm_id", "side", "split", "cluster"]
    try:
        settings.apply({**settings.DEFAULTS, "prior_combine": "odds", "borrow": 1.0})
        alt = evaluate(hist, a, lg, aub)
        m = base.detail.merge(alt.detail, on=keys, suffixes=("", "_alt"))
        assert len(m) == len(base.detail)
        for k in ("prior_whiff", "prior_chase", "prior_hard_hit"):
            assert m[f"{k}_alt"].between(0, 1).all()
            assert (m[k] - m[f"{k}_alt"]).abs().max() > 1e-4          # it does change something
        d = alt.detail[alt.detail.split == "all"].set_index(["batter_tm_id", "label"])
        ids = _name_ids(hist)
        rise = d.loc[(ids["Weak, Rise"], "Riseball")]
        assert rise.whiff > rise.pop_whiff + 0.05
        # release extension as a feature: runs, and pitches with missing extension aren't dropped
        settings.apply({**settings.DEFAULTS, "extension_weight": 0.5})
        h2 = hist.copy()
        h2.loc[h2.index[::3], "extension"] = np.nan
        ext = evaluate(h2, a, lg, aub)
        assert ext.population.sim_pitches.sum() > 0.5 * base.population.sim_pitches.sum()
    finally:
        settings.apply(settings.DEFAULTS)
    assert engine.PRIOR_COMBINE == "add" and engine.BORROW == 0 and shape.EXTENSION_WEIGHT == 0


def test_combine_math():
    from matchup import engine
    try:
        engine.PRIOR_COMBINE = "odds"
        # a hitter exactly at the base rate leaves the shape rate unchanged; extremes stay inside (0, 1)
        assert engine.combine("whiff", 0.40, 0.20, 0.20) == pytest.approx(0.40)
        assert 0.40 < engine.combine("whiff", 0.40, 0.30, 0.20) < 0.55
        assert engine.combine("whiff", 0.90, 0.60, 0.20) < 1.0
        assert engine.combine("rv", 0.01, 0.03, 0.0) == pytest.approx(0.04)   # run value stays additive
        engine.PRIOR_COMBINE = "add"
        assert engine.combine("whiff", 0.40, 0.30, 0.20) == pytest.approx(0.50)
    finally:
        engine.PRIOR_COMBINE = "add"


def test_pitcher_effect_beyond_shape(league, burnham):
    """A pitcher who gets more whiffs than her pitch shapes explain: the effect finds it (only when on),
    and a pitcher who is exactly as good as her shapes gets ~no effect."""
    from matchup import settings
    _, lg, hist = league
    a, _ = burnham
    aub = sorted(hist[hist.batter_team == "AUB_TIG_SB"].batter_tm_id.unique())
    h = hist.copy()
    contact = h.index[(h.pitcher_tm_id == a.pitcher_id) & h.is_swing & ~h.is_whiff]
    flip = contact[::2]                                   # half her contacted swings become misses
    h.loc[flip, "is_whiff"] = True
    try:
        settings.apply({**settings.DEFAULTS, "pitcher_effect": 1.0})
        on = evaluate(h, a, lg, aub).detail
        plain = evaluate(hist, a, lg, aub).detail
        settings.apply(settings.DEFAULTS)
        off = evaluate(h, a, lg, aub).detail
    finally:
        settings.apply(settings.DEFAULTS)
    assert (off.pitfx_whiff == 0).all()
    allc = on[on.split == "all"]
    assert allc.pitfx_whiff.min() > 0.03                 # every pitch type: she gets more whiffs than its shape
    assert on[on.split == "2k"].pitfx_whiff.mean() > 0   # thinner two-strike samples: shrunk, same direction
    k = ["batter_tm_id", "side", "split", "cluster"]
    m = allc.merge(off, on=k, suffixes=("", "_off"))
    assert (m.prior_whiff > m.prior_whiff_off).all()     # every hitter's starting point vs her moves up
    assert plain.pitfx_whiff.abs().max() < 0.05          # unaltered pitcher: about what her shapes predict
