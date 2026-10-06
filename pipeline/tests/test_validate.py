"""Backtest machinery on the synthetic league."""
import numpy as np
import pytest

from matchup import engine, recency, settings
from matchup import validate as V
from matchup.data import ensure_columns
from tests.synth import make_league


@pytest.fixture(scope="module")
def prep():
    df = ensure_columns(make_league(n_rounds=8))
    return V.prepare(df, V.Config(n_pitchers=10, min_train=150, min_test=80, boot=100), verbose=False)


@pytest.fixture(autouse=True)
def restore_settings():
    yield
    settings.apply(settings.DEFAULTS)


def test_no_future_leaks_into_training(prep):
    assert max(prep.train.game_date) < prep.cfg.split
    for t in prep.tests.values():
        assert min(t.game_date) >= prep.cfg.split


def test_predictions_cover_every_test_pitch(prep):
    pred = V.predict(prep, verbose=False)
    assert len(pred) == sum(len(t) for t in prep.tests.values())
    for m in V.METRIC_ORDER:
        for n in V.PREDICTORS:
            assert pred[f"p_{n}_{m}"].notna().mean() > 0.99, (n, m)
    assert pred["has_history"].mean() > 0.8


def test_scores_and_layers(prep):
    pred, sc = V.run_with(prep, settings.DEFAULTS)
    assert set(sc.metric) == set(V.METRIC_ORDER)
    assert (sc.skill_league == 0).all()
    whiff = sc.set_index("metric").loc["whiff"]
    # Whiff depends on pitch type in the synthetic league: knowing the pitch shape must beat the league rate.
    assert whiff.skill_shape > 0 and whiff.skill_prior > 0
    assert isinstance(V.verdict(whiff), str)


def test_noise_metrics_prefer_heavier_shrinkage(prep):
    """Chase doesn't depend on hitter or pitch in the synthetic league: stronger shrinkage must not hurt."""
    _, light = V.run_with(prep, {**settings.DEFAULTS, "prior_scale": 0.5})
    _, heavy = V.run_with(prep, {**settings.DEFAULTS, "prior_scale": 8.0})
    c_light = light.set_index("metric").loc["chase", "skill_model"]
    c_heavy = heavy.set_index("metric").loc["chase", "skill_model"]
    assert c_heavy > c_light


def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PATH", tmp_path / "settings.json")
    vals = {"bandwidth": 0.6, "half_life_days": float("inf"), "prior_scale": {"whiff": 2.0, "chase": 8.0}}
    settings.save(vals, {"whiff": "adds value"})
    got = settings.load()
    assert engine.BANDWIDTH == 0.6 and np.isinf(recency.HALF_LIFE_DAYS)
    assert engine.prior_strength("chase") == engine.METRICS["chase"][2] * 8.0
    assert engine.prior_strength("rv") == engine.METRICS["rv"][2]
    assert settings.validation() == {"whiff": "adds value"}
    assert got["bandwidth"] == 0.6


def test_verdict_requires_beating_every_simpler_model():
    base = {f"skill_{n}": 1.0 for n in V.PREDICTORS}
    row = {**base, "skill_batter": 3.5, "skill_model": 2.9, "skill_prior": 2.3,
           "model_vs_prior_lo": 0.1, "model_vs_prior_hi": 1.0, "model_vs_batter_lo": -1.2, "model_vs_batter_hi": -0.1,
           "prior_vs_batter_lo": -2, "prior_vs_batter_hi": -0.5}
    assert V.verdict(row) == "hitter's overall rate is the best guide"
    row.update(model_vs_batter_lo=0.05, model_vs_batter_hi=0.4, skill_model=3.9)
    assert V.verdict(row).startswith("validated")
    noise = {**{f"skill_{n}": -0.3 for n in V.PREDICTORS}, "skill_league": 0.0,
             **{k: 0 for k in ("model_vs_prior_lo", "model_vs_prior_hi", "model_vs_batter_lo", "model_vs_batter_hi",
                               "prior_vs_batter_lo", "prior_vs_batter_hi")}}
    assert V.verdict(noise).startswith("mostly noise")


def test_batter_baseline_knows_the_count(prep):
    """The hitter's-own-rate baseline must carry the two-strike shift like the league baseline does."""
    pred, _ = V.run_with(prep, settings.DEFAULTS)
    two = pred[pred.is_two_strike]
    one = pred[~pred.is_two_strike]
    lift_league = two.p_league_chase.mean() - one.p_league_chase.mean()
    lift_batter = two.p_batter_chase.mean() - one.p_batter_chase.mean()
    assert lift_league > 0.05
    assert abs(lift_batter - lift_league) < 0.03


def test_hurt_verdict():
    row = {**{f"skill_{n}": 0.1 for n in V.PREDICTORS}, "skill_shape": 0.14,
           "model_vs_prior_lo": -0.07, "model_vs_prior_hi": -0.03, "model_vs_batter_lo": -0.1,
           "model_vs_batter_hi": 0.2, "prior_vs_batter_lo": 0.01, "prior_vs_batter_hi": 0.3}
    assert "HURTS" in V.verdict(row)


def test_log5_baseline():
    # pitcher at league average -> log5 is just the hitter; all three equal -> unchanged
    assert V._log5(np.array([0.3]), np.array([0.2]), np.array([0.2]), "whiff")[0] == pytest.approx(0.3)
    assert V._log5(np.array([0.3]), np.array([0.3]), np.array([0.2]), "whiff")[0] > 0.3
    assert V._log5(0.02, 0.01, 0.0, "rv") == pytest.approx(0.03)


def test_backtest_reports_log5(prep):
    _, sc = V.run_with(prep, settings.DEFAULTS)
    assert "skill_log5" in sc and "model_vs_log5" in sc
    assert "model-log5" in V.describe(sc, "t")


def test_optional_features_need_a_real_gain():
    assert V._keep_default({0.0: 1.00, 0.5: 1.01}, 0.0) == 0.0     # inside noise: keep it off
    assert V._keep_default({0.0: 1.00, 0.5: 1.10}, 0.0) == 0.5
    assert V._keep_default({"add": 1.0, "odds": 1.5}, "add") == "odds"
