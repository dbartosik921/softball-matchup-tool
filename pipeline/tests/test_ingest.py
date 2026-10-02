from pathlib import Path

import numpy as np
import pytest

from matchup.ingest import parse_file, season_for

FIXTURE = Path(__file__).parent / "fixtures" / "20260424-BoglePark-1_SB.csv"


@pytest.fixture(scope="module")
def pf():
    return parse_file(FIXTURE)


def test_game_fields(pf):
    assert pf.game["game_uid"] == "5ca95647-9a33-4467-b411-ccad8e2db357"
    assert pf.game["season"] == "2025-26"
    assert pf.game["league"] == "SEC"
    assert len(pf.pitches) == pf.rows_read == 219


def test_season_flip():
    from datetime import date
    assert season_for(date(2026, 6, 30)) == "2025-26"
    assert season_for(date(2026, 7, 1)) == "2026-27"


def test_ids_stay_text(pf):
    ids = set(pf.pitches["batter_tm_id"])
    assert "1000000000570" in ids  # 13-digit ID
    assert "100000002415" in ids   # 12-digit ID
    assert all(isinstance(i, str) for i in ids)


def test_frames_are_consistent(pf):
    """PlateLocSide, RelSide, HorzBreak and HorzApprAngle share one frame (pitcher view, + toward RHH)."""
    t = pf.pitches[pf.pitches.pitch_tracked]
    dist = 43 - t.extension
    pred = t.rel_side + np.tan(np.radians(t.horz_rel_angle)) * dist + t.horz_break / 12
    assert (pred - t.plate_loc_side).abs().mean() < 0.1
    assert np.corrcoef(t.horz_appr_angle, t.plate_loc_side - t.rel_side)[0, 1] > 0.9


def test_handedness_features(pf):
    p = pf.pitches
    rhh, lhh = p[p.b_side == "R"], p[p.b_side == "L"]
    assert np.allclose(rhh.loc_in.dropna(), rhh.plate_loc_side.dropna())
    assert np.allclose(lhh.loc_in.dropna(), -lhh.plate_loc_side.dropna())
    # all pitchers in this game are RHP, so arm side == raw horizontal break
    assert (p.p_throws == "R").all()
    assert np.allclose(p.hb_arm.dropna(), p.horz_break.dropna())
    assert p.same_side.sum() == len(rhh)


def test_outcome_flags(pf):
    p = pf.pitches
    assert p.is_whiff.sum() == 14
    assert p.is_called_strike.sum() == 42
    assert p.is_bip.sum() == 46
    assert p.is_swing.sum() == 14 + 31 + 2 + 46
    assert p.in_zone.isna().sum() == 7


def test_pa_results(pf):
    counts = pf.pitches.pa_result.value_counts().to_dict()
    assert counts["K"] == 5 and counts["BB"] == 6 and counts["1B"] == 13
    assert sum(counts.values()) == pf.pitches.pa_ending.sum()


def test_untracked_pitches_kept(pf):
    assert (~pf.pitches.pitch_tracked).sum() == 7
    assert any("without full tracking" in w for w in pf.warnings)


def test_damaged_id_recovered_from_same_file(tmp_path):
    import pandas as pd
    raw = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    raw.loc[0, "BatterId"] = "1.00E+11"   # Waits, Addy bats again later with a clean ID
    damaged = tmp_path / "damaged.csv"
    raw.to_csv(damaged, index=False)
    pf = parse_file(damaged)
    assert len(pf.pitches) == 219
    assert pf.pitches.loc[pf.pitches.pitch_uid == raw.loc[0, "PitchUID"], "batter_tm_id"].item() == "1000000000570"
    assert any("1 recovered from this file" in w for w in pf.warnings)


def test_pitch_key(pf):
    k = pf.pitches.pitch_key.dropna()
    assert len(k) == 219 and k.is_unique
    assert k.iloc[0] == "2026-04-24T22:01:45|burnham, payton"
