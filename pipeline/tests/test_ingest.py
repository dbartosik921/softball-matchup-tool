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


def _variant(tmp_path, name, mutate):
    import pandas as pd
    raw = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    raw = mutate(raw)
    out = tmp_path / name
    raw.to_csv(out, index=False)
    return out


def test_missing_uid_columns_generated(tmp_path):
    f = _variant(tmp_path, "no_uids.csv", lambda r: r.drop(columns=["PitchUID", "GameUID"]))
    pf = parse_file(f)
    assert len(pf.pitches) == 219 and pf.pitches.pitch_uid.is_unique
    assert pf.game["game_uid"] == "20260424-BoglePark-1"  # falls back to GameID
    assert parse_file(f).pitches.pitch_uid.tolist() == pf.pitches.pitch_uid.tolist()  # stable


def test_missing_uid_and_gameid_generated(tmp_path):
    f = _variant(tmp_path, "bare.csv", lambda r: r.drop(columns=["PitchUID", "GameUID", "GameID"]))
    pf = parse_file(f)
    assert pf.game["game_uid"].startswith("gen:2026-04-24|UNI_ARK_SB|MIS_TIG_SB|")
    assert any("generated a game ID" in w for w in pf.warnings)


def test_blank_date_falls_back_to_timestamps(tmp_path):
    def blank(r):
        r["Date"] = ""
        return r
    pf = parse_file(_variant(tmp_path, "nodate.csv", blank))
    assert str(pf.game["game_date"]) == "2026-04-24"


def test_blank_gameid_column_is_fine(tmp_path):
    def blank(r):
        r["GameID"] = ""
        return r
    assert len(parse_file(_variant(tmp_path, "nogameid.csv", blank)).pitches) == 219


def test_empty_and_foreign_files(tmp_path):
    from matchup.ingest import NotTrackmanFile
    empty = _variant(tmp_path, "empty.csv", lambda r: r.iloc[0:0])
    with pytest.raises(ValueError, match="no pitch rows"):
        parse_file(empty)
    roster = tmp_path / "roster.csv"
    roster.write_text("Name,Team\nSmith,ARK\n")
    with pytest.raises(NotTrackmanFile):
        parse_file(roster)


def test_blank_ids_recovered_by_name(tmp_path):
    def blank(r):
        r.loc[r.Batter == "Waits, Addy", "BatterId"] = ""
        r.loc[0, "BatterId"] = "1000000000570"  # one clean appearance left in the file
        return r
    pf = parse_file(_variant(tmp_path, "blankids.csv", blank))
    assert len(pf.pitches) == 219


@pytest.mark.parametrize("name,bp", [
    ("20260414-BoglePark-BP-1_SB_unverified", True), ("20260824-BoglePark-BP-2_SB_unverified", True),
    ("BP 9.12.26", True), ("Ark_BP", True),
    ("20260424-BoglePark-1_SB", False), ("Ark Live ABs 9.11.26", False), ("BPA_Arkansas_03012026", False),
])
def test_batting_practice_detection(name, bp):
    from matchup.ingest import is_batting_practice
    assert is_batting_practice(name) == bp


def test_bp_file_excluded(tmp_path):
    import shutil
    from matchup.ingest import ExcludedFile
    f = tmp_path / "20260414-BoglePark-BP-1_SB_unverified.csv"
    shutil.copy(FIXTURE, f)
    with pytest.raises(ExcludedFile):
        parse_file(f)


def test_game_types(tmp_path):
    assert parse_file(FIXTURE).game["game_type"] == "regular"
    intrasquad = _variant(tmp_path, "intrasquad.csv", lambda r: r.assign(PitcherTeam="UNI_ARK_SB", BatterTeam="UNI_ARK_SB"))
    assert parse_file(intrasquad).game["game_type"] == "fall"
    fall = _variant(tmp_path, "fall.csv", lambda r: r.assign(Date="2026-10-03"))
    pf = parse_file(fall)
    assert pf.game["game_type"] == "fall" and pf.game["season"] == "2026-27"


def test_multi_game_file_split(tmp_path):
    import pandas as pd
    from matchup.ingest import parse_path
    from tests.test_dedup import reexport

    a = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    b = pd.read_csv(reexport(tmp_path, "g2", shift_seconds=4 * 3600), dtype=str, keep_default_na=False)
    combo = tmp_path / "Florida Trackman Data as of 3.9.26.csv"
    pd.concat([a, b]).to_csv(combo, index=False)
    parts = parse_path(combo)
    assert len(parts) == 2
    assert {p.game["game_uid"] for p in parts} == {"5ca95647-9a33-4467-b411-ccad8e2db357", "uid-g2"}
    assert all(len(p.pitches) == 219 for p in parts)
    assert len({p.sha256 for p in parts}) == 2
    with pytest.raises(ValueError, match="contains 2 games"):
        parse_file(combo)


def test_compilation_with_junk_gameuid_splits_by_gameid_and_date(tmp_path):
    import pandas as pd
    from matchup.ingest import parse_path
    from tests.test_dedup import reexport

    a = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    b = pd.read_csv(reexport(tmp_path, "g2", shift_seconds=24 * 3600), dtype=str, keep_default_na=False)
    b["Date"] = "2026-04-25"
    b["GameID"] = "20260425-BoglePark-1"
    combo = pd.concat([a, b])
    combo["GameUID"] = "V3 - Softball"          # junk label in every row
    combo.iloc[:11, combo.columns.get_loc("GameUID")] = ""
    f = tmp_path / "Arkansas Trackman Data as of 3.5.26.csv"
    combo.to_csv(f, index=False)
    parts = parse_path(f)
    assert len(parts) == 2
    assert sorted(len(p.pitches) for p in parts) == [219, 219]
    assert {p.game["game_uid"] for p in parts} == {"20260424-BoglePark-1", "20260425-BoglePark-1"}
    assert {str(p.game["game_date"]) for p in parts} == {"2026-04-24", "2026-04-25"}


def test_blank_uid_rows_join_their_game(tmp_path):
    import pandas as pd
    from matchup.ingest import parse_path
    from tests.test_dedup import reexport

    a = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    a.iloc[:5, a.columns.get_loc("GameUID")] = ""   # a few rows lost their UID
    b = pd.read_csv(reexport(tmp_path, "g2", shift_seconds=4 * 3600), dtype=str, keep_default_na=False)
    b["GameUID"] = "0b0b0b0b-1111-2222-3333-444455556666"
    b["GameID"] = "20260424-BoglePark-2"
    f = tmp_path / "combo.csv"
    pd.concat([a, b]).to_csv(f, index=False)
    parts = parse_path(f)
    assert sorted(len(p.pitches) for p in parts) == [219, 219]
