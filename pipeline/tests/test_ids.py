from pathlib import Path

import pandas as pd
import pytest

from matchup.ids import RegistryLookup, classify_id, name_key
from matchup.ingest import parse_file

FIXTURE = Path(__file__).parent / "fixtures" / "20260424-BoglePark-1_SB.csv"


@pytest.mark.parametrize("raw,expected", [
    ("100000002415", "ok"), ("1000000000570", "ok"),
    ("1E+11", "damaged"), ("1.00E+11", "damaged"), ("100000002415.0", "damaged"),
    ("100,000,002,415", "damaged"), ("100000000000", "damaged"), ("1000000000000", "damaged"),
    ("", "missing"), (None, "missing"), ("NULL", "missing"),
])
def test_classify(raw, expected):
    assert classify_id(raw) == expected


def test_name_key_matches_registry_rule():
    assert name_key("  Waits,  Addy ") == "waits, addy"
    assert name_key("Addy Waits") == "waits, addy"


def _fully_damaged(tmp_path) -> Path:
    """Every ID in the file destroyed, like Washington_USF_05162026.csv."""
    raw = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    for c in ("PitcherId", "BatterId", "CatcherId"):
        raw[c] = "1E+11"
    out = tmp_path / "fully_damaged.csv"
    raw.to_csv(out, index=False)
    return out


def _registry_from_fixture() -> RegistryLookup:
    raw = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    lk = RegistryLookup()
    for n, i, t in [("Pitcher", "PitcherId", "PitcherTeam"), ("Batter", "BatterId", "BatterTeam")]:
        for name, pid, team in raw[[n, i, t]].drop_duplicates().itertuples(index=False):
            lk.add(name_key(name), team, "2025-26", pid)
    return lk


def test_fully_damaged_without_registry_loads_nothing(tmp_path):
    pf = parse_file(_fully_damaged(tmp_path))
    assert len(pf.pitches) == 0
    assert any("registry not connected" in w for w in pf.warnings)


def test_fully_damaged_recovered_from_registry(tmp_path):
    clean = parse_file(FIXTURE)
    pf = parse_file(_fully_damaged(tmp_path), registry=_registry_from_fixture())
    assert len(pf.pitches) == 219
    merged = clean.pitches.merge(pf.pitches, on="pitch_uid", suffixes=("", "_r"))
    assert (merged.pitcher_tm_id == merged.pitcher_tm_id_r).all()
    assert (merged.batter_tm_id == merged.batter_tm_id_r).all()
    assert any("219 from the registry" in w for w in pf.warnings)


def test_registry_falls_back_to_any_season():
    lk = RegistryLookup()
    lk.add("waits, addy", "MIS_TIG_SB", "2024-25", "1000000000570")
    assert lk.resolve("waits, addy", "MIS_TIG_SB", "2025-26") == "1000000000570"
    assert lk.resolve("waits, addy", "UNI_ARK_SB", "2025-26") is None  # wrong team: no match


def test_ambiguous_names_are_never_guessed(tmp_path):
    lk = _registry_from_fixture()
    lk.add("waits, addy", "MIS_TIG_SB", "2025-26", "999999999991")  # a second Addy Waits on the team
    pf = parse_file(_fully_damaged(tmp_path), registry=lk)
    assert not (pf.pitches.batter_tm_id == "999999999991").any()
    assert "Waits, Addy" not in set(pf.pitches.batter_name)
    assert any("unrecoverable" in w and "Waits, Addy" in w for w in pf.warnings)


def test_rounded_zero_ids_are_not_trusted(tmp_path):
    """'100000000000' passes a digits-only check but is Excel rounding; must not merge players."""
    raw = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    raw["BatterId"] = "100000000000"
    out = tmp_path / "rounded.csv"
    raw.to_csv(out, index=False)
    pf = parse_file(out, registry=_registry_from_fixture())
    assert "100000000000" not in set(pf.pitches.batter_tm_id)
    assert pf.pitches.batter_tm_id.nunique() > 10
