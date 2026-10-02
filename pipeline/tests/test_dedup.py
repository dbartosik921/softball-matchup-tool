from pathlib import Path

import pandas as pd

from matchup.dedup import group_within_run
from matchup.ingest import parse_file

FIXTURE = Path(__file__).parent / "fixtures" / "20260424-BoglePark-1_SB.csv"


def reexport(tmp_path, name, *, new_uids=True, drop=0, shift_seconds=0) -> Path:
    """Same game, different file: new GameUID/PitchUIDs (re-processed export), optionally incomplete."""
    raw = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    if new_uids:
        raw["GameUID"] = f"uid-{name}"
        raw["PitchUID"] = raw["PitchUID"] + f"-{name}"
    if shift_seconds:
        t = pd.to_datetime(raw["UTCDateTime"].str.slice(0, 19)) + pd.Timedelta(seconds=shift_seconds)
        raw["UTCDateTime"] = t.dt.strftime("%Y-%m-%dT%H:%M:%S.0000000Z")
    if drop:
        raw = raw.iloc[drop:]
    out = tmp_path / f"{name}.csv"
    raw.to_csv(out, index=False)
    return out


def test_same_game_different_uids_is_one_game(tmp_path):
    files = [parse_file(FIXTURE), parse_file(reexport(tmp_path, "copy"))]
    keep, dups = group_within_run(files)
    assert len(keep) == 1 and len(dups) == 1


def test_most_complete_copy_is_kept(tmp_path):
    partial = parse_file(reexport(tmp_path, "partial", drop=50))
    full = parse_file(FIXTURE)
    keep, dups = group_within_run([partial, full])
    assert keep == [full]
    assert dups[0].file is partial and dups[0].kept is full


def test_different_game_same_teams_and_date_is_not_a_duplicate(tmp_path):
    """Doubleheader: same teams, same day, different pitch times."""
    game2 = parse_file(reexport(tmp_path, "game2", shift_seconds=3 * 3600))
    keep, dups = group_within_run([parse_file(FIXTURE), game2])
    assert len(keep) == 2 and not dups


def test_same_gameuid_files_grouped(tmp_path):
    same = parse_file(reexport(tmp_path, "same", new_uids=False, drop=10))
    keep, dups = group_within_run([parse_file(FIXTURE), same])
    assert len(keep) == 1 and len(dups) == 1
