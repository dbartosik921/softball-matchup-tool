"""Read one Trackman game file into normalized game + pitch records.

Every column is read as text first, so IDs are never turned into floats. IDs that Excel has damaged
(1.00E+11, 100000002415.0) are rejected with a warning instead of being loaded wrong.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .features import add_features
from .ids import Resolver, name_key, recover_ids

_ID_RE = re.compile(C.ID_PATTERN)


@dataclass
class ParsedFile:
    file_name: str
    sha256: str
    game: dict
    pitches: pd.DataFrame
    rows_read: int
    warnings: list[str] = field(default_factory=list)


def season_for(d: date) -> str:
    """July 1 flip, labelled like the player ID registry: 2026-04-24 -> '2025-26'."""
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def _hand(v: str | None) -> str | None:
    v = (v or "").strip().lower()
    return {"right": "R", "r": "R", "left": "L", "l": "L"}.get(v)


def _read_raw(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if suffix in (".xlsx", ".xlsm"):
        sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False)
        return max(sheets.values(), key=len)  # the sheet with the most rows
    raise ValueError(f"Unsupported file type: {path.name}")


class NotTrackmanFile(ValueError):
    """A file in the folder that isn't a Trackman pitch log (scouting report, roster, ...)."""


class ExcludedFile(ValueError):
    """A Trackman file deliberately left out (batting practice)."""


# 'BP' as its own token in the file name or GameID: 20260414-BoglePark-BP-1_SB_unverified.csv
_BP = re.compile(r"(?:^|[-_ ])BP(?:[-_ .\d]|$)")
FALL_MONTHS = range(8, 13)  # Aug-Dec


def is_batting_practice(*names) -> bool:
    return any(n and _BP.search(str(n)) for n in names)


def game_type_for(raw: pd.DataFrame, gdate: date) -> str:
    """'fall' for fall ball and intrasquad (live ABs), else 'regular'."""
    if gdate.month in FALL_MONTHS:
        return "fall"
    if "PitcherTeam" in raw and "BatterTeam" in raw:
        both = raw[["PitcherTeam", "BatterTeam"]].dropna()
        if len(both) and (both["PitcherTeam"] == both["BatterTeam"]).all():
            return "fall"
    return "regular"


def _first(raw: pd.DataFrame, col: str):
    if col not in raw:
        return None
    s = raw[col].dropna()
    s = s[s.astype(str).str.strip() != ""]
    return s.iloc[0] if len(s) else None


def _parse_one_date(v) -> date | None:
    v = ("" if v is None else str(v)).strip()
    if not v:
        return None
    for fmt, text in (("%Y-%m-%d", v[:10]), ("%m/%d/%Y", v.split(" ")[0]), ("%m/%d/%y", v.split(" ")[0]),
                      ("%Y%m%d", v[:8])):
        try:
            d = pd.to_datetime(text, format=fmt)
        except (ValueError, TypeError):
            continue
        if not pd.isna(d):
            return d.date()
    return None


def _game_date(raw: pd.DataFrame) -> date | None:
    """Date column first, then the pitch timestamps, then the GameID prefix (20260424-...)."""
    for col in ("Date", "LocalDateTime", "UTCDateTime", "GameID"):
        d = _parse_one_date(_first(raw, col))
        if d:
            return d
    return None


class EmptyFile(ExcludedFile):
    """A Trackman export with a header and no pitches (common for _unverified exports)."""


def _load(path: Path) -> tuple[pd.DataFrame, str]:
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = _read_raw(path)
    missing = [c for c in C.REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise NotTrackmanFile(f"not a Trackman pitch log (missing {', '.join(missing)})")
    raw = raw.replace({"": None, "NULL": None, "null": None}).dropna(how="all")
    if len(raw) == 0:
        raise EmptyFile("file has no pitch rows")
    return raw, sha


_UID = re.compile(r"^[A-Za-z0-9_:.-]{4,}$")


def _valid_uid(v) -> bool:
    """Trackman GameUIDs are UUIDs; other systems use similar tokens. Edited exports sometimes carry a
    label instead ('V3 - Softball'): anything with spaces isn't an identifier."""
    return v is not None and bool(_UID.match(str(v).strip()))


def _game_groups(raw: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Season compilation files ('Florida Trackman Data as of 3.9.26.csv') hold many games.
    Group by every game identifier present: a real (UUID) GameUID, the GameID and the date. Using all of
    them keeps a junk or blank GameUID from merging different games. A single-game file returns one group."""
    parts = []
    if "GameUID" in raw:
        parts.append(raw["GameUID"].where(raw["GameUID"].map(_valid_uid), ""))
    if "GameID" in raw:
        parts.append(raw["GameID"].fillna(""))
    if "Date" in raw:
        parts.append(raw["Date"].fillna(""))
    if not parts:
        return [("", raw)]
    key = parts[0].astype(str)
    for p in parts[1:]:
        key = key + "|" + p.astype(str)
    if key.nunique() <= 1:
        return [("", raw)]
    # Rows missing the UID but sharing GameID+Date with a UID'd group belong to that game.
    if "GameUID" in raw:
        tail = key.str.split("|", n=1).str[1]
        uid_for_tail = (
            pd.DataFrame({"uid": parts[0], "tail": tail})
            .query("uid != ''").drop_duplicates("tail").set_index("tail")["uid"]
        )
        fill = tail.map(uid_for_tail).fillna("")
        key = key.where(parts[0] != "", fill + "|" + tail)
    out = []
    for k, g in raw.groupby(key, sort=False):
        label = next((x for x in str(k).split("|") if x), "unlabeled")
        out.append((label if len(label) < 40 else label[:8], g))
    return out


def parse_path(path: str | Path, registry: Resolver | None = None) -> list[ParsedFile]:
    """Parse a file into one ParsedFile per game it contains."""
    path = Path(path)
    raw, sha = _load(path)
    groups = _game_groups(raw)
    if len(groups) == 1:
        return [_parse_frame(raw, path, sha, path.name, registry)]
    out = []
    for key, part in groups:
        part_sha = hashlib.sha256(f"{sha}:{key}".encode()).hexdigest()
        try:
            pf = _parse_frame(part, path, part_sha, f"{path.name} [{key}]", registry)
        except ExcludedFile:
            continue
        pf.warnings.insert(0, f"one of {len(groups)} games in a multi-game file")
        out.append(pf)
    return out


def parse_file(path: str | Path, registry: Resolver | None = None) -> ParsedFile:
    """Parse a single-game file. Multi-game files: use parse_path."""
    parts = parse_path(path, registry)
    if len(parts) != 1:
        raise ValueError(f"{Path(path).name} contains {len(parts)} games; use parse_path")
    return parts[0]


def _parse_frame(raw: pd.DataFrame, path: Path, sha: str, display_name: str,
                 registry: Resolver | None) -> ParsedFile:
    warnings: list[str] = []
    raw = raw.reset_index(drop=True)
    rows_read = len(raw)
    df = pd.DataFrame(index=raw.index)

    for src, dst in C.TEXT_COLUMNS.items():
        df[dst] = raw[src].str.strip() if src in raw else None
    for src, dst in C.INT_COLUMNS.items():
        df[dst] = pd.to_numeric(raw[src], errors="coerce").astype("Int64") if src in raw else pd.NA
    for src, dst in C.FLOAT_COLUMNS.items():
        df[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw else np.nan

    df["p_throws"] = raw["PitcherThrows"].map(_hand)
    df["b_side"] = raw["BatterSide"].map(_hand)

    # Game-level fields.
    game_id = _first(raw, "GameID") or ""
    if is_batting_practice(path.stem, game_id):
        raise ExcludedFile("batting practice session (excluded)")
    gdate = _game_date(raw)
    if gdate is None:
        raise ValueError("could not read the game date (Date, timestamps and GameID are all empty or unreadable)")
    uid = _first(raw, "GameUID")
    if uid is not None and not _valid_uid(uid):
        warnings.append(f"GameUID '{uid}' is not a Trackman ID; ignored")
        uid = None
    game_uid = uid or game_id
    if not game_uid:
        # Some exports drop the UID columns. Build a stable one from what identifies the game.
        # No timestamps either: fall back to the file name so two same-day games can't merge.
        stamp = (_first(raw, "UTCDateTime") or _first(raw, "Time") or path.stem)
        game_uid = "gen:" + "|".join(str(x) for x in (gdate, _first(raw, "HomeTeam"), _first(raw, "AwayTeam"), stamp))
        warnings.append("no GameUID/GameID in file; generated a game ID from date, teams and first pitch time")
    game = {
        "game_uid": game_uid,
        "game_id": game_id or None,
        "game_date": gdate,
        "season": season_for(gdate),
        "home_team": _first(raw, "HomeTeam"),
        "away_team": _first(raw, "AwayTeam"),
        "stadium": _first(raw, "Stadium"),
        "level": _first(raw, "Level"),
        "league": _first(raw, "League"),
        "game_type": game_type_for(raw, gdate),
    }
    if df["pitch_uid"].isna().any():
        n_missing = int(df["pitch_uid"].isna().sum())
        seq = df["pitch_no"].astype("string").fillna(pd.Series(range(1, len(df) + 1), index=df.index).astype("string"))
        df["pitch_uid"] = df["pitch_uid"].fillna(f"{game_uid}#" + seq)
        warnings.append(f"{n_missing} pitches had no PitchUID; generated from game ID + pitch number")
    df["game_uid"] = game_uid
    df["game_date"] = gdate
    df["season"] = game["season"]

    # IDs: kept as text. Damaged ones (Excel) are recovered by name + team: same file first, then the
    # player registry. Anything still unresolved is skipped, never guessed.
    rec = recover_ids(raw, game["season"], registry)
    for src, dst in (("PitcherId", "pitcher_tm_id"), ("BatterId", "batter_tm_id"), ("CatcherId", "catcher_tm_id")):
        df[dst] = pd.Series(rec.ids.get(src, [None] * len(raw)), index=raw.index, dtype=object)
    for src in ("PitcherId", "BatterId"):
        n = rec.damaged.get(src, 0)
        if not n:
            continue
        ff, fr = rec.from_file.get(src, 0), rec.from_registry.get(src, 0)
        lost = n - ff - fr
        msg = f"{n} rows with a damaged or missing {src} (e.g. '{rec.example[src] or 'blank'}'): {ff} recovered from this file, {fr} from the registry"
        named = rec.by_name_only.get(src, [])
        if named:
            msg += f" ({len(named)} player(s) matched by name only, team code not in registry: {', '.join(named[:5])}{'...' if len(named) > 5 else ''})"
        if lost:
            names = rec.unresolved.get(src, [])
            msg += f", {lost} unrecoverable ({', '.join(names[:5])}{'...' if len(names) > 5 else ''})"
            if registry is None:
                msg += " [registry not connected]"
        warnings.append(msg)

    # Fingerprint for spotting the same game exported twice under different GameUIDs/PitchUIDs:
    # pitch release time to the second + pitcher name.
    ts = raw["UTCDateTime"] if "UTCDateTime" in raw else None
    if ts is None or ts.isna().all():
        ts = raw.get("Date", pd.Series(index=raw.index, dtype=object)).fillna("") + "T" + raw.get("Time", pd.Series(index=raw.index, dtype=object)).fillna("")
    ts = ts.fillna("").str.replace(" ", "T").str.slice(0, 19)
    pkey = raw["Pitcher"].map(name_key) if "Pitcher" in raw else pd.Series(None, index=raw.index)
    df["pitch_key"] = (ts + "|" + pkey.fillna("")).where((ts.str.len() == 19) & pkey.notna(), None)

    # Tracking quality: required metrics present and no Low-confidence flags.
    conf_bad = pd.Series(False, index=raw.index)
    for c in ("PitchReleaseConfidence", "PitchLocationConfidence", "PitchMovementConfidence"):
        if c in raw:
            conf_bad |= raw[c].isin(C.BAD_PITCH_CONF)
    df["pitch_tracked"] = df[C.TRACKING_REQUIRED].notna().all(axis=1) & ~conf_bad

    # Drop rows we can't attribute.
    keep = (
        df["pitch_uid"].notna() & df["pitcher_tm_id"].notna() & df["batter_tm_id"].notna()
        & df["p_throws"].notna() & df["b_side"].notna() & df["pitch_call"].notna()
        & (df["pitch_call"] != "Undefined")
    )
    dropped = int((~keep).sum())
    if dropped:
        reasons = []
        if df["p_throws"].isna().any() or df["b_side"].isna().any():
            reasons.append("unknown handedness")
        if (df["pitch_call"].isna() | (df["pitch_call"] == "Undefined")).any():
            reasons.append("undefined pitch call")
        warnings.append(f"{dropped} rows skipped ({', '.join(reasons) or 'unrecoverable IDs'})")
    df = df[keep].copy()

    dupes = df["pitch_uid"].duplicated()
    if dupes.any():
        warnings.append(f"{int(dupes.sum())} duplicate PitchUIDs dropped")
        df = df[~dupes]

    untracked = int((~df["pitch_tracked"]).sum())
    if untracked:
        warnings.append(f"{untracked} pitches without full tracking (kept for counts/outcomes, excluded from similarity)")

    df = add_features(df)
    return ParsedFile(display_name, sha, game, df.reset_index(drop=True), rows_read, warnings)
