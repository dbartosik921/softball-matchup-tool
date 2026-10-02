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


def _parse_date(raw: str, game_id: str) -> date | None:
    raw = (raw or "").strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return pd.to_datetime(raw[:10] if fmt == "%Y-%m-%d" else raw, format=fmt).date()
        except (ValueError, TypeError):
            pass
    m = re.match(r"^(\d{8})", game_id or "")
    return pd.to_datetime(m.group(1), format="%Y%m%d").date() if m else None


def parse_file(path: str | Path) -> ParsedFile:
    path = Path(path)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    raw = _read_raw(path)
    warnings: list[str] = []

    missing = [c for c in C.REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: not a Trackman pitch file (missing {', '.join(missing)})")

    raw = raw.replace({"": None, "NULL": None, "null": None})
    rows_read = len(raw)
    df = pd.DataFrame(index=raw.index)

    for src, dst in C.TEXT_COLUMNS.items():
        df[dst] = raw[src].str.strip() if src in raw else None
    for src, dst in C.INT_COLUMNS.items():
        df[dst] = pd.to_numeric(raw.get(src), errors="coerce").astype("Int64") if src in raw else pd.NA
    for src, dst in C.FLOAT_COLUMNS.items():
        df[dst] = pd.to_numeric(raw.get(src), errors="coerce") if src in raw else np.nan

    # IDs: text only, rejected if damaged.
    for src, dst in (("PitcherId", "pitcher_tm_id"), ("BatterId", "batter_tm_id"), ("CatcherId", "catcher_tm_id")):
        ids = raw[src].fillna("").str.strip() if src in raw else pd.Series("", index=raw.index)
        ok = ids.str.match(_ID_RE)
        df[dst] = ids.where(ok, None)
        bad = (~ok & (ids != "")).sum()
        if bad and dst != "catcher_tm_id":
            warnings.append(f"{bad} rows with a damaged {src} (e.g. '{ids[~ok & (ids != '')].iloc[0]}') were skipped")

    df["p_throws"] = raw["PitcherThrows"].map(_hand)
    df["b_side"] = raw["BatterSide"].map(_hand)

    # Game-level fields.
    game_id = (raw.get("GameID", pd.Series([""])).dropna().iloc[0] if "GameID" in raw else "") or ""
    gdate = _parse_date(raw["Date"].dropna().iloc[0] if raw["Date"].notna().any() else "", game_id)
    if gdate is None:
        raise ValueError(f"{path.name}: could not read the game date")
    game_uid = raw["GameUID"].dropna().iloc[0] if raw["GameUID"].notna().any() else game_id
    if not game_uid:
        raise ValueError(f"{path.name}: no GameUID or GameID")
    first = lambda c: (raw[c].dropna().iloc[0] if c in raw and raw[c].notna().any() else None)  # noqa: E731
    game = {
        "game_uid": game_uid,
        "game_id": game_id or None,
        "game_date": gdate,
        "season": season_for(gdate),
        "home_team": first("HomeTeam"),
        "away_team": first("AwayTeam"),
        "stadium": first("Stadium"),
        "level": first("Level"),
        "league": first("League"),
    }
    df["game_uid"] = df["game_uid"].fillna(game_uid)
    df["game_date"] = gdate
    df["season"] = game["season"]

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
        warnings.append(f"{dropped} rows skipped ({', '.join(reasons) or 'missing IDs'})")
    df = df[keep].copy()

    dupes = df["pitch_uid"].duplicated()
    if dupes.any():
        warnings.append(f"{int(dupes.sum())} duplicate PitchUIDs dropped")
        df = df[~dupes]

    untracked = int((~df["pitch_tracked"]).sum())
    if untracked:
        warnings.append(f"{untracked} pitches without full tracking (kept for counts/outcomes, excluded from similarity)")

    df = add_features(df)
    return ParsedFile(path.name, sha, game, df.reset_index(drop=True), rows_read, warnings)
