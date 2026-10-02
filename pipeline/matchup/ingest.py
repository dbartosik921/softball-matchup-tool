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


def _parse_date(raw: str, game_id: str) -> date | None:
    raw = (raw or "").strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return pd.to_datetime(raw[:10] if fmt == "%Y-%m-%d" else raw, format=fmt).date()
        except (ValueError, TypeError):
            pass
    m = re.match(r"^(\d{8})", game_id or "")
    return pd.to_datetime(m.group(1), format="%Y%m%d").date() if m else None


def parse_file(path: str | Path, registry: Resolver | None = None) -> ParsedFile:
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
        msg = f"{n} rows with a damaged {src} (e.g. '{rec.example[src]}'): {ff} recovered from this file, {fr} from the registry"
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
    return ParsedFile(path.name, sha, game, df.reset_index(drop=True), rows_read, warnings)
