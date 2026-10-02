"""Player ID hygiene and recovery.

Excel damages Trackman IDs when a CSV is opened and re-saved: 100000002415 becomes '1E+11',
'1.00E+11', '100000002415.0', or a rounded '100000000000'. The real digits are gone from that file,
so a damaged ID is recovered by WHO the player is (name + team), in this order:

  1. the same file: the same player in another row or role with a clean ID
     (a pitcher who also bats, a catcher who also bats);
  2. the player ID registry: name (any name the player has used) + team + season,
     then name + team in any season.

Only an unambiguous match is used. Two candidates (two players with the same name on one team)
means the rows are skipped, never guessed. Mirrors player-id-db/src/lib/ids.ts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

_CLEAN_ID = re.compile(r"^\d{6,15}$")
_EXCEL_DAMAGE = re.compile(r"[eE][+-]?\d|[.,]")
# Excel rounding to a few significant figures leaves a long tail of zeros (100000000000).
_SUSPICIOUS = re.compile(r"0{6,}$")

ROLE_COLUMNS = {
    # role: (name column, id column, team column)
    "pitcher": ("Pitcher", "PitcherId", "PitcherTeam"),
    "batter": ("Batter", "BatterId", "BatterTeam"),
    "catcher": ("Catcher", "CatcherId", "CatcherTeam"),
}


def classify_id(raw) -> str:
    v = ("" if raw is None else str(raw)).strip()
    if not v or v.lower() == "null":
        return "missing"
    if _EXCEL_DAMAGE.search(v) or not _CLEAN_ID.match(v):
        return "damaged"
    if _SUSPICIOUS.search(v):
        return "damaged"
    return "ok"


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def name_key(raw_name) -> str | None:
    """'Last, First' -> 'last, first' (same rule as the registry's generated name_key column)."""
    v = _clean("" if raw_name is None else str(raw_name))
    if not v:
        return None
    if "," in v:
        last, first = v.split(",", 1)
    else:
        parts = v.split(" ")
        last, first = parts[-1], " ".join(parts[:-1])
    return f"{_clean(last).lower()}, {_clean(first).lower()}"


def team_key(raw_team) -> str:
    return re.sub(r"[^A-Z0-9_]", "_", _clean("" if raw_team is None else str(raw_team)).upper())


class Resolver(Protocol):
    def resolve(self, nkey: str, team: str, season: str) -> str | None: ...


@dataclass
class RegistryLookup:
    """In-memory index of the registry: (name_key, team, season) and (name_key, team) -> Trackman IDs."""

    by_season: dict[tuple[str, str, str], set[str]] = field(default_factory=dict)
    by_team: dict[tuple[str, str], set[str]] = field(default_factory=dict)

    def add(self, nkey: str, team: str | None, season: str | None, tm_id: str):
        if not nkey or not tm_id or classify_id(tm_id) != "ok":
            return
        if team:
            self.by_team.setdefault((nkey, team), set()).add(tm_id)
            if season:
                self.by_season.setdefault((nkey, team, season), set()).add(tm_id)

    def resolve(self, nkey: str, team: str, season: str) -> str | None:
        for ids in (self.by_season.get((nkey, team, season)), self.by_team.get((nkey, team))):
            if ids:
                return next(iter(ids)) if len(ids) == 1 else None  # ambiguous: never guess
        return None

    def __len__(self):
        return len(self.by_team)


REGISTRY_SQL = """
select n.name_key, s.external_id, ps.season, ps.team_code
from player_names n
join player_source_ids s on s.player_id = n.player_id and s.source = 'trackman'
left join player_seasons ps on ps.player_id = n.player_id
"""


def load_registry(url: str) -> RegistryLookup:
    from . import db

    conn = db.connect(url)
    try:
        rows = conn.query(REGISTRY_SQL)
    finally:
        conn.close()
    lk = RegistryLookup()
    for nkey, tm_id, season, team in rows:
        lk.add(nkey, team, season, tm_id)
    return lk


@dataclass
class RecoveryResult:
    ids: dict[str, list]                 # id column -> recovered values per row (None = unusable)
    damaged: dict[str, int]              # id column -> rows that were damaged
    from_file: dict[str, int]            # id column -> rows recovered from this file
    from_registry: dict[str, int]        # id column -> rows recovered from the registry
    unresolved: dict[str, list[str]]     # id column -> names that could not be resolved
    example: dict[str, str]              # id column -> an example damaged value


def recover_ids(raw, season: str, registry: Resolver | None = None) -> RecoveryResult:
    """raw: the file as a DataFrame of strings. Returns clean IDs for every role column."""
    present = {r: cols for r, cols in ROLE_COLUMNS.items() if cols[1] in raw.columns}

    # 1. Clean IDs seen anywhere in this file, by (name, team), across all roles.
    in_file: dict[tuple[str, str], set[str]] = {}
    status: dict[str, list[str]] = {}
    for role, (ncol, icol, tcol) in present.items():
        ids = raw[icol].fillna("").astype(str).str.strip()
        st = [classify_id(v) for v in ids]
        status[icol] = st
        names = raw[ncol] if ncol in raw.columns else [None] * len(raw)
        teams = raw[tcol] if tcol in raw.columns else [None] * len(raw)
        for v, s, n, t in zip(ids, st, names, teams):
            if s == "ok" and (nk := name_key(n)):
                in_file.setdefault((nk, team_key(t)), set()).add(v)

    res = RecoveryResult({}, {}, {}, {}, {}, {})
    cache: dict[tuple[str, str], tuple[str | None, str]] = {}
    for role, (ncol, icol, tcol) in present.items():
        ids = raw[icol].fillna("").astype(str).str.strip().tolist()
        names = raw[ncol].tolist() if ncol in raw.columns else [None] * len(raw)
        teams = raw[tcol].tolist() if tcol in raw.columns else [None] * len(raw)
        out, dmg, ff, fr, unresolved = [], 0, 0, 0, set()
        for v, s, n, t in zip(ids, status[icol], names, teams):
            if s == "ok":
                out.append(v)
                continue
            if s == "missing":
                out.append(None)
                continue
            dmg += 1
            res.example.setdefault(icol, v)
            nk, tk = name_key(n), team_key(t)
            if not nk:
                out.append(None)
                continue
            if (nk, tk) not in cache:
                cands = in_file.get((nk, tk), set())
                if len(cands) == 1:
                    cache[(nk, tk)] = (next(iter(cands)), "file")
                elif len(cands) == 0 and registry is not None:
                    hit = registry.resolve(nk, tk, season)
                    cache[(nk, tk)] = (hit, "registry") if hit else (None, "")
                else:
                    cache[(nk, tk)] = (None, "")
            rid, how = cache[(nk, tk)]
            out.append(rid)
            if rid is None:
                unresolved.add(_clean(str(n)))
            elif how == "file":
                ff += 1
            else:
                fr += 1
        res.ids[icol] = out
        if dmg:
            res.damaged[icol], res.from_file[icol], res.from_registry[icol] = dmg, ff, fr
            res.unresolved[icol] = sorted(unresolved)
    return res
