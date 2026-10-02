"""Duplicate-game detection.

The same game often arrives more than once: exported by both teams, downloaded twice, or re-saved
under a new file name. When the copies share Trackman's GameUID/PitchUID, the database already
ignores repeated pitches. When they don't (re-processed exports), the pitches would be counted twice
and every matchup stat for those players would be inflated.

Each pitch gets a fingerprint: release time to the second + pitcher name. Two files whose fingerprints
overlap by at least OVERLAP_THRESHOLD (of the smaller file) are the same game. Doubleheaders don't
collide because their pitch times differ.

Policy: keep the most complete copy (most usable pitches, then most tracked pitches). If a more complete
copy of a game already in the database arrives later, it replaces the stored one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .ingest import ParsedFile

OVERLAP_THRESHOLD = 0.5


def keyset(pf: ParsedFile) -> set[str]:
    return set(pf.pitches["pitch_key"].dropna())


def quality(pf: ParsedFile) -> tuple[int, int]:
    return len(pf.pitches), int(pf.pitches["pitch_tracked"].sum())


def overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


@dataclass
class Duplicate:
    file: ParsedFile
    kept: ParsedFile | None      # the copy kept from this run, if the duplicate is within the run
    existing_game_uid: str | None = None


def group_within_run(files: list[ParsedFile]) -> tuple[list[ParsedFile], list[Duplicate]]:
    """Return (files to keep, duplicates) for one batch of new files."""
    keys = [keyset(pf) for pf in files]
    parent = list(range(len(files)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_key: dict[str, set[int]] = {}
    for i, ks in enumerate(keys):
        for k in ks:
            by_key.setdefault(k, set()).add(i)
    for i, ks in enumerate(keys):
        candidates = set().union(*(by_key[k] for k in ks)) - {i} if ks else set()
        for j in candidates:
            if find(i) != find(j) and overlap(keys[i], keys[j]) >= OVERLAP_THRESHOLD:
                parent[find(j)] = find(i)
        # same GameUID is the same game even without timestamps
    uid_first: dict[str, int] = {}
    for i, pf in enumerate(files):
        uid = pf.game["game_uid"]
        if uid in uid_first and find(uid_first[uid]) != find(i):
            parent[find(i)] = find(uid_first[uid])
        uid_first.setdefault(uid, i)

    groups: dict[int, list[int]] = {}
    for i in range(len(files)):
        groups.setdefault(find(i), []).append(i)
    keep, dups = [], []
    for members in groups.values():
        best = max(members, key=lambda i: (quality(files[i]), -i))
        keep.append(files[best])
        dups += [Duplicate(files[i], files[best]) for i in members if i != best]
    return keep, dups


EXISTING_SQL = """
select game_uid, pitch_key from pitches
where pitch_key is not null
  and game_date in (select jsonb_array_elements_text(%s::jsonb)::date)
"""


def match_existing(conn, files: list[ParsedFile]) -> dict[int, tuple[str, int]]:
    """For each file (by index) that duplicates a game already stored under a different GameUID:
    (stored game_uid, stored fingerprinted pitch count)."""
    dates = sorted({pf.game["game_date"].isoformat() for pf in files})
    if not dates:
        return {}
    stored: dict[str, set[str]] = {}
    for uid, key in conn.query(EXISTING_SQL, (json.dumps(dates),)):
        stored.setdefault(uid, set()).add(key)
    out = {}
    for i, pf in enumerate(files):
        ks = keyset(pf)
        best = None
        for uid, sk in stored.items():
            if uid == pf.game["game_uid"]:
                continue
            ov = overlap(ks, sk)
            if ov >= OVERLAP_THRESHOLD and (best is None or ov > best[2]):
                best = (uid, len(sk), ov)
        if best:
            out[i] = (best[0], best[1])
    return out
