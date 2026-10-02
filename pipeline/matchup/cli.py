"""Command line entry point.

  python -m matchup migrate                 # create/upgrade tables
  python -m matchup sync ~/Trackman         # load every new game file in a folder (recursively)
  python -m matchup ingest a.csv b.csv      # load specific files
  python -m matchup check ~/Trackman        # parse only: report problems, write nothing
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from .ingest import parse_file

GAME_FILE_SUFFIXES = {".csv", ".xlsx", ".xlsm"}


def _load_env():
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def _files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        p = p.expanduser()
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in GAME_FILE_SUFFIXES and not f.name.startswith("~$"))
        elif p.exists():
            out.append(p)
        else:
            print(f"! not found: {p}", file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> int:
    _load_env()
    ap = argparse.ArgumentParser(prog="matchup")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate")
    for name in ("sync", "ingest", "check"):
        s = sub.add_parser(name)
        s.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)

    if args.cmd == "check":
        bad = 0
        for f in _files(args.paths):
            try:
                pf = parse_file(f)
                print(f"ok   {f.name}: {len(pf.pitches)}/{pf.rows_read} pitches" + "".join(f"\n       - {w}" for w in pf.warnings))
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FAIL {f.name}: {e}")
        return 1 if bad else 0

    from . import db

    with db.connect() as conn:
        if args.cmd == "migrate":
            applied = db.migrate(conn)
            print("applied: " + (", ".join(applied) or "nothing (up to date)"))
            return 0

        db.migrate(conn)
        seen = db.known_hashes(conn)
        new_games = skipped = failed = 0
        for f in _files(args.paths):
            if hashlib.sha256(f.read_bytes()).hexdigest() in seen:
                skipped += 1
                continue
            try:
                pf = parse_file(f)
                n = db.load(conn, pf, source="folder")
                new_games += 1
                print(f"+ {f.name}: {n} pitches" + "".join(f"\n    - {w}" for w in pf.warnings))
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"! {f.name}: {e}", file=sys.stderr)
        print(f"done: {new_games} loaded, {skipped} already loaded, {failed} failed")
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
