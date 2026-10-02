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
# Games per database round trip. Raw CSV size is a rough proxy for the JSON payload size.
BATCH_GAMES = 20
BATCH_BYTES = 6_000_000


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

    conn = db.connect()
    try:
        if args.cmd == "migrate":
            applied = db.migrate(conn)
            print(f"connected via {conn.transport}")
            print("applied: " + (", ".join(applied) or "nothing (up to date)"))
            return 0

        db.migrate(conn)
        seen = db.known_hashes(conn)
        stats = {"loaded": 0, "skipped": 0, "failed": 0, "pitches": 0}
        pending: list = []
        pending_bytes = 0

        def report(pf, n):
            stats["loaded"] += 1
            stats["pitches"] += n
            print(f"+ {pf.file_name}: {n} pitches" + "".join(f"\n    - {w}" for w in pf.warnings))

        def flush():
            nonlocal pending, pending_bytes
            if not pending:
                return
            try:
                for pf, n in zip(pending, db.load_many(conn, pending)):
                    report(pf, n)
            except Exception:  # noqa: BLE001 - retry one by one to find the bad file
                for pf in pending:
                    try:
                        report(pf, db.load(conn, pf))
                    except Exception as e:  # noqa: BLE001
                        stats["failed"] += 1
                        print(f"! {pf.file_name}: {e}", file=sys.stderr)
            pending, pending_bytes = [], 0

        for f in _files(args.paths):
            data = f.read_bytes()
            if hashlib.sha256(data).hexdigest() in seen:
                stats["skipped"] += 1
                continue
            try:
                pf = parse_file(f)
            except Exception as e:  # noqa: BLE001
                stats["failed"] += 1
                print(f"! {f.name}: {e}", file=sys.stderr)
                continue
            pending.append(pf)
            pending_bytes += len(data)
            if len(pending) >= BATCH_GAMES or pending_bytes >= BATCH_BYTES:
                flush()
        flush()
        print(f"done: {stats['loaded']} games loaded ({stats['pitches']} pitches), "
              f"{stats['skipped']} already loaded, {stats['failed']} failed")
        return 1 if stats["failed"] else 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
