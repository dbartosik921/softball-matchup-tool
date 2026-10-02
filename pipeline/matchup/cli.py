"""Command line entry point.

  python -m matchup migrate                 # create/upgrade tables
  python -m matchup sync ~/Trackman         # load every new game file in a folder (recursively)
  python -m matchup ingest a.csv b.csv      # load specific files
  python -m matchup check ~/Trackman        # parse only: report problems, write nothing
"""
from __future__ import annotations

import argparse
import hashlib
import re
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
        files = _files(args.paths)
        summary = {"ok": 0, "warn": 0, "fail": 0, "pitches": 0}
        problems: list[tuple[str, str]] = []
        warn_counts: dict[str, int] = {}
        for f in files:
            try:
                pf = parse_file(f)
            except Exception as e:  # noqa: BLE001
                summary["fail"] += 1
                problems.append((f.name, str(e)))
                print(f"FAIL {f.name}: {e}")
                continue
            n = len(pf.pitches)
            summary["pitches"] += n
            damaged = [w for w in pf.warnings if "damaged" in w]
            if n == 0 or damaged:
                status = "FAIL" if n == 0 else "WARN"
                summary["fail" if n == 0 else "warn"] += 1
                problems.append((f.name, f"{n}/{pf.rows_read} pitches usable; " + "; ".join(damaged or pf.warnings)))
            else:
                status = "ok  "
                summary["ok"] += 1
            for w in pf.warnings:
                key = re.sub(r"^\d+ ", "", w.split("(")[0]).replace(" were skipped", "").strip()
                warn_counts[key] = warn_counts.get(key, 0) + 1
            print(f"{status} {f.name}: {n}/{pf.rows_read} pitches" + "".join(f"\n       - {w}" for w in pf.warnings))

        print("\n" + "=" * 70)
        print(f"SUMMARY  {len(files)} files: {summary['ok']} ok, {summary['warn']} with warnings, "
              f"{summary['fail']} unusable  |  {summary['pitches']:,} usable pitches")
        if warn_counts:
            print("Warnings by type (files affected):")
            for k, v in sorted(warn_counts.items(), key=lambda kv: -kv[1]):
                print(f"  {v:>5}  {k}")
        if problems:
            print("Files needing attention:")
            for name, msg in problems:
                print(f"  - {name}: {msg}")
        return 1 if summary["fail"] else 0

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
