"""Command line entry point.

  python -m matchup migrate                 # create/upgrade tables
  python -m matchup check ~/Trackman        # parse only: report problems and duplicates, write nothing
  python -m matchup sync ~/Trackman         # load every new game file (recursively)
  python -m matchup ingest a.csv b.csv      # same as sync, for specific files

Damaged player IDs are recovered from the player registry when REGISTRY_DATABASE_URL is set.
The same game exported more than once is loaded once (the most complete copy).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

from .ingest import NotTrackmanFile, ParsedFile, parse_file

GAME_FILE_SUFFIXES = {".csv", ".xlsx", ".xlsm"}
# Games per database round trip, and a cap on the JSON payload per request.
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


def _registry():
    url = os.environ.get("REGISTRY_DATABASE_URL")
    if not url or "@host/" in url:
        print("registry: not configured (REGISTRY_DATABASE_URL) - damaged IDs can only be recovered from the same file")
        return None
    from .ids import load_registry

    try:
        lk = load_registry(url)
    except Exception as e:  # noqa: BLE001
        print(f"registry: could not connect ({e}) - continuing without it")
        return None
    print(f"registry: {len(lk):,} player/team names loaded for ID recovery")
    return lk


def _parse_all(files: list[Path], registry, skip_hashes: set[str] = frozenset()):
    parsed: list[ParsedFile] = []
    failures: list[tuple[str, str]] = []
    ignored: list[str] = []
    skipped = 0
    for f in files:
        if skip_hashes and hashlib.sha256(f.read_bytes()).hexdigest() in skip_hashes:
            skipped += 1
            continue
        try:
            parsed.append(parse_file(f, registry))
        except NotTrackmanFile:
            ignored.append(f.name)
        except Exception as e:  # noqa: BLE001
            failures.append((f.name, f"{type(e).__name__}: {e}" if not isinstance(e, ValueError) else str(e)))
    return parsed, failures, ignored, skipped


def _warn_key(w: str) -> str:
    w = w.split("(")[0].split(":")[0]
    return re.sub(r"^\d+ ", "", w).replace(" were skipped", "").strip()


def cmd_check(args) -> int:
    from .dedup import group_within_run

    files = _files(args.paths)
    registry = _registry()
    parsed, failures, ignored, _ = _parse_all(files, registry)
    keep, dups = group_within_run(parsed)
    dup_names = {d.file.file_name: d.kept.file_name for d in dups}

    summary = {"ok": 0, "warn": 0, "fail": len(failures)}
    problems = list(failures)
    warn_counts: dict[str, int] = {}
    for n, m in failures:
        print(f"FAIL {n}: {m}")
    for pf in parsed:
        n = len(pf.pitches)
        lost = [w for w in pf.warnings if "unrecoverable" in w]
        if pf.file_name in dup_names:
            status = "DUP "
        elif n == 0:
            status = "FAIL"
            summary["fail"] += 1
            problems.append((pf.file_name, f"0/{pf.rows_read} pitches usable; " + "; ".join(pf.warnings)))
        elif lost:
            status = "WARN"
            summary["warn"] += 1
            problems.append((pf.file_name, f"{n}/{pf.rows_read} pitches usable; " + "; ".join(lost)))
        else:
            status = "ok  "
            summary["ok"] += 1
        for w in pf.warnings:
            warn_counts[_warn_key(w)] = warn_counts.get(_warn_key(w), 0) + 1
        extra = f"  [duplicate of {dup_names[pf.file_name]}]" if pf.file_name in dup_names else ""
        print(f"{status} {pf.file_name}: {n}/{pf.rows_read} pitches{extra}" + "".join(f"\n       - {w}" for w in pf.warnings))

    usable = sum(len(pf.pitches) for pf in keep)
    print("\n" + "=" * 70)
    print(f"SUMMARY  {len(files)} files -> {len(keep)} unique games, {len(dups)} duplicate files  |  {usable:,} usable pitches")
    print(f"         {summary['ok']} ok, {summary['warn']} with unrecoverable rows, {summary['fail']} unusable, "
          f"{len(ignored)} ignored (not Trackman pitch logs)")
    if warn_counts:
        print("Warnings by type (files affected):")
        for k, v in sorted(warn_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {v:>5}  {k}")
    if dups:
        print("Duplicate files (same game; the most complete copy is kept):")
        for d in dups:
            print(f"  - {d.file.file_name}  ->  kept {d.kept.file_name}")
    if ignored:
        print("Ignored (not Trackman pitch logs):")
        for name in ignored:
            print(f"  - {name}")
    if problems:
        print("Files needing attention:")
        for name, msg in problems:
            print(f"  - {name}: {msg}")
    return 1 if summary["fail"] else 0


def cmd_sync(args, conn) -> int:
    from . import db
    from .dedup import group_within_run, match_existing

    db.migrate(conn)
    registry = _registry()
    parsed, failures, ignored, skipped = _parse_all(_files(args.paths), registry, db.known_hashes(conn))
    if ignored:
        print(f"ignored {len(ignored)} files that aren't Trackman pitch logs: {', '.join(ignored[:10])}{'...' if len(ignored) > 10 else ''}")
    for name, msg in failures:
        print(f"! {name}: {msg}", file=sys.stderr)

    keep, dups = group_within_run(parsed)
    empty = [pf for pf in keep if not len(pf.pitches)]
    for pf in empty:
        print(f"! {pf.file_name}: no usable pitches; " + "; ".join(pf.warnings), file=sys.stderr)
    keep = [pf for pf in keep if len(pf.pitches)]
    existing = match_existing(conn, keep)

    stats = {"loaded": 0, "replaced": 0, "dup": 0, "failed": len(failures) + len(empty), "pitches": 0}
    to_load: list[ParsedFile] = []
    dup_records: list = []  # (file, kept game_uid)
    for i, pf in enumerate(keep):
        if i in existing:
            old_uid, old_n = existing[i]
            if len(pf.pitches) > old_n:
                try:
                    n = db.replace_game(conn, old_uid, pf)
                    stats["replaced"] += 1
                    stats["pitches"] += n
                    print(f"~ {pf.file_name}: replaced a less complete copy of this game ({old_n} -> {n} pitches)")
                except Exception as e:  # noqa: BLE001
                    stats["failed"] += 1
                    print(f"! {pf.file_name}: {e}", file=sys.stderr)
            else:
                stats["dup"] += 1
                dup_records.append((pf, old_uid))
                print(f"= {pf.file_name}: this game is already loaded from another file, skipped")
            continue
        to_load.append(pf)

    def report(pf, n):
        stats["loaded"] += 1
        stats["pitches"] += n
        print(f"+ {pf.file_name}: {n} pitches" + "".join(f"\n    - {w}" for w in pf.warnings))

    def flush(batch):
        if not batch:
            return
        try:
            for pf, n in zip(batch, db.load_many(conn, batch)):
                report(pf, n)
        except Exception:  # noqa: BLE001 - retry one by one to find the bad file
            for pf in batch:
                try:
                    report(pf, db.load(conn, pf))
                except Exception as e:  # noqa: BLE001
                    stats["failed"] += 1
                    print(f"! {pf.file_name}: {e}", file=sys.stderr)

    batch, size = [], 0
    for pf in to_load:
        batch.append(pf)
        size += len(pf.pitches) * 900  # ~bytes of JSON per pitch
        if len(batch) >= BATCH_GAMES or size >= BATCH_BYTES:
            flush(batch)
            batch, size = [], 0
    flush(batch)

    for d in dups:
        stats["dup"] += 1
        dup_records.append((d.file, d.kept.game["game_uid"]))
        print(f"= {d.file.file_name}: duplicate of {d.kept.file_name}, skipped")
    if dup_records:
        try:
            conn.batch([db.duplicate_statement(pf, uid) for pf, uid in dup_records])
        except Exception as e:  # noqa: BLE001
            print(f"(could not record duplicate files: {e})", file=sys.stderr)

    print(f"done: {stats['loaded']} games loaded, {stats['replaced']} replaced with more complete copies, "
          f"{stats['pitches']:,} pitches | {stats['dup']} duplicate files skipped, {skipped} already loaded, "
          f"{stats['failed']} failed")
    return 1 if stats["failed"] else 0


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
        return cmd_check(args)

    from . import db

    conn = db.connect()
    try:
        if args.cmd == "migrate":
            applied = db.migrate(conn)
            print(f"connected via {conn.transport}")
            print("applied: " + (", ".join(applied) or "nothing (up to date)"))
            return 0
        return cmd_sync(args, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
