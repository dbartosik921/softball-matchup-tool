"""Command line entry point.

  python -m matchup migrate                 # create/upgrade tables
  python -m matchup check ~/Trackman        # parse only: report problems and duplicates, write nothing
  python -m matchup sync ~/Trackman         # load every new game file (recursively)
  python -m matchup ingest a.csv b.csv      # same as sync, for specific files
  python -m matchup league                  # league calibration: run values, hard-hit line, baselines
  python -m matchup matchup --pitcher "Burnham, Payton" --team AUB_TIG_SB   # HTML matchup report
  python -m matchup backtest [--tune] [--apply]   # does the model predict later games? tune its settings

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

from .ingest import EmptyFile, ExcludedFile, NotTrackmanFile, ParsedFile, parse_path

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
            parts = parse_path(f, registry)
            parsed += [pf for pf in parts if pf.sha256 not in skip_hashes]
            skipped += sum(1 for pf in parts if pf.sha256 in skip_hashes)
        except (NotTrackmanFile, ExcludedFile) as e:
            why = "empty" if isinstance(e, EmptyFile) else "batting practice" if isinstance(e, ExcludedFile) else "not a pitch log"
            ignored.append(f"{f.name} ({why})")
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
        tag = "  [fall]" if pf.game["game_type"] == "fall" else ""
        print(f"{status} {pf.file_name}: {n}/{pf.rows_read} pitches{tag}{extra}" + "".join(f"\n       - {w}" for w in pf.warnings))

    usable = sum(len(pf.pitches) for pf in keep)
    print("\n" + "=" * 70)
    print(f"SUMMARY  {len(files)} files -> {len(keep)} unique games, {len(dups)} duplicate files  |  {usable:,} usable pitches")
    print(f"         {summary['ok']} ok, {summary['warn']} with unrecoverable rows, {summary['fail']} unusable, "
          f"{len(ignored)} ignored (empty, batting practice, or not pitch logs)")
    multi = len({pf.file_name.split(" [")[0] for pf in parsed if " [" in pf.file_name})
    if multi:
        print(f"         ({multi} multi-game files were split into individual games)")

    fall = sum(1 for pf in keep if pf.game["game_type"] == "fall")
    print(f"         {len(keep) - fall} regular-season games, {fall} fall/intrasquad sessions")
    if warn_counts:
        print("Warnings by type (files affected):")
        for k, v in sorted(warn_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {v:>5}  {k}")
    if dups:
        print("Duplicate files (same game; the most complete copy is kept):")
        for d in dups:
            print(f"  - {d.file.file_name}  ->  kept {d.kept.file_name}")
    if ignored:
        print("Ignored (empty, batting practice, or not Trackman pitch logs):")
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
        print(f"ignored {len(ignored)} files (empty, batting practice, or not Trackman pitch logs): {', '.join(ignored[:10])}{'...' if len(ignored) > 10 else ''}")
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


def cmd_backtest(args, conn) -> int:
    import time
    from datetime import date

    from . import settings
    from . import validate as V
    from .data import load_pitches

    current = settings.load()
    df = load_pitches(conn, use_cache=not args.refresh)
    cfg = V.Config(n_pitchers=args.pitchers, split=date.fromisoformat(args.split) if args.split else None)
    t0 = time.time()
    prep = V.prepare(df, cfg)
    out_dir = Path(__file__).resolve().parents[1] / "reports"
    out_dir.mkdir(exist_ok=True)
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    if args.tune:
        best, trials = V.tune(prep, current, log=log)
        pd_ = __import__("pandas")
        pd_.DataFrame(trials).to_csv(out_dir / "backtest_trials.csv", index=False)
        log(f"best settings: {V.json_safe(best)}")
        values = best
    else:
        values = current
    pred, sc = V.run_with(prep, values, verbose=True)
    base = V.run_with(prep, dict(settings.DEFAULTS))[1] if args.tune else None
    settings.apply(values)
    log(V.describe(sc, f"\nAll hitters (settings: {V.json_safe(values)})"))
    hist_only = V.score(pred, prep.league.hard_hit_mph, boot=cfg.boot, history_only=True)
    log(V.describe(hist_only, "\nHitters with history vs that pitcher hand and side"))
    if base is not None:
        log(f"\nobjective: defaults {V.objective(base):+.3f} -> tuned {V.objective(sc):+.3f}")
    verdicts = {r["metric"]: V.verdict(r) for _, r in sc.iterrows()}
    log(f"\n{len(prep.arsenals)} pitchers, {len(pred):,} test pitches, {time.time() - t0:.0f}s")
    path = out_dir / f"backtest_{date.today():%Y%m%d}.txt"
    path.write_text("\n".join(lines))
    print(f"saved {path}")
    if args.apply:
        saved = settings.save(values, verdicts)
        print(f"applied: {saved} (reports use these settings from now on)")
    elif args.tune:
        print("not applied: re-run with --tune --apply to use these settings")
    return 0


def cmd_model(args, conn) -> int:
    from . import settings
    from .calibrate import calibrate, describe
    from .data import load_pitches

    settings.load()

    df = load_pitches(conn, use_cache=not args.refresh)
    regular = df[df["game_type"].fillna("regular") == "regular"]
    print("calibrating league...")
    lg, hist = calibrate(regular)
    if args.cmd == "publish":
        import os
        from .publish import publish
        home = args.home or os.environ.get("HOME_TEAM")
        if not home:
            raise SystemExit("Which team is home? Use --home TEAM_CODE (or HOME_TEAM=... in pipeline/.env).")
        publish(conn, df, lg, hist, home, min_pitches=args.min_pitches, opponents=not args.no_opponents)
        return 0
    if args.cmd == "league":
        print(describe(lg))
        out = Path(__file__).resolve().parents[1] / ".cache" / "league.json"
        out.write_text(lg.to_json())
        print(f"saved {out}")
        return 0

    from .run import run, write

    print(f"building {args.pitcher} vs {args.team}...")
    r = run(df, args.pitcher, args.team, league=(lg, hist))
    path = write(r)
    a = r.result.arsenal
    print(f"{a.pitcher_name} ({a.throws}HP): {a.n_pitches} tracked pitches, "
          + ", ".join(f"{c.label} {100 * c.share:.0f}%" for c in a.clusters))
    pop = r.result.population
    pop = pop[pop["split"] == "all"]
    print("similar league pitches per pitch type (vs LHH / vs RHH):")
    for c in a.clusters:
        n = {sd: pop[(pop["cluster"] == c.cid) & (pop["side"] == sd)]["sim_pitches"].sum() for sd in ("L", "R")}
        print(f"  {c.label:<34}{n['L']:>9,.0f} / {n['R']:,.0f}")
    b = r.result.batters.sort_values("score", ascending=False)
    print(f"{'Batter':<26}{'Bats':<5}{'Adv':>5}{'xRV/100':>9}{'Whiff':>7}{'Chase':>7}{'Hard':>7}  Sample")
    for _, x in b.iterrows():
        f = lambda v, pct=True: "  -  " if v != v else (f"{100 * v:5.0f}%" if pct else f"{v:+.2f}")  # noqa: E731
        print(f"{str(x['batter_name'])[:25]:<26}{str(x['side']):<5}{x['score']:>5.0f}{f(x['xrv100'], False):>9}"
              f"{f(x['whiff']):>7}{f(x['chase']):>7}{f(x['hard_hit']):>7}  {x['confidence']}")
    print(f"report: {path}")
    if not args.no_open:
        import webbrowser
        webbrowser.open(path.as_uri())
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_env()
    ap = argparse.ArgumentParser(prog="matchup")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate")
    for name in ("sync", "ingest", "check"):
        s = sub.add_parser(name)
        s.add_argument("paths", nargs="+")
    lgp = sub.add_parser("league")
    lgp.add_argument("--refresh", action="store_true", help="re-download pitches instead of using the cache")
    mp = sub.add_parser("matchup")
    mp.add_argument("--pitcher", required=True, help="name ('Burnham, Payton') or Trackman ID")
    mp.add_argument("--team", required=True, help="opponent team code, e.g. AUB_TIG_SB")
    mp.add_argument("--refresh", action="store_true")
    mp.add_argument("--no-open", action="store_true", help="don't open the report in the browser")
    pp = sub.add_parser("publish", help="precompute matchups for the dashboard and write them to Neon")
    pp.add_argument("--home", help="your team code (default: HOME_TEAM in .env)")
    pp.add_argument("--min-pitches", type=int, default=150, help="opponent pitchers need this many tracked pitches")
    pp.add_argument("--no-opponents", action="store_true", help="only your own pitchers (fast)")
    pp.add_argument("--refresh", action="store_true")
    bp = sub.add_parser("backtest")
    bp.add_argument("--pitchers", type=int, default=60, help="pitchers to test (most pitches after the split first)")
    bp.add_argument("--split", help="split date YYYY-MM-DD (default: 60%% of the season before it)")
    bp.add_argument("--tune", action="store_true", help="search for better settings (about 15x longer)")
    bp.add_argument("--apply", action="store_true", help="save the tuned settings for future reports")
    bp.add_argument("--refresh", action="store_true")
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
        if args.cmd in ("league", "matchup", "publish"):
            return cmd_model(args, conn)
        if args.cmd == "backtest":
            return cmd_backtest(args, conn)
        return cmd_sync(args, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
