# Softball Matchup Tool

Pitcher vs batter matchup analysis from Trackman softball game logs. A pitcher's arsenal is described by
movement and shape; a batter is evaluated against pitches she has seen that profile similarly, with
handedness built into every comparison.

```
db/migrations/   Postgres schema (Neon)
pipeline/        Python: ingest, features, models, validation
web/             Next.js dashboard on Vercel (private, single admin login)
```

## Data conventions

Verified against Trackman softball exports. Breaking these silently flips inside and outside.

| Field | Frame |
|---|---|
| `HorzBreak`, `RelSide`, `PlateLocSide`, `HorzApprAngle` | Pitcher's view, **positive = toward the RHH box** |
| `x0`, `pfxx` | Catcher's view (opposite sign). Not used. |

Derived features (stored on every pitch):

| Column | Meaning |
|---|---|
| `hb_arm`, `rel_side_arm` | Arm-relative: positive = arm side. Compares arsenals across RHP and LHP. |
| `hb_in`, `loc_in`, `haa_in` | Batter-relative: positive = toward the batter / inside. Used for batter history. |
| `same_side` | RHP vs RHH or LHP vs LHH |
| `in_zone` | Fixed zone: \|side\| ≤ 0.71 ft, 1.5 ≤ height ≤ 3.0 ft |

Displays flip location at render time only (pitcher's view by default).

Other rules:
- Trackman IDs are stored as **text**. 12- and 13-digit IDs are both valid. IDs damaged by Excel
  (`1.00E+11`, `123.0`) are rejected with a warning, never loaded wrong.
- Pitch type tags are unreliable and used as labels only. Similarity is built on physical metrics.
- Pitches missing tracking (or flagged Low confidence) still count for balls, strikes and outcomes but are
  excluded from similarity.
- Exit-velo metrics use balls in play with High or Medium launch confidence.
- Seasons flip on July 1 and are labelled `2025-26`, matching the player ID registry.

## Pipeline

```bash
cd pipeline
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # paste your Neon DATABASE_URL

python -m matchup check ~/path/to/Trackman            # parse only, report problems, writes nothing
python -m matchup migrate                             # create / upgrade tables
python -m matchup sync  ~/path/to/Trackman            # load every new game file (re-running is safe)
```

**Connection:** for Neon (`*.neon.tech`) the pipeline talks to the database over HTTPS (port 443), the
same protocol as Neon's serverless driver, because many campus networks block Postgres's port 5432.
Other hosts (local Postgres, CI) use a normal connection. Force either with `DB_TRANSPORT=http` or
`DB_TRANSPORT=tcp` in `.env`.

`sync` walks the folder recursively, accepts `.csv`, `.xlsx` and `.xlsm`, and skips any file it has
already loaded (by file hash) and any pitch already stored (by `PitchUID`). Games are sent 20 per
round trip; each game is written atomically, and a failing batch is retried file by file so one bad
file never blocks the rest.

### Damaged IDs and duplicate games

- **Damaged IDs** (Excel turned `100000002415` into `1E+11`, `1.00E+11` or a rounded `100000000000`) are
  recovered by name + team: first from the same player's clean ID elsewhere in the file, then from the
  player ID registry (`REGISTRY_DATABASE_URL`, read-only), matching any name the player has used, for
  that season and then any season. Ambiguous matches (two same-named players on one team) are skipped,
  never guessed.
- **Duplicate games** (the same game exported more than once under different file names, including
  re-processed exports with new GameUIDs) are detected by pitch fingerprint: release time to the second
  + pitcher. The most complete copy is loaded; if a more complete copy arrives later it replaces the
  stored one. Skipped copies are recorded in `ingest_files` (`status = 'duplicate'`).
  `check` lists duplicates without writing anything.

### What gets loaded

- **Batting practice** (`BP` as its own word in the file name or GameID) is never loaded.
- **Fall ball, live ABs and intrasquad** (games Aug-Dec, or pitcher and batter on the same team) are loaded
  with `games.game_type = 'fall'`. Opponent matchups use `regular` games by default.
- **Season compilation files** (many games in one file) are split into one game each, then deduplicated
  against the single-game files.
- **Empty exports** and non-Trackman files (scouting reports, rosters) are listed as ignored.

### Tests

```bash
pytest -q                                                       # parser and feature tests
TEST_DATABASE_URL=postgresql://... pytest -q                    # plus database load tests
```

CI runs both on every push (`.github/workflows/pipeline-tests.yml`).

## Player ID registry

Names, rosters and team names come from the separate player ID registry (`player-id-db`) through a
read-only connection (`REGISTRY_DATABASE_URL`). Matchup data is keyed on Trackman IDs and joined to the
registry's `player_source_ids` (source `trackman`).

## Roadmap

1. ~~Ingest and handedness-aware features~~
2. Run values per pitch event (count-based), hard-hit threshold from the data
3. Pitcher arsenal clustering; batter similarity lookup with recency weighting and shrinkage
4. Matchup tables: overall score, whiff %, chase %, called strike %, hard hit %, OPS (all counts and two strikes)
5. Validation: first-half / second-half backtest against baselines
6. Dashboard: pitcher vs batter, pitcher vs team (lineup weighted by expected PA)
7. Upload page on the dashboard (in addition to folder sync)
