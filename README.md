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

### Matchup reports

```bash
python -m matchup league                                              # calibration summary
python -m matchup matchup --pitcher "Burnham, Payton" --team AUB_TIG_SB  # writes + opens an HTML report
```

The first run downloads every pitch (about a minute) and caches it in `pipeline/.cache/`; later runs reuse
the cache until new games are synced (`--refresh` forces a download). Reports go to `pipeline/reports/`
(git-ignored: they contain player data).

How the numbers are built (`matchup/engine.py`):

1. **League calibration** (`calibrate.py`): run values per PA result (half-inning regression) and per count,
   hard-hit line (top quarter of D1 exit velo), VAA-location slope, feature scales, baselines by hand matchup.
2. **Arsenal** (`arsenal.py`): her pitches clustered on arm-relative shape (velo, IVB, arm-side break,
   location-adjusted VAA); recency-weighted usage vs LHH / RHH, all counts and two strikes.
3. **Similar pitches**: every pitch a hitter saw from a same-handed pitcher is weighted by how typical it would
   be of each cluster, in the hitter's frame (break and release side toward/away from the hitter), times recency.
4. **Shrinkage**: rates are pulled toward how all same-side hitters did against that shape, shifted by the
   hitter's own overall skill vs that hand. Prior strengths live in `engine.METRICS`.
5. **Pitcher advantage** = expected runs per 100 pitches vs her arsenal, as a percentile among qualified hitters.
6. **Shape fit** = the matchup without the talent: for each pitch type, (hitter's shrunk rate − what her overall
   level predicts) × how often that outcome can happen × its run value (`calibrate.event_values`), summed over
   **only the outcomes the backtest validated** (read from `settings.json`; whiff + hard hit until a backtest
   has been saved). Negative = she handles these shapes worse than usual. Hidden if nothing validated.

### Publishing to the dashboard

```bash
python -m matchup publish --home YOUR_TEAM_CODE        # or set HOME_TEAM in pipeline/.env
python -m matchup publish --no-opponents               # only your own staff (fast)
```

Precomputes every home pitcher vs every hitter in the latest season, and every opponent pitcher (150+ tracked
pitches) vs your hitters, then writes them to the `pub_*` tables. The site (`web/`) always shows the latest
complete publish; re-run after each `sync`. The team code accepts a unique fragment (`--home ARK`).

### Backtest (validation)

```bash
python -m matchup backtest                    # score current settings on later games (~2 min)
python -m matchup backtest --tune             # search better settings (~20-30 min)
python -m matchup backtest --tune --apply     # ...and save them to pipeline/settings.json for every report
```

Games before the split date (default: 60% of the season) are the only data the model sees; each later pitch
thrown by the sampled pitchers is predicted five ways and scored against what happened:

| predictor | what it knows |
|---|---|
| league | league rate for this pitcher hand x batter side |
| batter | the hitter's own rate vs this hand |
| shape | all same-side hitters vs pitches shaped like this one |
| prior | shape + hitter's overall skill |
| model | prior + the hitter's own history vs similar pitches (what the report shows) |

Skill = % lower error than the league baseline; a bootstrap over hitters gives 90% intervals. Each metric gets
a plain verdict ("validated", "pitch shape adds value; hitter-specific part unproven", "hitter's overall rate
is the best guide", "mostly noise"), which the HTML report prints in its Validation section. Tuning searches
similarity width, recency half-life and per-metric shrinkage strength. `settings.json` is committed on purpose:
it holds tuning results, not player data.

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
6. ~~Dashboard: pitcher vs batter, pitcher vs team (lineup weighted by expected PA)~~ (`web/`, Vercel)
7. Upload page on the dashboard (in addition to folder sync)
