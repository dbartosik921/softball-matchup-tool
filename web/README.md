# Matchup dashboard (Next.js on Vercel)

Private site: one admin password, then pick a pitcher and an opponent (or one batter) and see the same
tables as the HTML report. It only **reads** what the pipeline publishes; nothing is computed here.

```
Mac:     python -m matchup sync ~/Trackman      # new games -> Neon
         python -m matchup publish --home CODE  # matchups -> Neon (pub_* tables)
Vercel:  this app reads the latest complete publish
```

Coverage of a publish: every home-team pitcher vs every hitter on every roster in the latest season, and
every opponent pitcher with 150+ tracked pitches vs the home team's hitters.

## Pages

- **Matchup** (`/`, `/matchup`): one pitcher vs an opponent's lineup and bench, or vs one batter, with the full report.
- **Gameday** (`/gameday`): pick a pitching team and a hitting team; that team's pitchers run across the top and the
  batting order is set with 1-9 dropdowns (only that team's hitters; starts from their last lineup). Each cell shows
  Pitcher adv., xRV/100, Shape fit and Sample, with a PA-weighted lineup total per pitcher. Hitters not in the order
  are listed alphabetically below. The order is kept in the URL, so a card can be bookmarked or printed.
  Each pitcher block also has **Plan** (attack pitch / two-strike put-away, plus head-to-head history),
  her pitch mix vs L and R under her name, faint colours for low samples, and coach notes per hitter (saved in
  Neon, printed on the card). When another team is pitching, a **Her tendencies** block shows her mix on the
  first pitch, when behind and with two strikes.
  Coverage follows the publish: home pitchers vs any team, other teams' pitchers vs the home team only.
  **Printing** gives two pages for front/back: the card (letter landscape, scaled to fit) and a back page
  (letter portrait) with two 5x5 OPS zones per pitcher per hitter, vs pitches shaped like her arsenal and like
  her changeup (pitcher's view; see `pipeline/matchup/zones.py`). The back page defaults to the team's top 4
  pitchers by innings and lists the 1-9 order, then the most at-bats, up to 10 hitters.

## Environment variables (Vercel → Project → Settings → Environment Variables)

| Name | Value |
|---|---|
| `DATABASE_URL` | Neon connection string for the matchup database (read access is enough) |
| `ADMIN_PASSWORD` | the site password |
| `SESSION_SECRET` | optional; long random string that signs the login cookie |

Every page and API route checks the signed session cookie server-side; responses are `no-store` and
`noindex`.

## Local development

```bash
cd web && npm install
cp .env.example .env.local    # fill in DATABASE_URL and ADMIN_PASSWORD
npm run dev
```
