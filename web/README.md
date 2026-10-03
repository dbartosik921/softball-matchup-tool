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
  Coverage follows the publish: home pitchers vs any team, other teams' pitchers vs the home team only.

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
