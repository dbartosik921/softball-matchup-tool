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
