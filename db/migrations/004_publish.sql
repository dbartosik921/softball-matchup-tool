-- Precomputed matchups for the dashboard (written by `python -m matchup publish`, read by web/).
-- A publish writes a new run_id, marks it complete, then deletes older runs; the site always reads the
-- latest complete run, so a half-finished publish is never shown.

create table if not exists pub_runs (
  run_id       bigserial primary key,
  created_at   timestamptz not null default now(),
  complete     boolean not null default false,
  home_team    text not null,
  data_through date,
  meta         jsonb not null default '{}'::jsonb   -- n_games, hard_hit_mph, slot_pa, fit components, validation, settings
);

create table if not exists pub_pitchers (
  run_id        bigint not null references pub_runs(run_id) on delete cascade,
  pitcher_tm_id text not null,
  pitcher_name  text not null,
  team          text,
  throws        text not null,
  n_pitches     int not null,
  is_home       boolean not null,
  arsenal       jsonb not null,                      -- clusters: label, share, usage, means, n, tag_mix
  primary key (run_id, pitcher_tm_id)
);

create table if not exists pub_teams (
  run_id    bigint not null references pub_runs(run_id) on delete cascade,
  team      text not null,
  season    text,
  last_game date,
  lineup    jsonb not null,                          -- batter ids, batting order of the latest game
  roster    jsonb not null,                          -- [{id, name, pa, games, last_game}]
  primary key (run_id, team)
);

create table if not exists pub_matchups (
  run_id        bigint not null references pub_runs(run_id) on delete cascade,
  pitcher_tm_id text not null,
  batter_tm_id  text not null,
  batter_name   text,
  batter_team   text,
  side          text,
  summary       jsonb not null,                      -- array in pub_runs.meta.summary_keys order
  detail        jsonb not null,                      -- arrays in meta.detail_keys order, per pitch type x split
  primary key (run_id, pitcher_tm_id, batter_tm_id)
);

create index if not exists pub_matchups_batter on pub_matchups (run_id, batter_tm_id);
