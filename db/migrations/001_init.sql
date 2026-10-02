-- 001: games, pitches, ingest log.
-- Conventions (verified against Trackman softball exports):
--   horz_break, rel_side, plate_loc_side are PITCHER'S VIEW, positive = toward the RHH box (3B side).
--   Never store catcher-view values (x0, pfxx); flip for display only.
--   Every external ID is TEXT (12- and 13-digit Trackman IDs are both valid).

create table if not exists schema_migrations (
  version     text primary key,
  applied_at  timestamptz not null default now()
);

create table if not exists games (
  game_uid     text primary key,          -- Trackman GameUID
  game_id      text,                      -- e.g. 20260424-BoglePark-1
  game_date    date not null,
  season       text not null check (season ~ '^[0-9]{4}-[0-9]{2}$'),  -- July 1 flip, matches the player ID registry
  home_team    text,
  away_team    text,
  stadium      text,
  level        text,
  league       text,
  ingested_at  timestamptz not null default now()
);
create index if not exists games_date on games(game_date);

create table if not exists ingest_files (
  sha256       text primary key,           -- re-adding the same file is a no-op
  file_name    text not null,
  game_uid     text references games(game_uid) on delete cascade,
  rows_read    integer not null,
  rows_loaded  integer not null,
  warnings     jsonb not null default '[]',
  source       text not null default 'folder',   -- 'folder' | 'upload'
  ingested_at  timestamptz not null default now()
);

create table if not exists pitches (
  pitch_uid        text primary key,       -- Trackman PitchUID
  game_uid         text not null references games(game_uid) on delete cascade,
  game_date        date not null,
  season           text not null,
  pitch_no         integer,
  inning           integer,
  top_bottom       text,
  pa_of_inning     integer,
  pitch_of_pa      integer,
  outs             smallint,
  balls            smallint,
  strikes          smallint,

  pitcher_tm_id    text not null check (pitcher_tm_id ~ '^[0-9A-Za-z_-]+$'),
  pitcher_name     text,
  pitcher_team     text,
  p_throws         char(1) not null check (p_throws in ('L','R')),
  batter_tm_id     text not null check (batter_tm_id ~ '^[0-9A-Za-z_-]+$'),
  batter_name      text,
  batter_team      text,
  b_side           char(1) not null check (b_side in ('L','R')),
  catcher_tm_id    text,

  tagged_pitch_type text,
  pitch_call       text not null,
  kor_bb           text,
  tagged_hit_type  text,
  play_result      text,
  outs_on_play     smallint,
  runs_scored      smallint,

  -- raw pitch metrics (pitcher's view)
  rel_speed        real,
  spin_rate        real,
  spin_axis        real,
  rel_height       real,
  rel_side         real,
  extension        real,
  vert_rel_angle   real,
  horz_rel_angle   real,
  induced_vert_break real,
  vert_break       real,
  horz_break       real,
  plate_loc_height real,
  plate_loc_side   real,
  zone_speed       real,
  vert_appr_angle  real,
  horz_appr_angle  real,
  zone_time        real,
  pitch_tracked    boolean not null,       -- false when release/movement/location is missing or not High confidence

  -- batted ball
  exit_speed       real,
  launch_angle     real,
  direction        real,
  distance         real,
  hit_launch_conf  text,

  -- derived: handedness-relative features
  hb_arm           real,   -- + = arm side (pitcher-hand normalized)
  rel_side_arm     real,   -- + = arm side
  hb_in            real,   -- + = toward the batter's hands
  loc_in           real,   -- + = inside to this batter
  haa_in           real,   -- horizontal approach angle, + = angling in toward the batter
  same_side        boolean not null,   -- RHP vs RHH or LHP vs LHH

  -- derived: outcome flags
  in_zone          boolean,
  is_swing         boolean not null,
  is_whiff         boolean not null,
  is_called_strike boolean not null,
  is_bip           boolean not null,
  is_two_strike    boolean not null,
  bip_ev_valid     boolean not null,   -- in play with High/Medium launch confidence and an exit speed
  pa_ending        boolean not null,
  pa_result        text,               -- 1B 2B 3B HR BB HBP K OUT ROE FC SF SH (null when not PA-ending)

  ingested_at      timestamptz not null default now()
);
create index if not exists pitches_pitcher on pitches(pitcher_tm_id, game_date);
create index if not exists pitches_batter on pitches(batter_tm_id, game_date);
create index if not exists pitches_matchup on pitches(p_throws, b_side);
create index if not exists pitches_game on pitches(game_uid);

insert into schema_migrations(version) values ('001') on conflict do nothing;
