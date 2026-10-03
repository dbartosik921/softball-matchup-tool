-- Gameday card back page: innings pitched per pitcher (to pick the top 4) and 5x5 OPS zones per matchup.
alter table pub_pitchers add column if not exists ip real;
alter table pub_matchups add column if not exists zones jsonb;   -- [ops_all(25), pa_all(25), ops_ch(25)|null, pa_ch(25)|null]
