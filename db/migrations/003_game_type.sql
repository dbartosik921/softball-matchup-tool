-- 003: game type. 'fall' = fall ball / live ABs / intrasquad (Aug-Dec, or pitcher and batter on the
-- same team); 'regular' = everything else. Opponent matchups use regular games by default.
-- Batting practice sessions are never loaded.
alter table games add column if not exists game_type text not null default 'regular';
alter table games drop constraint if exists games_game_type_check;
alter table games add constraint games_game_type_check check (game_type in ('regular', 'fall'));
create index if not exists games_type_date on games(game_type, game_date);
