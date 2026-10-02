-- 002: duplicate-game detection.
-- pitch_key = release time to the second + pitcher name; identifies the same game exported twice
-- under different GameUIDs. ingest_files records files skipped as duplicates so they aren't re-read.

alter table pitches add column if not exists pitch_key text;
create index if not exists pitches_date_key on pitches(game_date, pitch_key);

alter table ingest_files add column if not exists status text not null default 'loaded';
alter table ingest_files add column if not exists duplicate_of text;
