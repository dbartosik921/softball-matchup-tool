-- Coach notes per hitter, written from the dashboard (Gameday card) and printed under the hitter's name.
-- Not tied to a publish run: notes survive every re-publish.
create table if not exists coach_notes (
  batter_tm_id text primary key,
  note         text not null,
  updated_at   timestamptz not null default now()
);
