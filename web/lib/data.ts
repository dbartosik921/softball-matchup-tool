import "server-only";
import { neon, neonConfig } from "@neondatabase/serverless";

// Local development only: point the driver at a stand-in for Neon's HTTP endpoint (pipeline/tests/neon_mock.py).
if (process.env.NEON_FETCH_ENDPOINT && process.env.NODE_ENV !== "production") {
  neonConfig.fetchEndpoint = process.env.NEON_FETCH_ENDPOINT;
}

function db() {
  const url = process.env.DATABASE_URL;
  if (!url) throw new Error("DATABASE_URL is not set");
  return neon(url);
}

export type Run = {
  run_id: number; home_team: string; data_through: string; created_at: string;
  meta: {
    n_games: number; hard_hit_mph: number; season: string; slot_pa: number[];
    fit_components: string[]; validation: Record<string, string>;
    summary_keys: string[]; detail_keys: string[];
  };
};
export type Cluster = {
  cid: number; label: string; share: number; n: number; usage: Record<string, number>;
  means: Record<string, number | null>; tag_mix: Record<string, number>;
};
export type Pitcher = {
  pitcher_tm_id: string; pitcher_name: string; team: string | null; throws: "L" | "R";
  n_pitches: number; is_home: boolean; arsenal: Cluster[];
};
export type RosterEntry = { id: string; name: string; pa: number; games: number; last_game: string };
export type Team = { team: string; season: string; last_game: string | null; lineup: string[]; roster: RosterEntry[] };
export type Num = number | null;
export type Summary = Record<string, Num | string>;
export type DetailRow = Record<string, Num | string>;
export type Matchup = {
  batter_tm_id: string; batter_name: string | null; batter_team: string | null; side: "L" | "R" | null;
  summary: Summary; detail: DetailRow[];
};

export async function latestRun(): Promise<Run | null> {
  const rows = await db().query(
    `select run_id, home_team, data_through::text as data_through, created_at::text as created_at, meta
       from pub_runs where complete order by run_id desc limit 1`);
  return (rows[0] as Run) ?? null;
}

export async function pitchers(run: number): Promise<Omit<Pitcher, "arsenal">[]> {
  return (await db().query(
    `select pitcher_tm_id, pitcher_name, team, throws, n_pitches, is_home from pub_pitchers
      where run_id = $1 order by is_home desc, team, pitcher_name`, [run])) as Omit<Pitcher, "arsenal">[];
}

export async function teamNames(run: number): Promise<string[]> {
  return (await db().query(`select team from pub_teams where run_id = $1 order by team`, [run])).map((r) => r.team as string);
}

export async function pitcher(run: number, id: string): Promise<Pitcher | null> {
  const r = await db().query(`select * from pub_pitchers where run_id = $1 and pitcher_tm_id = $2`, [run, id]);
  return (r[0] as Pitcher) ?? null;
}

export async function team(run: number, code: string): Promise<Team | null> {
  const r = await db().query(
    `select team, season, last_game::text as last_game, lineup, roster from pub_teams where run_id = $1 and team = $2`,
    [run, code]);
  return (r[0] as Team) ?? null;
}

export async function matchups(r: Run, pitcherId: string, batterIds: string[]): Promise<Matchup[]> {
  if (!batterIds.length) return [];
  const rows = await db().query(
    `select batter_tm_id, batter_name, batter_team, side, summary, detail from pub_matchups
      where run_id = $1 and pitcher_tm_id = $2 and batter_tm_id = any($3::text[])`,
    [r.run_id, pitcherId, batterIds]);
  const sk = r.meta.summary_keys, dk = r.meta.detail_keys;
  const zip = (keys: string[], vals: unknown[]) => Object.fromEntries(keys.map((k, i) => [k, vals[i] ?? null]));
  return rows.map((x) => ({
    batter_tm_id: x.batter_tm_id, batter_name: x.batter_name, batter_team: x.batter_team, side: x.side,
    summary: zip(sk, x.summary as unknown[]) as Summary,
    detail: (x.detail as unknown[][]).map((d) => zip(dk, d) as DetailRow),
  }));
}

// ---- Gameday card: every pitcher of one team x every hitter of another, four numbers per pair ----
export type GamedayCell = { score: Num; xrv100: Num; fit100: Num; sim: Num; conf: string | null; side: string | null };
export type Gameday = {
  pitchers: { id: string; name: string; throws: string; n: number }[];
  roster: RosterEntry[];
  lastLineup: string[];
  cells: Record<string, Record<string, GamedayCell>>;   // batter id -> pitcher id -> numbers
};

export async function pitchingTeams(run: Run): Promise<string[]> {
  const rows = await db().query(
    `select distinct coalesce(team, '') as team from pub_pitchers where run_id = $1 and team is not null order by 1`,
    [run.run_id]);
  const teams = rows.map((r) => r.team as string).filter(Boolean);
  return [run.home_team, ...teams.filter((t) => t !== run.home_team)];
}

export async function gameday(run: Run, pitchingTeam: string, battingTeam: string): Promise<Gameday | null> {
  const t = await team(run.run_id, battingTeam);
  if (!t) return null;
  const ps = await db().query(
    `select pitcher_tm_id as id, pitcher_name as name, throws, n_pitches as n from pub_pitchers
      where run_id = $1 and (case when $2::text = $3::text then is_home else team = $2::text and not is_home end)
      order by n_pitches desc`,
    [run.run_id, pitchingTeam, run.home_team]);
  const ids = t.roster.map((r) => r.id);
  const pids = ps.map((p) => p.id as string);
  const k = run.meta.summary_keys;
  const at = (name: string) => k.indexOf(name);
  const rows = pids.length && ids.length ? await db().query(
    `select pitcher_tm_id, batter_tm_id, side, summary from pub_matchups
      where run_id = $1 and pitcher_tm_id = any($2::text[]) and batter_tm_id = any($3::text[])`,
    [run.run_id, pids, ids]) : [];
  const cells: Gameday["cells"] = {};
  const n = (v: unknown) => (typeof v === "number" ? v : null);
  for (const r of rows) {
    const s = r.summary as unknown[];
    (cells[r.batter_tm_id] ??= {})[r.pitcher_tm_id] = {
      score: n(s[at("score")]), xrv100: n(s[at("xrv100")]), fit100: n(s[at("fit100")]),
      sim: n(s[at("sim_pitches")]), conf: (s[at("confidence")] as string) ?? null, side: r.side,
    };
  }
  return {
    pitchers: ps.map((p) => ({ id: p.id, name: p.name, throws: p.throws, n: p.n })),
    roster: t.roster, lastLineup: t.lineup, cells,
  };
}
