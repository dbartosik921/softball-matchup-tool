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
