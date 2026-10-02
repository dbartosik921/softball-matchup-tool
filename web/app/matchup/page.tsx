import Link from "next/link";
import Header from "@/components/Header";
import Picker from "@/components/Picker";
import Report from "@/components/Report";
import { requireAdmin } from "@/lib/auth";
import { latestRun, matchups, pitcher, pitchers, team, teamNames } from "@/lib/data";

export const dynamic = "force-dynamic";

type SP = Promise<{ pitcher?: string; team?: string; batter?: string }>;

export default async function MatchupPage({ searchParams }: { searchParams: SP }) {
  await requireAdmin();
  const q = await searchParams;
  const run = await latestRun();
  if (!run) return <><Header /><main><div className="empty">Nothing published yet.</div></main></>;
  const [p, t, ps, teams] = await Promise.all([
    q.pitcher ? pitcher(run.run_id, q.pitcher) : null,
    q.team ? team(run.run_id, q.team) : null,
    pitchers(run.run_id), teamNames(run.run_id),
  ]);
  const picker = <Picker key={`${q.pitcher}|${q.team}|${q.batter}`} pitchers={ps} teams={teams} home={run.home_team}
    initial={{ pitcher: q.pitcher, team: q.team, batter: q.batter }} />;
  let body: React.ReactNode;
  if (!p || !t) {
    body = <div className="empty">That pitcher or team isn&apos;t in the latest publish. <Link href="/">Pick again</Link>.</div>;
  } else if (!p.is_home && t.team !== run.home_team) {
    body = <div className="empty">Opponent pitchers are published only against {run.home_team}.</div>;
  } else {
    const ids = q.batter ? [q.batter] : t.roster.map((r) => r.id);
    const rows = await matchups(run, p.pitcher_tm_id, ids);
    body = rows.length
      ? <Report run={run} p={p} team={t} rows={rows} batter={q.batter || undefined} />
      : <div className="empty">No matchup rows for this pitcher and {q.batter ? "batter" : "team"}.</div>;
  }
  return (
    <>
      <Header through={run.data_through} />
      <main>
        <div className="noprint">{picker}</div>
        <div style={{ height: 16 }} />
        {body}
      </main>
    </>
  );
}
