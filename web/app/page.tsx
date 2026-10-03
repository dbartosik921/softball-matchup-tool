import Header from "@/components/Header";
import Picker from "@/components/Picker";
import { requireAdmin } from "@/lib/auth";
import { latestRun, pitchers, teamNames } from "@/lib/data";

export const dynamic = "force-dynamic";

export default async function Home() {
  await requireAdmin();
  const run = await latestRun();
  if (!run) {
    return (<><Header /><main><h1>Matchups</h1>
      <div className="empty">Nothing published yet. On your Mac run <code>python -m matchup publish --home YOUR_TEAM_CODE</code>, then reload.</div>
    </main></>);
  }
  const [ps, teams] = await Promise.all([pitchers(run.run_id), teamNames(run.run_id)]);
  const nHome = ps.filter((p) => p.is_home).length;
  return (
    <>
      <Header through={run.data_through} tab="matchup" />
      <main>
        <h1>Pitcher vs opponent</h1>
        <p className="sub">{run.home_team} pitchers vs any team · opponents&apos; pitchers vs {run.home_team} ·
          {" "}{run.meta.n_games.toLocaleString()} games, {run.meta.season} · published {run.created_at.slice(0, 16)}</p>
        <Picker pitchers={ps} teams={teams} home={run.home_team} />
        <p className="note">{nHome} {run.home_team} pitchers and {ps.length - nHome} opponent pitchers with enough tracked pitches.
          To refresh after new games: <code>python -m matchup sync …</code> then <code>python -m matchup publish</code>.</p>
      </main>
    </>
  );
}
