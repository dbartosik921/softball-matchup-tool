import Header from "@/components/Header";
import GamedayCard from "@/components/GamedayCard";
import GamedayPicker from "@/components/GamedayPicker";
import { requireAdmin } from "@/lib/auth";
import { gameday, latestRun, pitchingTeams, teamNames } from "@/lib/data";

export const dynamic = "force-dynamic";

type SP = Promise<{ pt?: string; bt?: string; l?: string }>;

export default async function GamedayPage({ searchParams }: { searchParams: SP }) {
  await requireAdmin();
  const q = await searchParams;
  const run = await latestRun();
  if (!run) return <><Header tab="gameday" /><main><div className="empty">Nothing published yet.</div></main></>;
  const home = run.home_team;
  const [pTeams, bTeamsAll] = await Promise.all([pitchingTeams(run), teamNames(run.run_id)]);
  const pt = q.pt && pTeams.includes(q.pt) ? q.pt : home;
  // Published coverage: home pitchers face everyone; other teams' pitchers only face the home team.
  const bTeams = pt === home ? bTeamsAll.filter((t) => t !== home) : [home];
  const bt = q.bt && bTeams.includes(q.bt) ? q.bt : (pt === home ? "" : home);
  const data = bt ? await gameday(run, pt, bt) : null;
  const lineup = q.l !== undefined ? q.l.split(",").slice(0, 9) : null;

  return (
    <>
      <Header through={run.data_through} tab="gameday" />
      <main style={{ maxWidth: "none" }}>
        <h1 className="noprint">Gameday card</h1>
        <GamedayPicker pTeams={pTeams} bTeams={bTeamsAll} home={home} pt={pt} bt={bt} />
        <p className="note noprint">{pt === home
          ? `${home} pitchers can face any team.`
          : `Opponent pitchers are published against ${home} only, and only pitchers with 150+ tracked pitches appear.`}
          {" "}Pitchers run across the top; set the batting order in the left column.</p>
        {!bt ? <div className="empty">Pick a hitting team to build the card.</div>
          : !data ? <div className="empty">{bt} isn&apos;t in the latest publish.</div>
          : !data.pitchers.length ? <div className="empty">No published pitchers for {pt}.</div>
          : <GamedayCard key={`${pt}|${bt}`} pt={pt} bt={bt} data={data} slotPa={run.meta.slot_pa}
              fitParts={run.meta.fit_components} initial={lineup} leagueOps={run.meta.league_ops} />}
      </main>
    </>
  );
}
