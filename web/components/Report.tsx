// Port of pipeline/matchup/report.py. Blue tint = better for the PITCHER than similar hitters,
// red = better for the HITTER, no tint = about average. Every number is printed; hover explains it.
import Link from "next/link";
import type { DetailRow, Matchup, Num, Pitcher, Run, Team } from "@/lib/data";

type Kind = "pct" | "ops" | "rv" | "n";
// metric -> [label, higher is good for the pitcher, difference that earns full tint, format]
const COLS: [string, string, boolean, number, Kind][] = [
  ["whiff", "Whiff %", true, 0.10, "pct"],
  ["chase", "Chase %", true, 0.08, "pct"],
  ["called_strike", "Called K %", true, 0.05, "pct"],
  ["hard_hit", "Hard-hit %", false, 0.10, "pct"],
  ["ops", "OPS*", false, 0.20, "ops"],
];
const FIT_NAMES: Record<string, string> = { whiff: "whiff", called_strike: "called strike", chase: "chase", hard_hit: "hard hit" };
const VAL_NAMES: Record<string, string> = {
  whiff: "Whiff %", chase: "Chase %", called_strike: "Called K %", hard_hit: "Hard-hit %", rv: "xRV / Pitcher adv.",
};

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

function fmt(v: unknown, kind: Kind): string {
  const x = num(v);
  if (x === null) return "–";
  if (kind === "pct") return `${Math.round(100 * x)}%`;
  if (kind === "ops") return x < 1 ? x.toFixed(3).replace(/^0/, "") : x.toFixed(3);
  if (kind === "rv") return (x >= 0 ? "+" : "") + x.toFixed(2);
  return x.toFixed(0);
}

function signed1(v: unknown): string {
  const x = num(v);
  return x === null ? "–" : (x >= 0 ? "+" : "") + x.toFixed(1);
}

function tint(v: unknown, pop: unknown, highGood: boolean, full: number): React.CSSProperties | undefined {
  const a = num(v), b = num(pop);
  if (a === null || b === null) return undefined;
  const d = (a - b) / full;
  if (Math.abs(d) < 0.15) return undefined;
  const goodForPitcher = (d > 0) === highGood;
  const alpha = Math.min(0.45, 0.1 + 0.35 * Math.min(1, Math.abs(d)));
  return { background: `rgba(var(${goodForPitcher ? "--pitch" : "--hit"}),${alpha.toFixed(2)})` };
}

function Cell({ v, pop, m, n, extra = "" }: { v: unknown; pop: unknown; m: number; n?: Num; extra?: string }) {
  const [, label, high, full, kind] = COLS[m];
  let title = `${label}: ${fmt(v, kind)} vs ${fmt(pop, kind)} for similar hitters`;
  if (num(n) !== null) title += `; ${Math.round(n as number)} similar pitches`;
  return <td style={tint(v, pop, high, full)} title={title + extra}>{fmt(v, kind)}</td>;
}

function ScoreCell({ s }: { s: unknown }) {
  const x = num(s);
  if (x === null) return <td title="No history vs this pitcher hand">–</td>;
  const d = (x - 50) / 50;
  const style = Math.abs(d) < 0.1 ? undefined
    : { background: `rgba(var(${d > 0 ? "--pitch" : "--hit"}),${Math.min(0.5, 0.1 + 0.4 * Math.abs(d)).toFixed(2)})` };
  return <td style={style} title={`Pitcher advantage: better matchup for the pitcher than ${x.toFixed(0)}% of qualified hitters`}>
    <b>{x.toFixed(0)}</b></td>;
}

function FitCell({ v, parts }: { v: unknown; parts: string }) {
  if (!parts) return <td title="No outcome passed the backtest, so Shape fit is not shown">–</td>;
  const x = num(v);
  if (x === null) return <td>–</td>;
  const title = `Runs per 100 pitches from ${parts} only (the outcomes that passed the backtest): how she does vs pitches `
    + "shaped like hers compared with what her overall level predicts. Negative = handles these shapes worse than usual (good for the pitcher).";
  return <td style={tint(x, 0, false, 0.3)} title={title}>{(x >= 0 ? "+" : "") + x.toFixed(2)}</td>;
}

function BatterRow({ m, slot, bench, parts, href }: { m: Matchup; slot?: number; bench: boolean; parts: string; href: string }) {
  const s = m.summary;
  const conf = String(s.confidence ?? "Low");
  return (
    <tr className={bench ? "bench" : ""}>
      <td className="l">{slot ?? ""}</td>
      <td className="l"><Link href={href}>{m.batter_name}</Link></td>
      <td className="l">{m.side ?? "–"}</td>
      <ScoreCell s={s.score} />
      <td title={`Expected runs per 100 pitches vs this arsenal (+ favors hitter); similar hitters ${fmt(100 * (num(s.pop_rv) ?? NaN), "rv")}`}>
        {fmt(s.xrv100, "rv")}</td>
      <FitCell v={s.fit100} parts={parts} />
      {COLS.map(([k], i) => <Cell key={k} v={s[k]} pop={s["pop_" + k]} m={i} n={num(s.sim_pitches)} />)}
      <Cell v={s.whiff_2k} pop={s.pop_whiff_2k} m={0} n={num(s.sim_pitches_2k)} extra=" (two strikes)" />
      <Cell v={s.chase_2k} pop={s.pop_chase_2k} m={1} n={num(s.sim_pitches_2k)} extra=" (two strikes)" />
      <td className={`conf-${conf}`} title="Usage-weighted similar pitches seen per pitch type">
        {conf} <span className="badge">{fmt(s.sim_pitches, "n")}</span></td>
      {num(s.direct_pa)
        ? <td className="l" title="Plate appearances vs this pitcher">{s.direct_pa} PA {String(s.direct_line ?? "")}</td>
        : <td className="l">–</td>}
    </tr>
  );
}

function Detail({ m, parts, open }: { m: Matchup; parts: string; open: boolean }) {
  if (!m.detail.length) return null;
  const bySplit = (split: string) => m.detail.filter((d) => d.split === split)
    .sort((a, b) => (num(b.usage) ?? 0) - (num(a.usage) ?? 0));
  const row = (r: DetailRow, i: number) => (
    <tr key={i}>
      <td className="l">{String(r.label)}</td>
      <td>{fmt(r.usage, "pct")}</td>
      <td>{fmt(r.sim_pitches, "n")}</td>
      {COLS.map(([k, , , , kind], j) => (
        <Cell key={k} v={r[k]} pop={r["pop_" + k]} m={j} n={num(r.sim_pitches)}
          extra={`; her raw rate ${fmt(r["raw_" + k], kind)} before shrinking`} />))}
      <FitCell v={r.fit100} parts={parts} />
      <td title="expected runs per 100 pitches (+ favors hitter)">{fmt(100 * (num(r.rv) ?? NaN), "rv")}</td>
    </tr>
  );
  return (
    <details open={open}>
      <summary><h3 style={{ display: "inline" }}>{m.batter_name}</h3>{" "}
        <span className="note">bats {m.side} · by pitch type</span></summary>
      <div className="wrap"><table>
        <thead><tr><th className="l">Pitch</th><th>Usage vs {m.side}HH</th><th>Similar pitches</th>
          {COLS.map(([k, label]) => <th key={k}>{label}</th>)}<th>Shape fit</th><th>xRV/100</th></tr></thead>
        <tbody>
          {(["all", "2k"] as const).map((split) => [
            <tr key={split}><td className="l" colSpan={10}><b>{split === "all" ? "All counts" : "Two strikes"}</b></td></tr>,
            ...bySplit(split).map(row),
          ])}
        </tbody>
      </table></div>
    </details>
  );
}

export default function Report({ run, p, team, rows, batter }: {
  run: Run; p: Pitcher; team: Team; rows: Matchup[]; batter?: string;
}) {
  const parts = (run.meta.fit_components ?? []).map((c) => FIT_NAMES[c] ?? c).join(" + ");
  const slotOf = new Map(team.lineup.map((id, i) => [id, i + 1]));
  const byId = new Map(rows.map((r) => [r.batter_tm_id, r]));
  const lineup = team.lineup.map((id) => byId.get(id)).filter((x): x is Matchup => Boolean(x));
  const bench = rows.filter((r) => !slotOf.has(r.batter_tm_id) && r.side)
    .sort((a, b) => (num(a.summary.score) ?? 999) - (num(b.summary.score) ?? 999));
  const shown = batter ? rows.filter((r) => r.batter_tm_id === batter) : [...lineup, ...bench];

  // Lineup-weighted team view: expected plate appearances by slot
  let wSum = 0, xrv = 0, score = 0;
  for (const r of lineup) {
    const x = num(r.summary.xrv100), s = num(r.summary.score);
    if (x === null || s === null) continue;
    const w = run.meta.slot_pa[(slotOf.get(r.batter_tm_id) ?? 1) - 1] ?? 1;
    wSum += w; xrv += w * x; score += w * s;
  }
  const hand = p.throws === "R" ? "RHP" : "LHP";
  const base = `/matchup?pitcher=${encodeURIComponent(p.pitcher_tm_id)}&team=${encodeURIComponent(team.team)}`;
  const one = batter ? byId.get(batter) : undefined;
  const head = (
    <thead><tr><th className="l">#</th><th className="l">Batter</th><th className="l">Bats</th><th>Pitcher adv.</th><th>xRV/100</th>
      <th>Shape fit</th>{COLS.map(([k, label]) => <th key={k}>{label}</th>)}
      <th>2K Whiff %</th><th>2K Chase %</th><th>Sample</th><th className="l">Vs her</th></tr></thead>
  );
  const val = run.meta.validation ?? {};

  return (
    <>
      <h1>{p.pitcher_name} ({hand}) vs {one ? `${one.batter_name} (${team.team})` : team.team}</h1>
      <p className="sub">Regular-season data through {run.data_through} · {run.meta.n_games.toLocaleString()} games
        {team.last_game ? ` · ${team.team} lineup from ${team.last_game}` : ""}</p>
      {one ? (
        <div className="kpis">
          <div className="kpi"><b>{fmt(one.summary.score, "n")}</b><span>Pitcher advantage (50 = avg hitter)</span></div>
          <div className="kpi"><b>{fmt(one.summary.xrv100, "rv")}</b><span>Expected runs / 100 pitches</span></div>
          <div className="kpi"><b>{fmt(one.summary.fit100, "rv")}</b><span>Shape fit ({parts || "not validated"})</span></div>
          <div className="kpi"><b>{p.n_pitches.toLocaleString()}</b><span>Tracked pitches describing her arsenal</span></div>
        </div>
      ) : (
        <div className="kpis">
          <div className="kpi"><b>{wSum ? (score / wSum).toFixed(0) : "–"}</b><span>Lineup pitcher advantage (50 = avg)</span></div>
          <div className="kpi"><b>{wSum ? fmt(xrv / wSum, "rv") : "–"}</b><span>Lineup expected runs / 100 pitches</span></div>
          <div className="kpi"><b>{p.n_pitches.toLocaleString()}</b><span>Tracked pitches describing her arsenal</span></div>
        </div>
      )}
      <div className="legend">
        <span><i className="sw" style={{ background: "rgba(var(--pitch),.4)" }} />better for the pitcher than similar hitters</span>
        <span><i className="sw" style={{ background: "rgba(var(--hit),.4)" }} />better for the hitter</span>
        <span>no tint = about average · hover any cell for the comparison</span>
        {one && <Link className="noprint" href={base}>← whole {team.team} lineup</Link>}
      </div>

      {!one && <h2>Lineup (most recent game) and bench</h2>}
      <div className="wrap"><table>{head}<tbody>
        {(one ? [one] : lineup).map((r) => (
          <BatterRow key={r.batter_tm_id} m={r} slot={slotOf.get(r.batter_tm_id)} bench={false} parts={parts}
            href={`${base}&batter=${encodeURIComponent(r.batter_tm_id)}`} />))}
        {!one && bench.map((r) => (
          <BatterRow key={r.batter_tm_id} m={r} bench parts={parts}
            href={`${base}&batter=${encodeURIComponent(r.batter_tm_id)}`} />))}
      </tbody></table></div>
      {!one && <p className="note">Lineup = batting order from {team.team}&apos;s most recent game; team numbers weight each slot by its expected
        plate appearances ({run.meta.slot_pa.map((x) => x.toFixed(1)).join(", ")}). Bench = everyone else who batted for them this season.
        Click a name for the one-batter view.</p>}

      <h2>Her arsenal (movement clusters)</h2>
      <div className="wrap"><table>
        <thead><tr><th className="l">Pitch</th><th>Overall</th><th>vs LHH</th><th>vs RHH</th><th>vs LHH 2K</th><th>vs RHH 2K</th>
          <th>Velo</th><th>IVB (in)</th><th>Arm-side HB (in)</th><th>Spin</th><th>VAA adj.</th><th>Rel. ht</th><th>n</th>
          <th className="l">Tagged as</th></tr></thead>
        <tbody>{p.arsenal.map((c) => (
          <tr key={c.cid}>
            <td className="l">{c.label}</td><td>{fmt(c.share, "pct")}</td>
            <td>{fmt(c.usage.L ?? 0, "pct")}</td><td>{fmt(c.usage.R ?? 0, "pct")}</td>
            <td>{fmt(c.usage.L2K ?? 0, "pct")}</td><td>{fmt(c.usage.R2K ?? 0, "pct")}</td>
            <td>{num(c.means.rel_speed)?.toFixed(1) ?? "–"}</td>
            <td>{signed1(c.means.induced_vert_break)}</td>
            <td>{signed1(c.means.hb_arm)}</td>
            <td>{fmt(c.means.spin_rate, "n")}</td>
            <td>{num(c.means.vaa_adj)?.toFixed(1) ?? "–"}</td>
            <td>{num(c.means.rel_height)?.toFixed(2) ?? "–"}</td>
            <td>{c.n}</td>
            <td className="l note">{Object.entries(c.tag_mix).map(([k, v]) => `${k} ${Math.round(100 * v)}%`).join(", ") || "untagged"}</td>
          </tr>))}</tbody>
      </table></div>
      <p className="note">Clusters are found from movement (velo, induced vertical break, arm-side break, location-adjusted approach angle),
        not pitch tags; a cluster is named after its tag only when at least half its pitches carry that tag (&quot;Tagged as&quot; shows the mix).
        Positive arm-side break = toward her arm side. Usage is recency-weighted.</p>

      <h2>By pitch type</h2>
      {shown.filter((r) => r.side).map((r) => <Detail key={r.batter_tm_id} m={r} parts={parts} open={Boolean(one)} />)}

      <h2>Validation</h2>
      {Object.keys(val).length ? (
        <>
          <p className="note">Backtest: matchups built from earlier games, scored on later games. Treat columns that aren&apos;t validated as context, not a prediction.</p>
          <ul className="note">
            {Object.entries(val).map(([k, v]) => <li key={k}><b>{VAL_NAMES[k] ?? k}</b>: {v}</li>)}
            <li><b>Shape fit</b>: {parts ? `built only from validated outcomes (${parts})` : "hidden (no outcome passed the backtest)"}</li>
          </ul>
        </>
      ) : <p className="note">Not backtested yet: run <code>python -m matchup backtest --tune --apply</code> to check which columns predict later games.</p>}

      <h2>How to read this</h2>
      <p className="note"><b>Similar pitches</b>: for each of her pitch clusters, every pitch a hitter has seen from a same-handed pitcher is weighted by
        how closely it matches that cluster&apos;s shape from the hitter&apos;s side (velo, vertical and horizontal break toward/away from the hitter,
        release height and side, approach angle) and by how recent it is. <b>Rates are shrunk</b> toward how all same-side hitters did against
        that shape, adjusted for the hitter&apos;s overall skill, so a 3-for-5 sample doesn&apos;t read as a trend; hover shows the raw rate.{" "}
        <b>Pitcher adv.</b> = expected runs per 100 pitches vs her arsenal, as a percentile among qualified D1 hitters (higher = better for her);
        it mostly reflects how good the hitter is overall. <b>Shape fit</b> isolates the matchup itself: how the hitter does against pitches shaped
        like hers compared with what her overall level predicts, converted to runs per 100 pitches. It uses only the outcomes that passed the
        backtest ({parts || "none"}) (negative / blue = she handles these shapes worse than usual).{" "}
        <b>Whiff %</b> = misses per swing; <b>Chase %</b> = swings at pitches outside the zone; <b>Called K %</b> = called strikes per pitch;{" "}
        <b>Hard-hit %</b> = balls in play at {run.meta.hard_hit_mph.toFixed(1)}+ mph (top quarter of D1); <b>OPS*</b> = OPS on plate appearances
        that ended on a similar pitch. <b>Sample</b>: High ≥ 100, Medium 30–99, Low &lt; 30 similar pitches per pitch type.</p>
    </>
  );
}
