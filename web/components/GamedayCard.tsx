"use client";
// Gameday card: the pitching team's staff across the top, the hitting team's batting order down the left.
// Per pitcher x batter: Pitcher adv. (percentile), xRV/100, Shape fit, Sample. The order lives in the URL
// (?l=id1,id2,...) so a finished card can be bookmarked, refreshed or printed.
import Link from "next/link";
import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import GamedayBack, { BACK_ROWS } from "./GamedayBack";
import type { Gameday, GamedayCell } from "@/lib/data";
import { num, runsStyle, scoreStyle, signed } from "@/lib/format";

const FIT_NAMES: Record<string, string> = { whiff: "whiff", called_strike: "called strike", chase: "chase", hard_hit: "hard hit" };

export default function GamedayCard({ pt, bt, data, slotPa, fitParts, initial, leagueOps }: {
  pt: string; bt: string; data: Gameday; slotPa: number[]; fitParts: string[]; initial: string[] | null;
  leagueOps?: number | null;
}) {
  const ids = useMemo(() => new Set(data.roster.map((r) => r.id)), [data.roster]);
  const byId = useMemo(() => new Map(data.roster.map((r) => [r.id, r])), [data.roster]);
  const alpha = useMemo(() => [...data.roster].sort((a, b) => a.name.localeCompare(b.name)), [data.roster]);
  const clean = (l: string[]) => Array.from({ length: 9 }, (_, i) => (l[i] && ids.has(l[i]) ? l[i] : ""));
  const [slots, setSlots] = useState<string[]>(() => clean(initial ?? data.lastLineup));

  useEffect(() => {
    const u = new URL(window.location.href);
    u.searchParams.set("pt", pt);
    u.searchParams.set("bt", bt);
    u.searchParams.set("l", slots.join(","));
    window.history.replaceState(null, "", u.toString());
  }, [slots, pt, bt]);

  // Back page: top 4 by innings (pitchers arrive sorted by IP), hitters = filled lineup slots in order,
  // then the most at-bats until the page is full.
  const [backPitchers, setBackPitchers] = useState<string[]>(() => {
    const ids = data.pitchers.slice(0, 4).map((p) => p.id);
    return [...ids, ...Array(4 - ids.length).fill("")];
  });
  const backRows = useMemo(() => {
    const inOrder = slots.filter(Boolean).map((id) => byId.get(id)!).filter(Boolean);
    const taken = new Set(inOrder.map((r) => r.id));
    const rest = data.roster.filter((r) => !taken.has(r.id)).sort((a, b) => (b.ab ?? b.pa) - (a.ab ?? a.pa));
    return [...inOrder, ...rest].slice(0, BACK_ROWS);
  }, [slots, byId, data.roster]);
  const slotOf = useMemo(() => new Map(slots.map((id, i) => [id, i + 1] as [string, number]).filter(([id]) => id)), [slots]);

  // Print: shrink the front table to fit one landscape page (wide staffs), restore afterwards.
  const front = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const before = () => {
      const el = front.current;
      if (!el) return;
      el.style.zoom = "1";
      // natural size of the table (not the 100%-wide screen layout), then fit letter landscape minus margins:
      // shrink a wide staff, enlarge a small one (max 1.4x)
      const t = el.querySelector("table");
      if (!t) return;
      const prev = t.style.width;
      t.style.width = "max-content";
      const w = t.getBoundingClientRect().width, h = el.getBoundingClientRect().height;
      t.style.width = prev;
      const z = Math.min((10.3 * 96) / w, (7.6 * 96) / h, 1.4);
      el.style.zoom = String(Math.round(z * 100) / 100);
    };
    const after = () => { if (front.current) front.current.style.zoom = ""; };
    window.addEventListener("beforeprint", before);
    window.addEventListener("afterprint", after);
    return () => { window.removeEventListener("beforeprint", before); window.removeEventListener("afterprint", after); };
  }, []);

  const set = (i: number, id: string) => setSlots((s) => s.map((x, j) => (j === i ? id : x === id && id ? "" : x)));
  const used = new Set(slots.filter(Boolean));
  const others = alpha.filter((r) => !used.has(r.id));
  const parts = fitParts.map((c) => FIT_NAMES[c] ?? c).join(" + ");

  // PA-weighted lineup totals per pitcher (slot i gets the league's expected PA for that slot)
  const totals = data.pitchers.map((p) => {
    let w = 0, s = 0, x = 0, f = 0, wf = 0;
    slots.forEach((id, i) => {
      const c = id ? data.cells[id]?.[p.id] : undefined;
      const sc = num(c?.score), xr = num(c?.xrv100), ft = num(c?.fit100);
      const wi = slotPa[i] ?? 1;
      if (sc !== null && xr !== null) { w += wi; s += wi * sc; x += wi * xr; }
      if (ft !== null) { wf += wi; f += wi * ft; }
    });
    return { score: w ? s / w : null, xrv: w ? x / w : null, fit: wf ? f / wf : null };
  });

  const bats = (id: string) => {
    const sides = new Set(Object.values(data.cells[id] ?? {}).map((c) => c.side).filter(Boolean));
    return sides.size > 1 ? "S" : ([...sides][0] ?? "–");
  };
  const cells = (id: string) => data.pitchers.map((p) => <Cells key={p.id} c={data.cells[id]?.[p.id]}
    href={`/matchup?pitcher=${encodeURIComponent(p.id)}&team=${encodeURIComponent(bt)}&batter=${encodeURIComponent(id)}`}
    parts={parts} />);

  const head = (label: string) => (
    <thead>
      <tr>
        <th className="stick" rowSpan={2}>{label}</th>
        <th className="l" rowSpan={2}>Bats</th>
        {data.pitchers.map((p) => (
          <th key={p.id} className="grp" colSpan={4} title={`${p.n.toLocaleString()} tracked pitches`}>
            {p.name} <span className="note">({p.throws}HP)</span>
          </th>))}
      </tr>
      <tr>
        {data.pitchers.map((p) => (
          <Fragment key={p.id}>
            <th className="g0" title="Pitcher advantage percentile (50 = average hitter, higher = better for the pitcher)">Adv</th>
            <th title="Expected runs per 100 pitches vs her arsenal (+ favors hitter)">xRV</th>
            <th title={parts ? `Shape fit (${parts})` : "Shape fit: no validated components"}>Fit</th>
            <th title="Similar pitches seen (usage-weighted)">Sample</th>
          </Fragment>))}
      </tr>
    </thead>
  );

  return (
    <>
      <div className="toolbar">
        <button type="button" className="ghost" onClick={() => setSlots(clean(data.lastLineup))}>Use {bt}&apos;s last lineup</button>
        <button type="button" className="ghost" onClick={() => setSlots(Array(9).fill(""))}>Clear lineup</button>
        <button type="button" className="ghost" onClick={() => window.print()}>Print</button>
        <span className="note">{pt} pitchers vs {bt}. Hover a header for its meaning; click a number for the full matchup.</span>
      </div>

      <div className="gd-front" ref={front}>
      <div className="print-only gd-title"><b>{pt} pitchers vs {bt}</b> · Gameday card</div>
      <div className="legend">
        <span><i className="sw" style={{ background: "rgba(var(--pitch),.4)" }} />better for the pitcher</span>
        <span><i className="sw" style={{ background: "rgba(var(--hit),.4)" }} />better for the hitter</span>
      </div>
      <div className="wrap"><table className="gd">
        {head("Lineup")}
        <tbody>
          {slots.map((id, i) => (
            <tr key={i}>
              <td className="stick">
                <span style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <b style={{ width: 14 }}>{i + 1}</b>
                  <select aria-label={`Batting ${i + 1}`} value={id} onChange={(e) => set(i, e.target.value)}>
                    <option value="">—</option>
                    {alpha.filter((r) => r.id === id || !used.has(r.id)).map((r) => (
                      <option key={r.id} value={r.id}>{r.name}</option>))}
                  </select>
                </span>
              </td>
              <td className="l">{id ? bats(id) : ""}</td>
              {id ? cells(id) : data.pitchers.map((p) => <Fragment key={p.id}><td className="g0" /><td /><td /><td /></Fragment>)}
            </tr>
          ))}
          <tr className="total">
            <td className="stick" title={`Weighted by expected plate appearances per slot (${slotPa.map((x) => x.toFixed(1)).join(", ")})`}>
              Lineup (PA-weighted)</td>
            <td />
            {totals.map((t, k) => (
              <Fragment key={k}>
                <td className="g0" style={scoreStyle(t.score)}>{t.score === null ? "–" : t.score.toFixed(0)}</td>
                <td>{signed(t.xrv)}</td>
                <td style={parts ? runsStyle(t.fit, 0.3) : undefined}>{parts ? signed(t.fit) : "–"}</td>
                <td />
              </Fragment>))}
          </tr>
          <tr className="gd-sec">
            <td className="stick"><b>Not in the lineup ({others.length})</b></td>
            <td colSpan={1 + 4 * data.pitchers.length} className="l note">
              {others.length ? "alphabetical" : "everyone on the roster is in the lineup"}</td>
          </tr>
          {others.map((r) => (
            <tr key={r.id}>
              <td className="stick">{r.name} <span className="note">{byId.get(r.id)?.pa ?? 0} PA</span></td>
              <td className="l">{bats(r.id)}</td>
              {cells(r.id)}
            </tr>))}
        </tbody>
      </table></div>

      </div>

      <div className="noprint">
      <p className="note"><b>Adv</b> = pitcher advantage, a percentile among qualified D1 hitters (50 = average, higher = better for the pitcher).{" "}
        <b>xRV</b> = expected runs per 100 pitches vs her arsenal (+ favors the hitter). <b>Fit</b> = Shape fit, the matchup without the
        hitter&apos;s overall talent ({parts || "hidden: nothing validated"}; negative / blue = she handles these shapes worse than usual).{" "}
        <b>Sample</b> = similar pitches seen per pitch type: High ≥ 100, Med 30–99, Low &lt; 30. Switch hitters (S) are evaluated from the
        side they bat against that pitcher&apos;s hand.</p>
      </div>

      <h2 className="noprint">Back page: location zones</h2>
      <GamedayBack pt={pt} bt={bt} data={data} rows={backRows} slotOf={slotOf} pitcherIds={backPitchers}
        setPitcherIds={setBackPitchers} center={leagueOps ?? 0.7} />
    </>
  );
}

function Cells({ c, href, parts }: { c?: GamedayCell; href: string; parts: string }) {
  if (!c) return <><td className="g0" title="No matchup published">–</td><td /><td /><td /></>;
  const conf = c.conf === "Medium" ? "Med" : (c.conf ?? "–");
  return (
    <>
      <td className="g0" style={scoreStyle(c.score)}><Link href={href}><b>{num(c.score)?.toFixed(0) ?? "–"}</b></Link></td>
      <td>{signed(c.xrv100)}</td>
      <td style={parts ? runsStyle(c.fit100, 0.3) : undefined}>{parts ? signed(c.fit100) : "–"}</td>
      <td className={`smp conf-${c.conf}`}>{conf} {num(c.sim) !== null ? Math.round(c.sim as number) : ""}</td>
    </>
  );
}
