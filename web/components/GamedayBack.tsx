"use client";
// Back page of the Gameday card (portrait): top pitchers across, hitters down; per pitcher two 5x5 OPS zones
// (vs her whole arsenal, vs her changeup), built only from pitches shaped like hers (pipeline/matchup/zones.py).
import { Fragment } from "react";
import type { Gameday, Num, RosterEntry, Zones } from "@/lib/data";

export const BACK_ROWS = 10;       // fits letter portrait at the zone size below
const OWN_PA = 2;                  // her own weighted PAs in a cell before its number prints dark

// Zone geometry (viewBox units): inner strike-zone cells are wide, the chase ring is a narrow band.
const RING = 12, CELL = 25.33;
const EDGES = [0, RING, RING + CELL, RING + 2 * CELL, RING + 3 * CELL, 2 * RING + 3 * CELL];
const SIZE = EDGES[5];

function opsColor(v: Num, center: number): string | undefined {
  if (v === null || !Number.isFinite(v)) return undefined;
  const d = (v - center) / 0.35;
  if (Math.abs(d) < 0.12) return "rgba(var(--mid-rgb),1)";
  const a = Math.min(0.8, 0.15 + 0.6 * Math.min(1, Math.abs(d)));
  return `rgba(var(${d > 0 ? "--hit" : "--pitch"}),${a.toFixed(2)})`;
}

const opsText = (v: Num) => (v === null || !Number.isFinite(v) ? "" : v < 1 ? v.toFixed(2).replace(/^0/, "") : v.toFixed(2));

export function Zone({ ops, pa, center, label }: { ops: Num[] | null; pa: Num[] | null; center: number; label: string }) {
  if (!ops) {
    return (
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="zone" role="img" aria-label={`${label}: no data`}>
        <rect x={0.5} y={0.5} width={SIZE - 1} height={SIZE - 1} className="zone-empty" />
        <text x={SIZE / 2} y={SIZE / 2 + 3} textAnchor="middle" className="zone-na">{label === "CH" ? "no CH" : "–"}</text>
      </svg>
    );
  }
  return (
    <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="zone" role="img" aria-label={`${label} OPS by location, pitcher's view`}>
      {ops.map((v, i) => {
        const r = Math.floor(i / 5), c = i % 5;
        const x = EDGES[c], y = EDGES[r], w = EDGES[c + 1] - x, h = EDGES[r + 1] - y;
        const inner = r >= 1 && r <= 3 && c >= 1 && c <= 3;
        const own = (pa?.[i] ?? 0) >= OWN_PA;
        return (
          <g key={i}>
            <rect x={x} y={y} width={w} height={h} fill={opsColor(v, center) ?? "transparent"} className="zc">
              <title>{`${label} OPS ${opsText(v) || "–"} · her own PAs here vs similar pitches: ${(pa?.[i] ?? 0).toFixed(1)}`}</title>
            </rect>
            {inner && <text x={x + w / 2} y={y + h / 2 + 3.6} textAnchor="middle" className={own ? "zt own" : "zt"}>{opsText(v)}</text>}
          </g>
        );
      })}
      <rect x={EDGES[1]} y={EDGES[1]} width={3 * CELL} height={3 * CELL} className="zone-k" />
    </svg>
  );
}

export default function GamedayBack({ pt, bt, data, rows, slotOf, pitcherIds, setPitcherIds, center }: {
  pt: string; bt: string; data: Gameday; rows: RosterEntry[]; slotOf: Map<string, number>;
  pitcherIds: string[]; setPitcherIds: (ids: string[]) => void; center: number;
}) {
  const byId = new Map(data.pitchers.map((p) => [p.id, p]));
  const ps = pitcherIds.map((id) => byId.get(id)).filter((p): p is Gameday["pitchers"][number] => Boolean(p));
  const anyZones = Object.values(data.zones).some((m) => Object.values(m).some(Boolean));
  const bats = (id: string) => {
    const s = new Set(Object.values(data.cells[id] ?? {}).map((c) => c.side).filter(Boolean));
    return s.size > 1 ? "S" : ([...s][0] ?? "–");
  };
  const ip = (v: number | null) => (v === null ? "" : `${Math.floor(v)}.${Math.round((v % 1) * 3)} IP`);

  return (
    <section className="backpage">
      <div className="toolbar noprint">
        <b>Back page pitchers</b>
        {[0, 1, 2, 3].map((k) => (
          <select key={k} aria-label={`Back page pitcher ${k + 1}`} value={pitcherIds[k] ?? ""}
            onChange={(e) => setPitcherIds(pitcherIds.map((x, j) => (j === k ? e.target.value : x === e.target.value ? pitcherIds[k] : x)))}>
            <option value="">—</option>
            {data.pitchers.map((p) => <option key={p.id} value={p.id}>{p.name}{p.ip !== null ? ` · ${ip(p.ip)}` : ""}</option>)}
          </select>))}
        <span className="note">Defaults to the top 4 by innings pitched this season.</span>
      </div>
      <div className="bp-head">
        <div>
          <h2 style={{ margin: 0 }}>{pt} pitchers vs {bt}: OPS by location vs similar pitches</h2>
          <div className="note">Pitcher&apos;s view (RHH stand on the right). Inner 3×3 = strike zone; outer band = chase.
            <b> All</b> = pitches shaped like her arsenal, <b>CH</b> = like her changeup.
            Dark numbers: 2+ of her own PAs there; gray: mostly the league pattern for these shapes plus her overall level.</div>
        </div>
        <div className="bp-scale" aria-label="Color scale">
          <span>OPS</span><i style={{ background: opsColor(center - 0.35, center) }} />
          <i style={{ background: opsColor(center - 0.15, center) }} /><i style={{ background: opsColor(center, center) }} />
          <i style={{ background: opsColor(center + 0.15, center) }} /><i style={{ background: opsColor(center + 0.35, center) }} />
          <span>{opsText(center - 0.35)} · {opsText(center)} · {opsText(center + 0.35)}</span>
        </div>
      </div>
      {!anyZones && <div className="empty">Location zones aren&apos;t in this publish yet. Run <code>python -m matchup migrate</code> and
        <code> python -m matchup publish</code> on your Mac.</div>}
      <table className="bp">
        <thead>
          <tr>
            <th className="l" rowSpan={2}>Batter</th>
            {ps.map((p) => <th key={p.id} colSpan={2} className="bp-p">{p.name} <span className="note">({p.throws}HP{p.ip !== null ? ` · ${ip(p.ip)}` : ""})</span></th>)}
          </tr>
          <tr>{ps.map((p) => <Fragment key={p.id}><th className="bp-p">All</th><th>{p.hasChangeup ? "CH" : "no CH"}</th></Fragment>)}</tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id}>
              <td className="l bp-name">
                {slotOf.has(r.id) ? <b>{slotOf.get(r.id)}. </b> : <span className="note">{r.ab ?? r.pa} AB · </span>}
                {r.name} <span className="note">({bats(r.id)})</span>
              </td>
              {ps.map((p) => {
                const z: Zones | null | undefined = data.zones[r.id]?.[p.id];
                return (
                  <Fragment key={p.id}>
                    <td className="bp-p"><Zone ops={z?.[0] ?? null} pa={z?.[1] ?? null} center={center} label="All" /></td>
                    <td><Zone ops={z?.[2] ?? null} pa={z?.[3] ?? null} center={center} label="CH" /></td>
                  </Fragment>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note bp-foot">OPS on plate appearances that ended on a pitch in that cell, weighted by how closely each pitch matched her
        shapes (same similarity as the front). Each cell is shrunk toward same-side hitters vs those shapes, adjusted for the hitter&apos;s overall
        OBP/SLG, so small samples stay near that. Location OPS has not been backtested: read it as tendencies, not predictions.</p>
    </section>
  );
}
