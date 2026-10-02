"use client";
import { useEffect, useMemo, useState } from "react";

type P = { pitcher_tm_id: string; pitcher_name: string; team: string | null; throws: string; is_home: boolean };
type R = { id: string; name: string; pa: number };

export default function Picker({ pitchers, teams, home, initial }: {
  pitchers: P[]; teams: string[]; home: string;
  initial?: { pitcher?: string; team?: string; batter?: string };
}) {
  const [pid, setPid] = useState(initial?.pitcher ?? pitchers.find((p) => p.is_home)?.pitcher_tm_id ?? "");
  const p = pitchers.find((x) => x.pitcher_tm_id === pid);
  const teamChoices = p && !p.is_home ? [home] : teams.filter((t) => t !== home || p?.is_home);
  const [team, setTeam] = useState(initial?.team ?? "");
  const [batter, setBatter] = useState(initial?.batter ?? "");
  const [roster, setRoster] = useState<R[]>([]);
  const effTeam = teamChoices.includes(team) ? team : teamChoices[0] ?? "";

  useEffect(() => {
    if (!effTeam) return;
    let live = true;
    fetch(`/api/roster?team=${encodeURIComponent(effTeam)}`)
      .then((r) => (r.ok ? r.json() : []))
      .then((x: R[]) => { if (live) setRoster(x); });
    return () => { live = false; };
  }, [effTeam]);

  const groups = useMemo(() => {
    const g = new Map<string, P[]>();
    for (const x of pitchers) {
      const k = x.is_home ? `${home} staff` : (x.team ?? "Unknown");
      g.set(k, [...(g.get(k) ?? []), x]);
    }
    return [...g.entries()];
  }, [pitchers, home]);

  return (
    <form className="pick" action="/matchup" method="get">
      <label>Pitcher
        <select name="pitcher" value={pid} onChange={(e) => { setPid(e.target.value); setBatter(""); }}>
          {groups.map(([g, ps]) => (
            <optgroup key={g} label={g}>
              {ps.map((x) => <option key={x.pitcher_tm_id} value={x.pitcher_tm_id}>{x.pitcher_name} ({x.throws}HP)</option>)}
            </optgroup>
          ))}
        </select>
      </label>
      <label>Opponent
        <select name="team" value={effTeam} onChange={(e) => { setTeam(e.target.value); setBatter(""); }}>
          {teamChoices.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
      </label>
      <label>Batter (optional)
        <select name="batter" value={batter} onChange={(e) => setBatter(e.target.value)}>
          <option value="">Whole lineup</option>
          {roster.map((r) => <option key={r.id} value={r.id}>{r.name} · {r.pa} PA</option>)}
        </select>
      </label>
      <button type="submit">Show matchup</button>
    </form>
  );
}
