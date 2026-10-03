"use client";
import { useState } from "react";

export default function GamedayPicker({ pTeams, bTeams, home, pt, bt }: {
  pTeams: string[]; bTeams: string[]; home: string; pt: string; bt: string;
}) {
  const [p, setP] = useState(pt);
  const [b, setB] = useState(bt);
  // Published coverage: home pitchers face everyone; other teams' pitchers only face the home team.
  const choices = p === home ? bTeams.filter((t) => t !== home) : [home];
  const bVal = choices.includes(b) ? b : (p === home ? "" : home);
  return (
    <form className="pick noprint" action="/gameday" method="get">
      <label>Pitching team
        <select name="pt" value={p} onChange={(e) => setP(e.target.value)}>
          {pTeams.map((t) => <option key={t} value={t}>{t}{t === home ? " (home)" : ""}</option>)}
        </select>
      </label>
      <label>Hitting team
        <select name="bt" value={bVal} onChange={(e) => setB(e.target.value)}>
          {p === home && <option value="">Choose…</option>}
          {choices.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
      </label>
      <button type="submit" disabled={!bVal}>Load card</button>
    </form>
  );
}
