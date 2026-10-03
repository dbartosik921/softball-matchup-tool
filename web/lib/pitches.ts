// Short pitch names for the card: Riseball -> RB, Changeup -> CH ... ('*' = named from shape, not tags).
import type { PitchUse } from "./data";

const BASE: [RegExp, string][] = [
  [/^rise/i, "RB"], [/^drop ?curve/i, "DC"], [/^drop/i, "DB"], [/^change/i, "CH"], [/^curve/i, "CB"],
  [/^screw/i, "SC"], [/^fast/i, "FB"], [/^slider/i, "SL"], [/^knuckle/i, "KN"], [/^cut/i, "CT"],
];

export function abbrevs(pitches: PitchUse[]): Map<number, string> {
  const raw = pitches.map((p) => {
    const hit = BASE.find(([re]) => re.test(p.label));
    const ab = hit ? hit[1] : p.label.replace(/[^A-Za-z]/g, "").slice(0, 2).toUpperCase();
    return [p.cid, ab + (/-type/.test(p.label) ? "*" : "")] as [number, string];
  });
  const count = new Map<string, number>();
  raw.forEach(([, a]) => count.set(a, (count.get(a) ?? 0) + 1));
  const seen = new Map<string, number>();
  return new Map(raw.map(([cid, a]) => {
    if ((count.get(a) ?? 0) < 2) return [cid, a];
    const i = (seen.get(a) ?? 0) + 1;
    seen.set(a, i);
    return [cid, `${a}${i}`];
  }));
}

/** "RB 40 · DB 35 · CH 25" for a usage key (e.g. 'L', 'RFP'). */
export function mix(pitches: PitchUse[], key: string, ab: Map<number, string>, top = 4): string {
  const xs = pitches.map((p) => [ab.get(p.cid) ?? "?", p.usage?.[key] ?? 0] as [string, number])
    .filter(([, u]) => u >= 0.05).sort((a, b) => b[1] - a[1]).slice(0, top);
  return xs.length ? xs.map(([a, u]) => `${a} ${Math.round(100 * u)}`).join(" · ") : "–";
}
