// Shared number formatting and tints (blue = good for the pitcher, red = good for the hitter).
import type { CSSProperties } from "react";

export const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

export function signed(v: unknown, d = 2): string {
  const x = num(v);
  return x === null ? "–" : (x >= 0 ? "+" : "") + x.toFixed(d);
}

/** Pitcher advantage percentile: 50 = average hitter, higher = better for the pitcher. */
/** Colour strength by sample: thin samples get a faint tint so they don't read like solid evidence. */
export const sampleWeight = (conf: string | null | undefined) => (conf === "High" ? 1 : conf === "Medium" ? 0.7 : 0.35);

export function scoreStyle(s: unknown, k = 1): CSSProperties | undefined {
  const x = num(s);
  if (x === null) return undefined;
  const d = (x - 50) / 50;
  if (Math.abs(d) < 0.1) return undefined;
  return { background: `rgba(var(${d > 0 ? "--pitch" : "--hit"}),${(k * Math.min(0.5, 0.1 + 0.4 * Math.abs(d))).toFixed(2)})` };
}

/** Runs-for-the-hitter scale (xRV, Shape fit): negative = good for the pitcher. */
export function runsStyle(v: unknown, full: number, k = 1): CSSProperties | undefined {
  const x = num(v);
  if (x === null) return undefined;
  const d = x / full;
  if (Math.abs(d) < 0.15) return undefined;
  const a = k * Math.min(0.45, 0.1 + 0.35 * Math.min(1, Math.abs(d)));
  return { background: `rgba(var(${d < 0 ? "--pitch" : "--hit"}),${a.toFixed(2)})` };
}
