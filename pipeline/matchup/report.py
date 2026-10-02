"""Self-contained HTML matchup report (one file, opens in any browser, prints cleanly).

Color is a secondary cue only: blue tint = better than similar hitters for the PITCHER, red tint = better
for the HITTER, gray/none = about average. Every number is printed; hovering a cell shows what it is
compared against and the sample behind it.
"""
from __future__ import annotations

import html
from datetime import datetime

import numpy as np
import pandas as pd

from .engine import Result

# metric -> (label, higher value is good for the pitcher?, difference that earns full tint, formatter)
COLS = {
    "whiff": ("Whiff %", True, 0.10, "pct"),
    "chase": ("Chase %", True, 0.08, "pct"),
    "called_strike": ("Called K %", True, 0.05, "pct"),
    "hard_hit": ("Hard-hit %", False, 0.10, "pct"),
    "ops": ("OPS*", False, 0.20, "ops"),
}

CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--mid:#f0efec;--pitch:42,120,214;--hit:227,73,72}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#383835;--mid:#383835;--pitch:57,135,229;--hit:230,103,103}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}h3{font-size:14px;margin:0}
.sub{color:var(--ink2);margin:0 0 16px}.note{color:var(--ink2);font-size:12px}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
.kpi{border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:170px}
.kpi b{display:block;font-size:22px}.kpi span{color:var(--ink2);font-size:12px}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{font-size:12px;color:var(--ink2);font-weight:600;position:sticky;top:0;background:var(--bg)}
td.l,th.l{text-align:left}tr.bench td{color:var(--ink2)}
.legend{display:flex;gap:16px;align-items:center;font-size:12px;color:var(--ink2);margin:6px 0}
.sw{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-3px;margin-right:4px}
details{border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:8px 12px}summary{cursor:pointer}
.conf-Low{color:var(--ink2)}.badge{font-size:11px;border:1px solid var(--line);border-radius:10px;padding:1px 6px;color:var(--ink2)}
@media print{details{break-inside:avoid}details>*{display:block}}
"""


def _fmt(v, kind):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if kind == "pct":
        return f"{100 * v:.0f}%"
    if kind == "ops":
        return f"{v:.3f}".lstrip("0") if v < 1 else f"{v:.3f}"
    if kind == "rv":
        return f"{v:+.2f}"
    return f"{v:.0f}"


def _tint(v, pop, pitcher_good_high, full):
    if v is None or pop is None or not (np.isfinite(v) and np.isfinite(pop)):
        return ""
    d = (v - pop) / full
    if abs(d) < 0.15:
        return ""
    good_for_pitcher = (d > 0) == pitcher_good_high
    a = min(0.45, 0.10 + 0.35 * min(1.0, abs(d)))
    return f"background:rgba(var({'--pitch' if good_for_pitcher else '--hit'}),{a:.2f})"


def _cell(v, pop, metric, n=None, extra=""):
    label, high_good, full, kind = COLS[metric]
    title = f"{label}: {_fmt(v, kind)} vs {_fmt(pop, kind)} for similar hitters"
    if n is not None:
        title += f"; {n:.0f} similar pitches"
    return f'<td style="{_tint(v, pop, high_good, full)}" title="{html.escape(title + extra)}">{_fmt(v, kind)}</td>'


def _score_cell(s):
    if s is None or not np.isfinite(s):
        return '<td title="No history vs this pitcher hand">–</td>'
    d = (s - 50) / 50
    style = "" if abs(d) < 0.1 else f"background:rgba(var({'--pitch' if d > 0 else '--hit'}),{min(0.5, 0.1 + 0.4 * abs(d)):.2f})"
    return f'<td style="{style}" title="Pitcher advantage: better matchup for the pitcher than {s:.0f}% of qualified hitters"><b>{s:.0f}</b></td>'


def _batter_rows(b: pd.DataFrame, slot_of: dict, bench: bool) -> str:
    out = []
    for _, r in b.iterrows():
        slot = slot_of.get(r["batter_tm_id"], "")
        cells = [f'<td class="l">{slot}</td>', f'<td class="l">{html.escape(str(r["batter_name"]))}</td>',
                 f'<td class="l">{r["side"] or "–"}</td>', _score_cell(r.get("score")),
                 f'<td title="Expected runs per 100 pitches vs this arsenal (+ favors hitter); similar hitters {_fmt(100 * r.get("pop_rv", np.nan), "rv")}">{_fmt(r.get("xrv100"), "rv")}</td>']
        for m in COLS:
            cells.append(_cell(r.get(m), r.get("pop_" + m), m, r.get("sim_pitches")))
        for m in ("whiff", "chase"):
            cells.append(_cell(r.get(m + "_2k"), r.get("pop_" + m + "_2k"), m, r.get("sim_pitches_2k"), " (two strikes)"))
        cells.append(f'<td class="conf-{r["confidence"]}" title="Usage-weighted similar pitches seen per pitch type">'
                     f'{r["confidence"]} <span class="badge">{r["sim_pitches"]:.0f}</span></td>')
        cells.append(f'<td class="l" title="Plate appearances vs this pitcher">{r["direct_pa"]} PA {html.escape(r["direct_line"])}</td>'
                     if r["direct_pa"] else '<td class="l">–</td>')
        out.append(f'<tr class="{"bench" if bench else ""}">' + "".join(cells) + "</tr>")
    return "\n".join(out)


def _detail(res: Result, bid: str, name: str, side: str) -> str:
    d = res.detail[(res.detail["batter_tm_id"] == bid) & (res.detail["side"] == side)]
    if not len(d):
        return ""
    rows = []
    for split, title in (("all", "All counts"), ("2k", "Two strikes")):
        rows.append(f'<tr><td class="l" colspan="9"><b>{title}</b></td></tr>')
        for _, r in d[d["split"] == split].sort_values("usage", ascending=False).iterrows():
            cells = [f'<td class="l">{html.escape(r["label"])}</td>', f"<td>{100 * r['usage']:.0f}%</td>",
                     f"<td>{r['sim_pitches']:.0f}</td>"]
            for m in COLS:
                raw = r.get("raw_" + m)
                cells.append(_cell(r[m], r["pop_" + m], m, r["sim_pitches"],
                                   f"; her raw rate {_fmt(raw, COLS[m][3])} before shrinking"))
            cells.append(f"<td title='runs per 100 pitches'>{_fmt(100 * r['rv'], 'rv')}</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
    head = "".join(f"<th>{COLS[m][0]}</th>" for m in COLS)
    return (f"<details><summary><h3 style='display:inline'>{html.escape(name)}</h3> "
            f"<span class='note'>bats {side} · by pitch type</span></summary><div class='wrap'><table>"
            f"<tr><th class='l'>Pitch</th><th>Usage vs {side}HH</th><th>Similar pitches</th>{head}<th>RV/100</th></tr>"
            + "\n".join(rows) + "</table></div></details>")


def render(res: Result, team: str, lineup: list[str], slot_pa: np.ndarray, lg, data_through, n_games: int) -> str:
    a = res.arsenal
    b = res.batters.copy()
    slot_of = {bid: i + 1 for i, bid in enumerate(lineup)}
    in_lineup = b[b["batter_tm_id"].isin(lineup)].copy()
    in_lineup["_o"] = in_lineup["batter_tm_id"].map(slot_of)
    in_lineup = in_lineup.sort_values("_o")
    bench = b[~b["batter_tm_id"].isin(lineup)].sort_values("score")

    # Lineup-weighted team view: expected plate appearances by slot.
    w = np.array([slot_pa[slot_of[x] - 1] for x in in_lineup["batter_tm_id"]]) if len(in_lineup) else np.array([])
    ok = in_lineup["xrv100"].notna().to_numpy() if len(in_lineup) else np.array([], bool)
    team_xrv = float((w[ok] * in_lineup["xrv100"].to_numpy()[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
    team_score = float((w[ok] * in_lineup["score"].to_numpy()[ok]).sum() / w[ok].sum()) if ok.any() else np.nan

    ars_rows = []
    for c in a.clusters:
        m = c.means
        ars_rows.append(
            f"<tr><td class='l'>{html.escape(c.label)}</td><td>{100 * c.share:.0f}%</td>"
            f"<td>{100 * c.usage.get('L', 0):.0f}%</td><td>{100 * c.usage.get('R', 0):.0f}%</td>"
            f"<td>{100 * c.usage.get('L2K', 0):.0f}%</td><td>{100 * c.usage.get('R2K', 0):.0f}%</td>"
            f"<td>{m['rel_speed']:.1f}</td><td>{m['induced_vert_break']:+.1f}</td><td>{m['hb_arm']:+.1f}</td>"
            f"<td>{m['spin_rate']:.0f}</td><td>{m['vaa_adj']:.1f}</td><td>{m['rel_height']:.2f}</td><td>{c.n}</td>"
            f"<td class='l note'>{html.escape(', '.join(f'{k} {100 * v:.0f}%' for k, v in c.tag_mix.items()) or 'untagged')}</td></tr>")

    head = ("<tr><th class='l'>#</th><th class='l'>Batter</th><th class='l'>Bats</th><th>Pitcher adv.</th><th>xRV/100</th>"
            + "".join(f"<th>{COLS[m][0]}</th>" for m in COLS)
            + "<th>2K Whiff %</th><th>2K Chase %</th><th>Sample</th><th class='l'>Vs her</th></tr>")
    details = "\n".join(_detail(res, r["batter_tm_id"], r["batter_name"], r["side"])
                        for _, r in pd.concat([in_lineup, bench]).iterrows() if r["side"])
    hand = "RHP" if a.throws == "R" else "LHP"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(a.pitcher_name)} vs {html.escape(team)}</title><style>{CSS}</style></head><body><main>
<h1>{html.escape(a.pitcher_name)} ({hand}) vs {html.escape(team)}</h1>
<p class="sub">Regular-season data through {data_through} · {n_games:,} games · generated {datetime.now():%b %d, %Y %H:%M}</p>
<div class="kpis">
 <div class="kpi"><b>{_fmt(team_score, 'n')}</b><span>Lineup pitcher advantage (50 = avg)</span></div>
 <div class="kpi"><b>{_fmt(team_xrv, 'rv')}</b><span>Lineup expected runs / 100 pitches</span></div>
 <div class="kpi"><b>{a.n_pitches:,}</b><span>Tracked pitches describing her arsenal</span></div>
</div>
<div class="legend"><span><i class="sw" style="background:rgba(var(--pitch),.4)"></i>better for the pitcher than similar hitters</span>
<span><i class="sw" style="background:rgba(var(--hit),.4)"></i>better for the hitter</span><span>no tint = about average · hover any cell for the comparison</span></div>

<h2>Lineup (most recent game) and bench</h2>
<div class="wrap"><table>{head}
{_batter_rows(in_lineup, slot_of, False)}
{_batter_rows(bench, slot_of, True)}
</table></div>
<p class="note">Lineup = batting order from {html.escape(team)}'s most recent game; team numbers weight each slot by its expected plate appearances
({", ".join(f"{x:.1f}" for x in slot_pa)}). Bench = everyone else who batted for them this season.</p>

<h2>Her arsenal (movement clusters)</h2>
<div class="wrap"><table><tr><th class="l">Pitch</th><th>Overall</th><th>vs LHH</th><th>vs RHH</th><th>vs LHH 2K</th><th>vs RHH 2K</th>
<th>Velo</th><th>IVB (in)</th><th>Arm-side HB (in)</th><th>Spin</th><th>VAA adj.</th><th>Rel. ht</th><th>n</th><th class="l">Tagged as</th></tr>
{"".join(ars_rows)}</table></div>
<p class="note">Clusters are found from movement (velo, induced vertical break, arm-side break, location-adjusted approach angle), not pitch tags;
a cluster is named after its tag only when at least half its pitches carry that tag ("Tagged as" shows the mix).
Positive arm-side break = toward her arm side. Usage is recency-weighted.</p>

<h2>By pitch type</h2>
{details}

<h2>How to read this</h2>
<p class="note"><b>Similar pitches</b>: for each of her pitch clusters, every pitch a hitter has seen from a same-handed pitcher is weighted by
how closely it matches that cluster's shape from the hitter's side (velo, vertical and horizontal break toward/away from the hitter,
release height and side, approach angle) and by how recent it is. <b>Rates are shrunk</b> toward how all same-side hitters did against
that shape, adjusted for the hitter's overall skill, so a 3-for-5 sample doesn't read as a trend; hover shows the raw rate.
<b>Pitcher adv.</b> = expected runs per 100 pitches vs her arsenal, as a percentile among qualified D1 hitters (higher = better for her).
<b>Whiff %</b> = misses per swing; <b>Chase %</b> = swings at pitches outside the zone; <b>Called K %</b> = called strikes per pitch;
<b>Hard-hit %</b> = balls in play at {lg.hard_hit_mph:.1f}+ mph (top quarter of D1); <b>OPS*</b> = OPS on plate appearances that ended on a
similar pitch. <b>Sample</b>: High ≥ 100, Medium 30–99, Low &lt; 30 similar pitches per pitch type.</p>
</main></body></html>"""
