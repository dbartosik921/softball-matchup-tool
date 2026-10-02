"""Synthetic softball league for testing the matchup engine.

Pitchers have known pitch shapes (arm-relative), both hands are represented, one switch hitter per team,
and three Auburn hitters carry planted tendencies the engine must recover:
  'Weak, Rise'    whiffs far more than average against riseballs
  'Weak, Change'  whiffs far more than average against changeups
  'Masher, Drop'  hits dropballs much harder than average
Rows go through the real feature code (matchup.features.add_features), so conventions match production.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from matchup.features import add_features
from matchup.ingest import season_for

# arm-relative shape: velo (mph), induced vertical break (in), horizontal break toward arm side (in), spin
SHAPES = {
    "Riseball": (65.0, 6.0, 1.0, 1400),
    "Dropball": (63.0, -4.0, 3.0, 1300),
    "Curveball": (60.0, 1.0, -8.0, 1450),
    "Changeup": (54.0, -3.0, 4.0, 1100),
}
BASE_WHIFF = {"Riseball": 0.26, "Dropball": 0.17, "Curveball": 0.22, "Changeup": 0.28}
TEAMS = ["UNI_ARK_SB", "AUB_TIG_SB", "TEAM_C_SB", "TEAM_D_SB", "TEAM_E_SB", "TEAM_F_SB"]
PLANTED = {"Weak, Rise": ("whiff", "Riseball"), "Weak, Change": ("whiff", "Changeup"), "Masher, Drop": ("power", "Dropball")}


def _roster(rng):
    pitchers, batters = [], []
    pid = 100000000100
    for t_i, team in enumerate(TEAMS):
        for k in range(3):
            pid += 1
            hand = "L" if k == 2 else "R"
            kinds = list(rng.choice(list(SHAPES), size=3, replace=False))
            name = f"P{t_i}{k}, Pitcher"
            if team == "UNI_ARK_SB" and k == 0:
                name, kinds = "Burnham, Payton", ["Riseball", "Dropball", "Changeup"]
            usage = rng.dirichlet(np.ones(3) * 4)
            offsets = {kd: rng.normal(0, [1.0, 0.8, 0.8]) for kd in kinds}
            pitchers.append(dict(id=str(pid), name=name, team=team, hand=hand, kinds=kinds, usage=usage, offsets=offsets))
        for k in range(10):
            pid += 1
            side = "S" if k == 8 else ("L" if k in (0, 3, 6, 7) else "R")
            name = f"B{t_i}{k}, Batter"
            planted = None
            if team == "AUB_TIG_SB" and k < 3:
                name = list(PLANTED)[k]
                planted = PLANTED[name]
            batters.append(dict(id=str(pid), name=name, team=team, side=side, skill=rng.normal(1.0, 0.12),
                                power=rng.normal(0, 3), planted=planted))
    return pitchers, batters


def make_league(n_rounds: int = 4, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pitchers, batters = _roster(rng)
    rows = []
    gday = date(2026, 2, 13)
    gnum = 0
    for _ in range(n_rounds):
        for i, home in enumerate(TEAMS):
            for away in TEAMS[i + 1:]:
                gnum += 1
                gday += timedelta(days=1)
                rows += _game(rng, gnum, gday, home, away, pitchers, batters)
    df = pd.DataFrame(rows)
    return add_features(df)


def _game(rng, gnum, gday, home, away, pitchers, batters):
    out, pitch_no = [], 0
    gid = f"synth-{gnum:04d}"
    staff = {t: [p for p in pitchers if p["team"] == t] for t in (home, away)}
    starter = {t: staff[t][gnum % 3] for t in (home, away)}
    lineup = {t: [b for b in batters if b["team"] == t][:9] for t in (home, away)}
    spot = {home: 0, away: 0}
    for inning in range(1, 8):
        for half, bat_team, field_team in (("Top", away, home), ("Bottom", home, away)):
            p = starter[field_team]
            outs, bases, pa_of_inning = 0, [], 0
            while outs < 3:
                b = lineup[bat_team][spot[bat_team] % 9]
                spot[bat_team] += 1
                pa_of_inning += 1
                side = ("R" if p["hand"] == "L" else "L") if b["side"] == "S" else b["side"]
                result, pitches = _pa(rng, p, b, side)
                runs = 0
                if result in ("Single", "Double", "HomeRun", "Walk"):
                    adv = {"Single": 1, "Walk": 1, "Double": 2, "HomeRun": 4}[result]
                    new = [r + adv for r in bases] + [adv]
                    runs = sum(1 for r in new if r >= 4)
                    bases = [r for r in new if r < 4]
                else:
                    outs += 1
                for j, pt in enumerate(pitches, 1):
                    pitch_no += 1
                    last = j == len(pitches)
                    pt.update(
                        pitch_uid=f"{gid}-{pitch_no}", game_uid=gid, game_date=gday, season=season_for(gday),
                        game_type="regular", inning=inning, top_bottom=half, pa_of_inning=pa_of_inning,
                        pitch_of_pa=j, pitch_no=pitch_no, outs=outs,
                        pitcher_tm_id=p["id"], pitcher_name=p["name"], pitcher_team=p["team"], p_throws=p["hand"],
                        batter_tm_id=b["id"], batter_name=b["name"], batter_team=b["team"], b_side=side,
                        runs_scored=runs if last else 0,
                    )
                    out.append(pt)
    return out


def _pa(rng, p, b, side):
    balls = strikes = 0
    pitches = []
    p_sign = 1.0 if p["hand"] == "R" else -1.0
    while True:
        kind = rng.choice(p["kinds"], p=p["usage"])
        velo, ivb, hb_arm, spin = SHAPES[kind]
        off = p["offsets"][kind]
        velo += off[0] + rng.normal(0, 1.0)
        ivb += off[1] + rng.normal(0, 1.2)
        hb_arm += off[2] + rng.normal(0, 1.2)
        in_target = rng.random() < 0.5
        side_loc = rng.normal(0, 0.45 if in_target else 0.95)
        height = rng.normal(2.25, 0.4 if in_target else 0.9)
        in_zone = abs(side_loc) <= 0.71 and 1.5 <= height <= 3.0
        rel_side = 0.6 * p_sign + rng.normal(0, 0.08)
        row = dict(
            tagged_pitch_type=kind, balls=balls, strikes=strikes,
            rel_speed=velo, spin_rate=spin + rng.normal(0, 60), rel_height=1.8 + rng.normal(0, 0.08),
            rel_side=rel_side, extension=5.3, induced_vert_break=ivb, horz_break=hb_arm * p_sign,
            plate_loc_height=height, plate_loc_side=side_loc,
            vert_appr_angle=-6.0 + 0.25 * ivb + 1.1 * (height - 2.25) + rng.normal(0, 0.3),
            horz_appr_angle=np.degrees(np.arctan((side_loc - rel_side) / 40.0)),
            pitch_tracked=rng.random() > 0.03, exit_speed=np.nan, launch_angle=np.nan, hit_launch_conf=None,
            kor_bb="Undefined", tagged_hit_type="Undefined", play_result="Undefined",
        )
        swing_p = (0.66 if in_zone else 0.27) + (0.12 if strikes == 2 else 0)
        whiff_p = BASE_WHIFF[kind] / b["skill"]
        power = b["power"]
        if b["planted"] == ("whiff", kind):
            whiff_p = 0.62
        if b["planted"] == ("power", kind):
            power += 14
        result = None
        if rng.random() < swing_p:
            if rng.random() < whiff_p:
                row["pitch_call"] = "StrikeSwinging"
                strikes += 1
            elif rng.random() < 0.45:
                row["pitch_call"] = "FoulBallNotFieldable"
                strikes = min(strikes + 1, 2) if strikes < 2 else 2
            else:
                ev = rng.normal(56 + power, 9)
                row.update(pitch_call="InPlay", exit_speed=ev, launch_angle=rng.normal(12, 18), hit_launch_conf="High")
                hit_p = 1 / (1 + np.exp(-(ev - 62) / 5)) * 0.75
                if rng.random() < hit_p:
                    result = "HomeRun" if ev > 74 else "Double" if ev > 66 else "Single"
                else:
                    result = "Out"
                row.update(play_result=result, tagged_hit_type="LineDrive")
        else:
            if in_zone and rng.random() < 0.9:
                row["pitch_call"] = "StrikeCalled"
                strikes += 1
            else:
                row["pitch_call"] = "BallCalled"
                balls += 1
        if result is None and strikes == 3:
            row["kor_bb"], result = "Strikeout", "Strikeout"
        if result is None and balls == 4:
            row["kor_bb"], result = "Walk", "Walk"
        if not row["pitch_tracked"]:
            for c in ("rel_speed", "induced_vert_break", "horz_break", "vert_appr_angle"):
                row[c] = np.nan
        pitches.append(row)
        if result:
            return result, pitches
