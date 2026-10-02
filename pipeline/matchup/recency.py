"""Recency weights.

Within the current season a pitch's weight halves every HALF_LIFE_DAYS before the as-of date.
Earlier seasons get a flat multiplier instead of continued decay: a pure exponential across a
seven-month offseason would erase last season entirely, which is exactly the data you need in February.
Both knobs are tuned by the validation step.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

HALF_LIFE_DAYS = 45.0
PRIOR_SEASON_MULT = (1.0, 0.4, 0.15)  # current, last season, two seasons back; older = 0


def season_start_year(season: str) -> int:
    return int(season[:4])


def recency_weights(
    game_date: pd.Series,
    season: pd.Series,
    as_of: date,
    as_of_season: str,
    half_life: float = HALF_LIFE_DAYS,
    season_mult: tuple[float, ...] = PRIOR_SEASON_MULT,
) -> np.ndarray:
    gd = pd.to_datetime(game_date)
    age_days = (pd.Timestamp(as_of) - gd).dt.days.to_numpy(dtype=float)
    seasons_back = season_start_year(as_of_season) - season.map(season_start_year).to_numpy()
    mult = np.zeros(len(gd))
    for i, m in enumerate(season_mult):
        mult[seasons_back == i] = m
    w = np.where(seasons_back == 0, 0.5 ** (np.clip(age_days, 0, None) / half_life), 1.0) * mult
    w[age_days < 0] = 0.0  # never use the future
    return w
