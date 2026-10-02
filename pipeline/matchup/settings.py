"""Tunable engine settings, persisted in pipeline/settings.json (written by `matchup backtest --apply`)."""
from __future__ import annotations

import json
from pathlib import Path

from . import engine, recency

PATH = Path(__file__).resolve().parents[1] / "settings.json"
DEFAULTS = {"bandwidth": 0.3, "half_life_days": 60.0, "prior_scale": 1.0}


def apply(values: dict) -> dict:
    v = {**DEFAULTS, **values}
    engine.FIT_COMPONENTS = ("whiff", "hard_hit")
    engine.BANDWIDTH = float(v["bandwidth"])
    recency.HALF_LIFE_DAYS = float(v["half_life_days"])   # 'inf' -> no decay within the season
    ps = v["prior_scale"]
    engine.PRIOR_SCALE = {k: float(x) for k, x in ps.items()} if isinstance(ps, dict) else float(ps)
    return v


FIT_CANDIDATES = ("whiff", "called_strike", "chase", "hard_hit")


def load() -> dict:
    """Apply saved settings (or defaults). Returns what is in effect."""
    saved = json.loads(PATH.read_text()) if PATH.exists() else {}
    v = apply({k: saved[k] for k in DEFAULTS if k in saved})
    verdicts = saved.get("validation", {})
    if verdicts:   # Shape fit uses exactly the components whose hitter-specific part was validated
        engine.FIT_COMPONENTS = tuple(m for m in FIT_CANDIDATES if str(verdicts.get(m, "")).startswith("validated"))
    return v


def save(values: dict, validation: dict | None = None) -> Path:
    from .validate import json_safe
    out = {k: json_safe(values[k]) for k in DEFAULTS}
    if validation:
        out["validation"] = validation
    PATH.write_text(json.dumps(out, indent=2))
    return PATH


def validation() -> dict:
    """Per-metric backtest verdicts saved with the settings (empty if never run)."""
    return json.loads(PATH.read_text()).get("validation", {}) if PATH.exists() else {}
