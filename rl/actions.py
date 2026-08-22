"""Discrete contest action contract and conversion to engine actions."""

from __future__ import annotations

import math

import numpy as np
from gymnasium import spaces

from core.engine import Action

MOVE_LUT = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
AIM_BINS = 16
ACTION_BOUNDS = np.asarray([9, AIM_BINS, 2], dtype=np.int64)


def contest_action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(ACTION_BOUNDS.copy())


def validate_action(action, name: str = "agent") -> np.ndarray:
    """Require exactly three finite integers within MultiDiscrete bounds."""
    raw = np.asarray(action)
    if raw.size != 3:
        raise ValueError(
            f"{name} must return exactly [move, aim, shoot]; got shape {raw.shape}")
    try:
        numeric = raw.astype(np.float64).reshape(3)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} action must contain numbers") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{name} action must contain finite integers; got {raw!r}")
    result = numeric.astype(np.int64)
    if np.any(result < 0) or np.any(result >= ACTION_BOUNDS):
        raise ValueError(
            f"{name} action {result.tolist()} is outside [0..8, 0..15, 0..1]")
    return result


def to_engine_action(action, *, name: str = "agent") -> Action:
    move_idx, aim_idx, shoot = validate_action(action, name)
    move_x, move_y = MOVE_LUT[int(move_idx)]
    aim = int(aim_idx) * (2.0 * math.pi / AIM_BINS)
    return Action(float(move_x), float(move_y), aim, bool(shoot))

