"""Minimal example agent showing the compact Gymnasium contract.

If a weights .npz produced by train_example.py exists it plays the learned
Q-table; otherwise it falls back to a simple hand-coded heuristic.
"""

from __future__ import annotations

import math
import os

import numpy as np

MOVE_LUT = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
AIM_BINS = 16


def discretize(obs: dict) -> tuple:
    """Compact state key: enemy sector, LOS, wall neighborhood, lives diff."""
    e = obs["enemy"]
    ang = int((math.atan2(e[1], e[0]) / (2 * math.pi) + 1.0) % 1.0 * 8)
    dist_bin = int((e[2] + 1) / 2 * 4)  # 0..3
    visible = 1 if e[4] > 0 else 0
    lg = obs["local_grid"]
    r = lg.shape[0] // 2
    walls = (int(lg[r, r + 1] > 0.5), int(lg[r, r - 1] > 0.5),
             int(lg[r + 1, r] > 0.5), int(lg[r - 1, r] > 0.5))
    lives_diff = int(np.clip(obs["self"][4] - e[3], -1, 1)) + 1
    return (ang, dist_bin, visible, walls, lives_diff)


class Agent:
    def __init__(self, weights_path: str | None = None):
        self.q = None
        if weights_path and os.path.exists(weights_path):
            data = np.load(weights_path, allow_pickle=True)
            self.q = data["q"].item()  # dict state -> (9*16*2) values
        self.rng = np.random.default_rng(0)

    def reset(self) -> None:
        pass

    def _heuristic(self, obs: dict) -> np.ndarray:
        e = obs["enemy"]
        ang = math.atan2(e[1], e[0])  # rel angle to enemy
        aim_bin = int(round(ang / (2 * math.pi) * AIM_BINS)) % AIM_BINS
        visible = e[4] > 0
        # chase when not visible, hold when visible
        if visible:
            move = 4  # stay
        else:
            mx = 1 if e[0] > 0.02 else (-1 if e[0] < -0.02 else 0)
            my = 1 if e[1] > 0.02 else (-1 if e[1] < -0.02 else 0)
            move = MOVE_LUT.index((mx, my))
        return np.array([move, aim_bin, 1 if visible else 0])

    def act(self, obs: dict) -> np.ndarray:
        if self.q is None:
            return self._heuristic(obs)
        key = discretize(obs)
        vals = self.q.get(key)
        if vals is None:
            return self._heuristic(obs)
        flat = int(np.argmax(vals))
        move = flat // (AIM_BINS * 2)
        rest = flat % (AIM_BINS * 2)
        return np.array([move, rest // 2, rest % 2])
