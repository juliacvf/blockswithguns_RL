"""Example slot-1 contest policy: direct deterministic pressure."""

from __future__ import annotations

import math


class Agent:
    def __init__(self, weights_path=None):
        # weights_path is the algo1/weights directory. This deterministic
        # example does not need a model file.
        self.weights_path = weights_path

    def reset(self):
        pass

    def act(self, obs):
        sx, sy = (float(obs["self"][i] + 1.0) * 50.0 for i in (0, 1))
        ex, ey = (float(obs["opponent"][i] + 1.0) * 50.0 for i in (0, 1))
        dx, dy = ex - sx, ey - sy
        distance = math.hypot(dx, dy)
        move_x = 0 if abs(dx) < 1.0 else (1 if dx > 0 else -1)
        move_y = 0 if abs(dy) < 1.0 else (1 if dy > 0 else -1)
        move = (move_y + 1) * 3 + (move_x + 1)
        aim = round((math.atan2(dy, dx) % (2 * math.pi)) * 16 / (2 * math.pi)) % 16
        visible = obs["self"][14] > 0
        return [move, aim, int(visible and distance <= 20.0)]

