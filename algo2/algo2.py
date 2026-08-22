"""Example slot-2 contest policy: deterministic orbit and counter-fire."""

from __future__ import annotations

import math


class Agent:
    def __init__(self, weights_path=None):
        self.weights_path = weights_path
        self.orbit = 1

    def reset(self):
        self.orbit = 1

    def act(self, obs):
        sx, sy = (float(obs["self"][i] + 1.0) * 50.0 for i in (0, 1))
        ex, ey = (float(obs["opponent"][i] + 1.0) * 50.0 for i in (0, 1))
        dx, dy = ex - sx, ey - sy
        distance = max(1e-6, math.hypot(dx, dy))
        radial = max(-1.0, min(1.0, (distance - 14.0) / 14.0))
        vx = dx / distance * radial - dy / distance * self.orbit
        vy = dy / distance * radial + dx / distance * self.orbit
        move_x = 0 if abs(vx) < 0.25 else (1 if vx > 0 else -1)
        move_y = 0 if abs(vy) < 0.25 else (1 if vy > 0 else -1)
        move = (move_y + 1) * 3 + (move_x + 1)
        aim = round((math.atan2(dy, dx) % (2 * math.pi)) * 16 / (2 * math.pi)) % 16
        visible = obs["self"][14] > 0
        return [move, aim, int(visible and distance <= 20.0)]

