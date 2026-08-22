"""Bot 4 - CamperBot: holds a dry defensive firing position.

Inspired by hasLineOfSight.js: it only pulls the trigger when the line
is clear, kites around walls and trees when rushed, leaves mud before
settling, and repositions toward the last seen enemy when sight is lost.
"""

from __future__ import annotations

import math

from core import constants as C
from core.engine import Action
from core.los import has_line_of_sight
from core.pathfinding import astar, wall_aware_velocity, wall_repulsion

from .base import Bot, View

HOLD_DIST = 26.0     # preferred engagement range
PANIC_DIST = 10.0    # too close: retreat
PATH_INTERVAL = 40
MUD_PENALTY = 3.0


class CamperBot(Bot):
    name = "camper"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self._last_seen: tuple[float, float] | None = None

    def reset(self) -> None:
        super().reset()
        self._last_seen = None

    def act(self, view: View) -> Action:
        me, en = view.me, view.enemy
        dx, dy = en.x - me.x, en.y - me.y
        dist = math.hypot(dx, dy) or 1e-6
        avoid = wall_repulsion(view.solid, me.x, me.y, radius=2)
        mud_avoid = self.mud_escape(view)
        in_mud = bool(view.mud[int(me.x), int(me.y)])
        los = has_line_of_sight(view.grid, me.x, me.y, en.x, en.y)
        if los:
            self._last_seen = (en.x, en.y)

        if in_mud:
            move = mud_avoid
        elif los and PANIC_DIST < dist < HOLD_DIST * 1.5:
            move = (0.0, 0.0)  # hold and shoot
        elif los and dist <= PANIC_DIST:
            move = (-dx / dist, -dy / dist)  # kite away
        else:
            target = self._last_seen or (en.x, en.y)
            planner = lambda grid, sx, sy, gx, gy: astar(
                grid, sx, sy, gx, gy, view.mud, MUD_PENALTY)
            move = self.follow_path(view, target[0], target[1], PATH_INTERVAL, planner)
            if los and dist > HOLD_DIST * 1.5:
                move = (move[0] + dx / dist * 0.5, move[1] + dy / dist * 0.5)

        if not in_mud:
            move = (move[0] + mud_avoid[0] * 0.35,
                    move[1] + mud_avoid[1] * 0.35)
        move = wall_aware_velocity(move, avoid, base_weight=0.5, avoid_weight=0.5)
        mag = math.hypot(*move)
        if mag > 1e-6:
            move = (move[0] / mag, move[1] / mag)

        evx, evy = self.enemy_velocity(view)
        aim = self.lead_angle(me, en.x, en.y, evx, evy, C.BULLET_SPEED)
        shoot = los and dist <= C.BULLET_RANGE
        return Action(move_x=move[0], move_y=move[1], aim=aim, shoot=shoot)
