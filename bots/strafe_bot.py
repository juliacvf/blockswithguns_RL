"""Bot 3 - StrafeBot: terrain-reactive mid-range orbiting.

Inspired by wallsLogic.js: constant perpendicular strafing mixed with
solid-obstacle avoidance, flips orbit direction on a seeded timer, and
breaks its orbit briefly to escape movement-slowing mud.
"""

from __future__ import annotations

import math

from core import constants as C
from core.engine import Action
from core.los import has_line_of_sight
from core.pathfinding import wall_aware_velocity, wall_repulsion, astar

from .base import Bot, View

ORBIT_DIST = 16.0
FLIP_MIN, FLIP_MAX = 40, 110  # ticks between orbit-direction flips
PATH_INTERVAL = 30
MUD_PENALTY = 0.75


class StrafeBot(Bot):
    name = "strafe"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self._orbit = 1.0
        self._next_flip = self.rng.randint(FLIP_MIN, FLIP_MAX)
        self._born_tick = 0

    def reset(self) -> None:
        super().reset()
        self._orbit = 1.0
        self._next_flip = self.rng.randint(FLIP_MIN, FLIP_MAX)

    def act(self, view: View) -> Action:
        me, en = view.me, view.enemy
        dx, dy = en.x - me.x, en.y - me.y
        dist = math.hypot(dx, dy) or 1e-6
        ux, uy = dx / dist, dy / dist          # toward enemy
        px, py = -uy * self._orbit, ux * self._orbit  # perpendicular

        if view.tick - self._born_tick >= self._next_flip:
            self._born_tick = view.tick
            self._next_flip = self.rng.randint(FLIP_MIN, FLIP_MAX)
            self._orbit *= -1.0

        # radial correction keeps the orbit near ORBIT_DIST
        radial = (dist - ORBIT_DIST) / ORBIT_DIST
        move = (px * 0.9 + ux * radial, py * 0.9 + uy * radial)

        los = has_line_of_sight(view.grid, me.x, me.y, en.x, en.y)
        if not los:
            # no shot: push toward the enemy through the maze
            planner = lambda grid, sx, sy, gx, gy: astar(
                grid, sx, sy, gx, gy, view.mud, MUD_PENALTY)
            pdx, pdy = self.follow_path(view, en.x, en.y, PATH_INTERVAL, planner)
            move = (move[0] * 0.3 + pdx * 0.7, move[1] * 0.3 + pdy * 0.7)

        mud_avoid = self.mud_escape(view)
        in_mud = bool(view.mud[int(me.x), int(me.y)])
        mud_weight = 1.1 if in_mud else 0.2
        move = (move[0] + mud_avoid[0] * mud_weight,
                move[1] + mud_avoid[1] * mud_weight)
        avoid = wall_repulsion(view.solid, me.x, me.y)
        move = wall_aware_velocity(move, avoid, base_weight=0.6, avoid_weight=0.4)
        mag = math.hypot(*move)
        if mag > 1e-6:
            move = (move[0] / mag, move[1] / mag)

        evx, evy = self.enemy_velocity(view)
        aim = self.lead_angle(me, en.x, en.y, evx, evy, C.BULLET_SPEED)
        return Action(move_x=move[0], move_y=move[1], aim=aim,
                      shoot=los and dist <= C.BULLET_RANGE)
