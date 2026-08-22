"""Bot 1 - DijkstraBot: cautious terrain-aware pursuit.

Chases along inexpensive dry routes, gives walls and trees a wide berth,
flees when crowded, and fires with deliberately jittered aim.
"""

from __future__ import annotations

import math

from core import constants as C
from core.engine import Action
from core.los import has_line_of_sight
from core.pathfinding import dijkstra, wall_aware_velocity, wall_repulsion

from .base import Bot, View

MAINTAIN_DIST = 20.0   # scaled from the JS 150px-on-500px-world
PATH_INTERVAL = 35     # ticks between replans
MUD_PENALTY = 4.0      # easy bot strongly prefers a longer dry route


class DijkstraBot(Bot):
    name = "dijkstra (easy)"

    def act(self, view: View) -> Action:
        me, en = view.me, view.enemy
        dx, dy = en.x - me.x, en.y - me.y
        dist = math.hypot(dx, dy)
        avoid = wall_repulsion(view.solid, me.x, me.y, radius=2)
        mud_avoid = self.mud_escape(view)

        if dist < MAINTAIN_DIST:
            flee = (-dx * 0.4, -dy * 0.4)
            move = wall_aware_velocity(flee, avoid)
            aim = math.atan2(dy, dx)
        else:
            planner = lambda grid, sx, sy, gx, gy: dijkstra(
                grid, sx, sy, gx, gy, view.mud, MUD_PENALTY)
            pd = self.follow_path(view, en.x, en.y, PATH_INTERVAL, planner)
            move = wall_aware_velocity(pd, avoid)
            # jittered aim, like the JS easy bot
            jx = self.rng.uniform(-1, 1)
            jy = self.rng.uniform(-1, 1)
            aim = math.atan2(dy + jy, dx + jx)

        move = (move[0] + mud_avoid[0] * 0.65,
                move[1] + mud_avoid[1] * 0.65)
        mag = math.hypot(*move)
        if mag > 1e-6:
            move = (move[0] / mag, move[1] / mag)
        shoot = (dist <= C.BULLET_RANGE
                 and has_line_of_sight(view.grid, me.x, me.y, en.x, en.y))
        return Action(move_x=move[0], move_y=move[1], aim=aim, shoot=shoot)
