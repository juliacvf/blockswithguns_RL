"""Bot 2 - AStarBot: balanced terrain-aware attacker.

A* chase with a modest mud cost, velocity-lead aim, and a direct attack
that accepts short muddy crossings when they save enough distance.
"""

from __future__ import annotations

import math

from core import constants as C
from core.engine import Action
from core.los import has_line_of_sight
from core.pathfinding import astar, wall_aware_velocity, wall_repulsion

from .base import Bot, View

MAINTAIN_DIST = 20.0
PATH_INTERVAL = 30
MUD_PENALTY = 1.25


class AStarBot(Bot):
    name = "astar (medium)"

    def act(self, view: View) -> Action:
        me, en = view.me, view.enemy
        dx, dy = en.x - me.x, en.y - me.y
        dist = math.hypot(dx, dy)
        avoid = wall_repulsion(view.solid, me.x, me.y)
        mud_avoid = self.mud_escape(view)
        evx, evy = self.enemy_velocity(view)
        aim = self.lead_angle(me, en.x, en.y, evx, evy, C.BULLET_SPEED)

        if dist < MAINTAIN_DIST:
            approach = (dx * (dist / 60.0), dy * (dist / 60.0))
            move = wall_aware_velocity(approach, avoid)
        else:
            planner = lambda grid, sx, sy, gx, gy: astar(
                grid, sx, sy, gx, gy, view.mud, MUD_PENALTY)
            pd = self.follow_path(view, en.x, en.y, PATH_INTERVAL, planner)
            move = wall_aware_velocity(pd, avoid)

        move = (move[0] + mud_avoid[0] * 0.25,
                move[1] + mud_avoid[1] * 0.25)
        mag = math.hypot(*move)
        if mag > 1e-6:
            move = (move[0] / mag, move[1] / mag)
        shoot = (has_line_of_sight(view.grid, me.x, me.y, en.x, en.y)
                 and dist <= C.BULLET_RANGE)
        return Action(move_x=move[0], move_y=move[1], aim=aim, shoot=shoot)
