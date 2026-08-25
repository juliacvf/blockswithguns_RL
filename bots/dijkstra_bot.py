"""Bot 1 - DijkstraBot: cautious terrain-aware pursuit.

Chases along inexpensive dry routes, gives walls and trees a wide berth,
flees when crowded, and fires with deliberately jittered aim. If the flee
and wall-repulsion vectors cancel out and pin it in a pocket, it detects
the stall, commits to one escape waypoint and paths its way out before
resuming normal behavior.
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

STUCK_WINDOW = 45      # ticks of position history used for stall detection
STUCK_DISP = 0.6       # blocks moved within the window below which it is trapped
ESCAPE_TICKS = 180     # max ticks spent pathing to one committed escape goal
                       # (long enough to cross the whole scan radius, even in mud)
ESCAPE_SCAN_RADIUS = 10  # cells scanned for a flee target when trapped
MAP_CENTER = (C.WORLD * 0.5, C.WORLD * 0.5)


class DijkstraBot(Bot):
    name = "dijkstra (easy)"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self._hist: list[tuple[float, float]] = []
        self._escape_goal: tuple[float, float] | None = None
        self._escape_until = -1

    def reset(self) -> None:
        super().reset()
        self._hist = []
        self._escape_goal = None
        self._escape_until = -1

    def _escape_target(self, view: View, dist: float) -> tuple[float, float]:
        """Where to run when trapped: the enemy when far away, otherwise the
        free cell that puts the most distance between us and the enemy."""
        me, en = view.me, view.enemy
        if dist >= MAINTAIN_DIST:
            return en.x, en.y
        cx, cy = int(me.x), int(me.y)
        best = None
        for dx in range(-ESCAPE_SCAN_RADIUS, ESCAPE_SCAN_RADIUS + 1):
            for dy in range(-ESCAPE_SCAN_RADIUS, ESCAPE_SCAN_RADIUS + 1):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < view.solid.shape[0] and 0 <= ny < view.solid.shape[1]):
                    continue
                if view.solid[nx, ny]:
                    continue
                # prefer cells far from the enemy, break ties by proximity
                enemy_d2 = (nx + 0.5 - en.x) ** 2 + (ny + 0.5 - en.y) ** 2
                self_d2 = (nx + 0.5 - me.x) ** 2 + (ny + 0.5 - me.y) ** 2
                candidate = (-enemy_d2, self_d2, nx, ny)
                if best is None or candidate < best:
                    best = candidate
        if best is None or (best[2] == cx and best[3] == cy):
            return MAP_CENTER   # walled in: head for the middle of the map
        return best[2] + 0.5, best[3] + 0.5

    def _escape_move(self, view: View, dist: float) -> tuple[float, float] | None:
        """Stall detection plus committed-goal escape pathing.

        Returns the escape move while escaping, or None during normal play.
        The goal is committed once and kept until reached or the burst ends —
        recomputing it every replan made the target flip between pockets on
        opposite sides, which was the actual trap.
        """
        me = view.me
        self._hist.append((me.x, me.y))
        if len(self._hist) > STUCK_WINDOW:
            self._hist.pop(0)
        stalled = (len(self._hist) == STUCK_WINDOW
                   and (me.x - self._hist[0][0]) ** 2
                   + (me.y - self._hist[0][1]) ** 2 < STUCK_DISP ** 2)

        if self._escape_goal is not None:
            gx, gy = self._escape_goal
            if ((me.x - gx) ** 2 + (me.y - gy) ** 2 < 1.0
                    or view.tick >= self._escape_until):
                self._escape_goal = None   # arrived, or gave up on this goal
                self._hist = []            # fresh window for the next stall check
        if stalled and self._escape_goal is None:
            self._escape_goal = self._escape_target(view, dist)
            self._escape_until = view.tick + ESCAPE_TICKS
            self._path = []   # force a replan toward the new goal
        if self._escape_goal is None:
            return None

        planner = lambda grid, sx, sy, gx, gy: dijkstra(grid, sx, sy, gx, gy)
        move = self.follow_path(view, *self._escape_goal, 5, planner)
        if move == (0.0, 0.0) and not self._path:
            # Unreachable goal: fall back to the map center once, then give up.
            if self._escape_goal != MAP_CENTER:
                self._escape_goal = MAP_CENTER
                self._path = []
                move = self.follow_path(view, *MAP_CENTER, 5, planner)
            if move == (0.0, 0.0) and not self._path:
                self._escape_goal = None
                return None
        return move

    def act(self, view: View) -> Action:
        me, en = view.me, view.enemy
        dx, dy = en.x - me.x, en.y - me.y
        dist = math.hypot(dx, dy)
        avoid = wall_repulsion(view.solid, me.x, me.y, radius=2)
        mud_avoid = self.mud_escape(view)

        escape = self._escape_move(view, dist)
        if escape is not None:
            # Trapped: path out toward the committed goal, replanning often.
            move = escape
            aim = math.atan2(dy, dx)
        elif dist < MAINTAIN_DIST:
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
        # slide along walls instead of wedging into corners
        move = self.slide_move(view, *move)
        shoot = (dist <= C.BULLET_RANGE
                 and has_line_of_sight(view.grid, me.x, me.y, en.x, en.y))
        return Action(move_x=move[0], move_y=move[1], aim=aim, shoot=shoot)
