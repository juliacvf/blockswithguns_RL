"""Bot interface shared by all deterministic bots and game/RL runners."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from core import constants as C
from core.engine import Action, Bullet, PlayerState
from core.pathfinding import path_direction
from core.powerups import Powerup

if TYPE_CHECKING:  # pragma: no cover
    from core.engine import Engine


@dataclass
class View:
    """What a bot sees each tick (full map knowledge, like the JS AI had)."""
    grid: np.ndarray           # walls only: line of sight, bullet blocking
    solid: np.ndarray          # walls + trees: movement and pathfinding
    mud: np.ndarray            # slowing terrain; does not block sight or shots
    me: PlayerState
    enemy: PlayerState
    bullets: list[Bullet]
    powerups: list[Powerup]
    time: float
    tick: int
    my_idx: int = 0


class Bot:
    name = "bot"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._last_enemy: tuple[float, float, float] | None = None  # x, y, time
        self._path: list[tuple[int, int]] = []
        self._path_tick = -10**9

    def reset(self) -> None:
        self._last_enemy = None
        self._path = []
        self._path_tick = -10**9

    # -- helpers ------------------------------------------------------- #
    def enemy_velocity(self, view: View) -> tuple[float, float]:
        x, y, t = view.enemy.x, view.enemy.y, view.time
        vx = vy = 0.0
        if self._last_enemy is not None:
            lx, ly, lt = self._last_enemy
            dt = max(1e-4, t - lt)
            vx, vy = (x - lx) / dt, (y - ly) / dt
        self._last_enemy = (x, y, t)
        return vx, vy

    def refresh_path(self, view: View, tx: float, ty: float, interval: int,
                     planner) -> list[tuple[int, int]]:
        if view.tick - self._path_tick >= interval or not self._path:
            self._path = planner(view.solid, view.me.x, view.me.y, tx, ty)
            self._path_tick = view.tick
        return self._path

    def follow_path(self, view: View, tx: float, ty: float, interval: int,
                    planner) -> tuple[float, float]:
        """Advance the cached path past reached cells, return move direction."""
        path = self.refresh_path(view, tx, ty, interval, planner)
        while len(path) > 1:
            cx, cy = path[0]
            if (view.me.x - (cx + 0.5)) ** 2 + (view.me.y - (cy + 0.5)) ** 2 < 0.8 ** 2:
                path.pop(0)
            else:
                break
        self._path = path
        return path_direction(path, view.me.x, view.me.y)

    @staticmethod
    def mud_escape(view: View, search_radius: int = 6) -> tuple[float, float]:
        """Return a deterministic push away from mud or toward nearby dry land."""
        x, y = view.me.x, view.me.y
        cx, cy = int(x), int(y)
        mud = view.mud
        if not mud[cx, cy]:
            # Avoid stepping into a pond when skirting its edge.
            rx = ry = 0.0
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = cx + dx, cy + dy
                    if ((dx or dy) and 0 <= nx < mud.shape[0]
                            and 0 <= ny < mud.shape[1] and mud[nx, ny]):
                        weight = 1.0 / max(1.0, math.hypot(dx, dy))
                        rx -= dx * weight
                        ry -= dy * weight
            mag = math.hypot(rx, ry)
            return (rx / mag, ry / mag) if mag > 1e-9 else (0.0, 0.0)

        # In a pond, head for the nearest dry, walkable cell.
        best: tuple[float, int, int] | None = None
        for radius in range(1, search_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < mud.shape[0] and 0 <= ny < mud.shape[1]):
                        continue
                    if mud[nx, ny] or view.solid[nx, ny]:
                        continue
                    candidate = ((nx + 0.5 - x) ** 2 + (ny + 0.5 - y) ** 2, nx, ny)
                    if best is None or candidate < best:
                        best = candidate
            if best is not None:
                break
        if best is None:
            return 0.0, 0.0
        _, nx, ny = best
        dx, dy = nx + 0.5 - x, ny + 0.5 - y
        mag = math.hypot(dx, dy)
        return (dx / mag, dy / mag) if mag > 1e-9 else (0.0, 0.0)

    @staticmethod
    def _circle_blocked(solid: np.ndarray, x: float, y: float, r: float) -> bool:
        """Same circle-vs-solid-cells test the engine applies to movement."""
        n = solid.shape[0]
        x0, x1 = int(math.floor(x - r)), int(math.floor(x + r))
        y0, y1 = int(math.floor(y - r)), int(math.floor(y + r))
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                if cx < 0 or cy < 0 or cx >= n or cy >= n or solid[cx, cy]:
                    nx = min(max(x, cx), cx + 1.0)
                    ny = min(max(y, cy), cy + 1.0)
                    if (x - nx) ** 2 + (y - ny) ** 2 < r * r:
                        return True
        return False

    def slide_move(self, view: View, mx: float, my: float) -> tuple[float, float]:
        """Direction closest to (mx, my) that the collision circle allows.

        The engine blocks each axis separately, so a diagonal push into a
        convex corner can freeze a bot completely; rotating the intent in
        45-degree steps until one fits lets it slide along walls instead.
        """
        if math.hypot(mx, my) <= 1e-6:
            return 0.0, 0.0
        me = view.me
        step = me.speed * C.FIXED_DT
        if view.mud[int(me.x), int(me.y)]:
            step *= C.MUD_SLOW
        base = math.atan2(my, mx)
        quarter = math.pi / 4.0
        for offset in (0.0, quarter, -quarter, 2 * quarter, -2 * quarter,
                       3 * quarter, -3 * quarter, math.pi):
            dx, dy = math.cos(base + offset), math.sin(base + offset)
            if not self._circle_blocked(
                    view.solid, me.x + dx * step, me.y + dy * step,
                    C.PLAYER_RADIUS):
                return dx, dy
        return 0.0, 0.0

    @staticmethod
    def lead_angle(me: PlayerState, ex: float, ey: float, evx: float, evy: float,
                   bullet_speed: float) -> float:
        dist = math.hypot(ex - me.x, ey - me.y)
        t = dist / bullet_speed
        return math.atan2(ey + evy * t - me.y, ex + evx * t - me.x)

    def act(self, view: View) -> Action:  # pragma: no cover - interface
        raise NotImplementedError


def make_view(engine: "Engine", idx: int) -> View:
    return View(
        grid=engine.grid,
        solid=engine.map.solid,
        mud=(engine.map.mud if engine.map.mud is not None
             else np.zeros_like(engine.grid)),
        me=engine.players[idx],
        enemy=engine.players[1 - idx],
        bullets=list(engine.bullets),
        powerups=list(engine.powerups.items),
        time=engine.time,
        tick=engine.tick,
        my_idx=idx,
    )
