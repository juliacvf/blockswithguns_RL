"""Bot 5 - MarkovBot: adaptive terrain-aware hard AI.

Keeps a timestamped memory of enemy positions and shots, blends a
physics prediction with a memory-based prediction, dodges incoming
bullets proportionally to a computed threat level, and changes how strongly
it avoids mud while pathing around walls and trees to predicted positions.
"""

from __future__ import annotations

import math
from collections import deque

from core import constants as C
from core.engine import Action
from core.pathfinding import astar, wall_repulsion

from .base import Bot, View

MEMORY_SECONDS = 30.0
MAINTAIN_DIST = 14.0
PATH_INTERVAL = 25
MAX_SPEED = 1.0  # movement output is normalized anyway


def _normalize(x: float, y: float) -> tuple[float, float]:
    m = math.hypot(x, y)
    return (x / m, y / m) if m > 1e-9 else (0.0, 0.0)


class MarkovBot(Bot):
    name = "markov (hard)"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.pos_memory: deque[tuple[float, float, float]] = deque(maxlen=512)
        self.shot_memory: deque[tuple[float, float, float, float]] = deque(maxlen=64)
        self._smooth = (0.0, 0.0)
        self._engage_start: float | None = None

    def reset(self) -> None:
        super().reset()
        self.pos_memory.clear()
        self.shot_memory.clear()
        self._smooth = (0.0, 0.0)
        self._engage_start = None

    # -- memory -------------------------------------------------------- #
    def _update_memory(self, view: View) -> None:
        en = view.enemy
        self.pos_memory.append((en.x, en.y, view.time))
        # remember freshly observed enemy shots (position + firing angle)
        if 0.0 <= view.time - en.last_shot_time < 0.05:
            self.shot_memory.append((en.x, en.y, en.aim, view.time))

    def _predict_from_memory(self) -> tuple[float, float] | None:
        """Decay-weighted linear extrapolation of remembered positions."""
        pts = list(self.pos_memory)
        if len(pts) < 4:
            return None
        recent = pts[-12:]
        n = len(recent)
        gamma = 0.85
        weights = [gamma ** (n - i - 1) for i in range(n)]
        wsum = sum(weights)
        avg_x = sum(p[0] * w for p, w in zip(recent, weights)) / wsum
        avg_y = sum(p[1] * w for p, w in zip(recent, weights)) / wsum
        # weighted velocity
        vx = vy = 0.0
        for i in range(1, n):
            w = weights[i]
            dt = max(1e-4, recent[i][2] - recent[i - 1][2])
            vx += w * (recent[i][0] - recent[i - 1][0]) / dt
            vy += w * (recent[i][1] - recent[i - 1][1]) / dt
        vx /= wsum
        vy /= wsum
        horizon = 0.6
        return avg_x + vx * horizon, avg_y + vy * horizon

    # -- threat / dodge -------------------------------------------------- #
    def _bullet_threat(self, view: View) -> tuple[float, tuple[float, float]]:
        """(threat 0..1, dodge vector) from live enemy bullets."""
        me = view.me
        worst = 0.0
        dodge = (0.0, 0.0)
        for b in view.bullets:
            if b.owner == view.my_idx:
                continue
            # closest approach of the bullet line to me
            rx, ry = me.x - b.x, me.y - b.y
            speed = math.hypot(b.vx, b.vy) or 1e-6
            ux, uy = b.vx / speed, b.vy / speed
            proj = rx * ux + ry * uy
            if proj < 0:
                continue  # moving away
            closest_x = b.x + ux * proj
            closest_y = b.y + uy * proj
            miss = math.hypot(me.x - closest_x, me.y - closest_y)
            tta = proj / speed
            threat = max(0.0, 1.0 - miss / 6.0) * max(0.0, 1.0 - tta / 2.0)
            if threat > 0.05:
                # dodge perpendicular to the bullet's path
                side = 1.0 if (rx * uy - ry * ux) > 0 else -1.0
                dodge = (dodge[0] + -uy * side * threat, dodge[1] + ux * side * threat)
            worst = max(worst, threat)
        return min(1.0, worst), _normalize(*dodge)

    # -- main ------------------------------------------------------------ #
    def act(self, view: View) -> Action:
        self._update_memory(view)
        me, en = view.me, view.enemy
        if self._engage_start is None:
            self._engage_start = view.time
        engaged = view.time - self._engage_start

        dx, dy = en.x - me.x, en.y - me.y
        dist = math.hypot(dx, dy)
        evx, evy = self.enemy_velocity(view)

        # hybrid prediction: physics lead vs memory extrapolation
        t_reach = dist / C.BULLET_SPEED
        phys = (en.x + evx * t_reach, en.y + evy * t_reach)
        mem = self._predict_from_memory()
        w_phys = 0.5 if engaged < 5.0 else 0.8
        if mem is not None:
            px = phys[0] * w_phys + mem[0] * (1 - w_phys)
            py = phys[1] * w_phys + mem[1] * (1 - w_phys)
        else:
            px, py = phys

        threat, dodge = self._bullet_threat(view)

        terrain_penalty = 0.5 + threat * 3.0
        planner = lambda grid, sx, sy, gx, gy: astar(
            grid, sx, sy, gx, gy, view.mud, terrain_penalty)
        path_dir = self.follow_path(view, px, py, PATH_INTERVAL, planner)
        pdx, pdy = _normalize(*path_dir)

        dodge_w = min(1.5, 0.8 + threat * 0.7)
        path_w = 1.2 - threat * 0.5
        wall_w = 1.0 - threat * 0.3
        avoid = wall_repulsion(view.solid, me.x, me.y)
        mud_avoid = self.mud_escape(view)
        in_mud = bool(view.mud[int(me.x), int(me.y)])
        mud_w = 1.2 if in_mud else 0.15 + threat * 0.65

        raw_x = (pdx * path_w + dodge[0] * dodge_w + avoid[0] * wall_w
                 + mud_avoid[0] * mud_w)
        raw_y = (pdy * path_w + dodge[1] * dodge_w + avoid[1] * wall_w
                 + mud_avoid[1] * mud_w)

        smoothing = 0.15 * (1 - threat * 0.5)
        self._smooth = (
            self._smooth[0] + (raw_x - self._smooth[0]) * smoothing,
            self._smooth[1] + (raw_y - self._smooth[1]) * smoothing,
        )
        move = _normalize(*self._smooth)
        if dist < MAINTAIN_DIST:
            mod = dist / (MAINTAIN_DIST * 2)
            move = (move[0] * mod - dx / max(dist, 1e-6) * 0.4,
                    move[1] * mod - dy / max(dist, 1e-6) * 0.4)
            move = _normalize(*move)

        # aim: high threat -> trust memory aim, else blend physics + memory
        if threat > 0.7 and mem is not None:
            aim = math.atan2(mem[1] - me.y, mem[0] - me.x)
        else:
            a_phys = math.atan2(phys[1] - me.y, phys[0] - me.x)
            if mem is not None:
                a_mem = math.atan2(mem[1] - me.y, mem[0] - me.x)
                w = 0.8 - threat * 0.3
                aim = math.atan2(
                    math.sin(a_phys) * w + math.sin(a_mem) * (1 - w),
                    math.cos(a_phys) * w + math.cos(a_mem) * (1 - w),
                )
            else:
                aim = a_phys

        from core.los import has_line_of_sight
        shoot = (has_line_of_sight(view.grid, me.x, me.y, en.x, en.y)
                 and dist <= C.BULLET_RANGE)
        return Action(move_x=move[0], move_y=move[1], aim=aim, shoot=shoot)
