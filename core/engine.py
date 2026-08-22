"""Core game engine: pure simulation, no rendering.

Deterministic given (map params, seed, action sequence). Used by the
pygame app, the Gymnasium/PettingZoo environments, bots, and contest runners.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from . import constants as C
from .mapgen import GameMap, SPAWN_CELLS, generate_map
from .powerups import PowerupManager


@dataclass
class Action:
    """One player's intent for a tick."""
    move_x: float = 0.0   # [-1, 1]
    move_y: float = 0.0   # [-1, 1]
    aim: float = 0.0      # radians
    shoot: bool = False


@dataclass
class PlayerState:
    x: float
    y: float
    aim: float = 0.0
    lives: int = C.PLAYER_LIVES
    cooldown: float = 0.0
    quickshot_timer: float = 0.0
    speed_timer: float = 0.0
    heat_timer: float = 0.0   # time spent inside the heat zone
    retreated: bool = False   # the once-per-match low-life retreat was used
    # bookkeeping for renderers / rewards
    last_hit_time: float = -10.0
    last_shot_time: float = -10.0

    @property
    def speed(self) -> float:
        return C.PLAYER_SPEED * (C.SPEED_MULTIPLIER if self.speed_timer > 0 else 1.0)

    @property
    def shoot_cooldown(self) -> float:
        return C.QUICKSHOT_COOLDOWN if self.quickshot_timer > 0 else C.SHOOT_COOLDOWN


@dataclass
class Bullet:
    x: float
    y: float
    vx: float
    vy: float
    owner: int
    ttl: float = C.BULLET_TTL
    dist: float = 0.0   # blocks travelled so far
    alive: bool = True


class Engine:
    """A 1v1 match on a generated map."""

    def __init__(
        self,
        seed: int = 0,
        map_seed: int | None = None,
        scale: float = 24.0,
        octaves: int = 3,
        threshold: float = 0.62,
        game_map: GameMap | None = None,
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.map = game_map or generate_map(
            seed=map_seed if map_seed is not None else seed,
            scale=scale, octaves=octaves, threshold=threshold,
        )
        self.grid: np.ndarray = self.map.grid
        self.players: list[PlayerState] = []
        self.bullets: list[Bullet] = []
        self.powerups = PowerupManager(self.map, self.rng)
        self.time = 0.0
        self.tick = 0
        self.winner: int | None = None   # None = running, -1 = draw
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self, swap_spawns: bool = False) -> None:
        a, b = (SPAWN_CELLS[1], SPAWN_CELLS[0]) if swap_spawns else SPAWN_CELLS
        self.players = [
            PlayerState(x=a[0] + 0.5, y=a[1] + 0.5, aim=0.0),
            PlayerState(x=b[0] + 0.5, y=b[1] + 0.5, aim=math.pi),
        ]
        self.bullets = []
        self.powerups.reset()
        self.time = 0.0
        self.tick = 0
        self.winner = None

    # ------------------------------------------------------------------ #
    def step(self, actions: tuple[Action, Action], dt: float = C.FIXED_DT) -> list[tuple]:
        """Advance one tick. Returns a list of events for renderers/audio."""
        events: list[tuple] = []
        if self.winner is not None:
            return events
        self.time += dt
        self.tick += 1

        for i, (pl, act) in enumerate(zip(self.players, actions)):
            self._move_player(pl, act, dt)
            pl.aim = act.aim % (2 * math.pi)
            pl.cooldown = max(0.0, pl.cooldown - dt)
            pl.quickshot_timer = max(0.0, pl.quickshot_timer - dt)
            pl.speed_timer = max(0.0, pl.speed_timer - dt)
            if act.shoot and pl.cooldown <= 0.0:
                self._fire(i, pl, events)

        self._update_bullets(dt, events)
        events += self.powerups.update(dt)
        events += self.powerups.try_pickup(self.players)
        self._update_heat(dt, events)

        for i, pl in enumerate(self.players):
            if pl.lives <= 0 and self.winner is None:
                self.winner = 1 - i
                events.append(("gameover", self.winner))

        if self.winner is None and self.time >= C.MAX_EPISODE_SECONDS - 1e-6:
            l0, l1 = self.players[0].lives, self.players[1].lives
            self.winner = -1 if l0 == l1 else (0 if l0 > l1 else 1)
            events.append(("gameover", self.winner))
        return events

    # ------------------------------------------------------------------ #
    def safe_half(self) -> float:
        """Half-width (in blocks) of the safe square centered on the map."""
        return max(0.0, C.WORLD * 0.5 - C.HEAT_SHRINK_RATE * self.time)

    def in_heat(self, x: float, y: float) -> bool:
        h = self.safe_half()
        c = C.WORLD * 0.5
        return abs(x - c) > h or abs(y - c) > h

    def _update_heat(self, dt: float, events: list[tuple]) -> None:
        for i, pl in enumerate(self.players):
            if self.in_heat(pl.x, pl.y):
                pl.heat_timer += dt
                while pl.heat_timer >= C.HEAT_DAMAGE_PERIOD:
                    pl.heat_timer -= C.HEAT_DAMAGE_PERIOD
                    pl.lives -= 1
                    pl.last_hit_time = self.time
                    events.append(("heat_hit", i))
            else:
                pl.heat_timer = 0.0

    # ------------------------------------------------------------------ #
    def _circle_hits_solid(self, x: float, y: float, r: float) -> bool:
        """Circle vs solid cells (walls and trees) — same logic as walls."""
        x0, x1 = int(math.floor(x - r)), int(math.floor(x + r))
        y0, y1 = int(math.floor(y - r)), int(math.floor(y + r))
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                if self.map.is_solid(cx, cy):
                    # closest point of the cell to the circle center
                    nx = min(max(x, cx), cx + 1.0)
                    ny = min(max(y, cy), cy + 1.0)
                    if (x - nx) ** 2 + (y - ny) ** 2 < r * r:
                        return True
        return False

    def _move_player(self, pl: PlayerState, act: Action, dt: float) -> None:
        mx = max(-1.0, min(1.0, act.move_x))
        my = max(-1.0, min(1.0, act.move_y))
        mag = math.hypot(mx, my)
        if mag > 1e-6:
            mx, my = mx / mag, my / mag
        else:
            return
        sp = pl.speed * dt
        if self.map.is_mud(int(pl.x), int(pl.y)):
            sp *= C.MUD_SLOW
        nx = pl.x + mx * sp
        if not self._circle_hits_solid(nx, pl.y, C.PLAYER_RADIUS):
            pl.x = nx
        ny = pl.y + my * sp
        if not self._circle_hits_solid(pl.x, ny, C.PLAYER_RADIUS):
            pl.y = ny

    def _fire(self, idx: int, pl: PlayerState, events: list[tuple]) -> None:
        pl.cooldown = pl.shoot_cooldown
        pl.last_shot_time = self.time
        ox = math.cos(pl.aim)
        oy = math.sin(pl.aim)
        bx = pl.x + ox * (C.PLAYER_RADIUS + C.BULLET_RADIUS + 0.05)
        by = pl.y + oy * (C.PLAYER_RADIUS + C.BULLET_RADIUS + 0.05)
        if self.map.is_wall(int(bx), int(by)):
            bx, by = pl.x, pl.y  # muzzle inside a wall: fire from center
        self.bullets.append(Bullet(bx, by, ox * C.BULLET_SPEED, oy * C.BULLET_SPEED, idx))
        events.append(("shot", idx))

    def _update_bullets(self, dt: float, events: list[tuple]) -> None:
        for b in self.bullets:
            if not b.alive:
                continue
            b.ttl -= dt
            if b.ttl <= 0:
                b.alive = False
                continue
            # sub-step so fast bullets cannot tunnel through 1-cell walls
            steps = max(1, int(math.ceil(C.BULLET_SPEED * dt / 0.4)))
            sdt = dt / steps
            for _ in range(steps):
                b.x += b.vx * sdt
                b.y += b.vy * sdt
                b.dist += C.BULLET_SPEED * sdt
                if b.dist >= C.BULLET_RANGE:
                    b.alive = False   # out of range: fizzles mid-air
                    break
                if self.map.is_wall(int(b.x), int(b.y)):
                    b.alive = False
                    events.append(("bullet_wall", b.x, b.y))
                    break
                target = self.players[1 - b.owner]
                dx = target.x - b.x
                dy = target.y - b.y
                rr = C.PLAYER_RADIUS + C.BULLET_RADIUS
                if dx * dx + dy * dy < rr * rr:
                    b.alive = False
                    target.lives -= 1
                    target.last_hit_time = self.time
                    events.append(("hit", 1 - b.owner, b.owner))
                    # once per match, and only when the hit lands exactly on
                    # RETREAT_LIVES (hits at 2 or 1 life do not trigger it)
                    if target.lives == C.RETREAT_LIVES and not target.retreated:
                        self._retreat(1 - b.owner, b.owner, events)
                    break
        self.bullets = [b for b in self.bullets if b.alive]

    def _retreat(self, victim: int, shooter: int, events: list[tuple]) -> None:
        """Low on lives: teleport the victim to a random free cell inside the
        heat-safe zone and away from the shooter. No-op if none is found."""
        foe = self.players[shooter]
        pl = self.players[victim]
        for _ in range(200):
            cx, cy = self.map.random_free_cell(self.rng)
            x, y = cx + 0.5, cy + 0.5
            if self.in_heat(x, y):
                continue
            if (x - foe.x) ** 2 + (y - foe.y) ** 2 < C.RETREAT_MIN_DIST ** 2:
                continue
            pl.x, pl.y = x, y
            pl.heat_timer = 0.0
            pl.retreated = True   # once per match
            events.append(("retreat", victim, x, y))
            return

    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict:
        """JSON-serializable frame for replays."""
        return {
            "tick": self.tick,
            "time": round(self.time, 4),
            "players": [
                {
                    "x": round(p.x, 3), "y": round(p.y, 3), "aim": round(p.aim, 3),
                    "lives": p.lives,
                    "quickshot": round(p.quickshot_timer, 2),
                    "speed": round(p.speed_timer, 2),
                }
                for p in self.players
            ],
            "bullets": [
                {"x": round(b.x, 3), "y": round(b.y, 3), "owner": b.owner}
                for b in self.bullets
            ],
            "powerups": [
                {"x": round(p.x, 2), "y": round(p.y, 2), "kind": p.kind}
                for p in self.powerups.items
            ],
            "safe_half": round(self.safe_half(), 2),
            "winner": self.winner,
        }

    def map_payload(self) -> dict:
        """Static map data for replays (sent once)."""
        return {
            "grid": self.grid.astype(int).tolist(),
            "mud": self.map.mud.astype(int).tolist() if self.map.mud is not None else [],
            "trees": [list(t) for t in self.map.trees],
            "torches": [list(t) for t in self.map.torches],
            "size": C.GRID_SIZE,
        }
