"""Powerup spawning and effects.

Every POWERUP_INTERVAL seconds (first after POWERUP_FIRST_DELAY) a powerup
appears at a random free cell and expires after POWERUP_TTL seconds if
nobody picks it up. Types: LIFE, QUICKSHOT, SPEED, SWAP.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import constants as C
from .mapgen import GameMap


@dataclass
class Powerup:
    x: float
    y: float
    kind: str  # one of C.POWERUP_TYPES
    alive: bool = True
    ttl: float = field(default_factory=lambda: C.POWERUP_TTL)


class PowerupManager:
    def __init__(self, game_map: GameMap, rng: random.Random):
        self.map = game_map
        self.rng = rng
        self.items: list[Powerup] = []
        self.timer = C.POWERUP_FIRST_DELAY

    def reset(self) -> None:
        self.items.clear()
        self.timer = C.POWERUP_FIRST_DELAY

    def update(self, dt: float) -> list[tuple]:
        """Tick spawn/expiry timers. Returns events."""
        events: list[tuple] = []
        for p in self.items:
            p.ttl -= dt
            if p.ttl <= 0.0:
                p.alive = False
                events.append(("powerup_expire", p.kind, p.x, p.y))
        self.items = [p for p in self.items if p.alive]
        self.timer -= dt
        # Tolerance makes exact fixed-tick cadences (such as 1.0 s at 30 Hz)
        # spawn on the intended tick despite floating-point subtraction noise.
        if self.timer <= 1e-9:
            self.timer = C.POWERUP_INTERVAL
            cx, cy = self.map.random_free_cell(self.rng)
            kind = self.rng.choice(C.POWERUP_TYPES)
            p = Powerup(cx + 0.5, cy + 0.5, kind)
            self.items.append(p)
            events.append(("powerup_spawn", p.kind, p.x, p.y))
        return events

    def try_pickup(self, players) -> list[tuple]:
        """Apply pickup effects for players touching a powerup. Returns events."""
        events: list[tuple] = []
        reach = C.POWERUP_RADIUS + C.PLAYER_RADIUS
        for p in self.items:
            if not p.alive:
                continue
            for idx, pl in enumerate(players):
                dx = pl.x - p.x
                dy = pl.y - p.y
                if dx * dx + dy * dy <= reach * reach:
                    p.alive = False
                    other = players[1 - idx]
                    if p.kind == "LIFE":
                        pl.lives = min(C.PLAYER_LIVES, pl.lives + 1)
                    elif p.kind == "QUICKSHOT":
                        pl.quickshot_timer = C.QUICKSHOT_DURATION
                    elif p.kind == "SPEED":
                        pl.speed_timer = C.SPEED_DURATION
                    elif p.kind == "SWAP":
                        pl.x, other.x = other.x, pl.x
                        pl.y, other.y = other.y, pl.y
                    events.append(("pickup", p.kind, idx))
                    break
        self.items = [p for p in self.items if p.alive]
        return events
