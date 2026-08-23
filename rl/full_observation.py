"""Complete fixed-shape observations for contest and multi-agent training."""

from __future__ import annotations

import math

import numpy as np
from gymnasium import spaces

from core import constants as C
from core.engine import Engine, PlayerState
from core.los import has_line_of_sight

MAP_CHANNEL_NAMES = (
    "walls", "trees", "mud", "safe_zone", "self", "opponent",
    "powerup_life", "powerup_quickshot", "powerup_speed", "powerup_swap",
    "self_bullets", "opponent_bullets",
)
PLAYER_FEATURE_NAMES = (
    "x", "y", "aim_sin", "aim_cos", "lives", "cooldown",
    "quickshot_timer", "speed_timer", "heat_timer", "retreated",
    "last_hit_recency", "last_shot_recency", "in_heat", "current_speed",
    "line_of_sight",
)
BULLET_FEATURE_NAMES = (
    "x", "y", "vx", "vy", "owned_by_self", "ttl", "distance",
)
POWERUP_FEATURE_NAMES = (
    "x", "y", "ttl", "life", "quickshot", "speed", "swap",
)
GAME_FEATURE_NAMES = (
    "elapsed", "remaining", "tick", "safe_half", "fixed_dt",
    "player_speed", "bullet_speed", "bullet_range", "shoot_cooldown",
    "quickshot_cooldown", "powerup_interval", "powerup_ttl",
    "heat_shrink_rate", "heat_damage_period", "retreat_lives",
    "retreat_min_distance", "mud_speed_multiplier", "speed_powerup_multiplier",
)

# Engine rules bound both counts. The extra capacity keeps the contract stable
# if timing constants are tuned later; normal gameplay must never be truncated.
MAX_FULL_BULLETS = 32
MAX_FULL_POWERUPS = max(32, math.ceil(C.POWERUP_TTL / C.POWERUP_INTERVAL) + 4)


def full_observation_space() -> spaces.Dict:
    """Return the fixed contest observation space (all arrays are float32)."""
    return spaces.Dict({
        "map": spaces.Box(0.0, 1.0,
                          (len(MAP_CHANNEL_NAMES), C.GRID_SIZE, C.GRID_SIZE),
                          dtype=np.float32),
        "self": spaces.Box(-1.0, 1.0, (len(PLAYER_FEATURE_NAMES),),
                           dtype=np.float32),
        "opponent": spaces.Box(-1.0, 1.0, (len(PLAYER_FEATURE_NAMES),),
                               dtype=np.float32),
        "bullets": spaces.Box(-1.0, 1.0,
                              (MAX_FULL_BULLETS, len(BULLET_FEATURE_NAMES)),
                              dtype=np.float32),
        "bullet_mask": spaces.Box(0.0, 1.0, (MAX_FULL_BULLETS,),
                                  dtype=np.float32),
        "powerups": spaces.Box(-1.0, 1.0,
                               (MAX_FULL_POWERUPS, len(POWERUP_FEATURE_NAMES)),
                               dtype=np.float32),
        "powerup_mask": spaces.Box(0.0, 1.0, (MAX_FULL_POWERUPS,),
                                   dtype=np.float32),
        "game": spaces.Box(0.0, 1.0, (len(GAME_FEATURE_NAMES),),
                           dtype=np.float32),
    })


def _signed(value: float, maximum: float) -> float:
    # plain float math: np.clip on scalars dominates encode() otherwise
    scaled = value / maximum
    if scaled <= 0.0:
        return -1.0
    if scaled >= 1.0:
        return 1.0
    return scaled * 2.0 - 1.0


class FullObservationEncoder:
    """Build complete observations without hiding either player's state."""

    def __init__(self, engine: Engine, max_seconds: float = C.MAX_EPISODE_SECONDS):
        self.engine = engine
        self.max_seconds = max(float(max_seconds), C.FIXED_DT)
        self._static_map = np.zeros(
            (len(MAP_CHANNEL_NAMES), C.GRID_SIZE, C.GRID_SIZE), dtype=np.float32)
        self._static_map[0] = engine.grid
        self._static_map[1] = np.logical_and(engine.map.solid, np.logical_not(engine.grid))
        if engine.map.mud is not None:
            self._static_map[2] = engine.map.mud

    def _player(self, player: PlayerState, visible: bool) -> np.ndarray:
        eng = self.engine
        hit_age = max(0.0, eng.time - player.last_hit_time)
        shot_age = max(0.0, eng.time - player.last_shot_time)
        return np.asarray([
            player.x / C.WORLD * 2.0 - 1.0,
            player.y / C.WORLD * 2.0 - 1.0,
            math.sin(player.aim),
            math.cos(player.aim),
            _signed(player.lives, C.PLAYER_LIVES),
            _signed(player.cooldown, C.SHOOT_COOLDOWN),
            _signed(player.quickshot_timer, C.QUICKSHOT_DURATION),
            _signed(player.speed_timer, C.SPEED_DURATION),
            _signed(player.heat_timer, C.HEAT_DAMAGE_PERIOD),
            1.0 if player.retreated else -1.0,
            1.0 - min(hit_age, 10.0) / 5.0,
            1.0 - min(shot_age, 10.0) / 5.0,
            1.0 if eng.in_heat(player.x, player.y) else -1.0,
            _signed(player.speed, C.PLAYER_SPEED * C.SPEED_MULTIPLIER),
            1.0 if visible else -1.0,
        ], dtype=np.float32)

    def encode(self, idx: int) -> dict[str, np.ndarray]:
        """Encode the entire state from player ``idx``'s perspective."""
        if idx not in (0, 1):
            raise ValueError("player index must be 0 or 1")
        eng = self.engine
        me, opponent = eng.players[idx], eng.players[1 - idx]
        visible = has_line_of_sight(eng.grid, me.x, me.y, opponent.x, opponent.y)

        world = self._static_map.copy()
        centers = np.arange(C.GRID_SIZE, dtype=np.float32) + 0.5
        safe_axis = np.abs(centers - C.WORLD * 0.5) <= eng.safe_half()
        world[3] = np.logical_and(safe_axis[:, None], safe_axis[None, :])
        world[4, int(me.x), int(me.y)] = 1.0
        world[5, int(opponent.x), int(opponent.y)] = 1.0

        if len(eng.bullets) > MAX_FULL_BULLETS:
            raise RuntimeError("bullet observation capacity exceeded")
        bullets = np.zeros(
            (MAX_FULL_BULLETS, len(BULLET_FEATURE_NAMES)), dtype=np.float32)
        bullet_mask = np.zeros(MAX_FULL_BULLETS, dtype=np.float32)
        for row, bullet in enumerate(eng.bullets):
            owner_is_self = bullet.owner == idx
            bullets[row] = (
                bullet.x / C.WORLD * 2.0 - 1.0,
                bullet.y / C.WORLD * 2.0 - 1.0,
                np.clip(bullet.vx / C.BULLET_SPEED, -1.0, 1.0),
                np.clip(bullet.vy / C.BULLET_SPEED, -1.0, 1.0),
                1.0 if owner_is_self else -1.0,
                _signed(bullet.ttl, C.BULLET_TTL),
                _signed(bullet.dist, C.BULLET_RANGE),
            )
            bullet_mask[row] = 1.0
            channel = 10 if owner_is_self else 11
            bx = min(C.GRID_SIZE - 1, max(0, int(bullet.x)))
            by = min(C.GRID_SIZE - 1, max(0, int(bullet.y)))
            world[channel, bx, by] = 1.0

        items = eng.powerups.items
        if len(items) > MAX_FULL_POWERUPS:
            raise RuntimeError("powerup observation capacity exceeded")
        powerups = np.zeros(
            (MAX_FULL_POWERUPS, len(POWERUP_FEATURE_NAMES)), dtype=np.float32)
        powerup_mask = np.zeros(MAX_FULL_POWERUPS, dtype=np.float32)
        for row, powerup in enumerate(items):
            kind_idx = C.POWERUP_TYPES.index(powerup.kind)
            one_hot = [0.0] * len(C.POWERUP_TYPES)
            one_hot[kind_idx] = 1.0
            powerups[row] = (
                powerup.x / C.WORLD * 2.0 - 1.0,
                powerup.y / C.WORLD * 2.0 - 1.0,
                _signed(powerup.ttl, C.POWERUP_TTL),
                *one_hot,
            )
            powerup_mask[row] = 1.0
            world[6 + kind_idx, int(powerup.x), int(powerup.y)] = 1.0

        elapsed = min(1.0, eng.time / self.max_seconds)
        max_ticks = self.max_seconds / C.FIXED_DT
        game = np.asarray([
            elapsed,
            1.0 - elapsed,
            min(1.0, eng.tick / max_ticks),
            eng.safe_half() / (C.WORLD * 0.5),
            min(1.0, C.FIXED_DT / 0.1),
            min(1.0, C.PLAYER_SPEED / 10.0),
            min(1.0, C.BULLET_SPEED / 20.0),
            min(1.0, C.BULLET_RANGE / C.WORLD),
            min(1.0, C.SHOOT_COOLDOWN),
            min(1.0, C.QUICKSHOT_COOLDOWN),
            min(1.0, C.POWERUP_INTERVAL / 10.0),
            min(1.0, C.POWERUP_TTL / 60.0),
            min(1.0, C.HEAT_SHRINK_RATE),
            min(1.0, C.HEAT_DAMAGE_PERIOD / 10.0),
            min(1.0, C.RETREAT_LIVES / C.PLAYER_LIVES),
            min(1.0, C.RETREAT_MIN_DIST / C.WORLD),
            min(1.0, C.MUD_SLOW),
            min(1.0, C.SPEED_MULTIPLIER / 2.0),
        ], dtype=np.float32)

        return {
            "map": world,
            "self": self._player(me, visible),
            "opponent": self._player(opponent, visible),
            "bullets": bullets,
            "bullet_mask": bullet_mask,
            "powerups": powerups,
            "powerup_mask": powerup_mask,
            "game": game,
        }

