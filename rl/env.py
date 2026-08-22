"""Gymnasium environment: BlocksWithGunsEnv.

Wraps core.engine. The learning agent is always player 0; the opponent
(player 1) is one of the built-in deterministic bots (or another agent
for self-play). No pygame needed for stepping.

Action space: MultiDiscrete([9, 16, 2])
    move:  3x3 grid (0..8) -> dx, dy in {-1, 0, 1}
    aim:   16 absolute bins -> angle = bin * 2pi / 16
    shoot: 0 = hold fire, 1 = fire

Observation (Dict, all float32):
    self:      [x, y, sin(aim), cos(aim), lives/PLAYER_LIVES, cooldown/max,
                quickshot/5s, speed/5s]                       (8,)
    enemy:     [rel_x/W, rel_y/W, dist/W, lives/PLAYER_LIVES, visible,
                reloading]                                    (6,)
    local_grid: 21x21 crop around the agent (1 = wall or tree, i.e. solid)
    bullets:   up to 8 nearest bullets [rel_x, rel_y, vx, vy] (normalized)
    powerups:  up to 4 powerups [rel_x, rel_y, one-hot kind x4]
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from bots import make_bot, make_view
from core import constants as C
from core.engine import Action, Engine
from core.los import has_line_of_sight

MOVE_LUT = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]  # 9 moves
AIM_BINS = 16
MAX_BULLETS = 8
MAX_POWERUPS = 4
LOCAL_R = 10  # local grid half-size -> 21x21


class BlocksWithGunsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        opponent: str | Any = "astar",
        map_seed: int = 0,
        scale: float = 24.0,
        octaves: int = 3,
        threshold: float = 0.62,
        max_seconds: float = C.MAX_EPISODE_SECONDS,
        swap_sides: bool = False,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.opponent_spec = opponent
        self.map_params = dict(map_seed=map_seed, scale=scale, octaves=octaves,
                               threshold=threshold)
        self.max_seconds = max_seconds
        self.swap_sides = swap_sides
        self.render_mode = render_mode

        self.action_space = spaces.MultiDiscrete([9, AIM_BINS, 2])
        self.observation_space = spaces.Dict({
            "self": spaces.Box(-1.0, 1.0, (8,), dtype=np.float32),
            "enemy": spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
            "local_grid": spaces.Box(0.0, 1.0, (2 * LOCAL_R + 1, 2 * LOCAL_R + 1),
                                     dtype=np.float32),
            "bullets": spaces.Box(-1.0, 1.0, (MAX_BULLETS, 4), dtype=np.float32),
            "powerups": spaces.Box(-1.0, 1.0, (MAX_POWERUPS, 6), dtype=np.float32),
        })

        self.engine: Engine | None = None
        self._opponent = None
        self._episode = 0
        self._screen = None
        self._clock = None
        self._renderer = None

    # ------------------------------------------------------------------ #
    def _make_opponent(self, seed: int):
        spec = self.opponent_spec
        if isinstance(spec, str):
            bot = make_bot(spec, seed=seed)
            return ("bot", bot)
        # user agent object with act(obs_enemy_view) -> gym action (self-play)
        return ("agent", spec)

    def _opp_action(self) -> Action:
        kind, opp = self._opponent
        if kind == "bot":
            return opp.act(make_view(self.engine, 1))
        # self-play: agent sees the mirrored obs
        obs = self._obs(1)
        act = np.asarray(opp.act(obs), dtype=int).ravel()
        return self._to_engine_action(act)

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        s = seed if seed is not None else self.map_params["map_seed"]
        self.engine = Engine(seed=s, map_seed=s,
                             scale=self.map_params["scale"],
                             octaves=self.map_params["octaves"],
                             threshold=self.map_params["threshold"])
        if self.swap_sides and self._episode % 2 == 1:
            self.engine.reset(swap_spawns=True)
        self._episode += 1
        self.engine.time = 0.0
        self._opponent = self._make_opponent(s + 1)
        if hasattr(self._opponent[1], "reset"):
            self._opponent[1].reset()
        return self._obs(0), {"map_seed": s}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_engine_action(act: np.ndarray) -> Action:
        move = MOVE_LUT[int(act[0]) % 9]
        aim = (int(act[1]) % AIM_BINS) * (2 * math.pi / AIM_BINS)
        return Action(move_x=float(move[0]), move_y=float(move[1]),
                      aim=aim, shoot=bool(int(act[2])))

    def step(self, action):
        eng = self.engine
        my = self._to_engine_action(np.asarray(action, dtype=int).ravel())
        opp = self._opp_action()
        events = eng.step((my, opp))

        reward = -0.005  # step penalty
        for e in events:
            if e[0] == "hit":
                victim, shooter = e[1], e[2]
                if shooter == 0:
                    reward += 1.0
                if victim == 0:
                    reward -= 1.0
            elif e[0] == "pickup" and e[2] == 0:
                reward += 0.5
            elif e[0] == "heat_hit" and e[1] == 0:
                reward -= 1.0
        terminated = eng.winner is not None
        if terminated:
            if eng.winner == 0:
                reward += 10.0
            elif eng.winner == 1:
                reward -= 10.0
        truncated = False
        if not terminated and eng.time >= self.max_seconds:
            terminated = True  # engine already decides by lives at 120 s;
            truncated = eng.winner is None  # shorter cap -> truncation
        info = {"winner": eng.winner, "time": eng.time,
                "lives": (eng.players[0].lives, eng.players[1].lives)}
        return self._obs(0), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _obs(self, idx: int) -> dict[str, np.ndarray]:
        eng = self.engine
        me = eng.players[idx]
        foe = eng.players[1 - idx]
        W = C.WORLD

        self_vec = np.array([
            me.x / W * 2 - 1, me.y / W * 2 - 1,
            math.sin(me.aim), math.cos(me.aim),
            me.lives / C.PLAYER_LIVES * 2 - 1,
            min(1.0, me.cooldown / C.SHOOT_COOLDOWN) * 2 - 1,
            min(1.0, me.quickshot_timer / C.QUICKSHOT_DURATION) * 2 - 1,
            min(1.0, me.speed_timer / C.SPEED_DURATION) * 2 - 1,
        ], dtype=np.float32)

        dx, dy = foe.x - me.x, foe.y - me.y
        dist = math.hypot(dx, dy)
        visible = has_line_of_sight(eng.grid, me.x, me.y, foe.x, foe.y)
        enemy_vec = np.array([
            np.clip(dx / W, -1, 1), np.clip(dy / W, -1, 1),
            min(1.0, dist / W) * 2 - 1,
            foe.lives / C.PLAYER_LIVES * 2 - 1,
            1.0 if visible else -1.0,
            min(1.0, foe.cooldown / C.SHOOT_COOLDOWN) * 2 - 1,
        ], dtype=np.float32)

        # local grid crop, padded with walls
        cx, cy = int(me.x), int(me.y)
        local = np.ones((2 * LOCAL_R + 1, 2 * LOCAL_R + 1), dtype=np.float32)
        for ix in range(-LOCAL_R, LOCAL_R + 1):
            for iy in range(-LOCAL_R, LOCAL_R + 1):
                gx, gy = cx + ix, cy + iy
                if 0 <= gx < C.GRID_SIZE and 0 <= gy < C.GRID_SIZE:
                    local[ix + LOCAL_R, iy + LOCAL_R] = float(eng.map.solid[gx, gy])

        # nearest bullets, relative
        bl = sorted(eng.bullets,
                    key=lambda b: (b.x - me.x) ** 2 + (b.y - me.y) ** 2)[:MAX_BULLETS]
        bullets = np.full((MAX_BULLETS, 4), -1.0, dtype=np.float32)
        for i, b in enumerate(bl):
            bullets[i] = [
                np.clip((b.x - me.x) / 40, -1, 1),
                np.clip((b.y - me.y) / 40, -1, 1),
                np.clip(b.vx / C.BULLET_SPEED, -1, 1),
                np.clip(b.vy / C.BULLET_SPEED, -1, 1),
            ]

        powerups = np.full((MAX_POWERUPS, 6), -1.0, dtype=np.float32)
        for i, p in enumerate(eng.powerups.items[:MAX_POWERUPS]):
            onehot = [0.0] * len(C.POWERUP_TYPES)
            onehot[C.POWERUP_TYPES.index(p.kind)] = 1.0
            powerups[i] = [
                np.clip((p.x - me.x) / 80, -1, 1),
                np.clip((p.y - me.y) / 80, -1, 1),
                *[v * 2 - 1 for v in onehot],
            ]

        return {"self": self_vec, "enemy": enemy_vec, "local_grid": local,
                "bullets": bullets, "powerups": powerups}

    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return None
        import os
        if self.render_mode != "human":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        from game.render_topdown import TopdownRenderer

        if self._renderer is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((640, 640))
                self._clock = pygame.time.Clock()
            else:
                self._screen = pygame.Surface((640, 640))
            self._renderer = TopdownRenderer(self.engine, px_per_cell=3)
        canvas = pygame.Surface((600, 600))
        self._renderer.draw(canvas, tick=self.engine.tick)
        self._screen.fill((24, 28, 24))
        self._screen.blit(canvas, (20, 20))
        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        return np.transpose(pygame.surfarray.array3d(self._screen), (1, 0, 2)).copy()

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
            self._renderer = None
