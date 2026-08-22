"""Trainable contest example: a small Q-table selects tactical behaviors."""

from __future__ import annotations

import math
import os

import numpy as np

from core.pathfinding import astar

NUM_OPTIONS = 8
CHASE, RETREAT, STRAFE_LEFT, STRAFE_RIGHT, POWERUP, EVADE, HOLD, SAFE_CENTER = range(
    NUM_OPTIONS)
OPTION_NAMES = (
    "chase", "retreat", "strafe_left", "strafe_right",
    "powerup", "evade", "hold", "safe_center",
)

# distance, line of sight, bullet threat, mud, health band, lives difference,
# heat, any powerup, tactical option
STATE_SHAPE = (5, 2, 2, 2, 3, 3, 2, 2)
Q_SHAPE = STATE_SHAPE + (NUM_OPTIONS,)


def _world_xy(vector: np.ndarray) -> tuple[float, float]:
    return float(vector[0] + 1.0) * 50.0, float(vector[1] + 1.0) * 50.0


def _lives(vector: np.ndarray) -> int:
    return int(round(float(vector[4] + 1.0) * 4.0))


def _distance(obs: dict) -> float:
    sx, sy = _world_xy(obs["self"])
    ex, ey = _world_xy(obs["opponent"])
    return math.hypot(ex - sx, ey - sy)


def _bullet_threat(obs: dict) -> bool:
    sx, sy = _world_xy(obs["self"])
    for row, valid in zip(obs["bullets"], obs["bullet_mask"]):
        if valid < 0.5 or row[4] > 0.0:
            continue
        bx, by = float(row[0] + 1.0) * 50.0, float(row[1] + 1.0) * 50.0
        if (bx - sx) ** 2 + (by - sy) ** 2 <= 8.0 ** 2:
            return True
    return False


def state_key(obs: dict) -> tuple[int, ...]:
    """Compress the full observation into a deliberately small RL state."""
    distance = _distance(obs)
    distance_bin = next(
        (idx for idx, edge in enumerate((8.0, 14.0, 20.0, 35.0))
         if distance < edge), 4)
    visible = int(obs["self"][14] > 0.0)
    threat = int(_bullet_threat(obs))
    sx, sy = _world_xy(obs["self"])
    on_mud = int(obs["map"][2, int(sx), int(sy)] > 0.5)
    my_lives, other_lives = _lives(obs["self"]), _lives(obs["opponent"])
    health = 0 if my_lives <= 3 else (1 if my_lives <= 5 else 2)
    lives_difference = 1 + (my_lives > other_lives) - (my_lives < other_lives)
    in_heat = int(obs["self"][12] > 0.0)
    has_powerup = int(np.any(obs["powerup_mask"] > 0.5))
    return (distance_bin, visible, threat, on_mud, health,
            lives_difference, in_heat, has_powerup)


def default_q() -> np.ndarray:
    """Untrained tactical priors; Q-learning is free to replace them."""
    q = np.zeros(Q_SHAPE, dtype=np.float32)
    for key in np.ndindex(STATE_SHAPE):
        distance, visible, threat, mud, health, lives_diff, heat, powerup = key
        q[key + (CHASE,)] = 0.12 if distance >= 2 else 0.04
        q[key + (STRAFE_LEFT,)] = 0.10 if visible and distance <= 2 else 0.0
        q[key + (STRAFE_RIGHT,)] = 0.09 if visible and distance <= 2 else 0.0
        # HOLD becomes an aim-alignment behavior when a target is visible.
        q[key + (HOLD,)] = 0.24 if visible and distance <= 2 else 0.0
        q[key + (POWERUP,)] = (0.16 if health < 2 else 0.03) if powerup else 0.0
        q[key + (RETREAT,)] = 0.13 if health == 0 or lives_diff == 0 else 0.0
        q[key + (EVADE,)] = 0.30 if threat else 0.0
        q[key + (SAFE_CENTER,)] = 0.35 if heat else 0.0
        if mud:
            q[key + (POWERUP,)] -= 0.03
            q[key + (CHASE,)] += 0.03
    return q


class Agent:
    """A compact Q-policy with deterministic movement and aiming helpers."""

    def __init__(self, weights_path=None):
        self.q = default_q()
        if weights_path:
            model_path = os.path.join(weights_path, "qtable.npz")
            if os.path.isfile(model_path):
                loaded = np.load(model_path, allow_pickle=False)["q"]
                if loaded.shape != Q_SHAPE:
                    raise ValueError(
                        f"qtable shape {loaded.shape} does not match {Q_SHAPE}")
                self.q = loaded.astype(np.float32)
        self.reset()

    def reset(self):
        self._path: list[tuple[int, int]] = []
        self._path_mode = ""
        self._path_goal: tuple[int, int] | None = None
        self._path_age = 10**9
        self._last_enemy: tuple[float, float] | None = None
        self._enemy_velocity = (0.0, 0.0)

    def option(self, obs: dict) -> int:
        return int(np.argmax(self.q[state_key(obs)]))

    @staticmethod
    def _safe_move(obs: dict, move_x: int, move_y: int) -> tuple[int, int]:
        if move_x == move_y == 0:
            return 0, 0
        sx, sy = _world_xy(obs["self"])
        cx, cy = int(sx), int(sy)
        solid = np.logical_or(obs["map"][0], obs["map"][1])
        candidates = (
            (move_x, move_y), (move_x, 0), (0, move_y),
            (-move_y, move_x), (move_y, -move_x), (0, 0),
        )
        for mx, my in candidates:
            nx, ny = cx + mx, cy + my
            if 0 <= nx < 100 and 0 <= ny < 100 and not solid[nx, ny]:
                return mx, my
        return 0, 0

    def _path_move(self, obs: dict, goal: tuple[float, float], mode: str) -> tuple[int, int]:
        sx, sy = _world_xy(obs["self"])
        goal_cell = (int(goal[0]), int(goal[1]))
        target_moved = (
            self._path_goal is None
            or abs(goal_cell[0] - self._path_goal[0]) + abs(goal_cell[1] - self._path_goal[1]) > 3
        )
        if mode != self._path_mode or not self._path or self._path_age >= 18 or target_moved:
            solid = np.logical_or(obs["map"][0], obs["map"][1]).astype(np.uint8)
            self._path = astar(solid, sx, sy, goal[0], goal[1])
            self._path_mode = mode
            self._path_goal = goal_cell
            self._path_age = 0
        self._path_age += 1
        while len(self._path) > 1:
            reached_x = self._path[1][0] + 0.5 - sx
            reached_y = self._path[1][1] + 0.5 - sy
            if reached_x * reached_x + reached_y * reached_y >= 0.35 ** 2:
                break
            self._path.pop(0)
        if len(self._path) < 2:
            return 0, 0
        nx, ny = self._path[1]
        target_x, target_y = nx + 0.5, ny + 0.5
        move_x = 0 if abs(target_x - sx) < 0.08 else (target_x > sx) - (target_x < sx)
        move_y = 0 if abs(target_y - sy) < 0.08 else (target_y > sy) - (target_y < sy)
        return self._safe_move(obs, move_x, move_y)

    @staticmethod
    def _powerup_target(obs: dict) -> tuple[float, float] | None:
        sx, sy = _world_xy(obs["self"])
        lives = _lives(obs["self"])
        best = None
        for row, valid in zip(obs["powerups"], obs["powerup_mask"]):
            if valid < 0.5:
                continue
            px, py = float(row[0] + 1.0) * 50.0, float(row[1] + 1.0) * 50.0
            score = math.hypot(px - sx, py - sy)
            if lives < 6 and row[3] > 0.5:  # prefer LIFE when damaged
                score -= 12.0
            if row[5] > 0.5:                # SPEED helps catch a fleeing Dijkstra
                score -= 10.0
            elif row[4] > 0.5:              # QUICKSHOT improves close pressure
                score -= 7.0
            candidate = (score, px, py)
            if best is None or candidate < best:
                best = candidate
        return None if best is None else (best[1], best[2])

    def action_for_option(self, obs: dict, option: int) -> np.ndarray:
        sx, sy = _world_xy(obs["self"])
        ex, ey = _world_xy(obs["opponent"])
        dx, dy = ex - sx, ey - sy
        distance = max(1e-6, math.hypot(dx, dy))

        if self._last_enemy is not None:
            instant_vx = np.clip((ex - self._last_enemy[0]) * 30.0, -5.5, 5.5)
            instant_vy = np.clip((ey - self._last_enemy[1]) * 30.0, -5.5, 5.5)
            self._enemy_velocity = (
                self._enemy_velocity[0] * 0.65 + float(instant_vx) * 0.35,
                self._enemy_velocity[1] * 0.65 + float(instant_vy) * 0.35,
            )
        self._last_enemy = (ex, ey)
        horizon = min(distance / 14.0, 0.8)
        lead_x = ex + self._enemy_velocity[0] * horizon
        lead_y = ey + self._enemy_velocity[1] * horizon
        aim_angle = math.atan2(lead_y - sy, lead_x - sx)
        aim = int(round((aim_angle % (2 * math.pi)) * 16 / (2 * math.pi))) % 16
        bin_angle = aim * (2 * math.pi / 16)
        aim_error = (aim_angle - bin_angle + math.pi) % (2 * math.pi) - math.pi

        if option == CHASE:
            mx, my = self._path_move(obs, (ex, ey), "enemy")
        elif option == RETREAT:
            mx, my = self._safe_move(
                obs, (sx > ex) - (sx < ex), (sy > ey) - (sy < ey))
        elif option in (STRAFE_LEFT, STRAFE_RIGHT):
            side = 1 if option == STRAFE_LEFT else -1
            mx = int(np.sign(-dy * side))
            my = int(np.sign(dx * side))
            mx, my = self._safe_move(obs, mx, my)
        elif option == POWERUP:
            target = self._powerup_target(obs)
            mx, my = (self._path_move(obs, target, "powerup")
                      if target is not None else self._path_move(obs, (ex, ey), "enemy"))
        elif option == EVADE:
            nearest = None
            for row, valid in zip(obs["bullets"], obs["bullet_mask"]):
                if valid < 0.5 or row[4] > 0.0:
                    continue
                bx, by = float(row[0] + 1.0) * 50.0, float(row[1] + 1.0) * 50.0
                candidate = ((bx - sx) ** 2 + (by - sy) ** 2, row)
                if nearest is None or candidate[0] < nearest[0]:
                    nearest = candidate
            if nearest is None:
                mx, my = self._safe_move(obs, int(np.sign(-dy)), int(np.sign(dx)))
            else:
                row = nearest[1]
                mx, my = self._safe_move(
                    obs, int(np.sign(-row[3])), int(np.sign(row[2])))
        elif option == SAFE_CENTER:
            mx, my = self._path_move(obs, (50.0, 50.0), "safe")
        elif option == HOLD and obs["self"][14] > 0.0:
            # Move perpendicular until the predicted target direction lines up
            # with one of the 16 legal aim bins, then hold the firing lane.
            if abs(aim_error) < 0.018:
                mx, my = 0, 0
            else:
                side = 1 if aim_error > 0.0 else -1
                mx = int(np.sign(-dy * side))
                my = int(np.sign(dx * side))
                mx, my = self._safe_move(obs, mx, my)
        else:
            mx, my = 0, 0

        move = (my + 1) * 3 + (mx + 1)
        tolerance = max(0.025, min(0.10, math.asin(min(0.5 / distance, 0.99))))
        shoot = int(
            obs["self"][14] > 0.0 and distance <= 19.5
            and abs(aim_error) <= tolerance)
        return np.asarray([move, aim, shoot], dtype=np.int64)

    def act(self, obs: dict) -> np.ndarray:
        return self.action_for_option(obs, self.option(obs))
