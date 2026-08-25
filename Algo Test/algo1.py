"""Trainable contest example: a small Q-table selects tactical behaviors.

The state key compresses the full observation into features that matter:
distance and line of sight to the opponent, incoming-bullet threat with
flight direction (not just proximity), mud underfoot, health and lives
balance, heat-zone margin, powerup proximity, and whether either player's
aim is aligned with the other. ``default_q`` seeds each feature's
contribution as a heuristic prior that Q-learning is free to override.
"""

from __future__ import annotations

import math
import os

import numpy as np

from core import constants as C
from core.pathfinding import astar

NUM_OPTIONS = 8
CHASE, RETREAT, STRAFE_LEFT, STRAFE_RIGHT, POWERUP, EVADE, HOLD, SAFE_CENTER = range(
    NUM_OPTIONS)
OPTION_NAMES = (
    "chase", "retreat", "strafe_left", "strafe_right",
    "powerup", "evade", "hold", "safe_center",
)

# distance band, line of sight, bullet threat (none/approaching/imminent),
# on mud, health band, lives difference, heat margin, powerup proximity,
# own aim aligned on the target, opponent aiming at us; then the option.
STATE_SHAPE = (5, 2, 3, 2, 3, 3, 3, 3, 2, 2)
Q_SHAPE = STATE_SHAPE + (NUM_OPTIONS,)


def _world_xy(vector: np.ndarray) -> tuple[float, float]:
    return float(vector[0] + 1.0) * 50.0, float(vector[1] + 1.0) * 50.0


def _lives(vector: np.ndarray) -> int:
    return int(round(float(vector[4] + 1.0) * 4.0))


def _distance(obs: dict) -> float:
    sx, sy = _world_xy(obs["self"])
    ex, ey = _world_xy(obs["opponent"])
    return math.hypot(ex - sx, ey - sy)


def _aim_error(aim_sin: float, aim_cos: float, dx: float, dy: float) -> float:
    """Absolute wrapped angle between a player's aim and a direction."""
    angle = math.atan2(float(aim_sin), float(aim_cos))
    target = math.atan2(dy, dx)
    return abs((angle - target + math.pi) % (2 * math.pi) - math.pi)


def _aim_tolerance(distance: float) -> float:
    """Angular slack that still lands a bullet on a 0.5-block target."""
    return max(0.05, min(0.12, math.asin(min(0.5 / max(distance, 1e-6), 0.99))))


def _threatening_bullets(obs: dict):
    """Enemy bullets whose flight line passes near us before the range cap.

    Yields (closing_distance, row) for each genuine threat: the bullet must
    be flying toward us, miss by less than 2 blocks at closest approach, and
    still have enough of its 20-block range left to get there.
    """
    sx, sy = _world_xy(obs["self"])
    for row, valid in zip(obs["bullets"], obs["bullet_mask"]):
        if valid < 0.5 or row[4] > 0.0:
            continue  # no bullet here, or one of ours
        bx, by = float(row[0] + 1.0) * 50.0, float(row[1] + 1.0) * 50.0
        vx, vy = float(row[2]) * C.BULLET_SPEED, float(row[3]) * C.BULLET_SPEED
        rx, ry = sx - bx, sy - by
        vv = vx * vx + vy * vy
        if vv < 1e-6:
            continue
        t_star = (rx * vx + ry * vy) / vv
        if t_star <= 0.0:
            continue  # already flying away from us
        travelled = (float(row[6]) + 1.0) * 0.5 * C.BULLET_RANGE
        closing = t_star * C.BULLET_SPEED
        if closing > C.BULLET_RANGE - travelled:
            continue  # fizzles out before reaching us
        mx, my = rx - vx * t_star, ry - vy * t_star
        if mx * mx + my * my > 2.0 ** 2:
            continue  # passes wide
        yield closing, row


def _bullet_threat(obs: dict) -> int:
    level = 0
    for closing, _row in _threatening_bullets(obs):
        level = max(level, 2 if closing < 8.0 else 1)
    return level


def _heat_band(obs: dict) -> int:
    """0 = burning outside the safe zone, 1 = safe but near the edge,
    2 = comfortably inside."""
    if obs["self"][12] > 0.0:
        return 0
    sx, sy = _world_xy(obs["self"])
    safe_half = float(obs["game"][3]) * C.WORLD * 0.5
    margin = safe_half - max(abs(sx - C.WORLD * 0.5), abs(sy - C.WORLD * 0.5))
    return 1 if margin < 6.0 else 2


def _powerup_band(obs: dict) -> int:
    """0 = none on the map, 1 = nearest is far, 2 = nearest is close."""
    sx, sy = _world_xy(obs["self"])
    best = None
    for row, valid in zip(obs["powerups"], obs["powerup_mask"]):
        if valid < 0.5:
            continue
        px, py = float(row[0] + 1.0) * 50.0, float(row[1] + 1.0) * 50.0
        d2 = (px - sx) ** 2 + (py - sy) ** 2
        if best is None or d2 < best:
            best = d2
    if best is None:
        return 0
    return 2 if best <= 15.0 ** 2 else 1


def state_key(obs: dict) -> tuple[int, ...]:
    """Compress the full observation into a deliberately small RL state."""
    sx, sy = _world_xy(obs["self"])
    ex, ey = _world_xy(obs["opponent"])
    distance = math.hypot(ex - sx, ey - sy)
    distance_bin = next(
        (idx for idx, edge in enumerate((8.0, 14.0, 20.0, 35.0))
         if distance < edge), 4)
    visible = int(obs["self"][14] > 0.0)
    threat = _bullet_threat(obs)
    on_mud = int(obs["map"][2, int(sx), int(sy)] > 0.5)
    my_lives, other_lives = _lives(obs["self"]), _lives(obs["opponent"])
    health = 0 if my_lives <= 3 else (1 if my_lives <= 5 else 2)
    lives_difference = 1 + (my_lives > other_lives) - (my_lives < other_lives)
    heat = _heat_band(obs)
    powerup = _powerup_band(obs)
    aligned = int(
        visible
        and _aim_error(obs["self"][2], obs["self"][3], ex - sx, ey - sy)
        <= _aim_tolerance(distance))
    enemy_aiming = int(
        visible
        and _aim_error(obs["opponent"][2], obs["opponent"][3], sx - ex, sy - ey)
        <= 0.25)
    return (distance_bin, visible, threat, on_mud, health,
            lives_difference, heat, powerup, aligned, enemy_aiming)


def default_q() -> np.ndarray:
    """Heuristic priors: every state feature adds its weight to each option.

    Q-learning starts from these educated guesses and is free to replace
    them with whatever the reward stream actually supports.
    """
    q = np.zeros(Q_SHAPE, dtype=np.float32)
    for key in np.ndindex(STATE_SHAPE):
        (distance, visible, threat, mud, health, lives_diff,
         heat, powerup, aligned, enemy_aiming) = key
        close = distance <= 2
        q[key + (CHASE,)] = 0.12 if distance >= 2 else 0.04
        q[key + (STRAFE_LEFT,)] = 0.10 if visible and close else 0.0
        q[key + (STRAFE_RIGHT,)] = 0.09 if visible and close else 0.0
        # HOLD is an aim-alignment behavior: worth most when already aligned.
        q[key + (HOLD,)] = (0.30 if aligned else 0.16) if visible and close else 0.0
        # EVADE scales with how real the incoming fire is.
        q[key + (EVADE,)] = (0.32 if threat == 2 else
                             0.15 if threat == 1 else 0.0)
        if enemy_aiming and visible and close:
            q[key + (EVADE,)] += 0.08
        # POWERUP scales with proximity; hurt players want it more.
        q[key + (POWERUP,)] = ((0.24 if health < 2 else 0.12) if powerup == 2 else
                               (0.10 if health < 2 else 0.04) if powerup == 1 else 0.0)
        q[key + (RETREAT,)] = 0.13 if health == 0 or lives_diff == 0 else 0.0
        if threat == 2 and health == 0:
            q[key + (RETREAT,)] += 0.08
        # SAFE_CENTER matters when burning and stays relevant near the edge.
        q[key + (SAFE_CENTER,)] = 0.40 if heat == 0 else (0.15 if heat == 1 else 0.0)
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

    def _evade_move(self, obs: dict) -> tuple[int, int]:
        """Step perpendicular to the most imminent incoming bullet, on the
        side that widens its miss; strafe around a staring opponent instead."""
        sx, sy = _world_xy(obs["self"])
        ex, ey = _world_xy(obs["opponent"])
        threats = sorted(_threatening_bullets(obs), key=lambda item: item[0])
        if not threats:
            dx, dy = ex - sx, ey - sy
            return self._safe_move(obs, int(np.sign(-dy)), int(np.sign(dx)))
        _closing, row = threats[0]
        vx, vy = float(row[2]), float(row[3])
        bx, by = float(row[0] + 1.0) * 50.0, float(row[1] + 1.0) * 50.0
        rx, ry = sx - bx, sy - by
        vv = vx * vx + vy * vy
        t_star = (rx * vx + ry * vy) / vv
        # miss vector at closest approach; move along it, away from the lane
        mx, my = rx - vx * t_star, ry - vy * t_star
        if mx * mx + my * my < 1e-6:  # dead-center hit incoming: pick a side
            mx, my = -vy, vx
        return self._safe_move(obs, int(np.sign(mx)), int(np.sign(my)))

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
            mx, my = self._evade_move(obs)
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
