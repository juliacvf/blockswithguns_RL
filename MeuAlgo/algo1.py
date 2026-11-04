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
import heapq
from collections import deque
from pathlib import Path

import numpy as np

from core import constants as C
from core.pathfinding import astar
from core.los import has_line_of_sight

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
STATE_SHAPE = (5, 2, 3, 2, 3, 3, 3, 3, 2, 2, 3)
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
    elapsed = float(obs["game"][0])
    time_band = (
        0 if elapsed < 0.5
        else 1 if elapsed < 0.8
        else 2
        )
    return (distance_bin, visible, threat, on_mud, health,
            lives_difference, heat, powerup, aligned, enemy_aiming, time_band)


def default_q() -> np.ndarray:
    """Heuristic priors: every state feature adds its weight to each option.

    Q-learning starts from these educated guesses and is free to replace
    them with whatever the reward stream actually supports.
    """
    q = np.zeros(Q_SHAPE, dtype=np.float32)
    for key in np.ndindex(STATE_SHAPE):
        (distance, visible, threat, mud, health, lives_diff,
         heat, powerup, aligned, enemy_aiming, time_band) = key
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

        # No fim da partida, se estiver empatado ou perdendo,
        # prioriza ações ofensivas e reduz a passividade.
        if time_band == 2 and lives_diff <= 1:
            q[key + (CHASE,)] += 0.15
            q[key + (STRAFE_LEFT,)] += 0.05
            q[key + (STRAFE_RIGHT,)] += 0.05

            if not aligned:
                q[key + (HOLD,)] -= 0.08

            if powerup != 2:
                q[key + (POWERUP,)] -= 0.04
    return q


class Agent:
    """A compact Q-policy with deterministic movement and aiming helpers."""

    profile = "general"
    decision_ticks = 1
    learned_tactics = False

    def state_key(self, obs):
        return state_key(obs)

    def available_options(self, obs):
        return np.arange(NUM_OPTIONS)

    def best_option(self, obs, key=None):
        key = self.state_key(obs) if key is None else key
        options = self.available_options(obs)
        return int(options[np.argmax(self.q[key][options])])

    def learning_context(self, key):
        return None

    def _plan_path(self, solid, obs, sx, sy, gx, gy):
        return astar(solid, sx, sy, gx, gy)

    def __init__(self, weights_path=None):
        self.q = default_q()
        if weights_path:
            model_path = os.fspath(weights_path)
            if os.path.isdir(model_path):
                model_path = os.path.join(model_path, "qtable.npz")
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
        self._path_origin = None
        self._progress_age = 0
        self._blocked_cells = set()

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

        # Track progress across goal changes too: alternating tactics must
        # not reset the stuck detector on every tick.
        if self._path_origin is None:
            self._path_origin = (sx, sy)
        if self._progress_age >= 18:
            if (len(self._path) > 1
                    and math.hypot(sx - self._path_origin[0],
                                   sy - self._path_origin[1]) < 0.4):
                blocked = self._path[1]
                if blocked not in (goal_cell, (int(sx), int(sy))):
                    self._blocked_cells.add(blocked)
                    self._path = []
            self._path_origin = (sx, sy)
            self._progress_age = 0
        self._progress_age += 1

        target_moved = (
            self._path_goal is None
            or abs(goal_cell[0] - self._path_goal[0])
            + abs(goal_cell[1] - self._path_goal[1]) > 3
        )

        if mode != self._path_mode or not self._path or self._path_age >= 18 or target_moved:
            solid = np.logical_or(obs["map"][0], obs["map"][1]).astype(np.uint8)

            for bx, by in self._blocked_cells:
                if (bx, by) != (int(sx), int(sy)) and (bx, by) != goal_cell:
                    solid[bx, by] = 1

            self._path = self._plan_path(solid, obs, sx, sy, goal[0], goal[1])
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
        # Centering inside our current free cell does not enter a neighbor.
        # Checking that neighbor can reject this move beside a wall forever.
        if (nx, ny) == (int(sx), int(sy)):
            return move_x, move_y
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


STRAFE_PROFILE = "strafe_v1"
# distance, sight, incoming threat, health, score, heat, lateral motion,
# recent reversal, late match. Keep the state smaller than the general table.
STRAFE_STATE_SHAPE = (5, 2, 3, 3, 3, 3, 3, 2, 2)
STRAFE_Q_SHAPE = STRAFE_STATE_SHAPE + (NUM_OPTIONS,)
STRAFE_DECISION_TICKS = 6


def strafe_default_q():
    q = np.zeros(STRAFE_Q_SHAPE, dtype=np.float32)
    for key in np.ndindex(STRAFE_STATE_SHAPE):
        distance, sight, threat, health, score, heat, lateral, turning, late = key
        close = distance in (1, 2)
        q[key + (CHASE,)] = 0.50 if distance >= 3 or not sight else 0.16
        q[key + (RETREAT,)] = 0.55 if distance == 0 else 0.02
        q[key + (STRAFE_LEFT,)] = 0.32 if sight and close else 0.04
        q[key + (STRAFE_RIGHT,)] = 0.32 if sight and close else 0.04
        # Follow the observed tangent to reduce relative lateral motion.
        if lateral != 1 and sight and close:
            q[key + (STRAFE_LEFT if lateral == 2 else STRAFE_RIGHT,)] += 0.10
        q[key + (HOLD,)] = 0.46 if sight and close and not turning else 0.08
        q[key + (EVADE,)] = (0.10, 0.55, 1.10)[threat]
        q[key + (POWERUP,)] = 0.40 if health == 0 else 0.12
        q[key + (SAFE_CENTER,)] = (1.40, 0.60, 0.02)[heat]
        if late and score <= 1:
            q[key + (CHASE,)] += 0.10
    return q


def intercept_time(dx, dy, vx, vy):
    """Positive root for a constant-speed bullet meeting a moving target."""
    a = vx * vx + vy * vy - C.BULLET_SPEED ** 2
    b = 2.0 * (dx * vx + dy * vy)
    c = dx * dx + dy * dy
    disc = b * b - 4.0 * a * c
    if disc < 0.0 or abs(a) < 1e-9:
        return math.hypot(dx, dy) / C.BULLET_SPEED
    roots = [t for t in ((-b + math.sqrt(disc)) / (2 * a),
                         (-b - math.sqrt(disc)) / (2 * a)) if t >= 0]
    return min(roots) if roots else math.hypot(dx, dy) / C.BULLET_SPEED


class StrafeAgent(Agent):
    profile = STRAFE_PROFILE
    decision_ticks = STRAFE_DECISION_TICKS
    learned_tactics = True
    q_shape = STRAFE_Q_SHAPE
    preferred_range = 11.0
    navigation_options = (CHASE, POWERUP, SAFE_CENTER)

    def initial_q(self):
        return strafe_default_q()

    def __init__(self, weights_path=None):
        self.q = self.initial_q()
        if weights_path is not None:
            path = Path(weights_path)
            if path.is_dir():
                path /= "qtable.npz"
            with np.load(path, allow_pickle=False) as checkpoint:
                if "profile" not in checkpoint.files or str(checkpoint["profile"].item()) != self.profile:
                    raise ValueError(f"checkpoint does not match {self.profile}")
                q = checkpoint["q"]
                if q.shape != self.q_shape or not np.isfinite(q).all():
                    raise ValueError(f"invalid {self.profile} Q-table")
                self.q = q.astype(np.float32)
        self.reset()

    def reset(self):
        super().reset()
        self._observed_tick = None
        self._previous_time = None
        self._motion_velocity = (0.0, 0.0)
        self._lateral = 1
        self._turn_until = -1.0
        self._decision_time = -1.0
        self._selected_option = CHASE
        self._last_context = None
        self._route_recoveries = 0

    def _path_move(self, obs, goal, mode):
        move = super()._path_move(obs, goal, mode)
        if not self._path and self._blocked_cells:
            # Avoidance memory is provisional. If it disconnects us, retry
            # against actual obstacles instead of treating it as permanent walls.
            self._blocked_cells.clear()
            self._route_recoveries += 1
            move = super()._path_move(obs, goal, mode)
        return move

    def observe(self, obs):
        tick = float(obs["game"][2])
        if tick == self._observed_tick:
            return
        now = float(obs["game"][0]) * C.MAX_EPISODE_SECONDS
        ex, ey = _world_xy(obs["opponent"])
        if self._last_enemy is not None and self._previous_time is not None:
            dt = now - self._previous_time
            displacement = math.dist((ex, ey), self._last_enemy)
            if dt > 1e-5 and displacement <= 6.0 * dt + 1e-3:
                velocity = ((ex - self._last_enemy[0]) / dt,
                            (ey - self._last_enemy[1]) / dt)
                sx, sy = _world_xy(obs["self"])
                distance = max(math.hypot(ex - sx, ey - sy), 1e-6)
                lateral_speed = ((ex - sx) * velocity[1] - (ey - sy) * velocity[0]) / distance
                lateral = 0 if lateral_speed < -0.5 else (2 if lateral_speed > 0.5 else 1)
                reversal = lateral != 1 and self._lateral != 1 and lateral != self._lateral
                if reversal:
                    self._turn_until = now + 0.4
                gain = 1.0 if reversal else 0.65
                self._motion_velocity = tuple(
                    gain * v + (1 - gain) * old
                    for v, old in zip(velocity, self._motion_velocity))
                self._lateral = lateral
            else:
                # SWAP/retreat teleports are not enemy movement velocities.
                self._motion_velocity = (0.0, 0.0)
                self._lateral = 1
                self._turn_until = now + 0.4
        self._last_enemy = (ex, ey)
        self._previous_time = now
        self._observed_tick = tick

    def state_key(self, obs):
        self.observe(obs)
        sx, sy = _world_xy(obs["self"])
        ex, ey = _world_xy(obs["opponent"])
        distance = math.hypot(ex - sx, ey - sy)
        band = next((i for i, edge in enumerate((8., 12., 17., 22.)) if distance < edge), 4)
        mine, enemy = _lives(obs["self"]), _lives(obs["opponent"])
        return (band, int(obs["self"][14] > 0), _bullet_threat(obs),
                0 if mine <= 3 else (1 if mine <= 5 else 2),
                1 + int(mine > enemy) - int(mine < enemy), _heat_band(obs),
                self._lateral, int(self._previous_time < self._turn_until),
                int(obs["game"][0] >= 0.8))

    def learning_context(self, key):
        return key[1], key[2], key[5], key[7]

    def learning_potential(self, obs):
        return strafe_potential(obs)

    def learning_extra_reward(self, events):
        reward = 0.0
        for event in events:
            if event[0] == "hit":
                reward += 0.5 if event[2] == 0 else -0.5
            elif event[0] == "heat_hit" and event[1] == 0:
                reward -= 0.5
        return reward

    def option(self, obs):
        key = self.state_key(obs)
        context = self.learning_context(key), tuple(self.available_options(obs))
        now = self._previous_time
        # Commit for 0.2 s; reconsider immediately on threat/visibility/heat changes.
        if now - self._decision_time >= self.decision_ticks * C.FIXED_DT - 1e-5 or context != self._last_context:
            self._selected_option = self.best_option(obs, key)
            self._decision_time = now
            self._last_context = context
        return self._selected_option

    @staticmethod
    def _step_position(obs, mx, my):
        """Predict legal movement including body radius and wall sliding."""
        x, y = _world_xy(obs["self"])
        mag = math.hypot(mx, my)
        if not mag:
            return x, y
        # Feature 13 is signed-normalized against the maximum buffed speed.
        speed = (float(obs["self"][13]) + 1) * 0.5 * C.PLAYER_SPEED * C.SPEED_MULTIPLIER
        step = speed * C.FIXED_DT
        if obs["map"][2, int(x), int(y)] > 0.5:
            step *= C.MUD_SLOW
        solid = obs["map"][0] + obs["map"][1]

        def blocked(px, py):
            r = C.PLAYER_RADIUS
            for cx in range(math.floor(px - r), math.floor(px + r) + 1):
                for cy in range(math.floor(py - r), math.floor(py + r) + 1):
                    if not (0 <= cx < C.GRID_SIZE and 0 <= cy < C.GRID_SIZE):
                        return True
                    if solid[cx, cy] and ((px - min(max(px, cx), cx + 1)) ** 2
                                         + (py - min(max(py, cy), cy + 1)) ** 2 < r * r):
                        return True
            return False

        nx = x + mx / mag * step
        if not blocked(nx, y):
            x = nx
        ny = y + my / mag * step
        if not blocked(x, ny):
            y = ny
        return x, y

    @staticmethod
    def _safe_move(obs, move_x, move_y):
        if move_x == move_y == 0:
            return 0, 0
        sx, sy = _world_xy(obs["self"])
        candidates = []
        for my in (-1, 0, 1):
            for mx in (-1, 0, 1):
                x, y = StrafeAgent._step_position(obs, mx, my)
                dx, dy = x - sx, y - sy
                score = dx * move_x + dy * move_y
                score -= 0.05 * abs(dx * move_y - dy * move_x)
                candidates.append((score, -(mx * mx + my * my), mx, my))
        _, _, mx, my = max(candidates)
        return mx, my

    def _shot(self, obs, sx, sy):
        ex, ey = _world_xy(obs["opponent"])
        vx, vy = self._motion_velocity
        t = intercept_time(ex - sx, ey - sy, vx, vy)
        px, py = ex + vx * t, ey + vy * t
        angle = math.atan2(py - sy, px - sx)
        aim = round(angle * 16 / (2 * math.pi)) % 16
        a = aim * 2 * math.pi / 16
        rvx, rvy = vx - C.BULLET_SPEED * math.cos(a), vy - C.BULLET_SPEED * math.sin(a)
        closest = max(0., -( (ex - sx) * rvx + (ey - sy) * rvy) / max(rvx * rvx + rvy * rvy, 1e-9))
        miss = math.hypot(ex - sx + rvx * closest, ey - sy + rvy * closest)
        in_range = closest <= C.BULLET_RANGE / C.BULLET_SPEED
        shoot = bool(obs["self"][14] > 0 and in_range and miss <= C.PLAYER_RADIUS + C.BULLET_RADIUS
                     and 0 <= px < C.WORLD and 0 <= py < C.WORLD
                     and has_line_of_sight(obs["map"][0], sx, sy, px, py))
        return int(aim), int(shoot), miss

    def action_for_option(self, obs, option):
        self.observe(obs)
        sx, sy = _world_xy(obs["self"])
        ex, ey = _world_xy(obs["opponent"])
        dx, dy = ex - sx, ey - sy
        distance = max(math.hypot(dx, dy), 1e-6)
        if option == CHASE:
            if not obs["self"][14] > 0 or distance > 22:
                mx, my = self._path_move(obs, (ex, ey), "enemy")
            else:
                vx, vy = self._motion_velocity
                horizon = min(0.6, distance / C.BULLET_SPEED)
                mx, my = self._safe_move(obs, dx + vx * horizon, dy + vy * horizon)
        elif option in (STRAFE_LEFT, STRAFE_RIGHT):
            side = 1 if option == STRAFE_LEFT else -1
            radial = max(-1., min(1., (distance - self.preferred_range) / 5.))
            mx, my = self._safe_move(obs, radial * dx / distance - side * dy / distance,
                                    radial * dy / distance + side * dx / distance)
        elif option == HOLD and obs["self"][14] > 0:
            candidates = []
            for cy in (-1, 0, 1):
                for cx in (-1, 0, 1):
                    x, y = self._step_position(obs, cx, cy)
                    _, _, miss = self._shot(obs, x, y)
                    score = -min(miss, 5.) - 0.015 * abs(math.hypot(ex - x, ey - y) - self.preferred_range)
                    candidates.append((score, -(cx * cx + cy * cy), cx, cy))
            _, _, mx, my = max(candidates)
        elif option == RETREAT:
            mx, my = self._safe_move(obs, -dx, -dy)
        elif option == EVADE:
            mx, my = self._evade_move(obs)
        elif option == POWERUP:
            target = self._powerup_target(obs)
            mx, my = self._path_move(obs, target or (ex, ey), "powerup" if target else "enemy")
        elif option == SAFE_CENTER:
            mx, my = self._path_move(obs, (50., 50.), "safe")
        else:
            mx, my = self._path_move(obs, (ex, ey), "enemy")
        aim, shoot, _ = self._shot(obs, sx, sy)
        return np.asarray([(my + 1) * 3 + mx + 1, aim, shoot], dtype=np.int64)


def strafe_potential(obs):
    """Bounded positioning hint; differencing prevents farming proximity."""
    sx, sy = _world_xy(obs["self"])
    ex, ey = _world_xy(obs["opponent"])
    distance = math.hypot(ex - sx, ey - sy)
    value = -0.15 * min(abs(distance - 11.) / 20., 1.)
    if obs["self"][14] > 0 and distance < 20:
        value += 0.10
    if obs["self"][12] > 0:
        value -= 0.25
    return value


DIJKSTRA_PROFILE = "dijkstra_v1"
# distance, sight, bullet threat, health, score, heat, enemy radial motion,
# enemy outside the safe zone, late match, nearby useful pickup.
DIJKSTRA_STATE_SHAPE = (5, 2, 3, 2, 3, 3, 3, 2, 2, 2)
DIJKSTRA_Q_SHAPE = DIJKSTRA_STATE_SHAPE + (NUM_OPTIONS,)


def dijkstra_default_q():
    q = np.zeros(DIJKSTRA_Q_SHAPE, dtype=np.float32)
    for key in np.ndindex(DIJKSTRA_STATE_SHAPE):
        distance, sight, threat, health, score, heat, radial, enemy_heat, late, pickup = key
        firing_range = sight and distance in (1, 2)
        q[key + (CHASE,)] = 0.65 if not sight or distance >= 2 else 0.25
        if radial == 2 and distance >= 2:
            q[key + (CHASE,)] += 0.13
        q[key + (HOLD,)] = 0.60 if firing_range else 0.02
        q[key + (STRAFE_LEFT,)] = 0.34 if sight and distance <= 2 else 0.08
        q[key + (STRAFE_RIGHT,)] = 0.34 if sight and distance <= 2 else 0.08
        q[key + (EVADE,)] = (0.08, 0.55, 1.10)[threat]
        q[key + (RETREAT,)] = 0.35 if health == 0 and threat else 0.02
        q[key + (POWERUP,)] = (0.55 if health == 0 or radial == 2 else 0.20) if pickup else 0.01
        q[key + (SAFE_CENTER,)] = (1.40, 0.85, 0.02)[heat]
        if enemy_heat and score >= 1:
            q[key + (SAFE_CENTER,)] += 0.20
            q[key + (CHASE,)] -= 0.25
        if late and score == 0:
            q[key + (CHASE,)] += 0.15
            q[key + (HOLD,)] += 0.10
    return q


class DijkstraAgent(StrafeAgent):
    """Intercept observed retreats while retaining reachable, safe routes."""
    profile = DIJKSTRA_PROFILE
    q_shape = DIJKSTRA_Q_SHAPE
    preferred_range = 13.0

    def initial_q(self):
        return dijkstra_default_q()

    def reset(self):
        super().reset()
        self._reachable = None
        self._reachable_centers = None
        self._combat_tick = None
        self._combat_history = deque()
        self._combat_shots = 0
        self._combat_cooldown = None
        self._combat_enemy_lives = None
        self._combat_previous_positions = None
        self._reposition_goal = None
        self._reposition_until = -1.
        self._combat_repositions = 0
        self._return_checked = -1e9
        self._returning = False
        self._return_goal = None
        self._return_path = []
        self._return_forecasts = 0

    def _forecast_return(self, obs, allow_heat=False):
        """Earliest return with cell deadlines imposed by the shrinking zone."""
        sx, sy = _world_xy(obs["self"])
        start = int(sx), int(sy)
        solid = np.logical_or(obs["map"][0], obs["map"][1])
        mud = obs["map"][2]
        speed = min(C.PLAYER_SPEED, (float(obs["self"][13]) + 1) * .5 *
                    C.PLAYER_SPEED * C.SPEED_MULTIPLIER)
        speed = max(speed, .1)
        now = float(obs["game"][0]) * C.MAX_EPISODE_SECONDS
        lookahead = min(20., max(0., C.MAX_EPISODE_SECONDS - now))
        target_margin = 8. + C.HEAT_SHRINK_RATE * lookahead
        arrival = {start: 0.}
        came = {}
        queue = [(0., start)]
        while queue:
            seconds, cell = heapq.heappop(queue)
            if seconds != arrival[cell]:
                continue
            x, y = cell
            margin = self._safe_margin(obs, x + .5, y + .5) - C.HEAT_SHRINK_RATE * seconds
            if margin >= target_margin:
                path = [cell]
                while cell in came:
                    cell = came[cell]
                    path.append(cell)
                path.reverse()
                slack = self._safe_margin(obs, sx, sy) / C.HEAT_SHRINK_RATE
                if len(path) > 1:
                    slack = min(slack, min(self._safe_margin(obs, px + .5, py + .5) /
                                          C.HEAT_SHRINK_RATE - arrival[(px, py)]
                                          for px, py in path[1:]))
                return path, slack
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not (0 <= nx < C.GRID_SIZE and 0 <= ny < C.GRID_SIZE) or solid[nx, ny]:
                    continue
                # Half of each edge is spent in each cell's terrain.
                cost = (.5 / (C.MUD_SLOW if mud[x, y] else 1.) +
                        .5 / (C.MUD_SLOW if mud[nx, ny] else 1.)) / speed
                eta = seconds + cost
                future_margin = self._safe_margin(obs, nx + .5, ny + .5) - C.HEAT_SHRINK_RATE * eta
                if not allow_heat and future_margin < .5:
                    continue
                if eta < arrival.get((nx, ny), float("inf")):
                    arrival[nx, ny] = eta
                    came[(nx, ny)] = (x, y)
                    heapq.heappush(queue, (eta, (nx, ny)))
        return [], -float("inf")

    def _return_needed(self, obs):
        now = float(obs["game"][0]) * C.MAX_EPISODE_SECONDS
        sx, sy = _world_xy(obs["self"])
        margin = self._safe_margin(obs, sx, sy)
        if now - self._return_checked < .75 and (self._returning or margin >= 2.):
            return self._returning
        self._return_checked = now
        if margin >= 16. and not self._returning:
            return False
        path, slack = self._forecast_return(obs)
        needed = slack < 8. or margin < 2. or (self._returning and margin < 8.)
        if needed and not path:
            path, _ = self._forecast_return(obs, allow_heat=True)
        if needed and not self._returning:
            self._return_forecasts += 1
        self._returning = needed
        self._return_path = path
        self._return_goal = (path[-1][0] + .5, path[-1][1] + .5) if path else (50., 50.)
        return needed

    def _aim_after_move(self, obs, mx, my):
        # The engine moves the shooter before creating its bullet.
        x, y = self._step_position(obs, mx, my)
        aim, shoot, _ = self._shot(obs, x, y)
        return np.asarray([(my + 1) * 3 + mx + 1, aim, shoot], dtype=np.int64)

    def observe(self, obs):
        super().observe(obs)
        if self._combat_tick == self._observed_tick:
            return
        self._combat_tick = self._observed_tick
        now = self._previous_time
        position, enemy = _world_xy(obs["self"]), _world_xy(obs["opponent"])
        lives = _lives(obs["opponent"])
        teleported = (self._combat_previous_positions is not None and
                      any(math.dist(a, b) > 1. for a, b in
                          zip((position, enemy), self._combat_previous_positions)))
        if lives != self._combat_enemy_lives or teleported:
            self._combat_history.clear()
            self._reposition_goal = None
        if teleported:
            self._return_checked = -1e9
        cooldown = float(obs["self"][5])
        if self._combat_cooldown is not None and cooldown > self._combat_cooldown + .1:
            self._combat_shots += 1
        self._combat_cooldown = cooldown
        self._combat_enemy_lives = lives
        self._combat_previous_positions = position, enemy
        self._combat_history.append((now, position, self._combat_shots,
                                     math.dist(position, enemy) <= C.BULLET_RANGE,
                                     enemy, obs["self"][14] > 0))
        while len(self._combat_history) > 1 and self._combat_history[1][0] < now - 6.:
            self._combat_history.popleft()

    def _shot(self, obs, sx, sy, velocity=None):
        # Grid visibility does not describe the actual quantized bullet lane.
        ex, ey = _world_xy(obs["opponent"])
        if velocity is None:
            velocity = self._motion_velocity
            recent = [row[4] for row in self._combat_history
                      if row[0] >= self._previous_time - .5]
            # Tiny wall-induced reversals do not imply sustained future motion.
            if len(recent) >= 12 and max(math.dist(p, recent[0]) for p in recent) < .5:
                velocity = (0., 0.)
        vx, vy = velocity
        t = intercept_time(ex - sx, ey - sy, vx, vy)
        aim = round(math.atan2(ey + vy * t - sy, ex + vx * t - sx) * 16 / (2 * math.pi)) % 16
        angle = aim * 2 * math.pi / 16
        ux, uy = math.cos(angle), math.sin(angle)
        rvx, rvy = vx - C.BULLET_SPEED * ux, vy - C.BULLET_SPEED * uy
        closest = max(0., -((ex - sx) * rvx + (ey - sy) * rvy) /
                      max(rvx * rvx + rvy * rvy, 1e-9))
        miss = math.hypot(ex - sx + rvx * closest, ey - sy + rvy * closest)
        allowed_miss = C.PLAYER_RADIUS + C.BULLET_RADIUS - .15 * min(closest, 1.)
        shoot = closest <= C.BULLET_RANGE / C.BULLET_SPEED and miss <= allowed_miss
        if shoot:
            muzzle = C.PLAYER_RADIUS + C.BULLET_RADIUS + .05
            bx, by = sx + ux * muzzle, sy + uy * muzzle
            walls = obs["map"][0]
            if walls[min(99, max(0, int(bx))), min(99, max(0, int(by)))]:
                bx, by = sx, sy
            length = max(0., closest * C.BULLET_SPEED - math.dist((sx, sy), (bx, by)))
            steps = max(1, math.ceil(length / .15))
            for i in range(steps + 1):
                x, y = bx + ux * length * i / steps, by + uy * length * i / steps
                if not (0 <= x < C.WORLD and 0 <= y < C.WORLD) or walls[int(x), int(y)]:
                    shoot = False
                    break
        return aim, int(shoot), miss

    def _firing_position(self, obs):
        sx, sy = _world_xy(obs["self"])
        ex, ey = _world_xy(obs["opponent"])
        self._safe_goal(obs, (sx, sy))  # Populate the static reachable component.
        solid = np.logical_or(obs["map"][0], obs["map"][1])
        candidates = set()
        for radius in (3., 5., 7.):
            for direction in range(16):
                angle = direction * math.pi / 8
                candidates.add((int(sx + math.cos(angle) * radius),
                                int(sy + math.sin(angle) * radius)))
        best = None
        for cx, cy in sorted(candidates):
            x, y = cx + .5, cy + .5
            if not (0 <= cx < C.GRID_SIZE and 0 <= cy < C.GRID_SIZE):
                continue
            if not self._reachable[cx, cy] or self._safe_margin(obs, x, y) < 3.:
                continue
            distance = math.hypot(ex - x, ey - y)
            if not 5. <= distance <= 18.:
                continue
            _, shoot, miss = self._shot(obs, x, y, velocity=(0., 0.))
            if not shoot:
                continue
            path = self._plan_path(solid, obs, sx, sy, x, y)
            if len(path) < 2:
                continue
            cost = len(path) + sum(float(obs["map"][2, px, py]) for px, py in path)
            cost += .25 * abs(distance - self.preferred_range) + miss
            candidate = cost, x, y
            if best is None or candidate < best:
                best = candidate
        return None if best is None else (best[1], best[2])

    def _combat_move(self, obs):
        now = self._previous_time
        sx, sy = _world_xy(obs["self"])
        if self._safe_margin(obs, sx, sy) < 3.:
            self._reposition_goal = None
            return None
        if self._reposition_goal is not None:
            goal = self._reposition_goal
            if (now >= self._reposition_until or math.dist((sx, sy), goal) < .6
                    or self._safe_margin(obs, *goal) < 2.):
                self._reposition_goal = None
                self._combat_history.clear()
            else:
                if _bullet_threat(obs) == 2:
                    return self._evade_move(obs)
                return self._path_move(obs, goal, "firing_position")
        history = self._combat_history
        if (len(history) < 2 or now - history[0][0] < 5.9
                or (self._combat_shots - history[0][2] < 3
                    and sum(row[5] for row in history) < .5 * len(history))
                or sum(row[3] for row in history) < .7 * len(history)
                or max(math.dist(row[1], history[0][1]) for row in history) > 3.5):
            return None
        goal = self._firing_position(obs)
        history.clear()
        if goal is None:
            return None
        self._reposition_goal = goal
        self._reposition_until = now + 5.
        self._combat_repositions += 1
        return self._path_move(obs, goal, "firing_position")

    @staticmethod
    def _safe_margin(obs, x, y):
        return float(obs["game"][3]) * C.WORLD * 0.5 - max(abs(x - 50.), abs(y - 50.))

    def available_options(self, obs):
        sx, sy = _world_xy(obs["self"])
        if self._safe_margin(obs, sx, sy) < 2.0 or self._return_needed(obs):
            return np.asarray([SAFE_CENTER])
        return super().available_options(obs)

    def _powerup_target(self, obs):
        sx, sy = _world_xy(obs["self"])
        lives = _lives(obs["self"])
        best = None
        for row, valid in zip(obs["powerups"], obs["powerup_mask"]):
            if valid < 0.5:
                continue
            x, y = _world_xy(row)
            distance = math.hypot(x - sx, y - sy)
            if distance > 24. or self._safe_margin(obs, x, y) < 3.:
                continue
            benefit = 0.
            if row[3] > 0.5 and lives < C.PLAYER_LIVES:
                benefit = 10. if lives <= 3 else 6.
            elif row[5] > 0.5 and obs["self"][7] <= -0.99:
                benefit = 8.
            elif row[4] > 0.5 and obs["self"][6] <= -0.99:
                benefit = 3.
            if not benefit:
                continue
            candidate = (distance - benefit, x, y)
            if best is None or candidate < best:
                best = candidate
        return None if best is None else (best[1], best[2])

    def state_key(self, obs):
        self.observe(obs)
        sx, sy = _world_xy(obs["self"])
        ex, ey = _world_xy(obs["opponent"])
        dx, dy = ex - sx, ey - sy
        distance = max(math.hypot(dx, dy), 1e-6)
        band = next((i for i, edge in enumerate((8., 14., 20., 30.)) if distance < edge), 4)
        radial_speed = (dx * self._motion_velocity[0] + dy * self._motion_velocity[1]) / distance
        radial = 0 if radial_speed < -0.5 else 2 if radial_speed > 0.5 else 1
        mine, enemy = _lives(obs["self"]), _lives(obs["opponent"])
        target = self._powerup_target(obs)
        return (band, int(obs["self"][14] > 0), _bullet_threat(obs), int(mine > 3),
                1 + int(mine > enemy) - int(mine < enemy), _heat_band(obs), radial,
                int(obs["opponent"][12] > 0), int(obs["game"][0] >= 0.8),
                int(target is not None and math.dist((sx, sy), target) <= 12.))

    def learning_context(self, key):
        return key[1], key[2], key[5], key[6], key[7]

    def learning_potential(self, obs):
        sx, sy = _world_xy(obs["self"])
        ex, ey = _world_xy(obs["opponent"])
        distance = math.hypot(ex - sx, ey - sy)
        value = -0.15 * min(abs(distance - self.preferred_range) / 25., 1.)
        if obs["self"][14] > 0 and distance < C.BULLET_RANGE:
            value += 0.10
        value -= (0.75, 0.10, 0.)[_heat_band(obs)]
        return value

    def learning_extra_reward(self, events):
        reward = super().learning_extra_reward(events)
        reward -= 1.5 * sum(event[0] == "heat_hit" and event[1] == 0 for event in events)
        return reward

    def _safe_goal(self, obs, goal):
        sx, sy = _world_xy(obs["self"])
        if self._reachable is None:
            solid = np.logical_or(obs["map"][0], obs["map"][1])
            reachable = np.zeros_like(solid, dtype=bool)
            start = int(sx), int(sy)
            queue = deque([start])
            reachable[start] = True
            while queue:
                x, y = queue.popleft()
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if (0 <= nx < C.GRID_SIZE and 0 <= ny < C.GRID_SIZE
                            and not solid[nx, ny] and not reachable[nx, ny]):
                        reachable[nx, ny] = True
                        queue.append((nx, ny))
            self._reachable = reachable
            self._reachable_centers = np.argwhere(reachable).astype(float) + 0.5
        cells = self._reachable_centers
        half = max(0., float(obs["game"][3]) * 50. - 2.)
        safe = np.max(np.abs(cells - 50.), axis=1) <= half
        candidates = cells[safe] if np.any(safe) else cells
        target = np.clip(np.asarray(goal, dtype=float), 50. - half, 50. + half)
        score = np.sum((candidates - target) ** 2, axis=1)
        point = candidates[int(np.argmin(score))]
        return float(point[0]), float(point[1])

    def _plan_path(self, solid, obs, sx, sy, gx, gy):
        if self._returning and self._return_path:
            cell = int(sx), int(sy)
            if cell in self._return_path:
                return self._return_path[self._return_path.index(cell):]
        mud = obs["map"][2]
        if self._safe_margin(obs, sx, sy) >= 0:
            centers = np.arange(C.GRID_SIZE) + 0.5
            half = float(obs["game"][3]) * 50.
            axis = np.abs(centers - 50.) <= half
            confined = np.logical_or(solid, ~np.logical_and(axis[:, None], axis[None, :]))
            path = astar(confined, sx, sy, gx, gy, mud, 1.0)
            if path:
                return path
        return astar(solid, sx, sy, gx, gy, mud, 1.0)

    def _path_move(self, obs, goal, mode):
        sx, sy = _world_xy(obs["self"])
        if mode == "safe" and self._safe_margin(obs, sx, sy) < 0:
            goal = sx, sy  # Reach the nearest safe area before heading inward.
        return super()._path_move(obs, self._safe_goal(obs, goal), mode)

    def action_for_option(self, obs, option):
        self.observe(obs)
        options = self.available_options(obs)
        if option not in options:
            option = int(options[0])
        if self._return_needed(obs):
            self._reposition_goal = None
            mx, my = super()._path_move(obs, self._return_goal, "future_safe")
            return self._aim_after_move(obs, mx, my)
        move = self._combat_move(obs)
        if move is not None:
            mx, my = move
            return self._aim_after_move(obs, mx, my)
        if option == CHASE:
            sx, sy = _world_xy(obs["self"])
            ex, ey = _world_xy(obs["opponent"])
            distance = math.hypot(ex - sx, ey - sy)
            horizon = min(1.2, max(0.3, distance / C.BULLET_SPEED))
            vx, vy = self._motion_velocity
            goal = self._safe_goal(obs, (ex + vx * horizon, ey + vy * horizon))
            mx, my = self._safe_move(obs, goal[0] - sx, goal[1] - sy)
            nx, ny = self._step_position(obs, mx, my)
            if obs["self"][14] <= 0 or distance > 20. or math.hypot(nx - sx, ny - sy) < 0.05:
                mx, my = self._path_move(obs, goal, "intercept")
            return self._aim_after_move(obs, mx, my)
        if option == HOLD and obs["self"][14] <= 0:
            option = SAFE_CENTER if obs["opponent"][12] > 0 else CHASE
            return self.action_for_option(obs, option)
        action = super().action_for_option(obs, option)
        return self._aim_after_move(obs, int(action[0]) % 3 - 1, int(action[0]) // 3 - 1)


ASTAR_PROFILE = "astar_v1"
# distance, sight, bullet threat, health, score, heat, enemy radial motion,
# enemy aim near us, own mud, useful nearby pickup, late match.
ASTAR_STATE_SHAPE = (5, 2, 3, 2, 3, 3, 3, 2, 2, 2, 2)
ASTAR_Q_SHAPE = ASTAR_STATE_SHAPE + (NUM_OPTIONS,)


def astar_default_q():
    q = np.zeros(ASTAR_Q_SHAPE, dtype=np.float32)
    for key in np.ndindex(ASTAR_STATE_SHAPE):
        distance, sight, threat, health, score, heat, closing, aimed, mud, pickup, late = key
        close = distance == 0
        engaging = sight and distance <= 3
        q[key + (CHASE,)] = .75 if not sight or distance >= 3 else .10
        q[key + (STRAFE_LEFT,)] = .76 if engaging else .05
        q[key + (STRAFE_RIGHT,)] = .76 if engaging else .05
        if aimed and engaging:
            q[key + (STRAFE_LEFT,)] += .08
            q[key + (STRAFE_RIGHT,)] += .08
        q[key + (RETREAT,)] = .95 if close and sight else .25 if health == 0 else .02
        q[key + (EVADE,)] = (.05, .90, 1.45)[threat]
        q[key + (HOLD,)] = .60 if sight and distance in (1, 2) and closing != 0 else .02
        q[key + (POWERUP,)] = (.70 if health == 0 else .35) if pickup else .01
        q[key + (SAFE_CENTER,)] = (1.5, .85, .02)[heat]
        if mud and engaging:
            q[key + (RETREAT,)] += .10
        if late and score == 0 and not engaging:
            q[key + (CHASE,)] += .15
    return q


class AStarAgent(DijkstraAgent):
    """Control an advancing opponent's range while reacting to actual shots."""
    profile = ASTAR_PROFILE
    q_shape = ASTAR_Q_SHAPE
    preferred_range = 14.0

    def initial_q(self):
        return astar_default_q()

    def state_key(self, obs):
        self.observe(obs)
        sx, sy = _world_xy(obs["self"])
        ex, ey = _world_xy(obs["opponent"])
        dx, dy = ex - sx, ey - sy
        distance = max(math.hypot(dx, dy), 1e-6)
        band = next((i for i, edge in enumerate((8., 12., 17., 22.)) if distance < edge), 4)
        radial = (dx * self._motion_velocity[0] + dy * self._motion_velocity[1]) / distance
        closing = 0 if radial < -.5 else 2 if radial > .5 else 1
        mine, enemy = _lives(obs["self"]), _lives(obs["opponent"])
        row = obs["opponent"]
        aimed = int(_aim_error(row[2], row[3], -dx, -dy) < .35)
        target = self._powerup_target(obs)
        return (band, int(obs["self"][14] > 0), _bullet_threat(obs), int(mine > 3),
                1 + int(mine > enemy) - int(mine < enemy), _heat_band(obs), closing, aimed,
                int(obs["map"][2, int(sx), int(sy)] > .5),
                int(target is not None and math.dist((sx, sy), target) < 10.),
                int(obs["game"][0] >= .8))

    def learning_context(self, key):
        return key[1], key[2], key[5], key[6], key[7], key[8]

    def action_for_option(self, obs, option):
        if option not in (STRAFE_LEFT, STRAFE_RIGHT):
            return super().action_for_option(obs, option)
        self.observe(obs)
        if self._return_needed(obs) or option not in self.available_options(obs):
            return super().action_for_option(obs, SAFE_CENTER)
        move = self._combat_move(obs)
        if move is not None:
            return self._aim_after_move(obs, *move)
        if obs["self"][14] <= 0:
            return super().action_for_option(obs, CHASE)
        sx, sy = _world_xy(obs["self"])
        ex, ey = _world_xy(obs["opponent"])
        dx, dy = ex - sx, ey - sy
        distance = max(math.hypot(dx, dy), 1e-6)
        incoming = (dx * self._motion_velocity[0] + dy * self._motion_velocity[1]) / distance
        radial = max(-2., min(1., (distance - self.preferred_range) / 4. + incoming / C.PLAYER_SPEED))
        side = 1 if option == STRAFE_LEFT else -1
        move = self._safe_move(obs, radial * dx / distance - side * dy / distance,
                               radial * dy / distance + side * dx / distance)
        return self._aim_after_move(obs, *move)


def policy_class(profile):
    """Select an implementation by checkpoint schema, never by opponent identity."""
    if profile == "general":
        return Agent
    if profile in ("strafe", STRAFE_PROFILE):
        return StrafeAgent
    if profile in ("dijkstra", DIJKSTRA_PROFILE):
        return DijkstraAgent
    if profile in ("astar", ASTAR_PROFILE):
        return AStarAgent
    raise ValueError(f"unsupported policy profile: {profile}")


def make_agent(weights_path=None):
    """Load an original weights directory or an official specialist NPZ file."""
    if weights_path is None:
        return Agent()
    path = Path(weights_path)
    model = path / "qtable.npz" if path.is_dir() else path
    if not model.is_file():
        if path.suffix == ".npz":
            raise FileNotFoundError(path)
        return Agent(weights_path)
    with np.load(model, allow_pickle=False) as checkpoint:
        profile = str(checkpoint["profile"].item()) if "profile" in checkpoint.files else "general"
    return policy_class(profile)(model)
