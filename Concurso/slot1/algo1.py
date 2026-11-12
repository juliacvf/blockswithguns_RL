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

class C:
    BULLET_RADIUS = 0.15
    BULLET_RANGE = 20.0
    BULLET_SPEED = 14.0
    BULLET_TTL = 6.0
    FIXED_DT = 0.03333333333333333
    GRID_SIZE = 100
    HEAT_DAMAGE_PERIOD = 5.0
    HEAT_SHRINK_RATE = 0.25
    MAX_EPISODE_SECONDS = 120.0
    MUD_SLOW = 0.5
    PLAYER_LIVES = 8
    PLAYER_RADIUS = 0.35
    PLAYER_SPEED = 3.5
    SPEED_DURATION = 5.0
    SPEED_MULTIPLIER = 1.5
    WORLD = 100.0
    POWERUP_TTL = 25.0
    QUICKSHOT_COOLDOWN = 0.35
    SHOOT_COOLDOWN = 0.65


_INF = float("inf")

def _to_cell(v: float) -> int:
    return int(v)

def _free_cell(grid: np.ndarray, cx: int, cy: int) -> tuple[int, int]:
    """Clamp into bounds; if the cell is a wall, spiral out to a free one."""
    n = grid.shape[0]
    cx = min(max(cx, 0), n - 1)
    cy = min(max(cy, 0), n - 1)
    if grid[cx, cy] == 0:
        return cx, cy
    for r in range(1, 12):
        for dx in range(-r, r + 1):
            for dy in (-r, r):
                for x, y in ((cx + dx, cy + dy), (cx + dy, cy + dx)):
                    if 0 <= x < n and 0 <= y < n and grid[x, y] == 0:
                        return x, y
    return cx, cy

def _reconstruct(came: dict, cur: int, n: int) -> list[tuple[int, int]]:
    path = [(cur // n, cur % n)]
    while cur in came:
        cur = came[cur]
        path.append((cur // n, cur % n))
    path.reverse()
    return path

def astar(grid: np.ndarray, sx: float, sy: float, gx: float, gy: float,
          terrain: np.ndarray | None = None,
          terrain_penalty: float = 0.0) -> list[tuple[int, int]]:
    """Find a shortest path, optionally charging extra for terrain cells."""
    n = grid.shape[0]
    start = _free_cell(grid, _to_cell(sx), _to_cell(sy))
    goal = _free_cell(grid, _to_cell(gx), _to_cell(gy))
    if start == goal:
        return [start]
    s = start[0] * n + start[1]
    gcell = goal[0] * n + goal[1]
    gx2, gy2 = goal

    g_score = {s: 0.0}
    came: dict[int, int] = {}
    closed = bytearray(n * n)
    counter = 0
    open_heap = [(abs(start[0] - gx2) + abs(start[1] - gy2), 0, 0.0, s)]
    push = heapq.heappush
    pop = heapq.heappop

    while open_heap:
        _, _, g, cur = pop(open_heap)
        if closed[cur]:
            continue
        if cur == gcell:
            return _reconstruct(came, cur, n)
        closed[cur] = 1
        cx, cy = cur // n, cur % n
        for nb in (cur + n, cur - n, cur + 1, cur - 1):
            nx, ny = nb // n, nb % n
            # stay in bounds and skip walls / wrapped rows
            if nx < 0 or nx >= n or ny < 0 or ny >= n or abs(nx - cx) + abs(ny - cy) != 1:
                continue
            if grid[nx, ny] or closed[nb]:
                continue
            ng = g + 1.0
            if terrain is not None and terrain[nx, ny]:
                ng += terrain_penalty
            if ng < g_score.get(nb, _INF):
                g_score[nb] = ng
                came[nb] = cur
                counter += 1
                f = ng + abs(nx - gx2) + abs(ny - gy2)
                push(open_heap, (f, counter, ng, nb))
    return []

def has_line_of_sight(grid: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> bool:
    """True when the straight world-space line crosses no wall cell."""
    gx1, gy1 = int(x1), int(y1)
    gx2, gy2 = int(x2), int(y2)
    dx = gx2 - gx1
    dy = gy2 - gy1
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return True
    n = grid.shape[0]
    for i in range(steps + 1):
        x = gx1 + round(dx * i / steps)
        y = gy1 + round(dy * i / steps)
        if x < 0 or y < 0 or x >= n or y >= n or grid[x, y] == 1:
            return False
    return True


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


class _BaseAgent:
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


class StrafeAgent(_BaseAgent):
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
    return_safety_seconds = 8.
    return_release_margin = 8.
    return_lookahead_seconds = 20.
    return_min_margin = 8.

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
        lookahead = min(self.return_lookahead_seconds, max(0., C.MAX_EPISODE_SECONDS - now))
        target_margin = self.return_min_margin + C.HEAT_SHRINK_RATE * lookahead
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
        release_margin = min(self.return_release_margin, 8. + C.HEAT_SHRINK_RATE *
                             min(self.return_lookahead_seconds, max(0., C.MAX_EPISODE_SECONDS - now)))
        needed = (slack < self.return_safety_seconds or margin < 2.
                  or (self._returning and margin < release_margin))
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


MARKOV_PROFILE = "markov_v1"
BREAK_PATTERN = NUM_OPTIONS
MARKOV_NUM_OPTIONS = NUM_OPTIONS + 1
# distance, sight, threat, health, score, heat, enemy radial motion,
# enemy turn, our projectile pressure, predictable own motion, enemy heat, late.
MARKOV_STATE_SHAPE = (5, 2, 3, 2, 3, 3, 3, 2, 2, 2, 2, 2)
MARKOV_Q_SHAPE = MARKOV_STATE_SHAPE + (MARKOV_NUM_OPTIONS,)


def markov_default_q():
    q = np.zeros(MARKOV_Q_SHAPE, dtype=np.float32)
    for key in np.ndindex(MARKOV_STATE_SHAPE):
        distance, sight, threat, health, score, heat, radial, turn, pressure, predictable, enemy_heat, late = key
        engaging = sight and distance <= 3
        q[key + (CHASE,)] = .75 if not sight or distance >= 3 else .50 if distance == 2 else .15
        if pressure and turn and distance >= 2:
            q[key + (CHASE,)] += .20
        q[key + (STRAFE_LEFT,)] = .65 if engaging else .05
        q[key + (STRAFE_RIGHT,)] = .65 if engaging else .05
        q[key + (HOLD,)] = .72 if sight and distance in (1, 2) and not turn and not pressure else .05
        q[key + (RETREAT,)] = .95 if sight and distance == 0 else .25 if health == 0 else .02
        q[key + (EVADE,)] = (.05, .90, 1.50)[threat]
        q[key + (POWERUP,)] = .55 if health == 0 else .10
        q[key + (SAFE_CENTER,)] = (1.6, .95, .02)[heat]
        if enemy_heat and score >= 1:
            q[key + (SAFE_CENTER,)] += .95
        q[key + (BREAK_PATTERN,)] = .95 if predictable and sight and distance in (1, 2, 3) else .08
        if late and score == 0 and not engaging:
            q[key + (CHASE,)] += .15
    return q


class MarkovAgent(DijkstraAgent):
    """Vary established motion using public history while respecting danger."""
    profile = MARKOV_PROFILE
    q_shape = MARKOV_Q_SHAPE
    preferred_range = 11.0
    return_safety_seconds = 12.
    return_release_margin = 18.
    return_lookahead_seconds = 40.
    option_names = OPTION_NAMES + ("break_pattern",)

    def initial_q(self):
        return markov_default_q()

    def reset(self):
        super().reset()
        self._pattern_tick = None
        self._own_motion = deque()
        self._last_break = -1e9
        self._break_until = -1.
        self._break_vector = (0., 0.)
        self._pattern_breaks = 0

    def observe(self, obs):
        super().observe(obs)
        if self._pattern_tick == self._observed_tick:
            return
        self._pattern_tick = self._observed_tick
        now = self._previous_time
        position = _world_xy(obs["self"])
        if self._own_motion and math.dist(position, self._own_motion[-1][1]) > 1.:
            self._own_motion.clear()
            self._last_break = -1e9
            self._break_until = -1.
        self._own_motion.append((now, position))
        while len(self._own_motion) > 1 and self._own_motion[1][0] < now - .85:
            self._own_motion.popleft()

    def _predictable_motion(self):
        rows = self._own_motion
        if len(rows) < 2 or rows[-1][0] - rows[0][0] < .65:
            return False
        positions = [r[1] for r in rows]
        net = math.dist(positions[0], positions[-1])
        travelled = sum(math.dist(a, b) for a, b in zip(positions, positions[1:]))
        return travelled < .25 or (net > .6 and net >= .9 * travelled)

    @staticmethod
    def _target_under_pressure(obs):
        ex, ey = _world_xy(obs["opponent"])
        for row, valid in zip(obs["bullets"], obs["bullet_mask"]):
            if valid < .5 or row[4] <= 0:
                continue
            bx, by = _world_xy(row)
            vx, vy = float(row[2]) * C.BULLET_SPEED, float(row[3]) * C.BULLET_SPEED
            vv = vx * vx + vy * vy
            if vv < 1e-6:
                continue
            t = ((ex - bx) * vx + (ey - by) * vy) / vv
            remaining = C.BULLET_RANGE * (1. - (float(row[6]) + 1.) * .5)
            if 0 < t < .9 and t * C.BULLET_SPEED <= remaining:
                if math.hypot(ex - bx - vx * t, ey - by - vy * t) < 3.:
                    return True
        return False

    def state_key(self, obs):
        self.observe(obs)
        sx, sy = _world_xy(obs["self"])
        ex, ey = _world_xy(obs["opponent"])
        dx, dy = ex - sx, ey - sy
        distance = max(math.hypot(dx, dy), 1e-6)
        band = next((i for i, edge in enumerate((8., 12., 17., 22.)) if distance < edge), 4)
        radial_speed = (dx * self._motion_velocity[0] + dy * self._motion_velocity[1]) / distance
        radial = 0 if radial_speed < -.5 else 2 if radial_speed > .5 else 1
        mine, enemy = _lives(obs["self"]), _lives(obs["opponent"])
        return (band, int(obs["self"][14] > 0), _bullet_threat(obs), int(mine > 3),
                1 + int(mine > enemy) - int(mine < enemy), _heat_band(obs), radial,
                int(self._previous_time < self._turn_until), int(self._target_under_pressure(obs)),
                int(self._predictable_motion()), int(obs["opponent"][12] > 0),
                int(obs["game"][0] >= .8))

    def learning_context(self, key):
        return key[1], key[2], key[5], key[6], key[7], key[8], key[9], key[10]

    def available_options(self, obs):
        common = super().available_options(obs)
        if len(common) == 1:
            return common
        now = float(obs["game"][0]) * C.MAX_EPISODE_SECONDS
        if now < self._break_until or now - self._last_break >= 1.2:
            return np.arange(MARKOV_NUM_OPTIONS)
        return common

    def _choose_break_vector(self, obs):
        sx, sy = _world_xy(obs["self"])
        ex, ey = _world_xy(obs["opponent"])
        first, last = self._own_motion[0], self._own_motion[-1]
        vx, vy = last[1][0] - first[1][0], last[1][1] - first[1][1]
        if math.hypot(vx, vy) < .3:
            vx, vy = ex - sx, ey - sy
        norm = max(math.hypot(vx, vy), 1e-6)
        candidates = []
        aim_x, aim_y = float(obs["opponent"][3]), float(obs["opponent"][2])
        for side in (-1, 1):
            dx, dy = -side * vy / norm, side * vx / norm
            mx, my = self._safe_move(obs, dx, dy)
            x, y = self._step_position(obs, mx, my)
            if math.dist((sx, sy), (x, y)) < .03 or self._safe_margin(obs, x, y) < 2.5:
                continue
            # Prefer leaving the opponent's current firing lane without
            # entering a wall or accepting unnecessary range changes.
            px, py = sx + (x - sx) * 6, sy + (y - sy) * 6
            lane = abs((px - ex) * aim_y - (py - ey) * aim_x)
            score = min(lane, 3.) - .06 * abs(math.hypot(ex - px, ey - py) - self.preferred_range)
            score -= .1 * float(obs["map"][2, int(x), int(y)])
            candidates.append((score, side, dx, dy))
        if not candidates:
            return None
        _, _, dx, dy = max(candidates)
        return dx, dy

    def _defensive_move(self, obs, baseline):
        """Compare reachable short trajectories, including during safe return."""
        sx, sy = _world_xy(obs["self"])
        horizon = .6
        projectiles = []
        for row, valid in zip(obs["bullets"], obs["bullet_mask"]):
            if valid < .5 or row[4] > 0:
                continue
            bx, by = _world_xy(row)
            vx, vy = float(row[2]) * C.BULLET_SPEED, float(row[3]) * C.BULLET_SPEED
            remaining = C.BULLET_RANGE * (1. - (float(row[6]) + 1.) * .5)
            if math.hypot(sx - bx, sy - by) > min(remaining, C.BULLET_SPEED * horizon) + 3.:
                continue
            # Walls stop bullets; trees only constrain our body.
            flight = []
            for tick in range(1, round(horizon / C.FIXED_DT) + 1):
                time = tick * C.FIXED_DT
                if C.BULLET_SPEED * time >= remaining:
                    break
                x, y = bx + vx * time, by + vy * time
                px, py = bx + vx * (time - C.FIXED_DT), by + vy * (time - C.FIXED_DT)
                samples = max(1, math.ceil(math.hypot(x - px, y - py) / .2))
                if any(not (0 <= qx < C.WORLD and 0 <= qy < C.WORLD)
                       or obs["map"][0, int(qx), int(qy)]
                       for qx, qy in ((px + (x - px) * i / samples,
                                       py + (y - py) * i / samples)
                                      for i in range(samples + 1))):
                    break
                flight.append((px, py, x, y))
            if flight:
                projectiles.append(flight)
        if not projectiles:
            return baseline

        solid = np.logical_or(obs["map"][0], obs["map"][1])
        speed = (float(obs["self"][13]) + 1.) * .5 * C.PLAYER_SPEED * C.SPEED_MULTIPLIER
        radius = C.PLAYER_RADIUS

        def blocked(x, y):
            for cx in range(math.floor(x - radius), math.floor(x + radius) + 1):
                for cy in range(math.floor(y - radius), math.floor(y + radius) + 1):
                    if not (0 <= cx < C.GRID_SIZE and 0 <= cy < C.GRID_SIZE):
                        return True
                    if solid[cx, cy] and ((x - min(max(x, cx), cx + 1)) ** 2
                                         + (y - min(max(y, cy), cy + 1)) ** 2 < radius ** 2):
                        return True
            return False

        def trajectory(move):
            mx, my = move
            norm = max(math.hypot(mx, my), 1.)
            x, y = sx, sy
            positions = []
            for _ in range(round(horizon / C.FIXED_DT)):
                step = speed * C.FIXED_DT * (C.MUD_SLOW if obs["map"][2, int(x), int(y)] else 1.)
                nx = x + mx / norm * step
                if not blocked(nx, y):
                    x = nx
                ny = y + my / norm * step
                if not blocked(x, ny):
                    y = ny
                positions.append((x, y))
            risk = 0.
            for flight in projectiles:
                closest = float("inf")
                for (x, y), (bx, by, ex, ey) in zip(positions, flight):
                    dx, dy = ex - bx, ey - by
                    along = max(0., min(1., ((x - bx) * dx + (y - by) * dy) / max(dx * dx + dy * dy, 1e-9)))
                    closest = min(closest, math.hypot(x - bx - along * dx, y - by - along * dy))
                risk += max(0., .75 - closest) / .75
                if closest < C.PLAYER_RADIUS + C.BULLET_RADIUS + .03:
                    risk += 4.
            return positions[-1], risk

        base_end, base_risk = trajectory(baseline)
        if base_risk == 0:
            return baseline
        candidates = []
        for my in (-1, 0, 1):
            for mx in (-1, 0, 1):
                end, risk = trajectory((mx, my))
                # Do not trade a dodge for a prolonged retreat into heat.
                heat_loss = max(0., self._safe_margin(obs, *base_end) - self._safe_margin(obs, *end))
                deviation = math.dist(end, base_end)
                score = risk * 10. + heat_loss * (2. if self._returning else .3) + .2 * deviation
                candidates.append((score, (mx, my) != baseline, mx * mx + my * my, mx, my, risk))
        _, _, _, mx, my, risk = min(candidates)
        return (mx, my) if risk < base_risk - .05 else baseline

    def action_for_option(self, obs, option):
        action = self._pattern_action(obs, option)
        baseline = int(action[0]) % 3 - 1, int(action[0]) // 3 - 1
        move = self._defensive_move(obs, baseline)
        return action if move == baseline else self._aim_after_move(obs, *move)

    def _pattern_action(self, obs, option):
        if option != BREAK_PATTERN:
            return super().action_for_option(obs, option)
        self.observe(obs)
        options = self.available_options(obs)
        if BREAK_PATTERN not in options:
            fallback = SAFE_CENTER if len(options) == 1 else self.best_option(obs)
            return super().action_for_option(obs, fallback)
        if self._return_needed(obs):
            return super().action_for_option(obs, SAFE_CENTER)
        move = self._combat_move(obs)
        if move is not None:
            return self._aim_after_move(obs, *move)
        if _bullet_threat(obs) == 2:
            return super().action_for_option(obs, EVADE)
        now = self._previous_time
        if now >= self._break_until:
            vector = self._choose_break_vector(obs)
            if vector is None:
                return super().action_for_option(obs, EVADE)
            self._break_vector = vector
            self._last_break = now
            self._break_until = now + .4
            self._pattern_breaks += 1
        return self._aim_after_move(obs, *self._safe_move(obs, *self._break_vector))


GENERALIST_PROFILE = "generalista_v1"
CONTROL_RANGE = MARKOV_NUM_OPTIONS
GENERALIST_NUM_OPTIONS = MARKOV_NUM_OPTIONS + 1
GENERALIST_Q_SHAPE = MARKOV_STATE_SHAPE + (GENERALIST_NUM_OPTIONS,)


def generalist_q_from_markov(q):
    """Keep the nine learned Markov actions and seed one new range action."""
    if q.shape != MARKOV_Q_SHAPE or not np.isfinite(q).all():
        raise ValueError("generalist initialization requires a valid Markov Q-table")
    result = np.empty(GENERALIST_Q_SHAPE, dtype=np.float32)
    result[..., :MARKOV_NUM_OPTIONS] = q
    lateral = np.maximum(q[..., STRAFE_LEFT], q[..., STRAFE_RIGHT])
    result[..., CONTROL_RANGE] = lateral - .10
    # Only prefer range control against an observed approach at combat range.
    # Keep defensive, heat-return and pattern-breaking actions independent.
    for band in (1, 2, 3):
        for threat in (0, 1):
            region = (band, 1, threat, slice(None), slice(None), slice(None), 0)
            result[region + (Ellipsis, CONTROL_RANGE)] = np.maximum(
                lateral[region], q[region + (Ellipsis, HOLD)]) + .03
    return result


class GeneralistAgent(MarkovAgent):
    """Learn when to combine Markov tactics with A* approach-range control."""
    profile = GENERALIST_PROFILE
    q_shape = GENERALIST_Q_SHAPE
    option_names = MarkovAgent.option_names + ("control_range",)
    control_range = 14.0
    defense_horizon = .8

    def initial_q(self):
        return generalist_q_from_markov(markov_default_q())

    def reset(self):
        super().reset()
        self._range_control_ticks = 0
        self._defense_tick = None
        self._defense_flights = ()
        self._defense_paths = {}
        self._defense_risks = {}
        self._defense_last_move = (0, 0)
        self._combat_side = 1
        self._defense_overrides = 0
        self._lateral_combat_ticks = 0
        self._early_repositions = 0
        self._safe_route_fallbacks = 0
        self.return_release_margin = type(self).return_release_margin

    def _forecast_return(self, obs, allow_heat=False):
        # Comfort is secondary to avoiding an unnecessary heat traversal.
        path, slack = super()._forecast_return(obs, allow_heat=False)
        if path:
            self.return_release_margin = type(self).return_release_margin
            return path, slack
        horizon = self.return_lookahead_seconds
        minimum = self.return_min_margin
        now = float(obs["game"][0]) * C.MAX_EPISODE_SECONDS
        remaining = max(0., C.MAX_EPISODE_SECONDS - now)
        self.return_min_margin = min(8., .75 + C.HEAT_SHRINK_RATE * remaining)
        self.return_lookahead_seconds = 0.
        try:
            path, slack = super()._forecast_return(obs, allow_heat=False)
            if path:
                self.return_release_margin = self.return_min_margin
                return path, slack
            if allow_heat:
                self.return_release_margin = self.return_min_margin
                return super()._forecast_return(obs, allow_heat=True)
            return path, slack
        finally:
            self.return_lookahead_seconds = horizon
            self.return_min_margin = minimum

    def _plan_path(self, solid, obs, sx, sy, gx, gy):
        path = super()._plan_path(solid, obs, sx, sy, gx, gy)
        if self._returning or self._safe_margin(obs, sx, sy) < 0 or not path:
            return path
        speed = max(.1, min(C.PLAYER_SPEED, (float(obs["self"][13]) + 1.) *
                            .5 * C.PLAYER_SPEED * C.SPEED_MULTIPLIER))
        eta = 0.
        previous = (sx, sy)
        previous_mud = bool(obs["map"][2, int(sx), int(sy)])
        for cx, cy in path:
            point = cx + .5, cy + .5
            mud = bool(obs["map"][2, cx, cy])
            eta += math.dist(previous, point) * (.5 / (C.MUD_SLOW if previous_mud else 1.) +
                                                .5 / (C.MUD_SLOW if mud else 1.)) / speed
            if self._safe_margin(obs, *point) - C.HEAT_SHRINK_RATE * eta < C.PLAYER_RADIUS + .1:
                # A pursuit route must not override a still-reachable safe
                # position just because the target lies beyond a closing exit.
                safe_path, _ = self._forecast_return(obs)
                if safe_path:
                    self._safe_route_fallbacks += 1
                    return safe_path
                break
            previous, previous_mud = point, mud
        return path

    def _firing_position(self, obs):
        """Search reachable local cells instead of sparse sampled circles."""
        sx, sy = _world_xy(obs["self"])
        ex, ey = _world_xy(obs["opponent"])
        solid = np.logical_or(obs["map"][0], obs["map"][1])
        start = int(sx), int(sy)
        costs = {start: 0.}
        queue = [(0., start)]
        best = None
        while queue:
            cost, cell = heapq.heappop(queue)
            if cost != costs[cell] or (best is not None and cost > best[0]):
                continue
            cx, cy = cell
            x, y = cx + .5, cy + .5
            eta = cost / C.PLAYER_SPEED
            if self._safe_margin(obs, x, y) - C.HEAT_SHRINK_RATE * eta < 3.:
                continue
            distance = math.hypot(ex - x, ey - y)
            if cell != start and 5. <= distance <= 18.:
                _, shoot, miss = self._shot(obs, x, y)
                if shoot:
                    candidate = cost + .25 * abs(distance - self.preferred_range) + miss, x, y
                    if best is None or candidate < best:
                        best = candidate
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if not (0 <= nx < C.GRID_SIZE and 0 <= ny < C.GRID_SIZE) or solid[nx, ny]:
                    continue
                if math.hypot(nx + .5 - sx, ny + .5 - sy) > 9.:
                    continue
                step = .5 / (C.MUD_SLOW if obs["map"][2, cx, cy] else 1.)
                step += .5 / (C.MUD_SLOW if obs["map"][2, nx, ny] else 1.)
                next_cost = cost + step
                if next_cost <= 16. and next_cost < costs.get((nx, ny), float("inf")):
                    costs[nx, ny] = next_cost
                    heapq.heappush(queue, (next_cost, (nx, ny)))
        return None if best is None else (best[1], best[2])

    def _combat_move(self, obs):
        move = super()._combat_move(obs)
        if move is not None or self._returning or self._return_needed(obs):
            return move
        if obs["opponent"][12] > 0 and _lives(obs["self"]) >= _lives(obs["opponent"]):
            return None  # Preserve a useful safe position against an exposed enemy.
        now = self._previous_time
        rows = [r for r in self._combat_history if r[0] >= now - 3.1]
        if (len(rows) < 2 or now - rows[0][0] < 2.95
                or sum(r[3] for r in rows) < .7 * len(rows)
                or max(math.dist(r[1], rows[-1][1]) for r in rows) > 1.5):
            return None
        goal = self._firing_position(obs)
        self._combat_history.clear()
        if goal is None:
            return None
        self._reposition_goal = goal
        self._reposition_until = now + 5.
        self._early_repositions += 1
        self._combat_repositions += 1
        self._path = []
        return self._path_move(obs, goal, "firing_position")

    def _incoming_flights(self, obs):
        """Predict the same live bullet substeps as Engine._update_bullets."""
        tick = float(obs["game"][2])
        if self._defense_tick == tick:
            return self._defense_flights
        self._defense_tick = tick
        self._defense_paths = {}
        self._defense_risks = {}
        flights = []
        sx, sy = _world_xy(obs["self"])
        dt = C.FIXED_DT
        ticks = round(self.defense_horizon / dt)
        substeps = max(1, math.ceil(C.BULLET_SPEED * dt / .4))
        sub_dt = dt / substeps
        walls = obs["map"][0]
        self._defense_solid = np.logical_or(walls, obs["map"][1])
        for row, valid in zip(obs["bullets"], obs["bullet_mask"]):
            if valid < .5 or row[4] > 0:
                continue
            x, y = _world_xy(row)
            vx, vy = float(row[2]) * C.BULLET_SPEED, float(row[3]) * C.BULLET_SPEED
            distance = (float(row[6]) + 1.) * .5 * C.BULLET_RANGE
            ttl = (float(row[5]) + 1.) * .5 * C.BULLET_TTL
            reach = min(C.BULLET_RANGE - distance, C.BULLET_SPEED * self.defense_horizon)
            if math.hypot(sx - x, sy - y) > reach + C.PLAYER_SPEED * C.SPEED_MULTIPLIER * self.defense_horizon + 1.:
                continue
            flight = []
            alive = True
            for step in range(ticks):
                ttl -= dt
                if ttl <= 1e-7:
                    break
                points = []
                for _ in range(substeps):
                    x += vx * sub_dt
                    y += vy * sub_dt
                    distance += C.BULLET_SPEED * sub_dt
                    # Range/TTL and walls are checked BEFORE body collision.
                    # Keep a valid first substep even if the second expires.
                    if (distance >= C.BULLET_RANGE - 1e-6 or
                            not (0 <= x < C.WORLD and 0 <= y < C.WORLD) or
                            walls[int(x), int(y)]):
                        alive = False
                        break
                    points.append((x, y))
                if points:
                    flight.append((step, tuple(points)))
                if not alive:
                    break
            if flight:
                flights.append(tuple(flight))
        self._defense_flights = tuple(flights)
        return self._defense_flights

    def _future_positions(self, obs, move):
        if move in self._defense_paths:
            return self._defense_paths[move]
        x, y = _world_xy(obs["self"])
        mx, my = move
        norm = max(math.hypot(mx, my), 1.)
        radius, solid = C.PLAYER_RADIUS, self._defense_solid
        speed_timer = (float(obs["self"][7]) + 1.) * .5 * C.SPEED_DURATION
        heat_timer = (float(obs["self"][8]) + 1.) * .5 * C.HEAT_DAMAGE_PERIOD
        positions, heat_hits = [], 0

        def blocked(px, py):
            for cx in range(math.floor(px - radius), math.floor(px + radius) + 1):
                for cy in range(math.floor(py - radius), math.floor(py + radius) + 1):
                    if not (0 <= cx < C.GRID_SIZE and 0 <= cy < C.GRID_SIZE):
                        return True
                    if solid[cx, cy] and ((px - min(max(px, cx), cx + 1)) ** 2 +
                                         (py - min(max(py, cy), cy + 1)) ** 2 < radius ** 2):
                        return True
            return False

        for step in range(round(self.defense_horizon / C.FIXED_DT)):
            speed = C.PLAYER_SPEED * (C.SPEED_MULTIPLIER if speed_timer > 1e-7 else 1.)
            stride = speed * C.FIXED_DT * (C.MUD_SLOW if obs["map"][2, int(x), int(y)] else 1.)
            nx = x + mx / norm * stride
            if not blocked(nx, y):
                x = nx
            ny = y + my / norm * stride
            if not blocked(x, ny):
                y = ny
            speed_timer = max(0., speed_timer - C.FIXED_DT)
            positions.append((x, y))
            safe = float(obs["game"][3]) * C.WORLD * .5 - C.HEAT_SHRINK_RATE * (step + 1) * C.FIXED_DT
            if max(abs(x - 50.), abs(y - 50.)) > safe:
                heat_timer += C.FIXED_DT
                if heat_timer >= C.HEAT_DAMAGE_PERIOD:
                    heat_hits += 1
                    heat_timer -= C.HEAT_DAMAGE_PERIOD
            else:
                heat_timer = 0.
        result = positions, heat_hits
        self._defense_paths[move] = result
        return result

    def _movement_risk(self, obs, move):
        if move in self._defense_risks:
            return self._defense_risks[move]
        positions, heat = self._future_positions(obs, move)
        hits, near, first = 0, 0., float("inf")
        radius = C.PLAYER_RADIUS + C.BULLET_RADIUS
        for flight in self._defense_flights:
            closest, collision = float("inf"), None
            for step, points in flight:
                x, y = positions[step]
                gap = min(math.hypot(x - bx, y - by) for bx, by in points)
                closest = min(closest, gap)
                if gap < radius + .025 and collision is None:
                    collision = (step + 1) * C.FIXED_DT
            near += max(0., radius + .25 - closest)
            if collision is not None:
                hits += 1
                first = min(first, collision)
        result = hits, heat, near, first, positions[-1]
        self._defense_risks[move] = result
        return result

    def _bullet_threat(self, obs):
        # A bullet need not reach our CENTER to hit our body, and our own
        # next movement can take us into its last valid substep.
        if not self._incoming_flights(obs):
            return 0
        standing = self._movement_risk(obs, (0, 0))
        moving = self._movement_risk(obs, self._defense_last_move)
        first = min(standing[3], moving[3])
        return 2 if first <= .3 else 1 if math.isfinite(first) else 0

    def state_key(self, obs):
        key = list(super().state_key(obs))
        key[2] = self._bullet_threat(obs)
        return tuple(key)

    def _defensive_move(self, obs, baseline):
        flights = self._incoming_flights(obs)
        sx, sy = _world_xy(obs["self"])
        ex, ey = _world_xy(obs["opponent"])
        dx, dy = ex - sx, ey - sy
        distance = max(math.hypot(dx, dy), 1e-6)
        shot_age = (1. - float(obs["opponent"][11])) * 5.
        aimed = _aim_error(obs["opponent"][2], obs["opponent"][3], -dx, -dy) < .3
        lateral_combat = (not self._returning and obs["self"][14] > 0 and
                          6. <= distance <= 21.5 and aimed and (shot_age <= 1.2 or bool(flights)))
        start_margin = self._safe_margin(obs, sx, sy)
        speed = C.PLAYER_SPEED * C.SPEED_MULTIPLIER
        planned_return = self._returning and bool(self._return_path)
        near_heat = (not planned_return and
                     start_margin < (speed + C.HEAT_SHRINK_RATE) * self.defense_horizon + .35)
        if not flights and not lateral_combat and not near_heat:
            self._defense_last_move = baseline
            return baseline
        base = self._movement_risk(obs, baseline)
        future_shrink = C.HEAT_SHRINK_RATE * self.defense_horizon
        base_margin = self._safe_margin(obs, *base[4])
        heat_risk = near_heat and base_margin - future_shrink < .35
        if not lateral_combat and base[0] == 0 and base[2] == 0 and not heat_risk:
            self._defense_last_move = baseline
            return baseline
        return_target = None
        if self._returning and self._return_path:
            nearest = min(range(len(self._return_path)), key=lambda i:
                          math.hypot(self._return_path[i][0] + .5 - sx,
                                     self._return_path[i][1] + .5 - sy))
            cell = self._return_path[min(nearest + 3, len(self._return_path) - 1)]
            return_target = cell[0] + .5, cell[1] + .5
        candidates = []
        for my in (-1, 0, 1):
            for mx in (-1, 0, 1):
                move = mx, my
                risk = self._movement_risk(obs, move)
                x, y = risk[4]
                margin = self._safe_margin(obs, x, y)
                # Respect an urgent safe-return route; allow a brief dodge
                # sideways but do not reverse into heat for optional strafing.
                if self._returning and margin < min(start_margin, base_margin) - .25:
                    continue
                heat_loss = max(0., base_margin - margin)
                deviation = math.dist(risk[4], base[4])
                score = risk[2] * 3. + heat_loss * (2. if self._returning else .35) + .12 * deviation
                # Keep optional movement inside the future zone; planned
                # return routes retain their necessary obstacle detours.
                if near_heat:
                    score += 4. * max(0., .35 + future_shrink - margin)
                if return_target is not None:
                    score += .8 * max(0., math.dist((x, y), return_target) - math.dist(base[4], return_target))
                if lateral_combat:
                    side = ((x - sx) * -dy + (y - sy) * dx) / distance
                    # Favor a sustained side step while closing to a useful
                    # firing distance, rather than charging down the barrel.
                    score += .22 * abs(math.hypot(ex - x, ey - y) - self.preferred_range)
                    score -= .45 * abs(side) + .08 * self._combat_side * side
                    if move == (0, 0):
                        score += .3
                # A later collision leaves another decision before impact.
                # Preserve total-damage priority and existing tactical scores
                # whenever neither move is hit on the next engine step.
                immediate = int(risk[3] <= C.FIXED_DT + 1e-6)
                candidates.append((risk[0] + risk[1], immediate, score, move != baseline,
                                   mx * mx + my * my, mx, my))
        if not candidates:
            move = baseline
        else:
            _, _, _, _, _, mx, my = min(candidates)
            move = mx, my
        if move != baseline:
            self._defense_overrides += 1
        if lateral_combat:
            lateral = (move[0] * -dy + move[1] * dx) / distance
            if abs(lateral) > .1:
                self._combat_side = 1 if lateral > 0 else -1
                self._lateral_combat_ticks += 1
        self._defense_last_move = move
        return move


    def available_options(self, obs):
        common = super().available_options(obs)
        if len(common) == 1 or obs["self"][14] <= 0 or self._bullet_threat(obs) == 2:
            return common
        self.observe(obs)
        sx, sy = _world_xy(obs["self"])
        ex, ey = _world_xy(obs["opponent"])
        dx, dy = ex - sx, ey - sy
        distance = max(math.hypot(dx, dy), 1e-6)
        radial = (dx * self._motion_velocity[0] + dy * self._motion_velocity[1]) / distance
        if 8. <= distance < 22. and radial < -.5:
            return np.append(common, CONTROL_RANGE)
        return common

    def _pattern_action(self, obs, option):
        # Markov.action_for_option still applies its trajectory-based bullet
        # defense to the movement returned here, including the new action.
        if option != CONTROL_RANGE:
            return super()._pattern_action(obs, option)
        self.observe(obs)
        options = self.available_options(obs)
        if CONTROL_RANGE not in options:
            fallback = SAFE_CENTER if len(options) == 1 else self.best_option(obs)
            return super()._pattern_action(obs, fallback)
        move = self._combat_move(obs)
        if move is not None:
            return self._aim_after_move(obs, *move)
        sx, sy = _world_xy(obs["self"])
        ex, ey = _world_xy(obs["opponent"])
        dx, dy = ex - sx, ey - sy
        distance = max(math.hypot(dx, dy), 1e-6)
        incoming = (dx * self._motion_velocity[0] + dy * self._motion_velocity[1]) / distance
        # Same radial controller as AStarAgent, without changing Markov's
        # preferred range for its existing actions.
        radial = max(-2., min(1., (distance - self.control_range) / 4. + incoming / C.PLAYER_SPEED))
        key = self.state_key(obs)
        side = 1 if self.q[key + (STRAFE_LEFT,)] >= self.q[key + (STRAFE_RIGHT,)] else -1
        move = self._safe_move(obs, radial * dx / distance - side * dy / distance,
                               radial * dy / distance + side * dx / distance)
        self._range_control_ticks += 1
        return self._aim_after_move(obs, *move)



_PursuitBase = GeneralistAgent


class GeneralistAgent(_PursuitBase):
    """Shape a pursuit into a useful firing lane, rather than disengage."""

    def reset(self):
        super().reset()
        self._tactic_tick = None
        self._tactic_history = deque()
        self._tactic_phase = ''
        self._tactic_until = -1.
        self._tactic_last = -1e9
        self._tactic_goal = None
        self._tactic_plan_time = -1e9
        self._tactic_events = []
        self._tactic_lives = None
        self._tactic_enemy_previous = None
        self._tactic_stable = 0.
        self._tactic_following = 0.
        self._tactic_attack_since = -1e9
        self._tactic_side = 0

    def _tactic_event(self, phase, reason, obs):
        self._tactic_phase = phase
        sx, sy = _world_xy(obs['self'])
        ex, ey = _world_xy(obs['opponent'])
        self._tactic_events.append(dict(t=round(self._previous_time,3), phase=phase,
            reason=reason, own=(sx,sy), enemy=(ex,ey), goal=self._tactic_goal,
            stable=round(self._tactic_stable,3), following=round(self._tactic_following,3)))

    def observe(self, obs):
        super().observe(obs)
        if self._tactic_tick == self._observed_tick:
            return
        self._tactic_tick = self._observed_tick
        now = self._previous_time
        own, enemy = _world_xy(obs['self']), _world_xy(obs['opponent'])
        lives = _lives(obs['self']), _lives(obs['opponent'])
        if self._tactic_enemy_previous is not None:
            before, when = self._tactic_enemy_previous
            dt = max(now-when, 1e-6)
            velocity = tuple((v-b)/dt for v,b in zip(enemy,before))
            delta = math.dist(velocity, self._motion_velocity)
            self._tactic_stable = self._tactic_stable+dt if delta<.65 else 0.
            dx, dy = enemy[0]-own[0], enemy[1]-own[1]
            closing = -(dx*velocity[0]+dy*velocity[1])/max(math.hypot(dx,dy),1e-6)
            self._tactic_following = min(1.,self._tactic_following+dt) if closing>.7 else max(0.,self._tactic_following-2*dt)
            if math.dist(enemy,before)>1. or (self._tactic_history and math.dist(own,self._tactic_history[-1][3])>1.):
                self._tactic_history.clear()
                self._tactic_goal = None
                if self._tactic_phase:self._tactic_event('', 'teleport', obs)
        if lives != self._tactic_lives:
            self._tactic_history.clear()
            if self._tactic_lives and lives[1]<self._tactic_lives[1] and self._tactic_phase:
                self._tactic_event('', 'attack_progress', obs)
                self._tactic_last=now
        self._tactic_lives=lives
        self._tactic_enemy_previous=enemy,now
        self._tactic_history.append((now,self._combat_shots,lives,own))
        while len(self._tactic_history)>1 and self._tactic_history[1][0]<now-9.:
            self._tactic_history.popleft()

    def _tactic_trigger(self, obs):
        now=self._previous_time
        sx,sy=_world_xy(obs['self']);ex,ey=_world_xy(obs['opponent'])
        distance=math.hypot(ex-sx,ey-sy)
        if now-self._tactic_last<12. or obs['self'][14]<=0 or not 6.<=distance<=19.:
            return None
        if obs['opponent'][12]>0 and _lives(obs['self'])>=_lives(obs['opponent']):
            return None
        recent_shot=(1.-float(obs['opponent'][11]))*5.<1.2
        aimed=_aim_error(obs['opponent'][2],obs['opponent'][3],sx-ex,sy-ey)<.3
        radial=((ex-sx)*self._motion_velocity[0]+(ey-sy)*self._motion_velocity[1])/max(distance,1e-6)
        pressure=recent_shot and aimed and distance<10. and (radial<-.7 or self._bullet_threat(obs)==2)
        h=self._tactic_history
        stalled=(len(h)>1 and now-h[0][0]>=8. and self._combat_shots-h[0][1]>=6 and recent_shot)
        return 'close_pressure' if pressure else 'ineffective_exchange' if stalled else None

    def _tactic_destination(self, obs):
        sx,sy=_world_xy(obs['self']);ex,ey=_world_xy(obs['opponent'])
        solid=np.logical_or(obs['map'][0],obs['map'][1])
        start=int(sx),int(sy);costs={start:0.};queue=[(0.,start)];best=None
        old_margin=self._safe_margin(obs,sx,sy)
        enemy_margin=self._safe_margin(obs,ex,ey)
        old_dx,old_dy=sx-ex,sy-ey
        old_distance=max(math.hypot(old_dx,old_dy),1e-6)
        while queue:
            cost,(cx,cy)=heapq.heappop(queue)
            if cost!=costs[cx,cy]:continue
            x,y=cx+.5,cy+.5
            eta=cost/C.PLAYER_SPEED
            margin=self._safe_margin(obs,x,y)-C.HEAT_SHRINK_RATE*(eta+3.)
            if margin<3.:continue
            distance=math.hypot(x-ex,y-ey)
            if 5.<=math.hypot(x-sx,y-sy)<=16. and 10.<=distance<=17.:
                # Lateral routes can swap who holds the inside of the zone;
                # raw separation is not a reward.
                bearing=(old_dx*(y-ey)-old_dy*(x-ex))/old_distance
                if self._tactic_side and bearing*self._tactic_side<0.:continue
                safety_gain=min(margin-old_margin,8.)
                inversion=max(0.,min(8.,margin-enemy_margin)) if old_margin<enemy_margin+2. else 0.
                # Estimate continued outward pursuit from observed velocity,
                # as an opportunity only; our own route must remain safe.
                toward=((x-ex)/distance,(y-ey)/distance)
                vx,vy=self._motion_velocity
                projected=(ex+vx*2.,ey+vy*2.)
                edge_pressure=max(0.,-self._safe_margin(obs,*projected))*max(0.,(toward[0]*vx+toward[1]*vy)/C.PLAYER_SPEED)
                visible=has_line_of_sight(obs['map'][0],x,y,ex,ey)
                _,_,miss=self._shot(obs,x,y,velocity=(toward[0]*C.PLAYER_SPEED,toward[1]*C.PLAYER_SPEED))
                score=.45*abs(bearing)+.4*safety_gain+.65*inversion+.4*edge_pressure+2.*visible
                score-=.22*cost+.3*abs(distance-13.)+min(miss,2.)
                candidate=score,-cost,x,y
                if best is None or candidate>best:best=candidate
            for nx,ny in ((cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)):
                if not (0<=nx<100 and 0<=ny<100) or solid[nx,ny]:continue
                if math.hypot(nx+.5-ex,ny+.5-ey)<min(7.,old_distance):continue
                edge=.5/(C.MUD_SLOW if obs['map'][2,cx,cy] else 1.)+.5/(C.MUD_SLOW if obs['map'][2,nx,ny] else 1.)
                new=cost+edge
                if new<=32. and new<costs.get((nx,ny),float('inf')):
                    costs[nx,ny]=new;heapq.heappush(queue,(new,(nx,ny)))
        return None if best is None else (best[2],best[3])

    def _tactic_move(self, obs):
        now=self._previous_time
        sx,sy=_world_xy(obs['self']);ex,ey=_world_xy(obs['opponent'])
        distance=max(math.hypot(ex-sx,ey-sy),1e-6)
        if self._return_needed(obs) or self._safe_margin(obs,sx,sy)<4.:
            if self._tactic_phase:self._tactic_event('', 'safe_return',obs);self._tactic_last=now
            return None
        if not self._tactic_phase:
            reason=self._tactic_trigger(obs)
            if reason is None:return None
            self._tactic_side=0
            self._tactic_goal=self._tactic_destination(obs)
            if self._tactic_goal is None:return None
            self._tactic_until=now+9.;self._tactic_last=now;self._tactic_plan_time=now
            cross=(sx-ex)*(self._tactic_goal[1]-ey)-(sy-ey)*(self._tactic_goal[0]-ex)
            self._tactic_side=1 if cross>=0 else -1
            self._tactic_following=0.
            self._tactic_event('shape',reason,obs)
        if now>=self._tactic_until or distance>22.:
            self._tactic_event('', 'expired_or_contact_lost',obs);self._tactic_last=now
            return None
        if self._safe_margin(obs,*self._tactic_goal)-C.HEAT_SHRINK_RATE*3.<3.:
            self._tactic_event('', 'unsafe_destination',obs);self._tactic_last=now
            return None
        # Fire as soon as pursuit becomes predictable. No mandatory flight
        # duration, minimum distance travelled or stationary waiting stage.
        attack=(self._tactic_following>=.3 and self._tactic_stable>=.3 and 9.<=distance<=17. and obs['self'][14]>0)
        if self._tactic_phase=='attack' and now-self._tactic_attack_since<.6 and 8.<=distance<=18.:attack=True
        phase='attack' if attack else 'shape'
        if phase=='attack' and self._tactic_phase!='attack':self._tactic_attack_since=now
        if phase!=self._tactic_phase:self._tactic_event(phase,'firing_window' if attack else 'recalculate',obs)
        if now-self._tactic_plan_time>4.:
            goal=self._tactic_destination(obs)
            if goal is not None:self._tactic_goal=goal
            self._tactic_plan_time=now
        desired=self._path_move(obs,self._tactic_goal,'pursuit_geometry')
        # Range follows the actual pursuer. Avoid outrunning him and avoid
        # blindly charging him at the counter-attack transition.
        vx,vy=self._motion_velocity
        radial_enemy=((ex-sx)*vx+(ey-sy)*vy)/distance
        target=12. if attack else 14.
        desired_range=max(-1.5,min(1.5,(distance-target)/3.+radial_enemy/C.PLAYER_SPEED))
        self._incoming_flights(obs)
        choices=[]
        for my in (-1,0,1):
            for mx in (-1,0,1):
                move=mx,my;risk=self._movement_risk(obs,move);x,y=risk[4]
                margin=self._safe_margin(obs,x,y)-C.HEAT_SHRINK_RATE*self.defense_horizon
                if margin<2.:continue
                dist=math.hypot(ex+vx*self.defense_horizon-x,ey+vy*self.defense_horizon-y)
                progress=(mx*desired[0]+my*desired[1])/max(math.hypot(mx,my),1.)
                radial=(mx*(ex-sx)+my*(ey-sy))/distance/max(math.hypot(mx,my),1.)
                _,shoot,miss=self._shot(obs,x,y)
                score=2.5*max(0.,10.-dist)+1.3*max(0.,dist-16.)
                score+=1.1*math.dist((x,y),self._tactic_goal)+.2*abs(radial-desired_range)+2.*risk[2]
                if attack:score+=1.5*min(miss,2.)-.6*shoot
                if move==(0,0):score+=.75
                choices.append((risk[0]+risk[1],score,mx,my))
        if not choices:return desired
        _,_,mx,my=min(choices)
        return mx,my

    def _pattern_action(self, obs, option):
        self.observe(obs)
        move=self._tactic_move(obs)
        return super()._pattern_action(obs,option) if move is None else self._aim_after_move(obs,*move)

    def _defensive_move(self, obs, baseline):
        # Tactical candidate scoring already includes physical bullet paths.
        return baseline if self._tactic_phase else super()._defensive_move(obs,baseline)

    def action_for_option(self, obs, option):
        action=super().action_for_option(obs,option)
        if self._tactic_phase:
            # In the shaping stage a genuinely good shot is still allowed;
            # uncertainty, not a timer, determines whether to hold fire.
            ex,ey=_world_xy(obs['opponent']);sx,sy=_world_xy(obs['self'])
            flight=math.dist((sx,sy),(ex,ey))/C.BULLET_SPEED
            if self._tactic_stable<min(.4,flight*.45):action[2]=0
        return action

_PressureBase = GeneralistAgent


class GeneralistAgent(_PressureBase):
    """Experimental forward pressure with lateral movement and safe exits."""

    def reset(self):
        super().reset()
        self._pressure_entry_lives = None

    def _tactic_destination(self, obs):
        sx,sy=_world_xy(obs['self']);ex,ey=_world_xy(obs['opponent'])
        solid=np.logical_or(obs['map'][0],obs['map'][1])
        start=int(sx),int(sy);costs={start:0.};queue=[(0.,start)];best=None
        distance=max(math.hypot(ex-sx,ey-sy),1e-6)
        # Hold the inside lane when feasible; do not chase an enemy into heat.
        center_length=max(math.hypot(50.-ex,50.-ey),1e-6)
        inside=((50.-ex)/center_length,(50.-ey)/center_length)
        while queue:
            cost,(cx,cy)=heapq.heappop(queue)
            if cost!=costs[cx,cy]:continue
            x,y=cx+.5,cy+.5
            margin=self._safe_margin(obs,x,y)-C.HEAT_SHRINK_RATE*(cost/C.PLAYER_SPEED+3.)
            if margin<3.:continue
            separation=math.hypot(x-ex,y-ey)
            if 5.5<=separation<=9.5 and math.hypot(x-sx,y-sy)<=16.:
                visible=has_line_of_sight(obs['map'][0],x,y,ex,ey)
                _,_,miss=self._shot(obs,x,y)
                lateral=((x-sx)*-(ey-sy)+(y-sy)*(ex-sx))/distance
                inside_lane=((x-ex)*inside[0]+(y-ey)*inside[1])/separation
                score=2.*visible-.9*min(miss,2.)-.4*abs(separation-7.5)
                score+=.65*inside_lane+.08*min(margin,15.)+.12*min(abs(lateral),6.)-.18*cost
                candidate=score,-cost,x,y
                if best is None or candidate>best:best=candidate
            for nx,ny in ((cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)):
                if not (0<=nx<100 and 0<=ny<100) or solid[nx,ny]:continue
                if math.hypot(nx+.5-ex,ny+.5-ey)<min(4.5,distance):continue
                edge=.5/(C.MUD_SLOW if obs['map'][2,cx,cy] else 1.)+.5/(C.MUD_SLOW if obs['map'][2,nx,ny] else 1.)
                new=cost+edge
                if new<=24. and new<costs.get((nx,ny),float('inf')):
                    costs[nx,ny]=new;heapq.heappush(queue,(new,(nx,ny)))
        return None if best is None else (best[2],best[3])

    def _tactic_move(self, obs):
        now=self._previous_time
        sx,sy=_world_xy(obs['self']);ex,ey=_world_xy(obs['opponent'])
        distance=max(math.hypot(ex-sx,ey-sy),1e-6)
        if self._return_needed(obs) or self._safe_margin(obs,sx,sy)<4.:
            if self._tactic_phase:self._tactic_event('', 'safe_return',obs);self._tactic_last=now
            return None
        if not self._tactic_phase:
            # Keep the pursuit prototype's trigger, duration and cooldown:
            # compare the posture rather than a different firing schedule.
            reason=self._tactic_trigger(obs)
            if reason is None:return None
            self._tactic_goal=self._tactic_destination(obs)
            if self._tactic_goal is None:return None
            self._pressure_entry_lives=_lives(obs['self'])
            self._tactic_until=now+9.;self._tactic_last=now;self._tactic_plan_time=now
            self._tactic_event('pressure',reason,obs)
        if (_lives(obs['self'])<=self._pressure_entry_lives-2 or now>=self._tactic_until or distance>22.):
            reason='damage_limit' if _lives(obs['self'])<=self._pressure_entry_lives-2 else 'expired_or_contact_lost'
            self._tactic_event('',reason,obs);self._tactic_last=now
            return None
        if now-self._tactic_plan_time>1.:
            goal=self._tactic_destination(obs)
            if goal is None:
                self._tactic_event('', 'no_safe_pressure_route',obs);self._tactic_last=now
                return None
            self._tactic_goal=goal;self._tactic_plan_time=now
        if self._safe_margin(obs,*self._tactic_goal)-C.HEAT_SHRINK_RATE*3.<3.:
            self._tactic_event('', 'unsafe_destination',obs);self._tactic_last=now
            return None
        desired=self._path_move(obs,self._tactic_goal,'forward_pressure')
        self._incoming_flights(obs)
        vx,vy=self._motion_velocity
        closing=((ex-sx)*vx+(ey-sy)*vy)/distance
        radial=max(-1.5,min(1.5,(distance-7.5)/3.+closing/C.PLAYER_SPEED))
        dx,dy=(ex-sx)/distance,(ey-sy)/distance
        choices=[]
        for my in (-1,0,1):
            for mx in (-1,0,1):
                move=mx,my;risk=self._movement_risk(obs,move);x,y=risk[4]
                margin=self._safe_margin(obs,x,y)-C.HEAT_SHRINK_RATE*self.defense_horizon
                if margin<2.:continue
                future=math.hypot(ex+vx*self.defense_horizon-x,ey+vy*self.defense_horizon-y)
                length=max(math.hypot(mx,my),1.)
                forward=(mx*dx+my*dy)/length
                sideways=(-mx*dy+my*dx)/length
                _,shoot,miss=self._shot(obs,x,y)
                score=1.2*math.dist((x,y),self._tactic_goal)+2.*risk[2]
                score+=3.*max(0.,5.5-future)+.8*max(0.,future-9.)
                score+=.45*abs(forward-radial)-.4*abs(sideways)+.7*min(miss,2.)-.3*shoot
                if move==(0,0):score+=.75
                choices.append((risk[0]+risk[1],score,mx,my))
        if not choices:
            self._tactic_event('', 'no_safe_pressure_move',obs);self._tactic_last=now
            return None
        _,_,mx,my=min(choices)
        return mx,my

    def action_for_option(self, obs, option):
        # Keep physical interception and line checks, but do not hold fire
        # waiting for pursuit: this experiment pressures while approaching.
        return _PursuitBase.action_for_option(self,obs,option)

_ResponsePressure = GeneralistAgent
_ResponseRetreat = _PressureBase


class GeneralistAgent(_ResponsePressure):
    """Select a posture from its observed effect within this match."""

    def reset(self):
        super().reset()
        self._response_tick = None
        self._response_previous = None
        self._response_rows = deque()
        self._response_forecasts = deque()
        self._response_errors = deque()
        self._response_next_forecast = -1e9
        self._response_mode = ''
        self._response_since = -1e9
        self._response_next_review = -1e9
        self._response_baseline = None
        self._response_tried = set()
        self._response_events = []
        self._response_best = None
        self._response_credit_mode = ''
        self._response_credit_until = -1e9
        self._response_last_move = (0,0)

    def observe(self, obs):
        super().observe(obs)
        if self._response_tick == self._observed_tick:return
        self._response_tick = self._observed_tick
        now=self._previous_time
        own,enemy=_world_xy(obs['self']),_world_xy(obs['opponent'])
        lives=(_lives(obs['self']),_lives(obs['opponent']))
        previous=self._response_previous
        if previous is not None:
            dt=now-previous['t']
            teleported=(dt>0 and any(math.dist(a,b)>6.*dt+.1 for a,b in zip((own,enemy),previous['positions'])))
            if teleported:
                self._response_rows.clear();self._response_errors.clear();self._response_forecasts.clear()
                self._response_mode='';self._response_best=None;self._response_credit_mode=''
                self._response_tried.clear()
            elif dt>0:
                dx,dy=enemy[0]-own[0],enemy[1]-own[1];distance=max(math.hypot(dx,dy),1e-6)
                ov=tuple((a-b)/dt for a,b in zip(own,previous['positions'][0]))
                ev=tuple((a-b)/dt for a,b in zip(enemy,previous['positions'][1]))
                forward=(ov[0]*dx+ov[1]*dy)/distance
                closing=-(ev[0]*dx+ev[1]*dy)/distance
                # Cures are not hits; heat and teleport damage are not
                # evidence that this combat posture improved our shooting.
                heat=[obs[k][12]>0 or previous['heat'][i] for i,k in enumerate(('self','opponent'))]
                lost=[max(0,previous['lives'][i]-lives[i]) if not heat[i] else 0 for i in (0,1)]
                mode=self._response_mode or (self._response_credit_mode if now<=self._response_credit_until else '')
                while self._response_forecasts and self._response_forecasts[0][0]<=now:
                    when,target,forecast_mode=self._response_forecasts.popleft()
                    self._response_errors.append((now,math.dist(target,enemy),forecast_mode))
                mx,my=self._response_last_move
                px,py=self._step_position(obs,mx,my)
                _,valid,miss=self._shot(obs,px,py)
                self._response_rows.append(dict(t=now,dt=min(dt,.1),mode=mode,forward=forward,closing=closing,
                    visible=obs['self'][14]>0,opportunity=bool(valid),miss=miss,distance=distance,
                    received=lost[0],landed=lost[1],own_margin=self._safe_margin(obs,*own),
                    advantage=self._safe_margin(obs,*own)-self._safe_margin(obs,*enemy)))
        while self._response_rows and self._response_rows[0]['t']<now-12.:self._response_rows.popleft()
        while self._response_errors and self._response_errors[0][0]<now-12.:self._response_errors.popleft()
        if now>=self._response_next_forecast:
            flight=max(.2,min(1.4,math.dist(own,enemy)/C.BULLET_SPEED))
            vx,vy=self._motion_velocity
            self._response_forecasts.append((now+flight,(enemy[0]+vx*flight,enemy[1]+vy*flight),self._response_mode))
            self._response_next_forecast=now+.15
        self._response_previous=dict(t=now,positions=(own,enemy),lives=lives,heat=(obs['self'][12]>0,obs['opponent'][12]>0))

    def _response_window(self, mode=None, since=None):
        now=self._previous_time
        since=now-2.5 if since is None else since
        rows=[r for r in self._response_rows if r['t']>=since and (mode is None or r['mode']==mode)]
        duration=sum(r['dt'] for r in rows)
        visible=sum(r['dt'] for r in rows if r['visible'])
        away=sum(r['dt'] for r in rows if r['forward']<-.5)
        followed=sum(r['dt'] for r in rows if r['forward']<-.5 and r['closing']>.7)
        errors=[r[1] for r in self._response_errors if r[0]>=since and (mode is None or r[2]==mode)]
        return dict(duration=duration,opportunity=sum(r['dt'] for r in rows if r['opportunity'])/max(duration,.001),
            visible=visible/max(duration,.001),away=away,pursued=followed/max(away,.001),
            error=float(np.median(errors)) if len(errors)>=3 else None,
            landed=sum(r['landed'] for r in rows),received=sum(r['received'] for r in rows),
            distance=sum(r['distance']*r['dt'] for r in rows)/max(duration,.001),
            closing=sum(r['closing']*r['dt'] for r in rows)/max(duration,.001))

    def _response_assess(self, mode, current, baseline):
        if current['received']>current['landed']:return 'unfavorable_trade'
        if current['landed']>0:return 'productive'
        if current['duration']<1.5:return 'unknown'
        radius=C.PLAYER_RADIUS+C.BULLET_RADIUS
        predictable=current['error'] is not None and current['error']<=radius
        better_prediction=(baseline['error'] is not None and current['error'] is not None and current['error']<baseline['error']-.15)
        better_shot=current['opportunity']>=baseline['opportunity']+.10
        if mode=='retreat':
            if current['away']<.3 or current['pursued']<.5:return 'not_followed'
            prediction_not_worse=(baseline['error'] is not None and current['error'] is not None and current['error']<=baseline['error'])
            if current['opportunity']>=.15 and (predictable or better_prediction or (better_shot and prediction_not_worse)):return 'productive'
            return 'no_firing_gain'
        useful=current['opportunity']>=.20 and current['visible']>=.5
        if useful and (better_shot or (predictable and better_prediction)):return 'productive'
        return 'no_firing_gain'

    def _response_start(self, obs, mode, reason):
        now=self._previous_time
        self._response_mode=mode
        goal=self._tactic_destination(obs)
        if goal is None:
            self._response_mode='';return False
        self._response_tried.add(mode)
        self._tactic_goal=goal;self._tactic_plan_time=now;self._tactic_until=now+9.
        self._pressure_entry_lives=_lives(obs['self']);self._tactic_following=0.
        sx,sy=_world_xy(obs['self']);ex,ey=_world_xy(obs['opponent'])
        cross=(sx-ex)*(goal[1]-ey)-(sy-ey)*(goal[0]-ex)
        self._tactic_side=1 if cross>=0 else -1
        self._response_since=now;self._response_next_review=now+2.5
        self._path=[];self._reposition_goal=None
        self._tactic_event('pressure' if mode=='pressure' else 'shape',reason,obs)
        self._response_events.append(dict(t=round(now,3),mode=mode,reason=reason))
        return True

    def _response_finish(self, obs, reason):
        now=self._previous_time
        recent=self._response_window(self._response_mode,since=max(self._response_since,now-2.5))
        if reason=='controller_completed' and recent['landed']>recent['received']:
            self._response_best=self._response_mode,now
        self._response_events.append(dict(t=round(now,3),mode='normal',reason=reason))
        self._response_credit_mode=self._response_mode;self._response_credit_until=now+2.
        self._response_mode='';self._tactic_last=now
        if self._tactic_phase:self._tactic_event('',reason,obs)

    def _tactic_destination(self, obs):
        implementation=_ResponseRetreat if self._response_mode=='retreat' else _ResponsePressure
        return implementation._tactic_destination(self,obs)

    def _tactic_move(self, obs):
        now=self._previous_time
        if self._response_mode and not self._tactic_phase:
            # Includes a successful hit, a teleport, or a controller exit.
            self._response_finish(obs,'controller_completed')
        if self._return_needed(obs) or self._safe_margin(obs,*_world_xy(obs['self']))<4.:
            if self._response_mode:self._response_finish(obs,'safe_return')
            return None
        if not self._response_mode:
            # Continue monitoring the position after the last maneuver.
            recent=self._response_window(since=now-2.)
            if recent['received']>recent['landed'] and self._response_best is not None:self._response_best=None
            reason=self._tactic_trigger(obs)
            if reason is None:return None
            self._response_baseline=self._response_window()
            self._response_tried=set()
            preferred='pressure'
            if self._response_best is not None and now-self._response_best[1]<25.:
                preferred=self._response_best[0]
            if not self._response_start(obs,preferred,'probe_'+reason):return None
        if now>=self._response_next_review:
            current=self._response_window(self._response_mode,since=max(self._response_since,now-2.5))
            assessment=self._response_assess(self._response_mode,current,self._response_baseline)
            self._response_events.append(dict(t=round(now,3),mode=self._response_mode,reason=assessment,metrics=current))
            if assessment=='productive':
                self._response_best=self._response_mode,now
                self._response_next_review=now+.75
            elif assessment=='unknown':self._response_next_review=now+.5
            else:
                alternative='retreat' if self._response_mode=='pressure' else 'pressure'
                # Judge the response to the new posture against the posture
                # it replaces, rather than a stale pre-combat reference.
                self._response_baseline=current
                if alternative in self._response_tried or not self._response_start(obs,alternative,'response_'+assessment):
                    self._response_finish(obs,assessment);return None
        implementation=_ResponseRetreat if self._response_mode=='retreat' else _ResponsePressure
        move=implementation._tactic_move(self,obs)
        if not self._tactic_phase:self._response_finish(obs,'controller_completed')
        return move

    def action_for_option(self, obs, option):
        action=_PursuitBase.action_for_option(self,obs,option)
        if self._response_mode=='retreat':
            sx,sy=_world_xy(obs['self']);ex,ey=_world_xy(obs['opponent'])
            if self._tactic_stable<min(.4,math.hypot(ex-sx,ey-sy)/C.BULLET_SPEED*.45):action[2]=0
        self._response_last_move=int(action[0])%3-1,int(action[0])//3-1
        return action

_FirstSelector = GeneralistAgent


class GeneralistAgent(_FirstSelector):
    """Keep the pressure controller's tempo; change only on adverse response."""

    def reset(self):
        super().reset()
        self._pressure_bad_since = None
        self._pressure_last_reason = None

    def _response_window(self, mode=None, since=None):
        current=super()._response_window(mode,since)
        cutoff=self._previous_time-2.5 if since is None else since
        rows=[r for r in self._response_rows if r['t']>=cutoff and (mode is None or r['mode']==mode)]
        current['exposure']=sum(r['dt'] for r in rows if r.get('danger',False))/max(current['duration'],.001)
        return current

    def _pressure_reaction(self, current):
        # Missing a few shots is not evidence that forward pressure is bad.
        # A real loss in the exchange, or sustained predicted exposure to a
        # closing opponent, is evidence worth reacting to.
        if current['received']>current['landed']:return 'unfavorable_trade'
        if (current['duration']>=1.5 and current['exposure']>=.35
                and current['closing']>.7):return 'sustained_exposure'
        return None

    def _response_finish(self, obs, reason):
        previous_mode=self._response_mode
        pressure_clock=self._tactic_last
        super()._response_finish(obs,reason)
        if previous_mode=='pressure':
            # The fixed policy does not restart its cooldown after a hit.
            # Observing that hit must not slow down otherwise identical play.
            self._tactic_last=pressure_clock
            self._response_credit_until=self._previous_time+4.
        self._pressure_bad_since=None;self._pressure_last_reason=None
        self._response_next_review=self._previous_time+.75

    def _try_retreat_response(self, obs, current, reason):
        now=self._previous_time
        old_mode=self._response_mode
        self._response_baseline=current
        if self._response_start(obs,'retreat','response_'+reason):
            self._tactic_last=now
            self._pressure_bad_since=None;self._pressure_last_reason=None
            return True
        self._response_mode=old_mode
        self._pressure_bad_since=None;self._pressure_last_reason=None
        self._response_next_review=now+2.
        self._response_events.append(dict(t=round(now,3),mode=old_mode or 'normal',reason='no_safe_retreat'))
        return False

    def _review_pressure(self, obs):
        now=self._previous_time
        if now<self._response_next_review:return False
        self._response_next_review=now+.75
        current=self._response_window(since=now-2.5)
        reason=self._pressure_reaction(current)
        if reason is None:
            self._pressure_bad_since=None;self._pressure_last_reason=None
            return False
        if self._pressure_bad_since is None or self._pressure_last_reason!=reason:
            self._pressure_bad_since=now;self._pressure_last_reason=reason
        flight=max(.6,min(1.4,current['distance']/C.BULLET_SPEED))
        # Give our already-fired bullet time to answer one received hit.
        confirmed=(current['received']>=2 or now-self._pressure_bad_since>=flight+.25)
        self._response_events.append(dict(t=round(now,3),mode=self._response_mode or 'normal',
            reason=reason if confirmed else 'await_exchange',metrics=current))
        return confirmed and self._try_retreat_response(obs,current,reason)

    def _retreat_response_move(self, obs):
        now=self._previous_time
        if now>=self._response_next_review:
            current=self._response_window('retreat',since=max(self._response_since,now-2.5))
            assessment=self._response_assess('retreat',current,self._response_baseline)
            self._response_events.append(dict(t=round(now,3),mode='retreat',reason=assessment,metrics=current))
            if assessment=='productive':
                self._response_next_review=now+.75
            elif assessment=='unknown':self._response_next_review=now+.5
            else:
                self._response_finish(obs,assessment)
                return None
        move=_ResponseRetreat._tactic_move(self,obs)
        if not self._tactic_phase:self._response_finish(obs,'controller_completed')
        return move

    def _tactic_move(self, obs):
        now=self._previous_time
        if self._response_mode and not self._tactic_phase:
            self._response_finish(obs,'controller_completed')
        if self._return_needed(obs) or self._safe_margin(obs,*_world_xy(obs['self']))<4.:
            if self._response_mode:self._response_finish(obs,'safe_return')
            return None
        if self._response_mode=='retreat':return self._retreat_response_move(obs)
        aftermath=(self._response_credit_mode=='pressure' and now<=self._response_credit_until)
        if (self._response_mode=='pressure' or aftermath) and self._review_pressure(obs):
            return self._retreat_response_move(obs)
        # Delegate starts, navigation, firing and cooldown to the successful
        # fixed policy. No extra probe, path reset or early no-gain timeout.
        move=_ResponsePressure._tactic_move(self,obs)
        if self._tactic_phase=='pressure' and self._response_mode!='pressure':
            self._response_mode='pressure';self._response_since=now
            self._response_next_review=now+.75
            self._pressure_bad_since=None;self._pressure_last_reason=None
            self._response_events.append(dict(t=round(now,3),mode='pressure',reason='pressure_default'))
        elif self._response_mode=='pressure' and not self._tactic_phase:
            self._response_finish(obs,'controller_completed')
        return move

    def action_for_option(self, obs, option):
        action=super().action_for_option(obs,option)
        if self._response_rows and abs(self._response_rows[-1]['t']-self._previous_time)<1e-5:
            self._incoming_flights(obs)
            move=int(action[0])%3-1,int(action[0])//3-1
            risk=self._movement_risk(obs,move)
            self._response_rows[-1]['danger']=risk[0]>0
        return action


_PositionBase = GeneralistAgent


class GeneralistAgent(_PositionBase):
    """Compare reachable maneuver endpoints, without changing posture triggers."""

    def reset(self):
        super().reset()
        self._exit_events=[]
        self._position_previous=None

    def _position_routes(self, obs):
        own=_world_xy(obs['self']);enemy=_world_xy(obs['opponent'])
        solid=np.logical_or(obs['map'][0],obs['map'][1])
        start=int(own[0]),int(own[1]);costs={start:0.};came={};queue=[(0.,start)]
        minimum=min(4.5,math.dist(own,enemy))
        while queue:
            cost,cell=heapq.heappop(queue)
            if cost!=costs[cell]:continue
            for nx,ny in ((cell[0]+1,cell[1]),(cell[0]-1,cell[1]),(cell[0],cell[1]+1),(cell[0],cell[1]-1)):
                if not (0<=nx<100 and 0<=ny<100) or solid[nx,ny]:continue
                if math.dist((nx+.5,ny+.5),enemy)<minimum:continue
                edge=(.5/(C.MUD_SLOW if obs['map'][2,cell[0],cell[1]] else 1.)+
                      .5/(C.MUD_SLOW if obs['map'][2,nx,ny] else 1.))/C.PLAYER_SPEED
                eta=cost+edge
                if eta>24./C.PLAYER_SPEED:continue
                if self._safe_margin(obs,nx+.5,ny+.5)-C.HEAT_SHRINK_RATE*(eta+3.)<3.:continue
                if eta<costs.get((nx,ny),float('inf')):
                    costs[nx,ny]=eta;came[nx,ny]=cell;heapq.heappush(queue,(eta,(nx,ny)))
        return costs,came

    def _position_return(self, obs, point, delay):
        """Travel seconds to a future interior refuge, plus enemy-crossing risk."""
        solid=np.logical_or(obs['map'][0],obs['map'][1]);enemy=_world_xy(obs['opponent'])
        start=int(point[0]),int(point[1]);queue=[(0.,0.,start)];best={start:(0.,0.)}
        while queue:
            seconds,crossing,cell=heapq.heappop(queue)
            if (seconds,crossing)!=best[cell]:continue
            margin=self._safe_margin(obs,cell[0]+.5,cell[1]+.5)-C.HEAT_SHRINK_RATE*(delay+seconds+3.)
            if margin>=6.:return seconds,crossing
            for nx,ny in ((cell[0]+1,cell[1]),(cell[0]-1,cell[1]),(cell[0],cell[1]+1),(cell[0],cell[1]-1)):
                if not (0<=nx<100 and 0<=ny<100) or solid[nx,ny]:continue
                edge=(.5/(C.MUD_SLOW if obs['map'][2,cell[0],cell[1]] else 1.)+
                      .5/(C.MUD_SLOW if obs['map'][2,nx,ny] else 1.))/C.PLAYER_SPEED
                eta=seconds+edge
                if eta>6.:continue
                if self._safe_margin(obs,nx+.5,ny+.5)-C.HEAT_SHRINK_RATE*(delay+eta)<.75:continue
                danger=max(crossing,max(0.,5.-math.dist((nx+.5,ny+.5),enemy)))
                value=eta,danger
                if value<best.get((nx,ny),(float('inf'),float('inf'))):
                    best[nx,ny]=value;heapq.heappush(queue,(eta,danger,(nx,ny)))
        return 7.,3.

    def _position_attack(self, obs, point):
        # Short projection only: do not assume several seconds of linear motion.
        ex,ey=_world_xy(obs['opponent']);vx,vy=self._motion_velocity
        quality=[]
        for delay in (0.,.35,.7):
            x=min(99.5,max(.5,ex+vx*delay));y=min(99.5,max(.5,ey+vy*delay))
            if obs['map'][0,int(x),int(y)] or obs['map'][1,int(x),int(y)]:x,y=ex,ey
            future=dict(obs);future['opponent']=obs['opponent'].copy()
            future['opponent'][0]=x/50.-1.;future['opponent'][1]=y/50.-1.
            _,shoot,miss=self._shot(future,*point)
            visible=has_line_of_sight(obs['map'][0],*point,x,y)
            quality.append(1.2*shoot+.6*visible-.6*min(miss,2.))
        return sum(quality)/len(quality)

    def _position_exposure(self, obs, path, costs):
        """Current bullets along the initial route; later bullets remain unknown."""
        flights=self._incoming_flights(obs)
        if not flights:return 0.
        own=_world_xy(obs['self']);positions=[];segment=1
        for step in range(round(self.defense_horizon/C.FIXED_DT)):
            when=(step+1)*C.FIXED_DT
            while segment<len(path) and costs[path[segment]]<when:segment+=1
            if segment>=len(path):positions.append((path[-1][0]+.5,path[-1][1]+.5));continue
            a=own if segment==1 else (path[segment-1][0]+.5,path[segment-1][1]+.5)
            b=path[segment][0]+.5,path[segment][1]+.5
            before=0. if segment==1 else costs[path[segment-1]]
            fraction=min(1.,max(0.,(when-before)/max(costs[path[segment]]-before,1e-6)))
            positions.append((a[0]+(b[0]-a[0])*fraction,a[1]+(b[1]-a[1])*fraction))
        risk=0.
        for flight in flights:
            gap=min(math.dist(positions[step],bullet) for step,points in flight for bullet in points)
            risk+=max(0.,C.PLAYER_RADIUS+C.BULLET_RADIUS+.25-gap)
        return risk

    def _position_score(self, obs, point, costs, came):
        cell=int(point[0]),int(point[1]);eta=costs[cell]
        path=[cell]
        while path[-1] in came:path.append(came[path[-1]])
        path.reverse()
        own=_world_xy(obs['self']);enemy=_world_xy(obs['opponent'])
        separation=math.dist(point,enemy);retreat=self._response_mode=='retreat'
        target=13. if retreat else 7.5
        margin=self._safe_margin(obs,*point)-C.HEAT_SHRINK_RATE*(eta+3.)
        attack=self._position_attack(obs,point)
        back,crossing=self._position_return(obs,point,eta)
        exposure=self._position_exposure(obs,path,costs)
        inside=((point[0]-enemy[0])*(50.-enemy[0])+(point[1]-enemy[1])*(50.-enemy[1]))/max(separation*math.dist(enemy,(50.,50.)),1e-6)
        score=attack-.4*abs(separation-target)-.18*eta*C.PLAYER_SPEED+.65*inside+.08*min(margin,15.)
        score-=.45*back+.55*crossing+2.*exposure
        return dict(score=score,attack=attack,return_seconds=back,crossing=crossing,exposure=exposure)

    def _tactic_destination(self, obs):
        original=super()._tactic_destination(obs)
        if original is None:return None
        costs,came=self._position_routes(obs)
        original_cell=int(original[0]),int(original[1])
        if original_cell not in costs:return original
        own=_world_xy(obs['self']);enemy=_world_xy(obs['opponent'])
        low,high,target=(10.,17.,13.) if self._response_mode=='retreat' else (5.5,9.5,7.5)
        ranked=[]
        for cell,eta in costs.items():
            point=cell[0]+.5,cell[1]+.5;distance=math.dist(point,enemy)
            if not (low<=distance<=high) or math.dist(point,original)>4.:continue
            _,shoot,miss=self._shot(obs,*point)
            quick=1.2*shoot-.6*min(miss,2.)-.4*abs(distance-target)-.18*eta*C.PLAYER_SPEED
            ranked.append((quick,point))
        choices=[original]
        previous=self._position_previous
        if (previous is not None and (int(previous[0]),int(previous[1])) in costs
                and math.dist(previous,original)<=4. and low<=math.dist(previous,enemy)<=high):choices.append(previous)
        for _,point in sorted(ranked,reverse=True):
            if all(math.dist(point,p)>=1.5 for p in choices):choices.append(point)
            if len(choices)>=7:break
        assessed=[(self._position_score(obs,p,costs,came),p) for p in choices]
        baseline=assessed[0][0];best,point=max(assessed,key=lambda row:row[0]['score'])
        if best['score']<baseline['score']+.65 or math.dist(point,original)<1.:return original
        if previous is not None:
            kept=next((q for q,p in assessed if p==previous),None)
            if kept is not None and kept['score']>=best['score']-.3 and kept['score']>=baseline['score']+.65:best,point=kept,previous
        self._position_previous=point
        self._exit_events.append(dict(t=round(self._previous_time,3),kind='position_changed',original=original,target=point,before=baseline,after=best))
        return point


_ProductiveBase = GeneralistAgent


class GeneralistAgent(_ProductiveBase):
    """Require stronger evidence before leaving a productive combat position."""

    def _position_productive(self, obs):
        current=self._response_window(since=self._previous_time-2.5)
        if current['duration']<1.25 or current['received'] or current['exposure']>.15:return False
        if current['visible']<.65 or current['opportunity']<.30:return False
        accurate=current['error'] is not None and current['error']<=C.PLAYER_RADIUS+C.BULLET_RADIUS+.1
        return current['landed']>0 or accurate

    def _tactic_destination(self, obs):
        original=_PositionBase._tactic_destination(self,obs)
        if original is None:return None
        before=len(self._exit_events)
        proposed=super()._tactic_destination(obs)
        if proposed==original or not self._position_productive(obs):return proposed
        own=_world_xy(obs['self'])
        # Never keep a successful firing lane at the expense of urgent return.
        if self._returning or self._safe_margin(obs,*own)-3.*C.HEAT_SHRINK_RATE<6.:return proposed
        if len(self._exit_events)==before:return proposed
        event=self._exit_events[-1]
        old,new=event['before'],event['after']
        gain=new['score']-old['score']
        if gain>=1.4 and new['exposure']<=old['exposure']+.05 and new['return_seconds']<=old['return_seconds']+.25:return proposed
        event['kind']='productive_preserved';event['proposed']=proposed;event['target']=original
        event['gain']=gain
        return original


class _CloseController(_ResponsePressure):
    """Experimental forward pressure with lateral movement and safe exits."""

    def reset(self):
        super().reset()
        self._pressure_entry_lives = None

    def _tactic_destination(self, obs):
        sx,sy=_world_xy(obs['self']);ex,ey=_world_xy(obs['opponent'])
        solid=np.logical_or(obs['map'][0],obs['map'][1])
        start=int(sx),int(sy);costs={start:0.};queue=[(0.,start)];best=None
        distance=max(math.hypot(ex-sx,ey-sy),1e-6)
        # Hold the inside lane when feasible; do not chase an enemy into heat.
        center_length=max(math.hypot(50.-ex,50.-ey),1e-6)
        inside=((50.-ex)/center_length,(50.-ey)/center_length)
        while queue:
            cost,(cx,cy)=heapq.heappop(queue)
            if cost!=costs[cx,cy]:continue
            x,y=cx+.5,cy+.5
            margin=self._safe_margin(obs,x,y)-C.HEAT_SHRINK_RATE*(cost/C.PLAYER_SPEED+3.)
            if margin<3.:continue
            separation=math.hypot(x-ex,y-ey)
            if 1.2<=separation<=3.0 and math.hypot(x-sx,y-sy)<=16.:
                visible=has_line_of_sight(obs['map'][0],x,y,ex,ey)
                _,_,miss=self._shot(obs,x,y)
                lateral=((x-sx)*-(ey-sy)+(y-sy)*(ex-sx))/distance
                inside_lane=((x-ex)*inside[0]+(y-ey)*inside[1])/separation
                score=2.*visible-.9*min(miss,2.)-.4*abs(separation-2.0)
                score+=.65*inside_lane+.08*min(margin,15.)+.12*min(abs(lateral),6.)-.18*cost
                candidate=score,-cost,x,y
                if best is None or candidate>best:best=candidate
            for nx,ny in ((cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)):
                if not (0<=nx<100 and 0<=ny<100) or solid[nx,ny]:continue
                if math.hypot(nx+.5-ex,ny+.5-ey)<min(1.2,distance):continue
                edge=.5/(C.MUD_SLOW if obs['map'][2,cx,cy] else 1.)+.5/(C.MUD_SLOW if obs['map'][2,nx,ny] else 1.)
                new=cost+edge
                if new<=24. and new<costs.get((nx,ny),float('inf')):
                    costs[nx,ny]=new;heapq.heappush(queue,(new,(nx,ny)))
        return None if best is None else (best[2],best[3])

    def _tactic_move(self, obs):
        now=self._previous_time
        sx,sy=_world_xy(obs['self']);ex,ey=_world_xy(obs['opponent'])
        distance=max(math.hypot(ex-sx,ey-sy),1e-6)
        if self._return_needed(obs) or self._safe_margin(obs,sx,sy)<4.:
            if self._tactic_phase:self._tactic_event('', 'safe_return',obs);self._tactic_last=now
            return None
        if not self._tactic_phase:
            # Keep the pursuit prototype's trigger, duration and cooldown:
            # compare the posture rather than a different firing schedule.
            reason=self._tactic_trigger(obs)
            if reason is None:return None
            self._tactic_goal=self._tactic_destination(obs)
            if self._tactic_goal is None:return None
            self._pressure_entry_lives=_lives(obs['self'])
            self._tactic_until=now+9.;self._tactic_last=now;self._tactic_plan_time=now
            self._tactic_event('pressure',reason,obs)
        if (_lives(obs['self'])<=self._pressure_entry_lives-2 or now>=self._tactic_until or distance>22.):
            reason='damage_limit' if _lives(obs['self'])<=self._pressure_entry_lives-2 else 'expired_or_contact_lost'
            self._tactic_event('',reason,obs);self._tactic_last=now
            return None
        if now-self._tactic_plan_time>1.:
            goal=self._tactic_destination(obs)
            if goal is None:
                self._tactic_event('', 'no_safe_pressure_route',obs);self._tactic_last=now
                return None
            self._tactic_goal=goal;self._tactic_plan_time=now
        if self._safe_margin(obs,*self._tactic_goal)-C.HEAT_SHRINK_RATE*3.<3.:
            self._tactic_event('', 'unsafe_destination',obs);self._tactic_last=now
            return None
        desired=self._path_move(obs,self._tactic_goal,'forward_pressure')
        self._incoming_flights(obs)
        vx,vy=self._motion_velocity
        closing=((ex-sx)*vx+(ey-sy)*vy)/distance
        radial=max(-1.5,min(1.5,(distance-2.0)/3.+closing/C.PLAYER_SPEED))
        dx,dy=(ex-sx)/distance,(ey-sy)/distance
        choices=[]
        for my in (-1,0,1):
            for mx in (-1,0,1):
                move=mx,my;risk=self._movement_risk(obs,move);x,y=risk[4]
                margin=self._safe_margin(obs,x,y)-C.HEAT_SHRINK_RATE*self.defense_horizon
                if margin<2.:continue
                future=math.hypot(ex+vx*self.defense_horizon-x,ey+vy*self.defense_horizon-y)
                length=max(math.hypot(mx,my),1.)
                forward=(mx*dx+my*dy)/length
                sideways=(-mx*dy+my*dx)/length
                _,shoot,miss=self._shot(obs,x,y)
                score=1.2*math.dist((x,y),self._tactic_goal)+2.*risk[2]
                score+=3.*max(0.,1.2-future)+.8*max(0.,future-3.)
                score+=.45*abs(forward-radial)-.4*abs(sideways)+.7*min(miss,2.)-.3*shoot
                if move==(0,0):score+=.75
                choices.append((risk[0]+risk[1],score,mx,my))
        if not choices:
            self._tactic_event('', 'no_safe_pressure_move',obs);self._tactic_last=now
            return None
        _,_,mx,my=min(choices)
        return mx,my



_CloseBase = GeneralistAgent


class GeneralistAgent(_CloseBase):
    """Shorten firing distance after a confirmed exchange without progress."""

    def reset(self):
        super().reset()
        self._close_pressure=False

    def _close_pressure_needed(self, obs):
        own=_world_xy(obs['self']);enemy=_world_xy(obs['opponent'])
        if self._returning or self._safe_margin(obs,*own)<8. or math.dist(own,enemy)>14.:return False
        history=[row for row in self._combat_history if row[0]>=self._previous_time-6.]
        if len(history)<2 or history[-1][0]-history[0][0]<5.5:return False
        if history[-1][2]-history[0][2]<4:return False
        current=self._response_window(since=self._previous_time-5.5)
        return current['landed']==0 and current['received']==0 and current['visible']>=.5 and current['exposure']<.35

    def _tactic_destination(self, obs):
        if self._close_pressure and self._response_mode!='retreat':
            controller=_BoostCloseController if self._boost_window(obs) in (1,2) else _CloseController
            return controller._tactic_destination(self,obs)
        return super()._tactic_destination(obs)

    def _tactic_move(self, obs):
        before=self._close_pressure
        self._close_pressure=self._close_pressure_needed(obs)
        if before!=self._close_pressure:
            self._exit_events.append(dict(t=round(self._previous_time,3),kind='close_pressure_start' if self._close_pressure else 'close_pressure_stop'))
            # The destination must use the same range as the move controller.
            self._tactic_plan_time=-1e9
        return self._adaptive_tactic_move(obs)


    def _adaptive_tactic_move(self, obs):
        now=self._previous_time
        controller=(_BoostCloseController if self._boost_window(obs) in (1,2) else _CloseController) if self._close_pressure else _ResponsePressure
        if self._response_mode and not self._tactic_phase:
            self._response_finish(obs,'controller_completed')
        if self._return_needed(obs) or self._safe_margin(obs,*_world_xy(obs['self']))<4.:
            if self._response_mode:self._response_finish(obs,'safe_return')
            return None
        if self._response_mode=='retreat':return self._retreat_response_move(obs)
        aftermath=(self._response_credit_mode=='pressure' and now<=self._response_credit_until)
        if (self._response_mode=='pressure' or aftermath) and self._review_pressure(obs):
            return self._retreat_response_move(obs)
        # Delegate starts, navigation, firing and cooldown to the successful
        # fixed policy. No extra probe, path reset or early no-gain timeout.
        move=controller._tactic_move(self,obs)
        if self._tactic_phase=='pressure' and self._response_mode!='pressure':
            self._response_mode='pressure';self._response_since=now
            self._response_next_review=now+.75
            self._pressure_bad_since=None;self._pressure_last_reason=None
            self._response_events.append(dict(t=round(now,3),mode='pressure',reason='pressure_default'))
        elif self._response_mode=='pressure' and not self._tactic_phase:
            self._response_finish(obs,'controller_completed')
        return move


_TieCloseBase = GeneralistAgent
class GeneralistAgent(_TieCloseBase):
    def _close_pressure_needed(self,obs):
        if _lives(obs['self'])>_lives(obs['opponent']):return False
        return super()._close_pressure_needed(obs)


_RecoveryBase = GeneralistAgent

class GeneralistAgent(_RecoveryBase):
    """Resolve close trades at projectile time scale, with a safe-route fallback."""
    def reset(self):
        super().reset()
        self._short_credit_until=-1e9
        self._short_trade_deadline=None

    def observe(self,obs):
        super().observe(obs)
        if not self._response_rows:
            # Includes the inherited history reset after a teleport.
            self._short_credit_until=-1e9
            self._short_trade_deadline=None

    def _pending_reply_delay(self,obs):
        ex,ey=_world_xy(obs['opponent']);evx,evy=self._motion_velocity
        delays=[]
        for row,valid in zip(obs['bullets'],obs['bullet_mask']):
            if valid<.5 or row[4]<=0:continue
            x,y=_world_xy(row);vx,vy=float(row[2])*C.BULLET_SPEED,float(row[3])*C.BULLET_SPEED
            rx,ry=ex-x,ey-y;rvx,rvy=evx-vx,evy-vy
            t=-(rx*rvx+ry*rvy)/max(rvx*rvx+rvy*rvy,1e-9)
            remaining=C.BULLET_RANGE*(1.-(float(row[6])+1.)*.5)
            ttl=(float(row[5])+1.)*.5*C.BULLET_TTL
            if not 0.<t<=min(ttl,remaining/C.BULLET_SPEED):continue
            if math.hypot(rx+rvx*t,ry+rvy*t)>C.PLAYER_RADIUS+C.BULLET_RADIUS+.25:continue
            steps=max(1,math.ceil(C.BULLET_SPEED*t/.2))
            lane=[(x+vx*t*i/steps,y+vy*t*i/steps) for i in range(steps+1)]
            if any(not (0<=px<100 and 0<=py<100) or obs['map'][0,int(px),int(py)] for px,py in lane):continue
            delays.append(t)
        enemy_interval=C.QUICKSHOT_COOLDOWN if obs['opponent'][6]>-1.+1e-6 else C.SHOOT_COOLDOWN
        return min(max(delays,default=0.),max(0.,enemy_interval-3.*C.FIXED_DT))

    def _tactic_move(self,obs):
        if self._close_pressure:self._short_credit_until=self._previous_time+C.BULLET_RANGE/C.BULLET_SPEED
        elif self._previous_time>self._short_credit_until:self._short_trade_deadline=None
        return super()._tactic_move(obs)

    def _review_pressure(self,obs):
        now=self._previous_time
        if now>self._short_credit_until:return super()._review_pressure(obs)
        current=self._response_window(since=now-2.5)
        if current['received']<=current['landed']:
            self._short_trade_deadline=None
            return super()._review_pressure(obs)
        if self._short_trade_deadline is None:
            wait=self._pending_reply_delay(obs)+2.*C.FIXED_DT
            self._short_trade_deadline=now+wait
            self._exit_events.append(dict(t=round(now,3),kind='close_trade_reply_wait',wait=wait))
        # This deadline is set once; another shot cannot postpone it forever.
        if now+1e-6<self._short_trade_deadline:return False
        self._short_trade_deadline=None
        self._close_pressure=False
        self._exit_events.append(dict(t=round(now,3),kind='close_trade_exit'))
        changed=self._try_retreat_response(obs,current,'short_trade_lost')
        if not changed:
            self._response_finish(obs,'short_trade_no_route')
            self._tactic_last=now
        return changed


_InitiativeBase = GeneralistAgent

class GeneralistAgent(_InitiativeBase):
    def _close_pressure_needed(self,obs):
        if not super()._close_pressure_needed(obs):return False
        distance=math.dist(_world_xy(obs['self']),_world_xy(obs['opponent']))
        if distance>6.:return True
        own=(float(obs['self'][5])+1.)*.5*C.SHOOT_COOLDOWN
        enemy=(float(obs['opponent'][5])+1.)*.5*C.SHOOT_COOLDOWN
        # Avoid the final approach while the enemy can answer substantially
        # earlier than us; keep normal pressure and firing in that interval.
        return own<=enemy+distance/C.BULLET_SPEED


_DeadlineEscapeBase = GeneralistAgent


class GeneralistAgent(_DeadlineEscapeBase):
    """Leave a pocket before its route to the final safe area closes."""

    def reset(self):
        super().reset()
        self._escape_checked=-1e9
        self._escape_active=False
        self._escape_events=[]

    def _end_safe_route(self,obs):
        horizon=self.return_lookahead_seconds
        minimum=self.return_min_margin
        now=float(obs['game'][0])*C.MAX_EPISODE_SECONDS
        try:
            self.return_lookahead_seconds=max(0.,C.MAX_EPISODE_SECONDS-now)
            self.return_min_margin=.75
            return DijkstraAgent._forecast_return(self,obs,allow_heat=False)
        finally:
            self.return_lookahead_seconds=horizon
            self.return_min_margin=minimum

    def _return_needed(self,obs):
        now=float(obs['game'][0])*C.MAX_EPISODE_SECONDS
        # The inherited observer invalidates return forecasts on teleport.
        invalidated=self._return_checked<self._escape_checked-1e-6
        if now-self._escape_checked<.75 and self._escape_active and not invalidated:
            return True
        regular=super()._return_needed(obs)
        if now-self._escape_checked<.75 and not invalidated:
            return regular
        self._escape_checked=now
        sx,sy=_world_xy(obs['self'])
        margin=self._safe_margin(obs,sx,sy)
        remaining=max(0.,C.MAX_EPISODE_SECONDS-now)
        # Already safe through the end, or too late for a heat-free rescue:
        # retain the normal return controller and its obstacle handling.
        if margin>=.75+C.HEAT_SHRINK_RATE*remaining or margin<.5:
            if self._escape_active:self._escape_events.append(dict(t=now,event='released'))
            self._escape_active=False
            return regular
        path,slack=self._end_safe_route(obs)
        if not path or len(path)==1:
            self._escape_active=False
            return regular
        if not self._escape_active and slack>=self.return_safety_seconds+.5:
            return regular
        if not self._escape_active:
            self._escape_events.append(dict(t=now,event='escape',slack=slack,path_length=len(path)))
        self._escape_active=True
        self._returning=True
        self._return_path=path
        self._return_goal=path[-1][0]+.5,path[-1][1]+.5
        return True


_PickupOpportunityBase=GeneralistAgent
class GeneralistAgent(_PickupOpportunityBase):
    pickup_budget=10.0

    def reset(self):
        super().reset()
        self._pickup_checked=-1e9
        self._pickup_choice=None

    def _nearby_advantage(self,obs):
        sx,sy=_world_xy(obs['self']);ex,ey=_world_xy(obs['opponent'])
        own=_lives(obs['self']);enemy=_lives(obs['opponent'])
        if (own>enemy and own<C.PLAYER_LIVES) or math.hypot(ex-sx,ey-sy)<4. or self._return_needed(obs):
            return None
        now=self._previous_time
        targets={}
        for row,valid in zip(obs['powerups'],obs['powerup_mask']):
            if valid<.5:continue
            x,y=_world_xy(row)
            if math.hypot(x-sx,y-sy)>self.pickup_budget or self._safe_margin(obs,x,y)<4.:continue
            kind='life' if row[3]>.5 and (own<C.PLAYER_LIVES or enemy<C.PLAYER_LIVES) else 'quick' if row[4]>.5 and obs['self'][6]<-.99 else 'speed' if row[5]>.5 and obs['self'][7]<-.99 else None
            if kind is None or (own>enemy and (kind!='life' or math.hypot(x-sx,y-sy)>5.)):continue
            ttl=(float(row[2])+1.)*.5*C.POWERUP_TTL
            targets[int(x),int(y)]=(x,y,kind,ttl)
        if not targets:
            self._pickup_choice=None
            return None
        if now-self._pickup_checked<.5 and self._pickup_choice is not None:
            choice=self._pickup_choice
            if (int(choice[0]),int(choice[1])) in targets:return choice[:2]
        self._pickup_checked=now
        start=int(sx),int(sy);costs={start:0.};queue=[(0.,start)];best=None
        solid=np.logical_or(obs['map'][0],obs['map'][1]);mud=obs['map'][2]
        while queue:
            cost,cell=heapq.heappop(queue)
            if cost!=costs[cell]:continue
            x,y=cell
            eta=cost/C.PLAYER_SPEED
            if self._safe_margin(obs,x+.5,y+.5)-C.HEAT_SHRINK_RATE*(eta+2.)<3.:continue
            if cell in targets:
                px,py,kind,ttl=targets[cell]
                if eta+.5<ttl:
                    score=cost-({'life':8.,'quick':4.,'speed':3.}[kind])
                    candidate=score,cost,px,py,kind
                    if best is None or candidate<best:best=candidate
            for nx,ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if not (0<=nx<100 and 0<=ny<100) or solid[nx,ny]:continue
                edge=.5/(C.MUD_SLOW if mud[x,y] else 1.)+.5/(C.MUD_SLOW if mud[nx,ny] else 1.)
                new=cost+edge
                if new<=self.pickup_budget and new<costs.get((nx,ny),float('inf')):
                    costs[nx,ny]=new;heapq.heappush(queue,(new,(nx,ny)))
        self._pickup_choice=None if best is None else best[2:]
        return None if best is None else best[2:4]

    def _tactic_move(self,obs):
        target=self._nearby_advantage(obs)
        if target is not None:
            self._tactic_phase=''
            self._response_mode=''
            self._tactic_last=min(self._tactic_last,self._previous_time-12.)
            return self._path_move(obs,target,'collect_advantage')
        return super()._tactic_move(obs)


class _BoostCloseController(_ResponsePressure):
    """Experimental forward pressure with lateral movement and safe exits."""

    def reset(self):
        super().reset()
        self._pressure_entry_lives = None

    def _tactic_destination(self, obs):
        sx,sy=_world_xy(obs['self']);ex,ey=_world_xy(obs['opponent'])
        solid=np.logical_or(obs['map'][0],obs['map'][1])
        start=int(sx),int(sy);costs={start:0.};queue=[(0.,start)];best=None
        distance=max(math.hypot(ex-sx,ey-sy),1e-6)
        # Hold the inside lane when feasible; do not chase an enemy into heat.
        center_length=max(math.hypot(50.-ex,50.-ey),1e-6)
        inside=((50.-ex)/center_length,(50.-ey)/center_length)
        while queue:
            cost,(cx,cy)=heapq.heappop(queue)
            if cost!=costs[cx,cy]:continue
            x,y=cx+.5,cy+.5
            margin=self._safe_margin(obs,x,y)-C.HEAT_SHRINK_RATE*(cost/C.PLAYER_SPEED+3.)
            if margin<3.:continue
            separation=math.hypot(x-ex,y-ey)
            if 0.6<=separation<=1.6 and math.hypot(x-sx,y-sy)<=16.:
                visible=has_line_of_sight(obs['map'][0],x,y,ex,ey)
                _,_,miss=self._shot(obs,x,y)
                lateral=((x-sx)*-(ey-sy)+(y-sy)*(ex-sx))/distance
                inside_lane=((x-ex)*inside[0]+(y-ey)*inside[1])/separation
                score=2.*visible-.9*min(miss,2.)-.4*abs(separation-1.0)
                score+=.65*inside_lane+.08*min(margin,15.)+.12*min(abs(lateral),6.)-.18*cost
                candidate=score,-cost,x,y
                if best is None or candidate>best:best=candidate
            for nx,ny in ((cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)):
                if not (0<=nx<100 and 0<=ny<100) or solid[nx,ny]:continue
                if math.hypot(nx+.5-ex,ny+.5-ey)<min(0.6,distance):continue
                edge=.5/(C.MUD_SLOW if obs['map'][2,cx,cy] else 1.)+.5/(C.MUD_SLOW if obs['map'][2,nx,ny] else 1.)
                new=cost+edge
                if new<=24. and new<costs.get((nx,ny),float('inf')):
                    costs[nx,ny]=new;heapq.heappush(queue,(new,(nx,ny)))
        return None if best is None else (best[2],best[3])

    def _tactic_move(self, obs):
        now=self._previous_time
        sx,sy=_world_xy(obs['self']);ex,ey=_world_xy(obs['opponent'])
        distance=max(math.hypot(ex-sx,ey-sy),1e-6)
        if self._return_needed(obs) or self._safe_margin(obs,sx,sy)<4.:
            if self._tactic_phase:self._tactic_event('', 'safe_return',obs);self._tactic_last=now
            return None
        if not self._tactic_phase:
            # Keep the pursuit prototype's trigger, duration and cooldown:
            # compare the posture rather than a different firing schedule.
            reason=self._tactic_trigger(obs)
            if reason is None:return None
            self._tactic_goal=self._tactic_destination(obs)
            if self._tactic_goal is None:return None
            self._pressure_entry_lives=_lives(obs['self'])
            self._tactic_until=now+9.;self._tactic_last=now;self._tactic_plan_time=now
            self._tactic_event('pressure',reason,obs)
        if (_lives(obs['self'])<=self._pressure_entry_lives-2 or now>=self._tactic_until or distance>22.):
            reason='damage_limit' if _lives(obs['self'])<=self._pressure_entry_lives-2 else 'expired_or_contact_lost'
            self._tactic_event('',reason,obs);self._tactic_last=now
            return None
        if now-self._tactic_plan_time>1.:
            goal=self._tactic_destination(obs)
            if goal is None:
                self._tactic_event('', 'no_safe_pressure_route',obs);self._tactic_last=now
                return None
            self._tactic_goal=goal;self._tactic_plan_time=now
        if self._safe_margin(obs,*self._tactic_goal)-C.HEAT_SHRINK_RATE*3.<3.:
            self._tactic_event('', 'unsafe_destination',obs);self._tactic_last=now
            return None
        desired=self._path_move(obs,self._tactic_goal,'forward_pressure')
        self._incoming_flights(obs)
        vx,vy=self._motion_velocity
        closing=((ex-sx)*vx+(ey-sy)*vy)/distance
        radial=max(-1.5,min(1.5,(distance-1.0)/3.+closing/C.PLAYER_SPEED))
        dx,dy=(ex-sx)/distance,(ey-sy)/distance
        choices=[]
        for my in (-1,0,1):
            for mx in (-1,0,1):
                move=mx,my;risk=self._movement_risk(obs,move);x,y=self._future_positions(obs,move)[0][5]
                margin=self._safe_margin(obs,x,y)-C.HEAT_SHRINK_RATE*self.defense_horizon
                if margin<2.:continue
                future=math.hypot(ex+vx*0.2-x,ey+vy*0.2-y)
                length=max(math.hypot(mx,my),1.)
                forward=(mx*dx+my*dy)/length
                sideways=(-mx*dy+my*dx)/length
                _,shoot,miss=self._shot(obs,x,y)
                score=1.2*math.dist((x,y),self._tactic_goal)+2.*risk[2]
                score+=3.*max(0.,0.6-future)+.8*max(0.,future-1.6)
                score+=.45*abs(forward-radial)-.4*abs(sideways)+.7*min(miss,2.)-.3*shoot
                if move==(0,0):score+=.75
                choices.append((risk[0]+risk[1],score,mx,my))
        if not choices:
            self._tactic_event('', 'no_safe_pressure_move',obs);self._tactic_last=now
            return None
        _,_,mx,my=min(choices)
        return mx,my





_BoostWindowBase=GeneralistAgent
class GeneralistAgent(_BoostWindowBase):
    def reset(self):
        super().reset()
        self._boost_last=0

    def _boost_window(self,obs):
        if _lives(obs['self'])>_lives(obs['opponent']) or self._return_needed(obs):return 0
        if obs['self'][7]>-.99 and obs['opponent'][7]<=-.99:return 1
        if obs['self'][6]>-.99 and obs['opponent'][6]<=-.99:return 2
        return 0

    def _tactic_trigger(self,obs):
        if self._boost_window(obs) and obs['self'][14]>0 and math.dist(_world_xy(obs['self']),_world_xy(obs['opponent']))<=22.:
            return 'temporary_power_advantage'
        return super()._tactic_trigger(obs)

    def _close_pressure_needed(self,obs):
        boost=self._boost_window(obs)
        if boost in (1,2):
            sx,sy=_world_xy(obs['self'])
            return obs['self'][14]>0 and self._safe_margin(obs,sx,sy)>=8. and math.dist((sx,sy),_world_xy(obs['opponent']))<=14.
        if boost==2:return False
        return super()._close_pressure_needed(obs)

    def _tactic_move(self,obs):
        boost=self._boost_window(obs)
        if boost!=self._boost_last:
            self._boost_last=boost;self._tactic_plan_time=-1e9
        return super()._tactic_move(obs)


_PocketEscapeBase = GeneralistAgent

class GeneralistAgent(_PocketEscapeBase):
    def reset(self):
        super().reset()
        self._pocket_checked=-1e9
        self._pocket_active=False
        self._pocket_events=[]

    def _pocket_route(self,obs):
        now=float(obs['game'][0])*C.MAX_EPISODE_SECONDS
        horizon,minimum=self.return_lookahead_seconds,self.return_min_margin
        try:
            self.return_lookahead_seconds=max(0.,C.MAX_EPISODE_SECONDS-now)
            self.return_min_margin=.75
            return DijkstraAgent._forecast_return(self,obs,allow_heat=True)
        finally:
            self.return_lookahead_seconds,self.return_min_margin=horizon,minimum

    def _return_needed(self,obs):
        now=float(obs['game'][0])*C.MAX_EPISODE_SECONDS
        invalidated=self._return_checked<self._pocket_checked-1e-6
        if now-self._pocket_checked<.75 and not invalidated:
            return True if self._pocket_active else super()._return_needed(obs)
        if invalidated:self._pocket_active=False
        self._pocket_checked=now
        sx,sy=_world_xy(obs['self'])
        margin=self._safe_margin(obs,sx,sy)
        remaining=max(0.,C.MAX_EPISODE_SECONDS-now)
        if margin>=.75+C.HEAT_SHRINK_RATE*remaining:
            self._pocket_active=False
            return super()._return_needed(obs)
        if not self._pocket_active:
            regular=super()._return_needed(obs)
            if regular or margin<.5:return regular
            free_path,_=self._end_safe_route(obs)
            if free_path:return regular
        else:
            free_path,_=self._end_safe_route(obs)
        if free_path:
            path=free_path;slack=0.
        else:
            path,slack=self._pocket_route(obs)
        if not path or len(path)<2 or (not self._pocket_active and slack>=self.return_safety_seconds):
            self._pocket_active=False
            return super()._return_needed(obs)
        if not self._pocket_active:self._pocket_events.append(dict(t=now,event='early_escape',slack=slack))
        self._pocket_active=True
        self._return_checked=now
        self._returning=True
        self._return_path=path
        self._return_goal=path[-1][0]+.5,path[-1][1]+.5
        return True


class Agent(GeneralistAgent):
    """Contest entry: corrected generalist with required, bundled weights."""

    def __init__(self, weights_path=None):
        directory = Path(weights_path) if weights_path is not None else Path(__file__).resolve().parent / "weights"
        model = directory if directory.suffix.lower() == ".npz" else directory / "qtable.npz"
        if not model.is_file():
            raise FileNotFoundError(f"required generalist checkpoint not found: {model}")
        super().__init__(model)

    def act(self, obs):
        action = np.asarray(super().act(obs))
        if (action.shape != (3,) or action.dtype.kind not in "iu" or
                np.any(action < 0) or np.any(action >= np.asarray([9, 16, 2]))):
            raise ValueError(f"invalid contest action: {action!r}")
        return action.astype(np.int64, copy=False)


def make_agent(weights_path=None):
    return Agent(weights_path)
