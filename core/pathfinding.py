"""A* and Dijkstra pathfinding on the wall grid (ports of the JS helpers).

Optimized: flat arrays indexed by x * n + y instead of dicts/sets, inlined
neighbor expansion. Paths are meant to be cached and followed for several
ticks (see bots.base.Bot.follow_path).
"""

from __future__ import annotations

import heapq

import numpy as np

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


def dijkstra(grid: np.ndarray, sx: float, sy: float, gx: float, gy: float,
             terrain: np.ndarray | None = None,
             terrain_penalty: float = 0.0) -> list[tuple[int, int]]:
    """Find a cheapest path, optionally charging extra for terrain cells."""
    n = grid.shape[0]
    start = _free_cell(grid, _to_cell(sx), _to_cell(sy))
    goal = _free_cell(grid, _to_cell(gx), _to_cell(gy))
    if start == goal:
        return [start]
    s = start[0] * n + start[1]
    gcell = goal[0] * n + goal[1]

    dist = {s: 0.0}
    came: dict[int, int] = {}
    closed = bytearray(n * n)
    open_heap = [(0.0, s)]
    push = heapq.heappush
    pop = heapq.heappop

    while open_heap:
        d, cur = pop(open_heap)
        if closed[cur]:
            continue
        if cur == gcell:
            return _reconstruct(came, cur, n)
        closed[cur] = 1
        cx, cy = cur // n, cur % n
        for nb in (cur + n, cur - n, cur + 1, cur - 1):
            nx, ny = nb // n, nb % n
            if nx < 0 or nx >= n or ny < 0 or ny >= n or abs(nx - cx) + abs(ny - cy) != 1:
                continue
            if grid[nx, ny] or closed[nb]:
                continue
            nd = d + 1.0
            if terrain is not None and terrain[nx, ny]:
                nd += terrain_penalty
            if nd < dist.get(nb, _INF):
                dist[nb] = nd
                came[nb] = cur
                push(open_heap, (nd, nb))
    return []


def path_direction(path: list[tuple[int, int]], x: float, y: float) -> tuple[float, float]:
    """Direction (not normalized) from world pos toward the next path cell."""
    if len(path) < 2:
        return 0.0, 0.0
    nx, ny = path[1]
    return nx + 0.5 - x, ny + 0.5 - y


def wall_repulsion(grid: np.ndarray, x: float, y: float, radius: int = 1) -> tuple[float, float]:
    """Normalized push away from adjacent wall cells (basicWallAvoidance port)."""
    gx, gy = int(x), int(y)
    rx = ry = 0.0
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            cx, cy = gx + dx, gy + dy
            if 0 <= cx < grid.shape[0] and 0 <= cy < grid.shape[1] and grid[cx, cy] == 1:
                rx -= dx / abs(dx) if dx else 0.0
                ry -= dy / abs(dy) if dy else 0.0
    mag = (rx * rx + ry * ry) ** 0.5
    if mag > 0:
        rx /= mag
        ry /= mag
    return rx, ry


def wall_aware_velocity(base: tuple[float, float], repulsion: tuple[float, float],
                        base_weight: float = 0.3, avoid_weight: float = 0.7) -> tuple[float, float]:
    return (base[0] * base_weight + repulsion[0] * avoid_weight,
            base[1] * base_weight + repulsion[1] * avoid_weight)
