"""Map generation: Perlin-noise walls, spawn carving, torch placement,
mud ponds and solid trees. The two spawns are always mutually reachable
(trees can never reseal the route), and random_free_cell only picks cells
inside that spawn-connected region so teleports never strand a player."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import constants as C
from .constants import GRID_SIZE, POWERUP_TYPES
from .perlin import perlin_field

SPAWN_CELLS = [(15, 15), (GRID_SIZE - 16, GRID_SIZE - 16)]
CARVE_RADIUS = 4  # free area carved around each spawn


@dataclass
class GameMap:
    grid: np.ndarray               # (GRID_SIZE, GRID_SIZE) uint8, 1 = wall
    torches: list[tuple[int, int]] = field(default_factory=list)  # wall cells holding a torch
    mud: np.ndarray | None = None  # (GRID_SIZE, GRID_SIZE) uint8, 1 = slowing mud pond
    trees: list[tuple[float, float]] = field(default_factory=list)  # solid: block movement like walls
    seed: int = 0
    scale: float = 24.0
    octaves: int = 3
    threshold: float = 0.62
    solid: np.ndarray = field(init=False)  # walls | tree cells (snapshot at build time)
    main_region: np.ndarray = field(init=False)  # walkable cells connected to the spawns

    def __post_init__(self) -> None:
        tree_cells = np.zeros_like(self.grid)
        for tx, ty in self.trees:
            tree_cells[int(tx), int(ty)] = 1
        self.solid = (self.grid | tree_cells).astype(np.uint8)
        # Resolved at call time: _flood_reachable is defined further down.
        self.main_region = _flood_reachable(self.solid, SPAWN_CELLS[0])

    def is_wall(self, cx: int, cy: int) -> bool:
        if cx < 0 or cy < 0 or cx >= GRID_SIZE or cy >= GRID_SIZE:
            return True
        return bool(self.grid[cx, cy])

    def is_solid(self, cx: int, cy: int) -> bool:
        """Wall or tree cell: blocks movement exactly like a wall."""
        if cx < 0 or cy < 0 or cx >= GRID_SIZE or cy >= GRID_SIZE:
            return True
        return bool(self.solid[cx, cy])

    def is_mud(self, cx: int, cy: int) -> bool:
        if self.mud is None:
            return False
        if cx < 0 or cy < 0 or cx >= GRID_SIZE or cy >= GRID_SIZE:
            return False
        return bool(self.mud[cx, cy])

    def random_free_cell(self, rng: random.Random) -> tuple[int, int]:
        # Only cells in the spawn-connected region: teleports (retreat) must
        # never drop a player into a sealed pocket with no way out.
        while True:
            x = rng.randrange(1, GRID_SIZE - 1)
            y = rng.randrange(1, GRID_SIZE - 1)
            if not self.solid[x, y] and self.main_region[x, y]:
                return x, y


def _carve_disc(grid: np.ndarray, cx: int, cy: int, r: int) -> None:
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                x, y = cx + dx, cy + dy
                if 1 <= x < GRID_SIZE - 1 and 1 <= y < GRID_SIZE - 1:
                    grid[x, y] = 0


def _flood_reachable(grid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    seen = np.zeros_like(grid, dtype=bool)
    if grid[start]:
        return seen
    q = deque([start])
    seen[start] = True
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and not seen[nx, ny] and not grid[nx, ny]:
                seen[nx, ny] = True
                q.append((nx, ny))
    return seen


def _carve_tunnel(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    """Carve a fat L-shaped tunnel so both spawns stay connected."""
    x, y = a
    tx, ty = b
    while x != tx:
        x += 1 if tx > x else -1
        _carve_disc(grid, x, y, 2)
    while y != ty:
        y += 1 if ty > y else -1
        _carve_disc(grid, x, y, 2)


def _place_torches(grid: np.ndarray, rng: random.Random, spacing: int = 7) -> list[tuple[int, int]]:
    """Torches hang on wall blocks that border open floor, spread out."""
    candidates = []
    for x in range(1, GRID_SIZE - 1):
        for y in range(1, GRID_SIZE - 1):
            if not grid[x, y]:
                continue
            open_sides = sum(
                not grid[x + dx, y + dy]
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
            if open_sides >= 2:
                candidates.append((x, y))
    rng.shuffle(candidates)
    torches: list[tuple[int, int]] = []
    for cell in candidates:
        if all(abs(cell[0] - t[0]) + abs(cell[1] - t[1]) >= spacing for t in torches):
            torches.append(cell)
    return torches


def generate_map(
    seed: int = 0,
    scale: float = 24.0,
    octaves: int = 3,
    threshold: float = 0.62,
) -> GameMap:
    """Build a GameMap. All knobs are changeable -> different dungeons."""
    noise = perlin_field(GRID_SIZE, seed=seed, scale=scale, octaves=octaves)
    grid = (noise > threshold).astype(np.uint8)

    # border walls
    grid[0, :] = 1
    grid[GRID_SIZE - 1, :] = 1
    grid[:, 0] = 1
    grid[:, GRID_SIZE - 1] = 1

    for sx, sy in SPAWN_CELLS:
        _carve_disc(grid, sx, sy, CARVE_RADIUS)

    # guarantee the two spawns are connected
    reach = _flood_reachable(grid, SPAWN_CELLS[0])
    if not reach[SPAWN_CELLS[1]]:
        _carve_tunnel(grid, SPAWN_CELLS[0], SPAWN_CELLS[1])

    rng = random.Random((seed, scale, octaves, threshold).__hash__() & 0xFFFFFFFF)
    torches = _place_torches(grid, rng)

    # mud ponds and trees use their own noise/rng streams so the wall grid
    # for a given (seed, scale, octaves, threshold) is unchanged
    mud_noise = perlin_field(GRID_SIZE, seed=seed + 91337, scale=C.MUD_SCALE, octaves=2)
    mud = ((mud_noise > C.MUD_THRESHOLD) & (grid == 0)).astype(np.uint8)
    for sx, sy in SPAWN_CELLS:  # keep spawn areas clean
        mud[max(0, sx - CARVE_RADIUS - 1):sx + CARVE_RADIUS + 2,
            max(0, sy - CARVE_RADIUS - 1):sy + CARVE_RADIUS + 2] = 0

    # Never use Python's salted string/tuple hash here: contest workers in
    # separate processes must generate identical trees for the same map seed.
    tree_seed = (int(seed) * 1_000_003 + 0x5EEDBEEF) & 0xFFFFFFFF
    trng = random.Random(tree_seed)
    trees: list[tuple[float, float]] = []
    for x in range(1, GRID_SIZE - 1):
        for y in range(1, GRID_SIZE - 1):
            if grid[x, y] or mud[x, y]:
                continue
            if any((x - sx) ** 2 + (y - sy) ** 2 <= (CARVE_RADIUS + 1) ** 2
                   for sx, sy in SPAWN_CELLS):
                continue
            if trng.random() < C.TREE_DENSITY:
                trees.append((x + 0.5 + trng.uniform(-0.25, 0.25),
                              y + 0.5 + trng.uniform(-0.25, 0.25)))

    # The wall grid guarantees connected spawns, but trees are placed
    # afterwards and can seal the route again. When that happens, drop the
    # trees standing on one wall-free spawn-to-spawn path so both players
    # can always reach each other. Seeds without a seal are untouched.
    tree_cells = np.zeros_like(grid)
    for tx, ty in trees:
        tree_cells[int(tx), int(ty)] = 1
    if not _flood_reachable(grid | tree_cells, SPAWN_CELLS[0])[SPAWN_CELLS[1]]:
        from .pathfinding import dijkstra
        path = dijkstra(grid, SPAWN_CELLS[0][0] + 0.5, SPAWN_CELLS[0][1] + 0.5,
                        SPAWN_CELLS[1][0] + 0.5, SPAWN_CELLS[1][1] + 0.5)
        on_path = {(cx, cy) for cx, cy in path}
        trees = [t for t in trees if (int(t[0]), int(t[1])) not in on_path]

    return GameMap(grid=grid, torches=torches, mud=mud, trees=trees, seed=seed,
                   scale=scale, octaves=octaves, threshold=threshold)
