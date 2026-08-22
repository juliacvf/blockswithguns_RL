"""Line of sight on the grid (port of hasLineOfSight.js)."""

from __future__ import annotations

import numpy as np


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
