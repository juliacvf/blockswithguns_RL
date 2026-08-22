"""Seeded Perlin-style (fBm gradient) noise over the grid.

All parameters are changeable: seed, scale (feature size in cells),
octaves and threshold are passed in from the caller, so the game UI and
the env can regenerate maps with different looks.
"""

from __future__ import annotations

import numpy as np


def _fade(t: np.ndarray) -> np.ndarray:
    # Perlin's smoothstep 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _gradient_noise(rng: np.random.Generator, size: int, cell_size: int) -> np.ndarray:
    """Single octave of 2D gradient noise on a size x size grid."""
    if cell_size < 1:
        cell_size = 1
    # Lattice of random gradients; +2 so the borders have a full cell
    gw = size // cell_size + 2
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(gw, gw))
    gx = np.cos(angles)
    gy = np.sin(angles)

    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    fx = xs / cell_size
    fy = ys / cell_size
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    tx = _fade(fx - x0)
    ty = _fade(fy - y0)

    def dot(ix: int, iy: int, ox: np.ndarray, oy: np.ndarray) -> np.ndarray:
        return gx[iy, ix] * (fx - x0 - ox) + gy[iy, ix] * (fy - y0 - oy)

    n00 = dot(x0, y0, 0.0, 0.0)
    n10 = dot(x0 + 1, y0, 1.0, 0.0)
    n01 = dot(x0, y0 + 1, 0.0, 1.0)
    n11 = dot(x0 + 1, y0 + 1, 1.0, 1.0)

    nx0 = n00 + tx * (n10 - n00)
    nx1 = n01 + tx * (n11 - n01)
    return nx0 + ty * (nx1 - nx0)


def perlin_field(
    size: int,
    seed: int = 0,
    scale: float = 24.0,
    octaves: int = 3,
    persistence: float = 0.5,
) -> np.ndarray:
    """fBm gradient noise normalized to [0, 1], shape (size, size).

    scale: approximate feature size in cells (bigger = larger blobs).
    """
    rng = np.random.default_rng(seed)
    total = np.zeros((size, size), dtype=np.float64)
    amp = 1.0
    amp_sum = 0.0
    for o in range(max(1, octaves)):
        cell = max(1.0, scale / (2 ** o))
        total += amp * _gradient_noise(rng, size, int(round(cell)))
        amp_sum += amp
        amp *= persistence
    total /= amp_sum
    # normalize to [0, 1]
    lo, hi = total.min(), total.max()
    if hi - lo < 1e-9:
        return np.zeros((size, size), dtype=np.float64)
    return (total - lo) / (hi - lo)
