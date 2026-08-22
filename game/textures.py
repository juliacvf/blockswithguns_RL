"""Procedural blocky textures: bright dungeon (grass, cobblestone, torches).

Everything is generated with pygame surfaces at startup - no external art.
Texture size is 16x16 pixels, deliberately blocky/pixelated.
"""

from __future__ import annotations

import random

import pygame

TEX = 16  # texture size in pixels

# Bright palette
GRASS_TOP = (110, 200, 90)
GRASS_DARK = (92, 175, 76)
GRASS_LIGHT = (132, 215, 110)
COBBLE = (150, 152, 158)
COBBLE_DARK = (118, 120, 128)
COBBLE_LIGHT = (176, 178, 184)
MORTAR = (96, 98, 106)
MOSS = (104, 160, 92)
MUD_TOP = (122, 92, 58)
MUD_DARK = (96, 70, 44)
MUD_LIGHT = (146, 114, 74)
TREE_LEAF = (52, 122, 50)
TREE_LEAF_DARK = (36, 92, 38)
TREE_LEAF_LIGHT = (72, 150, 64)
TREE_TRUNK = (110, 76, 44)
TORCH_STICK = (120, 84, 48)
TORCH_FLAME = (255, 214, 90)
TORCH_FLAME_HOT = (255, 244, 170)
SKY = (188, 224, 255)

_cache: dict[str, pygame.Surface] = {}
_torch_frames: list[pygame.Surface] = []


def _noise_speckle(surf: pygame.Surface, rng: random.Random, colors, count: int) -> None:
    for _ in range(count):
        x = rng.randrange(TEX)
        y = rng.randrange(TEX)
        surf.set_at((x, y), rng.choice(colors))


def grass_texture(seed: int = 7) -> pygame.Surface:
    key = "grass"
    if key in _cache:
        return _cache[key]
    rng = random.Random(seed)
    s = pygame.Surface((TEX, TEX))
    s.fill(GRASS_TOP)
    _noise_speckle(s, rng, [GRASS_DARK, GRASS_LIGHT], 42)
    # a couple of brighter 2x2 tufts
    for _ in range(3):
        x, y = rng.randrange(TEX - 1), rng.randrange(TEX - 1)
        pygame.draw.rect(s, GRASS_LIGHT, (x, y, 2, 1))
    _cache[key] = s
    return s


def mud_texture(seed: int = 23) -> pygame.Surface:
    """Wet brown puddle tile for mud ponds."""
    key = "mud"
    if key in _cache:
        return _cache[key]
    rng = random.Random(seed)
    s = pygame.Surface((TEX, TEX))
    s.fill(MUD_TOP)
    _noise_speckle(s, rng, [MUD_DARK, MUD_LIGHT], 56)
    # puddle glints
    for _ in range(3):
        x, y = rng.randrange(TEX - 3), rng.randrange(TEX - 2)
        pygame.draw.rect(s, (168, 142, 104), (x, y, 3, 1))
    _cache[key] = s
    return s


def make_tree_billboard() -> pygame.Surface:
    """Blocky conifer for the 3D view (taller than wide, alpha background)."""
    s = pygame.Surface((16, 24), pygame.SRCALPHA)
    # trunk
    pygame.draw.rect(s, TREE_TRUNK, (7, 17, 3, 7))
    pygame.draw.rect(s, (84, 58, 34), (7, 17, 1, 7))
    # three stacked blocky leaf tiers
    pygame.draw.rect(s, TREE_LEAF_DARK, (2, 12, 12, 5))
    pygame.draw.rect(s, TREE_LEAF, (3, 12, 10, 4))
    pygame.draw.rect(s, TREE_LEAF_DARK, (4, 7, 8, 5))
    pygame.draw.rect(s, TREE_LEAF, (5, 7, 6, 4))
    pygame.draw.rect(s, TREE_LEAF_LIGHT, (6, 3, 4, 4))
    pygame.draw.rect(s, TREE_LEAF, (7, 2, 2, 2))
    return s


def make_tree_topdown() -> pygame.Surface:
    """Round canopy blob for the topdown view."""
    s = pygame.Surface((TEX, TEX), pygame.SRCALPHA)
    pygame.draw.rect(s, TREE_LEAF_DARK, (2, 2, 12, 12))
    pygame.draw.rect(s, TREE_LEAF, (3, 3, 10, 10))
    pygame.draw.rect(s, TREE_LEAF_LIGHT, (5, 4, 4, 3))
    pygame.draw.rect(s, TREE_TRUNK, (7, 7, 2, 2))
    return s


def cobble_texture(mossy: bool = False, seed: int = 11) -> pygame.Surface:
    key = f"cobble_{mossy}"
    if key in _cache:
        return _cache[key]
    rng = random.Random(seed + (99 if mossy else 0))
    s = pygame.Surface((TEX, TEX))
    s.fill(MORTAR)
    # blocky rounded stones: 2x2-ish blobs on a mortar grid
    stones = [(0, 0, 7, 7), (8, 0, 8, 6), (0, 8, 6, 8), (7, 7, 9, 7), (9, 14, 7, 2)]
    for i, (x, y, w, h) in enumerate(stones):
        base = [COBBLE, COBBLE_DARK, COBBLE_LIGHT][i % 3]
        if mossy and i % 2 == 0:
            base = MOSS
        pygame.draw.rect(s, base, (x + 1, y + 1, max(1, w - 2), max(1, h - 2)))
        pygame.draw.rect(s, tuple(min(255, c + 18) for c in base),
                         (x + 1, y + 1, max(1, w - 2), 1))  # top highlight
    _noise_speckle(s, rng, [COBBLE_DARK, COBBLE_LIGHT], 14)
    _cache[key] = s
    return s


def torch_frames() -> list[pygame.Surface]:
    """3 animated torch frames (blocky stick + flame)."""
    global _torch_frames
    if _torch_frames:
        return _torch_frames
    frames = []
    for f in range(3):
        s = pygame.Surface((TEX, TEX), pygame.SRCALPHA)
        # stick
        pygame.draw.rect(s, TORCH_STICK, (7, 8, 3, 8))
        pygame.draw.rect(s, (90, 60, 34), (7, 8, 1, 8))
        # flame block, wobbles per frame
        fx = 6 + (f % 2)
        pygame.draw.rect(s, TORCH_FLAME, (fx, 2 + (f == 1), 5, 6))
        pygame.draw.rect(s, TORCH_FLAME_HOT, (fx + 1, 3 + (f == 2), 3, 3))
        frames.append(s)
    _torch_frames = frames
    return frames


def wall_texture_for(cell_value: int, seed_hint: int = 0) -> pygame.Surface:
    """Pick a wall texture; ~15% of walls are mossy cobble (seeded per cell)."""
    mossy = (cell_value * 2654435761 + seed_hint) % 100 < 15
    return cobble_texture(mossy=mossy)


def make_player_sprite(color: tuple[int, int, int]) -> pygame.Surface:
    """Blocky knight-ish player block with a face."""
    s = pygame.Surface((TEX, TEX), pygame.SRCALPHA)
    dark = tuple(max(0, c - 55) for c in color)
    pygame.draw.rect(s, color, (2, 2, 12, 12))
    pygame.draw.rect(s, dark, (2, 2, 12, 2))      # helmet shade
    pygame.draw.rect(s, dark, (2, 12, 12, 2))     # bottom shade
    pygame.draw.rect(s, (30, 30, 34), (4, 6, 3, 3))   # eyes
    pygame.draw.rect(s, (30, 30, 34), (10, 6, 3, 3))
    return s


def make_bullet_sprite() -> pygame.Surface:
    s = pygame.Surface((6, 6), pygame.SRCALPHA)
    pygame.draw.rect(s, (255, 236, 140), (0, 0, 6, 6))
    pygame.draw.rect(s, (255, 170, 60), (1, 1, 4, 4))
    return s


def make_gun_sprite() -> pygame.Surface:
    """Simple blocky gun pointing right (for topdown/first person overlay)."""
    s = pygame.Surface((16, 8), pygame.SRCALPHA)
    pygame.draw.rect(s, (60, 60, 66), (0, 2, 12, 4))   # barrel
    pygame.draw.rect(s, (88, 60, 36), (10, 3, 6, 5))   # grip
    pygame.draw.rect(s, (40, 40, 46), (0, 1, 4, 6))    # muzzle
    return s


POWERUP_COLORS = {
    "LIFE": (235, 80, 90),
    "QUICKSHOT": (250, 210, 70),
    "SPEED": (90, 190, 250),
    "SWAP": (190, 120, 250),
}


def make_powerup_sprite(kind: str) -> pygame.Surface:
    color = POWERUP_COLORS[kind]
    s = pygame.Surface((TEX, TEX), pygame.SRCALPHA)
    dark = tuple(max(0, c - 60) for c in color)
    pygame.draw.rect(s, dark, (1, 1, 14, 14))
    pygame.draw.rect(s, color, (2, 2, 12, 12))
    c = (255, 255, 255)
    if kind == "LIFE":  # heart-ish
        pygame.draw.rect(s, c, (4, 5, 3, 3))
        pygame.draw.rect(s, c, (9, 5, 3, 3))
        pygame.draw.rect(s, c, (5, 8, 6, 3))
        pygame.draw.rect(s, c, (7, 11, 2, 2))
    elif kind == "QUICKSHOT":  # lightning
        pygame.draw.polygon(s, c, [(9, 2), (5, 9), (8, 9), (6, 14), (11, 7), (8, 7)])
    elif kind == "SPEED":  # chevrons
        pygame.draw.polygon(s, c, [(3, 4), (8, 8), (3, 12), (5, 12), (10, 8), (5, 4)])
        pygame.draw.polygon(s, c, [(8, 4), (13, 8), (8, 12), (10, 12), (15, 8), (10, 4)])
    else:  # SWAP: two arrows
        pygame.draw.polygon(s, c, [(3, 5), (7, 2), (7, 4), (12, 4), (12, 6), (7, 6), (7, 8)])
        pygame.draw.polygon(s, c, [(13, 11), (9, 14), (9, 12), (4, 12), (4, 10), (9, 10), (9, 8)])
    return s


def make_heart_sprite(filled: bool = True) -> pygame.Surface:
    s = pygame.Surface((TEX, TEX), pygame.SRCALPHA)
    color = (235, 80, 90) if filled else (50, 50, 56)
    pygame.draw.rect(s, color, (3, 4, 4, 4))
    pygame.draw.rect(s, color, (9, 4, 4, 4))
    pygame.draw.rect(s, color, (2, 6, 12, 4))
    pygame.draw.rect(s, color, (4, 10, 8, 2))
    pygame.draw.rect(s, color, (6, 12, 4, 2))
    if filled:
        pygame.draw.rect(s, (255, 160, 165), (4, 5, 2, 2))
    return s
