"""Classic topdown renderer (blocky bright dungeon) + minimap widget."""

from __future__ import annotations

import math

import pygame

from core import constants as C
from core.engine import Engine

from . import textures as T

PLAYER_COLORS = [(90, 170, 250), (250, 120, 110)]  # P1 blue, P2 red


class TopdownRenderer:
    """Renders the whole map (like the original game) at px_per_cell scale."""

    def __init__(self, engine: Engine, px_per_cell: int = 4):
        self.engine = engine
        self.ppc = px_per_cell
        self.size = C.GRID_SIZE * px_per_cell
        self._static: pygame.Surface | None = None
        self.player_sprites = [T.make_player_sprite(c) for c in PLAYER_COLORS]
        self.bullet_sprite = T.make_bullet_sprite()
        self.gun_sprite = T.make_gun_sprite()
        self.tree_sprite = T.make_tree_topdown()
        self.powerup_sprites = {k: T.make_powerup_sprite(k) for k in C.POWERUP_TYPES}

    # ------------------------------------------------------------------ #
    def _build_static(self) -> pygame.Surface:
        ppc = self.ppc
        surf = pygame.Surface((self.size, self.size))
        grass = pygame.transform.scale(T.grass_texture(), (ppc, ppc))
        mud = pygame.transform.scale(T.mud_texture(), (ppc, ppc))
        cobble = pygame.transform.scale(T.cobble_texture(), (ppc, ppc))
        mossy = pygame.transform.scale(T.cobble_texture(mossy=True), (ppc, ppc))
        mudgrid = self.engine.map.mud
        for x in range(C.GRID_SIZE):
            for y in range(C.GRID_SIZE):
                if mudgrid is not None and mudgrid[x, y] and not self.engine.grid[x, y]:
                    surf.blit(mud, (x * ppc, y * ppc))
                else:
                    surf.blit(grass, (x * ppc, y * ppc))
        grid = self.engine.grid
        for x in range(C.GRID_SIZE):
            for y in range(C.GRID_SIZE):
                if grid[x, y]:
                    is_mossy = (x * 31 + y * 17) % 100 < 15
                    surf.blit(mossy if is_mossy else cobble, (x * ppc, y * ppc))
                    # bright top edge when the cell above is open (fake 3D block)
                    if y > 0 and not grid[x, y - 1]:
                        pygame.draw.line(surf, (210, 212, 218),
                                         (x * ppc, y * ppc), (x * ppc + ppc - 1, y * ppc))
        return surf

    # ------------------------------------------------------------------ #
    def draw(self, surf: pygame.Surface, particles=None, tick: int = 0,
             offset: tuple[int, int] = (0, 0)) -> None:
        if self._static is None:
            self._static = self._build_static()
        ox, oy = offset
        ppc = self.ppc
        surf.blit(self._static, (ox, oy))

        # torches (animated flame)
        frames = T.torch_frames()
        frame = frames[(tick // 8) % len(frames)]
        torch_img = pygame.transform.scale(frame, (ppc, ppc))
        for tx, ty in self.engine.map.torches:
            surf.blit(torch_img, (tx * ppc + ox, ty * ppc + oy))

        # trees (decorative)
        tree_img = pygame.transform.scale(self.tree_sprite, (ppc, ppc))
        for tx, ty in self.engine.map.trees:
            surf.blit(tree_img, (tx * ppc - ppc // 2 + ox, ty * ppc - ppc // 2 + oy))

        # powerups (pulsing)
        pulse = 1.0 + 0.2 * math.sin(tick * 0.15)
        for p in self.engine.powerups.items:
            img = self.powerup_sprites[p.kind]
            w = int(ppc * pulse)
            img2 = pygame.transform.scale(img, (w, w))
            surf.blit(img2, (p.x * ppc - w // 2 + ox, p.y * ppc - w // 2 + oy))

        # players + guns
        for i, pl in enumerate(self.engine.players):
            px, py = pl.x * ppc + ox, pl.y * ppc + oy
            hit_recent = self.engine.time - pl.last_hit_time < 0.15
            img = self.player_sprites[i]
            w = int(ppc * 2.2)
            body = pygame.transform.scale(img, (w, w))
            if hit_recent and (tick // 2) % 2 == 0:
                body = body.copy()
                body.fill((255, 90, 90, 160), special_flags=pygame.BLEND_RGBA_ADD)
            # gun rotated toward aim
            gw = int(ppc * 1.8)
            gun = pygame.transform.scale(self.gun_sprite, (gw, int(gw * 0.5)))
            gun = pygame.transform.rotate(gun, -math.degrees(pl.aim))
            gx = px + math.cos(pl.aim) * ppc * 0.6
            gy = py + math.sin(pl.aim) * ppc * 0.6
            surf.blit(gun, gun.get_rect(center=(gx, gy)))
            surf.blit(body, body.get_rect(center=(px, py)))
            # muzzle flash
            if self.engine.time - pl.last_shot_time < 0.08:
                mx = px + math.cos(pl.aim) * ppc * 1.2
                my = py + math.sin(pl.aim) * ppc * 1.2
                s = max(2, ppc // 2)
                pygame.draw.rect(surf, (255, 240, 150),
                                 (mx - s // 2, my - s // 2, s, s))

        # bullets
        bw = max(2, ppc // 2)
        bullet_img = pygame.transform.scale(self.bullet_sprite, (bw, bw))
        for b in self.engine.bullets:
            surf.blit(bullet_img, (b.x * ppc - bw // 2 + ox, b.y * ppc - bw // 2 + oy))

        if particles is not None:
            particles.draw(surf, ppc, cam_x=-ox / ppc, cam_y=-oy / ppc)

        self._draw_heat(surf, ox, oy)

    def _draw_heat(self, surf: pygame.Surface, ox: int, oy: int) -> None:
        """Red haze outside the safe square + its shrinking border."""
        half = self.engine.safe_half()
        c = C.WORLD * 0.5
        ppc = self.ppc
        x0, y0 = ox + (c - half) * ppc, oy + (c - half) * ppc
        x1, y1 = ox + (c + half) * ppc, oy + (c + half) * ppc
        haze = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        haze.fill((200, 40, 20, 70))
        if x1 > x0 and y1 > y0:
            # carve the safe square back out of the haze
            haze.fill((0, 0, 0, 0), pygame.Rect(x0, y0, x1 - x0, y1 - y0))
        surf.blit(haze, (0, 0))
        pygame.draw.rect(surf, (255, 120, 40),
                         pygame.Rect(x0, y0, max(1, x1 - x0), max(1, y1 - y0)), 2)


class Minimap:
    """Small live map for the bottom-right corner of the raycast view."""

    def __init__(self, engine: Engine, size: int = 200):
        self.engine = engine
        self.size = size
        self._static: pygame.Surface | None = None
        self.scale = size / C.GRID_SIZE

    def _build_static(self) -> pygame.Surface:
        s = pygame.Surface((self.size, self.size))
        s.fill((60, 130, 66))
        grid = self.engine.grid
        cell = max(1, int(self.scale))
        wall = pygame.Surface((cell, cell))
        wall.fill((150, 152, 158))
        mud = self.engine.map.mud
        for x in range(C.GRID_SIZE):
            for y in range(C.GRID_SIZE):
                if grid[x, y]:
                    s.blit(wall, (int(x * self.scale), int(y * self.scale)))
                elif mud is not None and mud[x, y]:
                    s.fill((122, 92, 58), (int(x * self.scale), int(y * self.scale),
                                           cell, cell))
        for tx, ty in self.engine.map.trees:
            s.fill((36, 92, 38), (int(tx * self.scale), int(ty * self.scale), 2, 2))
        return s

    def draw(self, surf: pygame.Surface, corner: tuple[int, int]) -> None:
        if self._static is None:
            self._static = self._build_static()
        x0, y0 = corner
        frame = pygame.Rect(x0 - 2, y0 - 2, self.size + 4, self.size + 4)
        pygame.draw.rect(surf, (30, 30, 36), frame)
        surf.blit(self._static, (x0, y0))
        s = self.scale
        for p in self.engine.powerups.items:
            col = T.POWERUP_COLORS[p.kind]
            pygame.draw.rect(surf, col, (x0 + p.x * s - 1, y0 + p.y * s - 1, 3, 3))
        for b in self.engine.bullets:
            pygame.draw.rect(surf, (255, 236, 140), (x0 + b.x * s, y0 + b.y * s, 2, 2))
        for i, pl in enumerate(self.engine.players):
            col = PLAYER_COLORS[i]
            pygame.draw.rect(surf, col, (x0 + pl.x * s - 2, y0 + pl.y * s - 2, 5, 5))
            ex = x0 + pl.x * s + math.cos(pl.aim) * 6
            ey = y0 + pl.y * s + math.sin(pl.aim) * 6
            pygame.draw.line(surf, col, (x0 + pl.x * s, y0 + pl.y * s), (ex, ey), 2)
        # heat zone: safe square border
        half = self.engine.safe_half()
        c = C.WORLD * 0.5
        pygame.draw.rect(surf, (255, 120, 40),
                         (x0 + (c - half) * s, y0 + (c - half) * s,
                          max(1, 2 * half * s), max(1, 2 * half * s)), 1)
