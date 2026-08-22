"""Quake-inspired software 3D polygon renderer (split-screen, stacked views).

Replaces the old DDA raycaster with a true polygon pipeline:
- perspective-correct floor casting (grass tiles, mud ponds, heat-zone haze)
- wall cells rendered as textured cuboids (exposed side quads, near-plane
  clipped, per-pixel z-buffer)
- trees / torches / powerups / bullets as alpha-tested billboards
- the opponent as a flat-shaded cube, the camera floating slightly above
  cube-top height with a slight downward pitch

Renders into a small numpy buffer (blocky pixels), then scales up.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from core import constants as C
from core.engine import Engine

from . import textures as T
from .render_topdown import PLAYER_COLORS

BUF_W = 288           # low-res buffer width per player view
FOV = math.radians(75.0)
EYE_H = 0.9           # camera height (player cubes are 0.62 tall)
PITCH = math.radians(7.0)     # slight downward tilt
MAX_DEPTH = 18.0
NEAR = 0.08
WALL_H = 1.0
CUBE_HALF = 0.31
CUBE_H = 0.62


def _surf_to_array(surf: pygame.Surface) -> np.ndarray:
    return pygame.surfarray.array3d(surf).astype(np.uint8)  # (x, y, 3)


class _Textures:
    def __init__(self):
        self.wall = _surf_to_array(T.cobble_texture())
        self.wall_mossy = _surf_to_array(T.cobble_texture(mossy=True))
        self.floor = _surf_to_array(T.grass_texture())
        self.mud = _surf_to_array(T.mud_texture())
        tree = T.make_tree_billboard()
        self.tree = pygame.surfarray.array3d(tree).astype(np.uint8)
        self.tree_mask = pygame.surfarray.array_alpha(tree) > 0
        self.torch = [pygame.surfarray.array3d(f).astype(np.uint8) for f in T.torch_frames()]
        self.torch_mask = [pygame.surfarray.array_alpha(f) > 0 for f in T.torch_frames()]
        self.powerup = {}
        self.powerup_mask = {}
        for k in C.POWERUP_TYPES:
            s = T.make_powerup_sprite(k)
            self.powerup[k] = pygame.surfarray.array3d(s).astype(np.uint8)
            self.powerup_mask[k] = pygame.surfarray.array_alpha(s) > 0


_TEX: _Textures | None = None


def tex() -> _Textures:
    global _TEX
    if _TEX is None:
        _TEX = _Textures()
    return _TEX


class _WorldGeo:
    """Static wall quads for a map, built once: exposed sides of wall cells."""

    def __init__(self, engine: Engine):
        grid = engine.grid
        n = C.GRID_SIZE
        verts, uvs, texs, normals, cents = [], [], [], [], []
        uv_side = [(0.0, 16.0), (16.0, 16.0), (16.0, 0.0), (0.0, 0.0)]
        for gx in range(n):
            for gz in range(n):
                if not grid[gx, gz]:
                    continue
                sides = []
                if gx + 1 < n and not grid[gx + 1, gz]:
                    sides.append(([(gx + 1, 0, gz), (gx + 1, 0, gz + 1),
                                   (gx + 1, WALL_H, gz + 1), (gx + 1, WALL_H, gz)],
                                  (1.0, 0.0, 0.0)))
                if gx - 1 >= 0 and not grid[gx - 1, gz]:
                    sides.append(([(gx, 0, gz + 1), (gx, 0, gz),
                                   (gx, WALL_H, gz), (gx, WALL_H, gz + 1)],
                                  (-1.0, 0.0, 0.0)))
                if gz + 1 < n and not grid[gx, gz + 1]:
                    sides.append(([(gx + 1, 0, gz + 1), (gx, 0, gz + 1),
                                   (gx, WALL_H, gz + 1), (gx + 1, WALL_H, gz + 1)],
                                  (0.0, 0.0, 1.0)))
                if gz - 1 >= 0 and not grid[gx, gz - 1]:
                    sides.append(([(gx, 0, gz), (gx + 1, 0, gz),
                                   (gx + 1, WALL_H, gz), (gx, WALL_H, gz)],
                                  (0.0, 0.0, -1.0)))
                mossy = (gx * 31 + gz * 17) % 100 < 15
                for quad, nrm in sides:
                    verts.append(quad)
                    uvs.append(uv_side)
                    texs.append(1 if mossy else 0)
                    normals.append(nrm)
                    cents.append(np.mean(np.array(quad), axis=0))
        self.verts = np.array(verts, dtype=np.float64)      # (M, 4, 3)
        self.uvs = np.array(uvs, dtype=np.float64)          # (M, 4, 2)
        self.texs = np.array(texs, dtype=np.int8)           # (M,)
        self.normals = np.array(normals, dtype=np.float64)  # (M, 3)
        self.cents = np.array(cents, dtype=np.float64)      # (M, 3)


class PolyView:
    """One player's first-person polygon view."""

    def __init__(self, engine: Engine, geo: _WorldGeo, w: int, h: int):
        self.engine = engine
        self.geo = geo
        self.w, self.h = w, h
        self.frame = np.zeros((w, h, 3), dtype=np.uint8)   # (x, y) like surfarray
        self.zbuf = np.full((w, h), MAX_DEPTH, dtype=np.float64)
        self.tan_h = math.tan(FOV / 2.0)
        self.tan_v = self.tan_h * h / w
        self.horizon = int(round(h * 0.5 * (1.0 - math.tan(PITCH) / self.tan_v)))
        self._sky = self._v_gradient(T.SKY, (140, 190, 235), max(1, self.horizon))
        self._cols_s = (2.0 * (np.arange(w) + 0.5) / w - 1.0) * self.tan_h

    @staticmethod
    def _v_gradient(top, bottom, rows: int) -> np.ndarray:
        t = np.linspace(0, 1, rows)[:, None]
        return (np.array(top) * (1 - t) + np.array(bottom) * t).astype(np.uint8)

    # ------------------------------------------------------------------ #
    def render(self, player_idx: int, tick: int, recoil: float = 0.0,
               bob: float = 0.0) -> np.ndarray:
        eng = self.engine
        pl = eng.players[player_idx]
        w, h = self.w, self.h
        self.frame[:, :self.horizon] = self._sky[None, :, :]
        self.frame[:, self.horizon:] = (96, 170, 80)
        self.zbuf[:] = MAX_DEPTH

        eye = EYE_H + math.sin(bob) * 0.03
        pitch = PITCH + recoil * 0.05
        cos_a, sin_a = math.cos(pl.aim), math.sin(pl.aim)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        self._draw_floor(pl.x, pl.y, eye, cos_a, sin_a, cos_p, sin_p)
        self._draw_walls(pl.x, pl.y, eye, cos_a, sin_a, cos_p, sin_p)
        self._draw_sprites(pl, player_idx, tick, eye, cos_a, sin_a, cos_p, sin_p)
        self._draw_cube(pl, 1 - player_idx, eye, cos_a, sin_a, cos_p, sin_p)
        return self.frame

    # ------------------------------------------------------------------ #
    def _draw_floor(self, px, pz, eye, cos_a, sin_a, cos_p, sin_p) -> None:
        """Perspective-correct ground: grass/mud tiles + heat haze.

        Fully vectorized: for a flat floor each screen row has a constant
        depth, and world X/Z vary linearly across the row.
        """
        t = tex()
        w, h = self.w, self.h
        rows = np.arange(self.horizon, h) + 0.5
        if len(rows) == 0:
            return
        tt = (h * 0.5 - rows) / (h * 0.5) * self.tan_v          # (R,)
        cz = eye * (cos_p + tt * sin_p) / (sin_p - tt * cos_p)  # depth per row
        valid = (cz > 0) & (cz <= MAX_DEPTH)
        out = np.empty((len(rows), w, 3), dtype=np.float64)
        out[:] = (96, 170, 80)   # base grass fill beyond MAX_DEPTH
        zc = np.where(valid, cz, MAX_DEPTH)
        self.zbuf[:, self.horizon:] = zc[None, :]
        if not valid.any():
            self.frame[:, self.horizon:] = out.astype(np.uint8).transpose(1, 0, 2)
            return
        czv = cz[valid]                                          # (Rv,)
        cx = self._cols_s[None, :] * czv[:, None]                # (Rv, w)
        wx = px + czv[:, None] * cos_a - cx * sin_a
        wz = pz + czv[:, None] * sin_a + cx * cos_a
        u = ((wx % 1.0) * 16).astype(int)
        v = ((wz % 1.0) * 16).astype(int)
        col = t.floor[u, v].astype(np.float64)                   # (Rv, w, 3)
        mud = self.engine.map.mud
        if mud is not None:
            gx = np.clip(wx.astype(int), 0, C.GRID_SIZE - 1)
            gz = np.clip(wz.astype(int), 0, C.GRID_SIZE - 1)
            mmask = mud[gx, gz] > 0
            if mmask.any():
                col[mmask] = t.mud[u[mmask], v[mmask]]
        # heat zone: haze outside the safe square, bright border line
        cen = C.WORLD * 0.5
        half = self.engine.safe_half()
        d = np.maximum(np.abs(wx - cen), np.abs(wz - cen))
        hot = d > half
        if hot.any():
            col[hot] = col[hot] * 0.45 + np.array((200, 40, 20)) * 0.55
        band = np.abs(d - half) < 0.3
        if band.any():
            col[band] = np.array((255, 120, 40))
        out[valid] = col
        self.frame[:, self.horizon:] = out.astype(np.uint8).transpose(1, 0, 2)

    # ------------------------------------------------------------------ #
    def _project(self, verts, px, pz, eye, cos_a, sin_a, cos_p, sin_p):
        """World (...,3) -> camera (cx, cy, cz) with yaw+pitch. No division."""
        dx = verts[..., 0] - px
        dz = verts[..., 2] - pz
        cz = dx * cos_a + dz * sin_a
        cx = -dx * sin_a + dz * cos_a
        cy = verts[..., 1] - eye
        cy2 = cy * cos_p + cz * sin_p
        cz2 = cz * cos_p - cy * sin_p
        return np.stack([cx, cy2, cz2], axis=-1)

    def _clip_near(self, verts: list[tuple]) -> list[tuple]:
        """Sutherland-Hodgman against the near plane cz >= NEAR (cam space)."""
        out = []
        n = len(verts)
        for i in range(n):
            a, b = verts[i], verts[(i + 1) % n]
            ain, bin_ = a[2] >= NEAR, b[2] >= NEAR
            if ain:
                out.append(a)
            if ain != bin_:
                f = (NEAR - a[2]) / (b[2] - a[2])
                out.append(tuple(a[j] + f * (b[j] - a[j]) for j in range(5)))
        return out

    def _raster_poly(self, poly: list[tuple], texture: np.ndarray | None,
                     amask: np.ndarray | None, color, shade: float) -> None:
        """poly: clipped cam-space verts (cx, cy, cz, u, v). Fan-triangulate."""
        w, h = self.w, self.h
        w2, h2 = w * 0.5, h * 0.5
        pts = []
        for cx, cy, cz, u, v in poly:
            iz = 1.0 / cz
            sx = w2 * (1.0 + cx * iz / self.tan_h)
            sy = h2 * (1.0 - cy * iz / self.tan_v)
            pts.append((sx, sy, iz, u * iz, v * iz))
        for i in range(1, len(pts) - 1):
            tri = (pts[0], pts[i], pts[i + 1])
            (ax, ay, aiz, au, av), (bx, by, biz, bu, bv), (cx2, cy2, ciz, cu, cv) = tri
            x0 = max(0, int(min(ax, bx, cx2)))
            x1 = min(w, int(max(ax, bx, cx2)) + 1)
            y0 = max(0, int(min(ay, by, cy2)))
            y1 = min(h, int(max(ay, by, cy2)) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            det = (by - cy2) * (ax - cx2) + (cx2 - bx) * (ay - cy2)
            if abs(det) < 1e-9:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5,
                                 np.arange(y0, y1) + 0.5, indexing="ij")
            la = ((by - cy2) * (gx - cx2) + (cx2 - bx) * (gy - cy2)) / det
            lb = ((cy2 - ay) * (gx - cx2) + (ax - cx2) * (gy - cy2)) / det
            lc = 1.0 - la - lb
            mask = (la >= -1e-6) & (lb >= -1e-6) & (lc >= -1e-6)
            if not mask.any():
                continue
            iz = la * aiz + lb * biz + lc * ciz
            z = 1.0 / iz
            if texture is not None:
                u = (la * au + lb * bu + lc * cu) / iz
                v = (la * av + lb * bv + lc * cv) / iz
                ti = np.clip(u.astype(int), 0, texture.shape[0] - 1)
                tj = np.clip(v.astype(int), 0, texture.shape[1] - 1)
                col = texture[ti, tj].astype(np.float64)
                if amask is not None:
                    mask &= amask[ti, tj]
                    if not mask.any():
                        continue
            else:
                col = np.empty(gx.shape + (3,))
                col[:] = color
            col = col * shade
            zreg = self.zbuf[x0:x1, y0:y1]
            draw = mask & (z < zreg)
            if draw.any():
                self.frame[x0:x1, y0:y1][draw] = col.astype(np.uint8)[draw]
                zreg[draw] = z[draw]

    # ------------------------------------------------------------------ #
    def _draw_walls(self, px, pz, eye, cos_a, sin_a, cos_p, sin_p) -> None:
        geo = self.geo
        cam = np.array([px, eye, pz])
        d = geo.cents - cam
        dist2 = d[:, 0] ** 2 + d[:, 2] ** 2
        facing = (geo.normals * d).sum(axis=1) < 0  # front-facing only
        keep = (dist2 < MAX_DEPTH * MAX_DEPTH) & facing
        idxs = np.nonzero(keep)[0]
        # far-to-near order is irrelevant with a z-buffer; near first is faster
        idxs = idxs[np.argsort(dist2[idxs])]
        t = tex()
        for i in idxs:
            cv = self._project(geo.verts[i], px, pz, eye, cos_a, sin_a, cos_p, sin_p)
            poly5 = [(cv[j, 0], cv[j, 1], cv[j, 2],
                      geo.uvs[i, j, 0], geo.uvs[i, j, 1]) for j in range(4)]
            poly = self._clip_near(poly5)
            if len(poly) < 3:
                continue
            nx = geo.normals[i]
            shade = 1.0 if nx[0] > 0 else (0.85 if nx[0] < 0 else
                                           (0.92 if nx[2] > 0 else 0.78))
            self._raster_poly(poly, t.wall_mossy if geo.texs[i] else t.wall,
                              None, None, shade)

    # ------------------------------------------------------------------ #
    def _draw_billboard(self, wx, wz, base, height, width, texture, amask,
                        px, pz, eye, cos_a, sin_a, cos_p, sin_p,
                        brightness: float = 1.0) -> None:
        # camera-right direction in world space (X, Z)
        rx, rz = -sin_a * width * 0.5, cos_a * width * 0.5
        verts = [(wx - rx, base, wz - rz), (wx + rx, base, wz + rz),
                 (wx + rx, base + height, wz + rz), (wx - rx, base + height, wz - rz)]
        cv = self._project(np.array(verts, dtype=np.float64), px, pz, eye,
                           cos_a, sin_a, cos_p, sin_p)
        tw, th = texture.shape[0], texture.shape[1]
        poly5 = [(cv[j, 0], cv[j, 1], cv[j, 2],
                  (0, tw - 1, tw - 1, 0)[j], (th - 1, th - 1, 0, 0)[j])
                 for j in range(4)]
        poly = self._clip_near(poly5)
        if len(poly) >= 3:
            self._raster_poly(poly, texture, amask, None, brightness)

    def _draw_sprites(self, pl, player_idx, tick, eye, cos_a, sin_a, cos_p, sin_p) -> None:
        eng = self.engine
        t = tex()
        px, pz = pl.x, pl.y
        args = (px, pz, eye, cos_a, sin_a, cos_p, sin_p)
        r2 = MAX_DEPTH * MAX_DEPTH

        for tx_, ty in eng.map.trees:
            if (tx_ - px) ** 2 + (ty - pz) ** 2 > r2:
                continue
            self._draw_billboard(tx_, ty, 0.0, 1.5, 1.0, t.tree, t.tree_mask, *args)
        torch_frame = tick // 8 % 3
        for tx_, ty in eng.map.torches:
            if (tx_ - px) ** 2 + (ty - pz) ** 2 > r2:
                continue
            self._draw_billboard(tx_ + 0.5, ty + 0.5, 0.35, 0.55, 0.55,
                                 t.torch[torch_frame], t.torch_mask[torch_frame],
                                 *args, brightness=1.15)
        pulse = 0.55 + 0.1 * math.sin(tick * 0.15)
        for p in eng.powerups.items:
            if (p.x - px) ** 2 + (p.y - pz) ** 2 > r2:
                continue
            self._draw_billboard(p.x, p.y, 0.0, pulse, pulse,
                                 t.powerup[p.kind], t.powerup_mask[p.kind], *args)
        for b in eng.bullets:
            if (b.x - px) ** 2 + (b.y - pz) ** 2 > r2:
                continue
            img = np.full((4, 4, 3), (255, 226, 120), dtype=np.uint8)
            img[1:3, 1:3] = (255, 180, 80)
            msk = np.ones((4, 4), dtype=bool)
            self._draw_billboard(b.x, b.y, 0.3, 0.22, 0.22, img, msk,
                                 *args, brightness=1.2)

    # ------------------------------------------------------------------ #
    def _draw_cube(self, viewer, other_idx: int, eye, cos_a, sin_a, cos_p, sin_p) -> None:
        """The opponent as a simple flat-shaded cube (+ aim nub)."""
        other = self.engine.players[other_idx]
        ox, oz = other.x, other.y
        hlf, top = CUBE_HALF, CUBE_H
        v = [(ox - hlf, 0, oz - hlf), (ox + hlf, 0, oz - hlf),
             (ox + hlf, 0, oz + hlf), (ox - hlf, 0, oz + hlf),
             (ox - hlf, top, oz - hlf), (ox + hlf, top, oz - hlf),
             (ox + hlf, top, oz + hlf), (ox - hlf, top, oz + hlf)]
        faces = [  # (quad indices, normal, shade)
            ((4, 5, 6, 7), (0, 1, 0), 1.0),    # top
            ((1, 2, 6, 5), (1, 0, 0), 0.88),   # +X
            ((3, 0, 4, 7), (-1, 0, 0), 0.72),  # -X
            ((2, 3, 7, 6), (0, 0, 1), 0.80),   # +Z
            ((0, 1, 5, 4), (0, 0, -1), 0.64),  # -Z
        ]
        color = np.array(PLAYER_COLORS[other_idx], dtype=np.float64)
        px, pz = viewer.x, viewer.y
        for quad, nrm, shade in faces:
            cent = np.mean([v[i] for i in quad], axis=0)
            if np.dot(np.array(nrm, dtype=np.float64),
                      cent - np.array([px, eye, pz])) >= 0:
                continue  # backface
            cv = self._project(np.array([v[i] for i in quad], dtype=np.float64),
                               px, pz, eye, cos_a, sin_a, cos_p, sin_p)
            poly5 = [(cv[j, 0], cv[j, 1], cv[j, 2], 0.0, 0.0) for j in range(4)]
            poly = self._clip_near(poly5)
            if len(poly) >= 3:
                self._raster_poly(poly, None, None, color, shade)
        # dark nub showing where the cube is aiming
        mx = ox + math.cos(other.aim) * (hlf + 0.14)
        mz = oz + math.sin(other.aim) * (hlf + 0.14)
        nub = np.full((4, 4, 3), (38, 38, 44), dtype=np.uint8)
        msk = np.ones((4, 4), dtype=bool)
        self._draw_billboard(mx, mz, top * 0.45, 0.14, 0.14, nub, msk,
                             px, pz, eye, cos_a, sin_a, cos_p, sin_p)


class SplitScreenPoly:
    """Two first-person polygon views + gun overlays, scaled up blocky."""

    def __init__(self, engine: Engine, view_w: int = 512, view_h: int = 512):
        self.engine = engine
        self.view_w = view_w
        self.view_h = view_h
        geo = _WorldGeo(engine)
        buf_h = max(32, int(round(BUF_W * view_h / max(1, view_w))))
        self.views = [PolyView(engine, geo, BUF_W, buf_h),
                      PolyView(engine, geo, BUF_W, buf_h)]
        self.gun = T.make_gun_sprite()

    def render(self, surf: pygame.Surface, origins: tuple[tuple[int, int], tuple[int, int]],
               tick: int, recoil: tuple[float, float] = (0.0, 0.0),
               bob: tuple[float, float] = (0.0, 0.0)) -> None:
        for i in range(2):
            frame = self.views[i].render(i, tick, recoil[i], bob[i])
            view_surf = pygame.surfarray.make_surface(frame)
            view_surf = pygame.transform.scale(view_surf, (self.view_w, self.view_h))
            ox, oy = origins[i]
            surf.blit(view_surf, (ox, oy))
            self._draw_gun(surf, i, ox, oy, tick, recoil[i], bob[i])

    def _draw_gun(self, surf, i, ox, oy, tick, recoil, bob_phase) -> None:
        vw, vh = self.view_w, self.view_h
        gw = int(vh * 0.84)
        gun = pygame.transform.scale(self.gun, (gw, int(gw * 0.5)))
        if i == 1:
            gun = pygame.transform.flip(gun, True, False)
        bob_y = int(math.sin(bob_phase) * vh * 0.02)
        kick = int(recoil * vh * 0.12)
        gx = ox + vw // 2 - gw // 2 + (kick if i == 0 else -kick)
        gy = oy + vh - int(gw * 0.28) + bob_y + int(recoil * vh * 0.1)
        surf.blit(gun, (gx, gy))
        pl = self.engine.players[i]
        if self.engine.time - pl.last_shot_time < 0.07:
            fx = ox + vw // 2 + (gw // 2 + kick if i == 0 else -(gw // 2 + kick))
            fy = gy + int(gw * 0.06)
            s = int(vh * 0.1) + (tick % 2) * 3
            pygame.draw.rect(surf, (255, 240, 160), (fx - s // 2, fy - s // 2, s, s))
            pygame.draw.rect(surf, (255, 200, 90), (fx - s // 4, fy - s // 4, s // 2, s // 2))
        # crosshair
        cx, cy = ox + vw // 2, oy + vh // 2
        c = (255, 255, 255)
        pygame.draw.line(surf, c, (cx - 6, cy), (cx - 2, cy))
        pygame.draw.line(surf, c, (cx + 2, cy), (cx + 6, cy))
        pygame.draw.line(surf, c, (cx, cy - 6), (cx, cy - 2))
        pygame.draw.line(surf, c, (cx, cy + 2), (cx, cy + 6))
