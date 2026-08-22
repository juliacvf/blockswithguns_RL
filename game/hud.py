"""HUD: hearts, powerup timers, cooldown bar."""

from __future__ import annotations

import pygame

from core import constants as C

from . import textures as T

POWERUP_LABELS = {
    "QUICKSHOT": ("QUICK", T.POWERUP_COLORS["QUICKSHOT"]),
    "SPEED": ("SPEED", T.POWERUP_COLORS["SPEED"]),
}


class Hud:
    def __init__(self):
        self.heart_full = T.make_heart_sprite(True)
        self.heart_empty = T.make_heart_sprite(False)
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 48)
        self.powerup_icons = {k: T.make_powerup_sprite(k) for k in C.POWERUP_TYPES}

    def draw_hearts(self, surf: pygame.Surface, x: int, y: int, lives: int,
                    align_right: bool = False, tick: int = 0, pulse: bool = False) -> None:
        size = 22
        if pulse and lives > 0:  # pulse the newest lost heart position
            size = 22 + int(2 * abs(((tick % 30) / 30.0) - 0.5))
        for i in range(C.PLAYER_LIVES):
            img = self.heart_full if i < lives else self.heart_empty
            img = pygame.transform.scale(img, (size, size))
            hx = x - i * (size + 4) - size if align_right else x + i * (size + 4)
            surf.blit(img, (hx, y))

    def draw_powerups(self, surf: pygame.Surface, x: int, y: int, player) -> None:
        ox = x
        if player.quickshot_timer > 0:
            self._timer_chip(surf, ox, y, "QUICKSHOT", player.quickshot_timer,
                             C.QUICKSHOT_DURATION)
            ox += 92
        if player.speed_timer > 0:
            self._timer_chip(surf, ox, y, "SPEED", player.speed_timer,
                             C.SPEED_DURATION)

    def _timer_chip(self, surf, x, y, kind, remaining, total) -> None:
        icon = pygame.transform.scale(self.powerup_icons[kind], (18, 18))
        surf.blit(icon, (x, y))
        frac = max(0.0, remaining / total)
        pygame.draw.rect(surf, (40, 40, 46), (x + 22, y + 4, 60, 10))
        pygame.draw.rect(surf, T.POWERUP_COLORS[kind], (x + 22, y + 4, int(60 * frac), 10))

    def draw_cooldown(self, surf: pygame.Surface, center, frac: float) -> None:
        """Small arc-ish bar under the crosshair."""
        w = 28
        x, y = center[0] - w // 2, center[1] + 12
        pygame.draw.rect(surf, (40, 40, 46), (x, y, w, 4))
        if frac > 0:
            pygame.draw.rect(surf, (250, 210, 70), (x, y, int(w * (1 - frac)), 4))
        else:
            pygame.draw.rect(surf, (120, 230, 120), (x, y, w, 4))

    def text(self, surf, s, x, y, color=(255, 255, 255), big=False, center=False) -> None:
        font = self.big_font if big else self.font
        img = font.render(s, True, color)
        rect = img.get_rect()
        if center:
            rect.center = (x, y)
        else:
            rect.topleft = (x, y)
        surf.blit(img, rect)
