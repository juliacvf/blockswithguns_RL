"""Blocky animation helpers: square particles, screen shake, flashes."""

from __future__ import annotations

import math
import random

import pygame


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "size")

    def __init__(self, x, y, vx, vy, life, color, size):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.life = self.max_life = life
        self.color = color
        self.size = size


class ParticleSystem:
    """World-space square particles (world units)."""

    def __init__(self):
        self.parts: list[Particle] = []
        self.rng = random.Random(1234)

    def burst(self, x: float, y: float, color, count: int = 14,
              speed: float = 6.0, life: float = 0.5, size: float = 0.22) -> None:
        for _ in range(count):
            a = self.rng.uniform(0, 6.2832)
            sp = self.rng.uniform(0.3, 1.0) * speed
            self.parts.append(Particle(
                x, y, math.cos(a) * sp, math.sin(a) * sp,
                self.rng.uniform(0.6, 1.0) * life, color,
                size * self.rng.uniform(0.7, 1.3),
            ))

    def update(self, dt: float) -> None:
        for p in self.parts:
            p.life -= dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vx *= 0.92
            p.vy *= 0.92
        self.parts = [p for p in self.parts if p.life > 0]

    def draw(self, surf: pygame.Surface, to_px, cam_x: float = 0.0, cam_y: float = 0.0) -> None:
        """to_px: world units -> pixels scale. cam: world offset of view."""
        for p in self.parts:
            t = p.life / p.max_life
            s = max(1, int(p.size * to_px * t))
            px = int((p.x - cam_x) * to_px)
            py = int((p.y - cam_y) * to_px)
            pygame.draw.rect(surf, p.color, (px - s // 2, py - s // 2, s, s))


class ScreenShake:
    def __init__(self):
        self.trauma = 0.0
        self.rng = random.Random(99)

    def add(self, amount: float = 0.5) -> None:
        self.trauma = min(1.0, self.trauma + amount)

    def update(self, dt: float) -> None:
        self.trauma = max(0.0, self.trauma - dt * 2.2)

    def offset(self) -> tuple[int, int]:
        if self.trauma <= 0:
            return 0, 0
        mag = self.trauma * self.trauma * 8
        return (int(self.rng.uniform(-mag, mag)), int(self.rng.uniform(-mag, mag)))


class Flash:
    """Timed full-view color flash (hit = red, pickup = white)."""

    def __init__(self):
        self.timer = 0.0
        self.duration = 0.18
        self.color = (255, 60, 60)
        self.max_alpha = 90

    def trigger(self, color=(255, 60, 60), duration: float = 0.18, alpha: int = 90) -> None:
        self.color = color
        self.duration = duration
        self.timer = duration
        self.max_alpha = alpha

    def update(self, dt: float) -> None:
        self.timer = max(0.0, self.timer - dt)

    def draw(self, surf: pygame.Surface) -> None:
        if self.timer <= 0:
            return
        alpha = int(self.max_alpha * (self.timer / self.duration))
        overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        overlay.fill((*self.color, alpha))
        surf.blit(overlay, (0, 0))
