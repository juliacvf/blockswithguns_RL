"""Shared pygame renderer used by the training environments."""

from __future__ import annotations

import os

import numpy as np

from core.engine import Engine


class TrainingRenderer:
    """Lazy top-down renderer for human and rgb_array training modes."""

    def __init__(self, render_mode: str | None):
        self.render_mode = render_mode
        self._screen = None
        self._clock = None
        self._renderer = None
        self._engine = None

    def render(self, engine: Engine):
        if self.render_mode not in ("human", "rgb_array"):
            return None
        if self.render_mode != "human":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame
        from game.render_topdown import TopdownRenderer

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((640, 640))
                pygame.display.set_caption("Blocks With Guns RL Training")
                self._clock = pygame.time.Clock()
            else:
                self._screen = pygame.Surface((640, 640))
        if self._engine is not engine:
            self._engine = engine
            self._renderer = TopdownRenderer(engine, px_per_cell=6)

        self._screen.fill((24, 28, 24))
        self._renderer.draw(self._screen, tick=engine.tick, offset=(20, 20))
        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(30)
            return None
        return np.transpose(
            pygame.surfarray.array3d(self._screen), (1, 0, 2)).copy()

    def close(self) -> None:
        if self._screen is not None:
            import pygame
            pygame.quit()
        self._screen = None
        self._clock = None
        self._renderer = None
        self._engine = None
