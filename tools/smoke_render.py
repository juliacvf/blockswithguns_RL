"""Headless smoke test: render both view modes and save screenshots."""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame  # noqa: E402

from game.app import App  # noqa: E402


def drive(app: App, frames: int, shoot_every: int = 0) -> None:
    for f in range(frames):
        for ev in pygame.event.get():
            if app.state == "play":
                for c in app.controllers:
                    if hasattr(c, "on_event"):
                        c.on_event(ev)
        if shoot_every and f % shoot_every == 0 and app.state == "play":
            ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(400, 300))
            for c in app.controllers:
                if hasattr(c, "on_event"):
                    c.on_event(ev)
        if app.state == "play":
            app._play_step(1.0 / 60.0)
        pygame.display.flip()


def main() -> None:
    app = App(sound=False, fps=0)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "replays")

    # --- topdown mode ---
    app.cfg["mode"] = "topdown"
    app.cfg["algo2"] = "strafe"
    app.start_match()
    drive(app, 60 * 14, shoot_every=45)   # ~14 simulated seconds: powerups spawned
    pygame.image.save(app.screen, os.path.join(out, "shot_topdown.png"))
    print("topdown ok, state:", app.state, "t=", round(app.engine.time, 1))

    # --- 3D polygon mode (algo vs algo spectate) ---
    app.cfg["mode"] = "poly"
    app.cfg["players"] = "ALGO VS ALGO"
    app.cfg["algo1"] = "astar"
    app.start_match()
    drive(app, 60 * 12, shoot_every=40)
    pygame.image.save(app.screen, os.path.join(out, "shot_poly.png"))
    print("poly ok, state:", app.state, "t=", round(app.engine.time, 1))
    app.cfg["players"] = "HUMAN VS ALGO"

    # --- menu ---
    app.end_to_menu()
    app._menu_draw()
    pygame.image.save(app.screen, os.path.join(out, "shot_menu.png"))
    print("menu ok")


if __name__ == "__main__":
    main()
