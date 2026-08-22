"""Interactive pygame viewer for human, contest, and built-in controllers.

Examples:
    python -m rl.visualize --player1 custom --player2 custom
    python -m rl.visualize --player1 custom --player2 markov
    python -m rl.visualize --player1 human --player2 custom
"""

from __future__ import annotations

import argparse

from bots import BOT_NAMES, make_bot, make_view
from core import constants as C
from core.engine import Action, Engine
from game.app import HumanController
from game.render_topdown import PLAYER_COLORS, TopdownRenderer
from rl.actions import to_engine_action
from rl.contest import ContestEntry, load_contest_entry
from rl.full_observation import FullObservationEncoder
from rl.rewards import winner_from_lives


def _controller(spec: str, idx: int, folder: str, seed: int):
    if spec == "human":
        if idx != 0:
            raise ValueError("the human controller is supported in player 1 only")
        return HumanController("topdown")
    if spec == "custom":
        return load_contest_entry(folder, idx + 1)
    if spec in BOT_NAMES:
        return make_bot(spec, seed=seed)
    raise ValueError(f"unknown controller {spec!r}")


def _action(controller, engine: Engine, encoder: FullObservationEncoder,
            idx: int, origin: tuple[int, int]) -> Action:
    if isinstance(controller, HumanController):
        return controller.build(engine, ppc=6.0, origin=origin)
    if isinstance(controller, ContestEntry):
        return to_engine_action(
            controller.act(encoder.encode(idx)), name=f"algo{idx + 1}")
    return controller.act(make_view(engine, idx))


def run_visual_match(
    player1: str = "custom",
    player2: str = "custom",
    algo1_folder: str = "algo1",
    algo2_folder: str = "algo2",
    seed: int = 100,
    max_seconds: float = C.MAX_EPISODE_SECONDS,
    fps: int = 60,
    linger: bool = True,
) -> dict:
    """Run one rendered match and return its winner/lives summary."""
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((900, 640))
    pygame.display.set_caption("Blocks With Guns - Contest Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    small = pygame.font.Font(None, 22)
    engine = Engine(seed=seed, map_seed=seed)
    encoder = FullObservationEncoder(engine, max_seconds)
    controllers = [
        _controller(player1, 0, algo1_folder, seed + 11),
        _controller(player2, 1, algo2_folder, seed + 22),
    ]
    for controller in controllers:
        if hasattr(controller, "reset"):
            controller.reset()
    renderer = TopdownRenderer(engine, px_per_cell=6)
    origin = (20, 20)
    accumulator = 0.0
    running = True
    finished = False
    result = None

    if isinstance(controllers[0], HumanController):
        pygame.mouse.set_visible(False)

    while running:
        frame_dt = min(clock.tick(fps) / 1000.0, 0.1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif finished and event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                running = False
            for controller in controllers:
                if isinstance(controller, HumanController):
                    controller.on_event(event)

        accumulator = min(accumulator + frame_dt, 5 * C.FIXED_DT)
        while not finished and accumulator >= C.FIXED_DT:
            accumulator -= C.FIXED_DT
            actions = tuple(
                _action(controller, engine, encoder, idx, origin)
                for idx, controller in enumerate(controllers)
            )
            engine.step(actions)
            if engine.winner is not None or engine.time >= max_seconds - 1e-9:
                lives = tuple(player.lives for player in engine.players)
                result = engine.winner if engine.winner is not None else winner_from_lives(lives)
                finished = True
                if not linger:
                    running = False

        screen.fill((22, 26, 22))
        renderer.draw(screen, tick=engine.tick, offset=origin)
        pygame.draw.rect(screen, (35, 40, 38), (640, 0, 260, 640))
        title = font.render("ALGO BATTLE", True, (255, 220, 100))
        screen.blit(title, (665, 30))
        names = (player1.upper(), player2.upper())
        for idx, player in enumerate(engine.players):
            y = 105 + idx * 130
            label = font.render(f"P{idx + 1}  {names[idx]}", True, PLAYER_COLORS[idx])
            screen.blit(label, (660, y))
            lives = small.render(f"Lives: {player.lives}/{C.PLAYER_LIVES}", True, (235, 235, 235))
            screen.blit(lives, (660, y + 34))
            status = small.render(
                f"x={player.x:05.1f}  y={player.y:05.1f}", True, (190, 205, 195))
            screen.blit(status, (660, y + 60))
        timer = font.render(f"TIME  {engine.time:05.1f}s", True, (230, 230, 230))
        screen.blit(timer, (660, 390))
        if player1 == "human":
            for line_idx, text in enumerate(("WASD: move", "Mouse: aim", "Click / E: shoot")):
                screen.blit(small.render(text, True, (185, 205, 185)),
                            (660, 455 + line_idx * 24))
        if finished:
            verdict = "DRAW" if result == -1 else f"PLAYER {result + 1} WINS"
            screen.blit(font.render(verdict, True, (255, 220, 100)), (660, 550))
            if linger:
                screen.blit(small.render("Enter to close", True, (210, 210, 210)), (660, 585))
        pygame.display.flip()

    pygame.mouse.set_visible(True)
    pygame.quit()
    lives = tuple(player.lives for player in engine.players)
    if result is None:
        result = winner_from_lives(lives)
    return {"winner": result, "lives": lives, "time": round(engine.time, 3)}


def main() -> None:
    choices = ["custom", *BOT_NAMES]
    parser = argparse.ArgumentParser(description="Visualize a contest or human match")
    parser.add_argument("--player1", choices=["human", *choices], default="custom")
    parser.add_argument("--player2", choices=choices, default="custom")
    parser.add_argument("--algo1-folder", default="algo1")
    parser.add_argument("--algo2-folder", default="algo2")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max-seconds", type=float, default=C.MAX_EPISODE_SECONDS)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--exit-on-finish", action="store_true")
    args = parser.parse_args()
    summary = run_visual_match(
        player1=args.player1, player2=args.player2,
        algo1_folder=args.algo1_folder, algo2_folder=args.algo2_folder,
        seed=args.seed, max_seconds=args.max_seconds, fps=args.fps,
        linger=not args.exit_on_finish)
    print(summary)


if __name__ == "__main__":
    main()
