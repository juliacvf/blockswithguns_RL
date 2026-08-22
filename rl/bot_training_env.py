"""Full-observation Gymnasium training environment versus built-in bots."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym

from bots import BOT_NAMES, make_bot, make_view
from core import constants as C
from core.engine import Engine
from rl.actions import contest_action_space, to_engine_action
from rl.full_observation import FullObservationEncoder, full_observation_space
from rl.rendering import TrainingRenderer
from rl.rewards import rewards_from_events, winner_from_lives


class BlocksWithGunsBotTrainingEnv(gym.Env):
    """Train player 0 against one bot or rotate through all five bots."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        opponent: str | Sequence[str] = "all",
        map_seed: int = 0,
        scale: float = 24.0,
        octaves: int = 3,
        threshold: float = 0.62,
        max_seconds: float = C.MAX_EPISODE_SECONDS,
        render_mode: str | None = None,
    ):
        super().__init__()
        if opponent == "all":
            names = list(BOT_NAMES)
        elif isinstance(opponent, str):
            names = [opponent]
        else:
            names = list(opponent)
        unknown = sorted(set(names) - set(BOT_NAMES))
        if not names or unknown:
            raise ValueError(f"opponents must come from {BOT_NAMES}; invalid={unknown}")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")

        self.opponent_names = names
        self.map_params = {
            "map_seed": map_seed, "scale": scale, "octaves": octaves,
            "threshold": threshold,
        }
        self.max_seconds = float(max_seconds)
        self.render_mode = render_mode
        self.action_space = contest_action_space()
        self.observation_space = full_observation_space()
        self.engine: Engine | None = None
        self._encoder: FullObservationEncoder | None = None
        self._opponent = None
        self._opponent_name = ""
        self._episode = 0
        self._renderer = TrainingRenderer(render_mode)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        match_seed = self.map_params["map_seed"] + self._episode if seed is None else seed
        opponent_idx = (self._episode if seed is None else seed) % len(self.opponent_names)
        self._opponent_name = self.opponent_names[opponent_idx]
        self._episode += 1
        self.engine = Engine(
            seed=match_seed,
            map_seed=match_seed,
            scale=self.map_params["scale"],
            octaves=self.map_params["octaves"],
            threshold=self.map_params["threshold"],
        )
        self._encoder = FullObservationEncoder(self.engine, self.max_seconds)
        self._opponent = make_bot(self._opponent_name, seed=match_seed + 1)
        self._opponent.reset()
        observation = self._encoder.encode(0)
        info = {"map_seed": match_seed, "opponent": self._opponent_name}
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        mine = to_engine_action(action, name="learning agent")
        theirs = self._opponent.act(make_view(self.engine, 1))
        events = self.engine.step((mine, theirs))
        natural_end = self.engine.winner is not None
        time_limit = not natural_end and self.engine.time >= self.max_seconds - 1e-9
        lives = tuple(player.lives for player in self.engine.players)
        result = self.engine.winner if natural_end else (
            winner_from_lives(lives) if time_limit else None)
        scored_winner = result if time_limit else self.engine.winner
        reward = rewards_from_events(events, scored_winner)[0]
        info = {
            "winner": result,
            "time": self.engine.time,
            "lives": lives,
            "events": tuple(events),
            "opponent": self._opponent_name,
        }
        observation = self._encoder.encode(0)
        if self.render_mode == "human":
            self.render()
        return observation, reward, natural_end, time_limit, info

    def render(self):
        if self.engine is None:
            return None
        return self._renderer.render(self.engine)

    def close(self) -> None:
        self._renderer.close()
