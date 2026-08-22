"""PettingZoo simultaneous-action environment for competitive self-play.

Run a random-policy smoke match with:
    python -m rl.pettingzoo_env
"""

from __future__ import annotations

from functools import lru_cache

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec

from core import constants as C
from core.engine import Engine
from rl.actions import contest_action_space, to_engine_action
from rl.full_observation import FullObservationEncoder, full_observation_space
from rl.rendering import TrainingRenderer
from rl.rewards import rewards_from_events, winner_from_lives

AGENTS = ("player_0", "player_1")


class BlocksWithGunsParallelEnv(ParallelEnv):
    """Two-agent PettingZoo ParallelEnv; both policies act every engine tick."""

    metadata = {
        "name": "blocks_with_guns_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
        "is_parallelizable": True,
    }

    def __init__(
        self,
        map_seed: int = 0,
        scale: float = 24.0,
        octaves: int = 3,
        threshold: float = 0.62,
        max_seconds: float = C.MAX_EPISODE_SECONDS,
        render_mode: str | None = None,
    ):
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        self.possible_agents = list(AGENTS)
        self.agents: list[str] = []
        self.map_params = {
            "map_seed": map_seed, "scale": scale, "octaves": octaves,
            "threshold": threshold,
        }
        self.max_seconds = float(max_seconds)
        self.render_mode = render_mode
        self.engine: Engine | None = None
        self._encoder: FullObservationEncoder | None = None
        self._action_spaces = {agent: contest_action_space() for agent in AGENTS}
        self._observation_spaces = {
            agent: full_observation_space() for agent in AGENTS
        }
        self._renderer = TrainingRenderer(render_mode)

    @lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return self._action_spaces[agent]

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None):
        del options
        match_seed = self.map_params["map_seed"] if seed is None else seed
        self.engine = Engine(
            seed=match_seed,
            map_seed=match_seed,
            scale=self.map_params["scale"],
            octaves=self.map_params["octaves"],
            threshold=self.map_params["threshold"],
        )
        self._encoder = FullObservationEncoder(self.engine, self.max_seconds)
        self.agents = list(self.possible_agents)
        for offset, agent in enumerate(self.possible_agents):
            self.action_space(agent).seed(match_seed + offset)
        observations = {
            agent: self._encoder.encode(idx)
            for idx, agent in enumerate(self.agents)
        }
        infos = {
            agent: {"map_seed": match_seed, "player_index": idx}
            for idx, agent in enumerate(self.agents)
        }
        if self.render_mode == "human":
            self.render()
        return observations, infos

    def step(self, actions: dict[str, object]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        acting_agents = list(self.agents)
        missing = set(acting_agents) - set(actions)
        extra = set(actions) - set(acting_agents)
        if missing or extra:
            raise ValueError(f"actions must match active agents; missing={missing}, extra={extra}")

        engine_actions = tuple(
            to_engine_action(actions[agent], name=agent) for agent in acting_agents)
        events = self.engine.step(engine_actions)
        natural_end = self.engine.winner is not None
        time_limit = not natural_end and self.engine.time >= self.max_seconds - 1e-9
        lives = tuple(player.lives for player in self.engine.players)
        result = self.engine.winner if natural_end else (
            winner_from_lives(lives) if time_limit else None)
        scored_winner = result if time_limit else self.engine.winner
        reward_values = rewards_from_events(events, scored_winner)

        rewards = {
            agent: reward_values[idx] for idx, agent in enumerate(acting_agents)
        }
        terminations = {agent: natural_end for agent in acting_agents}
        truncations = {agent: time_limit for agent in acting_agents}
        infos = {
            agent: {
                "winner": result,
                "time": self.engine.time,
                "lives": lives,
                "events": tuple(events),
                "player_index": idx,
            }
            for idx, agent in enumerate(acting_agents)
        }

        observations = {
            agent: self._encoder.encode(idx)
            for idx, agent in enumerate(acting_agents)
        }
        done = natural_end or time_limit
        if done:
            self.agents = []
        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.engine is None:
            return None
        return self._renderer.render(self.engine)

    def close(self) -> None:
        self._renderer.close()


def parallel_env(**kwargs) -> BlocksWithGunsParallelEnv:
    """PettingZoo-standard constructor for simultaneous self-play."""
    return BlocksWithGunsParallelEnv(**kwargs)


def env(**kwargs):
    """AEC conversion for frameworks that require the turn-based API."""
    return parallel_to_aec(parallel_env(**kwargs))


if __name__ == "__main__":
    game = parallel_env(max_seconds=5.0, render_mode="human")
    observations, _ = game.reset(seed=0)
    while game.agents:
        sampled = {agent: game.action_space(agent).sample() for agent in game.agents}
        observations, rewards, terms, truncs, infos = game.step(sampled)
    game.close()
