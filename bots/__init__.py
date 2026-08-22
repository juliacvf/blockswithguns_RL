"""Deterministic bots for Blocks With Guns RL."""

from .astar_bot import AStarBot
from .base import Bot, View, make_view
from .camper_bot import CamperBot
from .dijkstra_bot import DijkstraBot
from .markov_bot import MarkovBot
from .strafe_bot import StrafeBot

BOT_CLASSES = {
    "dijkstra": DijkstraBot,
    "astar": AStarBot,
    "strafe": StrafeBot,
    "camper": CamperBot,
    "markov": MarkovBot,
}

BOT_NAMES = list(BOT_CLASSES.keys())


def make_bot(name: str, seed: int = 0) -> Bot:
    return BOT_CLASSES[name](seed=seed)


__all__ = [
    "Bot", "View", "make_view", "make_bot", "BOT_CLASSES", "BOT_NAMES",
    "DijkstraBot", "AStarBot", "StrafeBot", "CamperBot", "MarkovBot",
]
