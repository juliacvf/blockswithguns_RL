"""Shared reward calculation for Gymnasium and PettingZoo environments."""

from __future__ import annotations


def rewards_from_events(events: list[tuple], winner: int | None) -> list[float]:
    """Return symmetric rewards for players 0 and 1."""
    rewards = [-0.005, -0.005]
    for event in events:
        if event[0] == "hit":
            victim, shooter = event[1], event[2]
            rewards[shooter] += 1.0
            rewards[victim] -= 1.0
        elif event[0] == "pickup":
            rewards[event[2]] += 0.5
        elif event[0] == "heat_hit":
            rewards[event[1]] -= 1.0
    if winner in (0, 1):
        rewards[winner] += 10.0
        rewards[1 - winner] -= 10.0
    return rewards


def winner_from_lives(lives: tuple[int, int]) -> int:
    """Resolve a time-limited match by lives, using -1 for a draw."""
    return -1 if lives[0] == lives[1] else (0 if lives[0] > lives[1] else 1)

