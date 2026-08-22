"""Evaluate an agent script (+ optional weights) against a built-in bot.

Run:  python -m rl.evaluate rl/example_agent.py --weights rl/qtable_weights.npz \
          --opponent astar --episodes 10
"""

from __future__ import annotations

import argparse

from .agent_interface import load_agent
from .env import BlocksWithGunsEnv


def evaluate(script: str, weights: str | None, opponent: str, episodes: int,
             max_seconds: float) -> dict:
    env = BlocksWithGunsEnv(opponent=opponent, max_seconds=max_seconds)
    agent = load_agent(script, weights_path=weights)
    results = {"wins": 0, "losses": 0, "draws": 0, "avg_reward": 0.0}
    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep)
        agent.reset()
        done = False
        total = 0.0
        while not done:
            obs, reward, terminated, truncated, info = env.step(agent.act(obs))
            total += reward
            done = terminated or truncated
        w = info["winner"]
        if w is None:  # truncated by the shorter cap: decide on lives
            l0, l1 = info["lives"]
            w = 0 if l0 > l1 else (1 if l1 > l0 else -1)
        results["wins"] += w == 0
        results["losses"] += w == 1
        results["draws"] += w == -1
        results["avg_reward"] += total / episodes
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--opponent", default="astar")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-seconds", type=float, default=60.0)
    args = ap.parse_args()
    res = evaluate(args.script, args.weights, args.opponent, args.episodes,
                   args.max_seconds)
    print(res)


if __name__ == "__main__":
    main()
