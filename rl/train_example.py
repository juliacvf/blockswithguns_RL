"""Tiny self-contained training example: tabular Q-learning vs a bot.

Run:  python -m rl.train_example --episodes 300 --opponent astar
Saves weights to rl/qtable_weights.npz, loadable by rl/example_agent.py.

This is deliberately dependency-light (gymnasium + numpy only). Swap in
stable-baselines3, RLlib, your own torch net, ... if you want more.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .env import AIM_BINS, BlocksWithGunsEnv
from .example_agent import discretize

N_ACTIONS = 9 * AIM_BINS * 2
WEIGHTS_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qtable_weights.npz")


def train(episodes: int, opponent: str, max_seconds: float, out: str) -> None:
    env = BlocksWithGunsEnv(opponent=opponent, max_seconds=max_seconds)
    q: dict[tuple, np.ndarray] = {}
    alpha, gamma = 0.2, 0.95
    eps_start, eps_end = 1.0, 0.05
    rng = np.random.default_rng(0)

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - ep / max(1, episodes * 0.8))
        done = False
        total = 0.0
        while not done:
            key = discretize(obs)
            if key not in q:
                q[key] = np.zeros(N_ACTIONS, dtype=np.float64)
            if rng.random() < eps:
                flat = int(rng.integers(N_ACTIONS))
            else:
                flat = int(np.argmax(q[key]))
            action = np.array([flat // (AIM_BINS * 2), (flat % (AIM_BINS * 2)) // 2, flat % 2])
            nobs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            nkey = discretize(nobs)
            if nkey not in q:
                q[nkey] = np.zeros(N_ACTIONS, dtype=np.float64)
            target = reward + (0.0 if terminated else gamma * float(np.max(q[nkey])))
            q[key][flat] += alpha * (target - q[key][flat])
            obs = nobs
            done = terminated or truncated
        if (ep + 1) % 25 == 0:
            print(f"episode {ep + 1}/{episodes}  return {total:+.1f}  states {len(q)}")

    np.savez_compressed(out, q=q)
    print("saved weights ->", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--opponent", type=str, default="astar")
    ap.add_argument("--max-seconds", type=float, default=45.0)
    ap.add_argument("--out", type=str, default=WEIGHTS_OUT)
    args = ap.parse_args()
    train(args.episodes, args.opponent, args.max_seconds, args.out)


if __name__ == "__main__":
    main()
