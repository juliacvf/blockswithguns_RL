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

from core import constants as C
from .env import AIM_BINS, BlocksWithGunsEnv
from .example_agent import discretize
from .progress import TrainingProgress

N_ACTIONS = 9 * AIM_BINS * 2
WEIGHTS_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qtable_weights.npz")


def _save_checkpoint(q: dict, out: str) -> None:
    np.savez_compressed(out, q=q)


def train(episodes: int, opponent: str, max_seconds: float, out: str) -> None:
    env = BlocksWithGunsEnv(opponent=opponent, max_seconds=max_seconds)
    q: dict[tuple, np.ndarray] = {}
    alpha, gamma = 0.2, 0.95
    eps_start, eps_end = 1.0, 0.05
    rng = np.random.default_rng(0)
    progress = TrainingProgress(episodes)
    print(f"training tabular Q-learning vs {opponent} "
          f"({episodes} full games, checkpoint after every game)")

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - ep / max(1, episodes * 0.8))
        done = False
        total = 0.0
        info: dict = {}
        while not done:
            key = discretize(obs)
            if key not in q:
                q[key] = np.zeros(N_ACTIONS, dtype=np.float64)
            if rng.random() < eps:
                flat = int(rng.integers(N_ACTIONS))
            else:
                flat = int(np.argmax(q[key]))
            action = np.array([flat // (AIM_BINS * 2), (flat % (AIM_BINS * 2)) // 2, flat % 2])
            nobs, reward, terminated, truncated, info = env.step(action)
            total += reward
            nkey = discretize(nobs)
            if nkey not in q:
                q[nkey] = np.zeros(N_ACTIONS, dtype=np.float64)
            target = reward + (0.0 if terminated else gamma * float(np.max(q[nkey])))
            q[key][flat] += alpha * (target - q[key][flat])
            obs = nobs
            done = terminated or truncated
        wins10 = progress.record(info.get("winner"))
        progress.show(ep + 1, wins10,
                      f"return {total:+8.2f}",
                      f"eps {eps:.3f}",
                      f"states {len(q)}")
        _save_checkpoint(q, out)
    progress.finish()
    print("result:", progress.summary())
    print("saved weights ->", out)
    env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--opponent", type=str, default="astar")
    ap.add_argument("--max-seconds", type=float, default=C.MAX_EPISODE_SECONDS,
                    help="safety cap; games normally end by KO or the engine's "
                         "own %.0fs limit" % C.MAX_EPISODE_SECONDS)
    ap.add_argument("--out", type=str, default=WEIGHTS_OUT)
    args = ap.parse_args()
    train(args.episodes, args.opponent, args.max_seconds, args.out)


if __name__ == "__main__":
    main()
