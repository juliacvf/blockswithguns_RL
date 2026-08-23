"""Train or evaluate the Algo Test Q-policy.

Run from the project root:
    .venv/bin/python "Algo Test/train.py" --episodes 250
    .venv/bin/python "Algo Test/train.py" --evaluate-only --opponent all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

from algo1 import Agent, NUM_OPTIONS, state_key  # noqa: E402
from bots import BOT_NAMES  # noqa: E402
from core import constants as C  # noqa: E402
from rl.bot_training_env import BlocksWithGunsBotTrainingEnv  # noqa: E402
from rl.progress import TrainingProgress  # noqa: E402

DEFAULT_WEIGHTS = HERE / "weights" / "qtable.npz"


def _shaping(previous: dict, current: dict) -> float:
    """Small learning hints; contest scoring still comes from the environment."""
    prev_self, now_self = previous["self"], current["self"]
    px, py = (float(prev_self[i] + 1.0) * 50.0 for i in (0, 1))
    pex, pey = (float(previous["opponent"][i] + 1.0) * 50.0 for i in (0, 1))
    nx, ny = (float(now_self[i] + 1.0) * 50.0 for i in (0, 1))
    nex, ney = (float(current["opponent"][i] + 1.0) * 50.0 for i in (0, 1))
    old_distance = ((pex - px) ** 2 + (pey - py) ** 2) ** 0.5
    new_distance = ((nex - nx) ** 2 + (ney - ny) ** 2) ** 0.5
    shaped = np.clip(old_distance - new_distance, -1.0, 1.0) * 0.003
    if current["self"][14] > 0 and new_distance <= 20.0:
        shaped += 0.002
    if current["self"][12] > 0:
        shaped -= 0.01
    return float(shaped)


def evaluate(q: np.ndarray, opponent: str, episodes: int,
             max_seconds: float, base_seed: int = 10_000) -> dict:
    opponents = BOT_NAMES if opponent == "all" else [opponent]
    results = {}
    for bot_name in opponents:
        env = BlocksWithGunsBotTrainingEnv(
            opponent=bot_name, max_seconds=max_seconds)
        policy = Agent()
        policy.q = q.copy()
        wins = losses = draws = 0
        total_reward = 0.0
        for episode in range(episodes):
            obs, _ = env.reset(seed=base_seed + episode)
            policy.reset()
            done = False
            while not done:
                obs, reward, terminated, truncated, info = env.step(policy.act(obs))
                total_reward += reward
                done = terminated or truncated
            winner = info["winner"]
            wins += winner == 0
            losses += winner == 1
            draws += winner == -1
        env.close()
        results[bot_name] = {
            "wins": wins, "losses": losses, "draws": draws,
            "avg_reward": round(total_reward / episodes, 3),
        }
    return results


def train(episodes: int, opponent: str, max_seconds: float, seed: int,
          output: Path, resume: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    policy = Agent(str(output.parent) if resume else None)
    q = policy.q
    visits = np.zeros_like(q, dtype=np.uint16)
    env = BlocksWithGunsBotTrainingEnv(opponent=opponent, max_seconds=max_seconds)
    gamma = 0.985
    progress = TrainingProgress(episodes)
    print(f"training Algo Test Q-policy vs {opponent} "
          f"({episodes} full games, checkpoint after every game)")
    output.parent.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(played: int) -> None:
        np.savez_compressed(
            output, q=q.astype(np.float32), episodes=np.asarray(played),
            opponent=np.asarray(opponent), seed=np.asarray(seed))

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        policy.reset()
        epsilon = max(0.04, 0.65 * (1.0 - episode / max(1, episodes * 0.9)))
        total_reward = 0.0
        done = False
        while not done:
            key = state_key(obs)
            if rng.random() < epsilon:
                option = int(rng.integers(NUM_OPTIONS))
            else:
                option = int(np.argmax(q[key]))
            action = policy.action_for_option(obs, option)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            learned_reward = reward + _shaping(obs, next_obs)
            next_key = state_key(next_obs)
            visits[key + (option,)] = min(65535, visits[key + (option,)] + 1)
            alpha = max(0.06, 0.35 / (1.0 + visits[key + (option,)] * 0.015))
            future = 0.0 if terminated or truncated else float(np.max(q[next_key]))
            target = learned_reward + gamma * future
            q[key + (option,)] += alpha * (target - q[key + (option,)])
            total_reward += reward
            obs = next_obs
            done = terminated or truncated
        wins10 = progress.record(step_info["winner"])
        progress.show(episode + 1, wins10,
                      f"opp {info['opponent']:<8}",
                      f"return {total_reward:+8.2f}",
                      f"eps {epsilon:.3f}")
        save_checkpoint(episode + 1)
    progress.finish()
    print("result:", progress.summary())
    print(f"saved trained weights -> {output}")
    env.close()
    return q


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Algo Test tactical Q-table (Dijkstra by default)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--opponent", choices=["all", *BOT_NAMES], default="dijkstra")
    parser.add_argument("--max-seconds", type=float, default=C.MAX_EPISODE_SECONDS,
                        help="safety cap; games normally end by KO or the engine's "
                             "own %.0fs limit" % C.MAX_EPISODE_SECONDS)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()

    if args.evaluate_only:
        if not args.output.is_file():
            parser.error(f"weights not found: {args.output}; train first")
        q = np.load(args.output, allow_pickle=False)["q"]
    else:
        q = train(args.episodes, args.opponent, args.max_seconds,
                  args.seed, args.output, args.resume)
    print("evaluation:")
    results = evaluate(q, args.opponent, args.eval_episodes, args.max_seconds)
    for opponent, stats in results.items():
        print(f"  {opponent:<9} {stats}")


if __name__ == "__main__":
    main()
