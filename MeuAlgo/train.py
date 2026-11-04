"""Train or evaluate MeuAlgo and its official specialist checkpoint profiles.

Training is headless and runs as fast as the simulator allows — no
rendering, no real-time pacing. Every episode is a complete match: it ends
only when the engine itself decides the game (knockout, or the lives
tiebreak at the built-in 120 s cap). There is no shorter time limit.

Run from the project root:
    .venv/Scripts/python.exe MeuAlgo/train.py --evaluate-only --opponent all
    .venv/Scripts/python.exe MeuAlgo/train.py --evaluate-only --opponent strafe \
        --output MeuAlgo/weights/especialistas/strafe.npz --eval-sides both
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

from algo1 import CHASE, POWERUP, SAFE_CENTER, policy_class, _world_xy  # noqa: E402
from bots import BOT_NAMES  # noqa: E402
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


def reset_evaluation_match(env, seed, swapped=False):
    """Swap spawn sides without changing map or initial bot RNG state."""
    obs, info = env.reset(seed=seed)
    if swapped:
        env.engine.reset(swap_spawns=True)
        obs = env._encoder.encode(0)
    return obs, info


def evaluate(q: np.ndarray, opponent: str, episodes: int,
             base_seed: int = 10_000, profile: str = "general",
             sides: str = "normal") -> dict:
    if episodes < 1 or sides not in ("normal", "swapped", "both"):
        raise ValueError("positive map count and valid spawn sides are required")
    opponents = BOT_NAMES if opponent == "all" else [opponent]
    spawn_sides = ("normal", "swapped") if sides == "both" else (sides,)
    results = {}
    for bot_name in opponents:
        env = BlocksWithGunsBotTrainingEnv(opponent=bot_name)
        policy = policy_class(profile)()
        if q.shape != policy.q.shape or not np.isfinite(q).all():
            raise ValueError("checkpoint does not match the selected policy profile")
        policy.q = q.copy()
        wins = losses = draws = 0
        total_reward = 0.0
        hits = damage = heat = alerts = recoveries = 0
        by_side = {side: dict(wins=0, draws=0, losses=0) for side in spawn_sides}
        try:
            for seed, side in ((base_seed + episode, side)
                               for episode in range(episodes) for side in spawn_sides):
                obs, _ = reset_evaluation_match(env, seed, swapped=side == "swapped")
                policy.reset()
                origin = _world_xy(obs["self"])
                ticks = navigation_intent = 0
                done = False
                while not done:
                    option = policy.option(obs)
                    action = policy.action_for_option(obs, option)
                    navigation_intent += int(option in (CHASE, POWERUP, SAFE_CENTER))
                    obs, reward, terminated, truncated, info = env.step(action)
                    for event in info["events"]:
                        hits += int(event[0] == "hit" and event[2] == 0)
                        damage += int(event[0] == "hit" and event[1] == 0)
                        heat += int(event[0] == "heat_hit" and event[1] == 0)
                    total_reward += reward
                    ticks += 1
                    if ticks % 90 == 0:
                        current = _world_xy(obs["self"])
                        alerts += int(navigation_intent >= 72 and
                                      ((current[0] - origin[0]) ** 2 +
                                       (current[1] - origin[1]) ** 2) < 0.4 ** 2)
                        origin, navigation_intent = current, 0
                    done = terminated or truncated
                winner = info["winner"]
                wins += winner == 0
                losses += winner == 1
                draws += winner == -1
                by_side[side]["wins" if winner == 0 else "losses" if winner == 1 else "draws"] += 1
                recoveries += getattr(policy, "_route_recoveries", 0)
        finally:
            env.close()
        if not np.array_equal(q, policy.q):
            raise AssertionError("evaluation modified the policy weights")
        results[bot_name] = {
            "wins": wins, "losses": losses, "draws": draws,
            "avg_reward": round(total_reward / (episodes * len(spawn_sides)), 3),
            "hits": hits, "damage": damage, "heat": heat,
            "navigation_alerts": alerts, "route_recoveries": recoveries,
            "by_side": by_side,
        }
    return results


def train(episodes: int, opponent: str, seed: int,
          output: Path, resume: bool, epsilon_start: float, alpha_start: float,
          profile: str = "general") -> np.ndarray:
    policy = policy_class(profile)()
    profile = policy.profile
    specialist = policy.learned_tactics
    q = policy.q
    visits = np.zeros_like(q, dtype=np.uint32 if specialist else np.uint16)
    completed_episodes = 0

    if resume:
        if not output.is_file():
            raise FileNotFoundError(f"checkpoint not found: {output}")

        with np.load(output, allow_pickle=False) as checkpoint:
            loaded_q = checkpoint["q"]

            if loaded_q.shape != q.shape:
                raise ValueError(
                    f"qtable shape {loaded_q.shape} does not match {q.shape}"
                )

            q = loaded_q.astype(np.float32)
            policy.q = q

            if "visits" in checkpoint.files:
                saved_visits = checkpoint["visits"]

                if saved_visits.shape == q.shape:
                    visits = saved_visits.astype(visits.dtype)

            if "episodes" in checkpoint.files:
                completed_episodes = int(checkpoint["episodes"])

    rng = np.random.default_rng(seed + completed_episodes)
    env = BlocksWithGunsBotTrainingEnv(opponent=opponent)
    gamma = 0.99
    progress = TrainingProgress(episodes)
    print(f"training MeuAlgo ({profile}) vs {opponent}: {episodes} full matches, "
          f"headless at full speed, checkpoint after every match")
    output.parent.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(played: int) -> None:
        np.savez_compressed(
            output,
            q=q.astype(np.float32),
            visits=visits,
            episodes=np.asarray(completed_episodes + played),
            opponent=np.asarray(opponent),
            seed=np.asarray(seed),
            epsilon_start=np.asarray(epsilon_start),
            alpha_start=np.asarray(alpha_start),
            profile=np.asarray(profile),
            decision_ticks=np.asarray(policy.decision_ticks),
        )
    for episode in range(episodes):
        global_episode = completed_episodes + episode

        obs, info = env.reset(seed=seed + global_episode)
        policy.reset()

        epsilon = max(
            0.04,
            epsilon_start * ((0.97 if specialist else 0.995) ** global_episode),
        )
        total_reward = 0.0
        done = False
        while not done:
            key = policy.state_key(obs)
            options = policy.available_options(obs)
            if rng.random() < epsilon:
                option = int(options[rng.integers(len(options))])
            else:
                option = policy.best_option(obs, key)
            learned_reward = episode_reward = 0.0
            discount = 1.0
            before_potential = policy.learning_potential(obs) if specialist else 0.0
            context = policy.learning_context(key), tuple(options)
            for _ in range(policy.decision_ticks):
                action = policy.action_for_option(obs, option)
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                extra = 0.0
                if specialist:
                    extra = policy.learning_extra_reward(step_info["events"])
                else:
                    extra = _shaping(obs, next_obs)
                learned_reward += discount * (reward + extra)
                episode_reward += reward
                discount *= gamma
                obs = next_obs
                next_key = policy.state_key(obs)
                next_context = policy.learning_context(next_key), tuple(policy.available_options(obs))
                if terminated or truncated or next_context != context:
                    break
            if specialist:
                learned_reward += discount * (0.0 if terminated or truncated else policy.learning_potential(obs)) - before_potential
            index = key + (option,)
            visits[index] = min(np.iinfo(visits.dtype).max, int(visits[index]) + 1)
            # gentle learning rate: noisy long-episode targets must not
            # trample the heuristic priors in rarely visited states
            alpha = max(0.005 if specialist else 0.03,
                        alpha_start / (1.0 + int(visits[index]) * (0.02 if specialist else 0.01)))
            future = 0.0 if terminated or truncated else float(np.max(q[next_key][policy.available_options(obs)]))
            target = learned_reward + discount * future
            q[key + (option,)] += alpha * (target - q[key + (option,)])
            total_reward += episode_reward
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
        description="Train or evaluate MeuAlgo checkpoint profiles over full matches "
                    "(Dijkstra by default)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--opponent", choices=["all", *BOT_NAMES], default="dijkstra")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=("auto", "general", "strafe", "dijkstra"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--eval-sides", choices=("normal", "swapped", "both"), default="normal")
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--alpha", type=float)
    args = parser.parse_args()

    if min(args.episodes, args.eval_episodes) < 1:
        parser.error("episode counts must be positive")
    args.output = args.output or (HERE / "weights/especialistas" / (args.profile + ".npz")
                                if args.profile in ("strafe", "dijkstra") else DEFAULT_WEIGHTS)
    profile = policy_class("general" if args.profile == "auto" else args.profile).profile
    if args.evaluate_only or args.resume:
        if not args.output.is_file():
            parser.error(f"weights not found: {args.output}; train first")
        with np.load(args.output, allow_pickle=False) as checkpoint:
            saved_profile = str(checkpoint["profile"].item()) if "profile" in checkpoint.files else "general"
        if args.profile != "auto" and saved_profile != profile:
            parser.error("selected profile does not match the checkpoint")
        profile = saved_profile
    specialist = policy_class(profile).learned_tactics
    epsilon = args.epsilon if args.epsilon is not None else (0.20 if specialist else 0.25)
    alpha = args.alpha if args.alpha is not None else (0.04 if specialist else 0.12)

    if args.evaluate_only:
        q = np.load(args.output, allow_pickle=False)["q"]
    else:
        q = train(args.episodes, args.opponent, args.seed,
                  args.output, args.resume, epsilon, alpha, profile=profile)
    print("evaluation:")
    results = evaluate(q, args.opponent, args.eval_episodes,
                       base_seed=args.eval_seed, profile=profile, sides=args.eval_sides)
    for opponent, stats in results.items():
        print(f"  {opponent:<9} {stats}")


if __name__ == "__main__":
    main()
