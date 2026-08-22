"""Validate contest folders and run strict algo1-vs-algo2 matches.

Run:
    python -m rl.contest --algo1-folder algo1 --algo2-folder algo2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from core import constants as C
from core.engine import Engine
from rl.actions import to_engine_action, validate_action
from rl.agent_interface import LoadedAgent, load_agent
from rl.full_observation import FullObservationEncoder, full_observation_space
from rl.rewards import rewards_from_events, winner_from_lives

MAX_CONTEST_FILES = 1_000
MAX_CONTEST_BYTES = 512 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPLAY_DIR = PROJECT_ROOT / "replays"


@dataclass
class ContestEntry:
    """A loaded slot-specific policy and its validated resources."""

    slot: int
    folder: Path
    script: Path
    weights: Path
    loaded: LoadedAgent

    @property
    def name(self) -> str:
        return self.folder.name

    def reset(self) -> None:
        self.loaded.reset()

    def act(self, observation: dict):
        raw = self.loaded.agent.act(observation)
        return validate_action(raw, f"algo{self.slot}")


def _folder_resources(folder: Path) -> tuple[int, int]:
    count = total = 0
    for path in folder.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"contest folders may not contain symlinks: {path}")
        if path.is_file():
            count += 1
            total += path.stat().st_size
    return count, total


def load_contest_entry(folder: str | os.PathLike, slot: int) -> ContestEntry:
    """Load ``algoN.py`` and pass its required ``weights/`` directory."""
    if slot not in (1, 2):
        raise ValueError("contest slot must be 1 or 2")
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"algo{slot} folder does not exist: {root}")
    script = root / f"algo{slot}.py"
    weights = root / "weights"
    if not script.is_file():
        raise ValueError(f"algo{slot} folder must contain {script.name}")
    if not weights.is_dir():
        raise ValueError(f"algo{slot} folder must contain a weights/ subfolder")
    count, total = _folder_resources(root)
    if count > MAX_CONTEST_FILES:
        raise ValueError(f"algo{slot} has too many files ({count} > {MAX_CONTEST_FILES})")
    if total > MAX_CONTEST_BYTES:
        raise ValueError(
            f"algo{slot} folder is too large ({total} > {MAX_CONTEST_BYTES} bytes)")
    loaded = load_agent(
        str(script), weights_path=str(weights), name=f"algo{slot}:{root.name}")
    return ContestEntry(slot, root, script, weights, loaded)


def validate_contest_folder(folder: str | os.PathLike, slot: int) -> dict:
    """Load a folder and smoke-test its reset/act/action contract."""
    entry = load_contest_entry(folder, slot)
    engine = Engine(seed=9700 + slot)
    encoder = FullObservationEncoder(engine)
    observation = encoder.encode(slot - 1)
    if not full_observation_space().contains(observation):
        raise ValueError("internal full observation does not match its declared space")
    entry.reset()
    action = entry.act(observation)
    to_engine_action(action, name=f"algo{slot}")
    count, total = _folder_resources(entry.folder)
    return {
        "slot": slot,
        "folder": str(entry.folder),
        "script": entry.script.name,
        "weights_dir": str(entry.weights),
        "files": count,
        "bytes": total,
        "sample_action": action.tolist(),
        "valid": True,
    }


def run_folder_match(
    algo1_folder: str | os.PathLike,
    algo2_folder: str | os.PathLike,
    episodes: int = 3,
    max_seconds: float = C.MAX_EPISODE_SECONDS,
    base_seed: int = 100,
    save_replay: bool = True,
) -> dict:
    """Run a deterministic, side-alternating contest series."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    entries = [
        load_contest_entry(algo1_folder, 1),
        load_contest_entry(algo2_folder, 2),
    ]
    wins = [0, 0]
    draws = 0
    total_rewards = [0.0, 0.0]
    replay_frames: list[dict] = []
    map_payload = None

    for episode in range(episodes):
        seed = base_seed + episode
        engine = Engine(seed=seed, map_seed=seed)
        if episode % 2:
            engine.reset(swap_spawns=True)
        encoder = FullObservationEncoder(engine, max_seconds)
        for entry in entries:
            entry.reset()
        frames: list[dict] = []
        episode_rewards = [0.0, 0.0]

        while engine.winner is None and engine.time < max_seconds - 1e-9:
            engine_actions = []
            for idx, entry in enumerate(entries):
                try:
                    action = entry.act(encoder.encode(idx))
                    engine_actions.append(to_engine_action(action, name=f"algo{idx + 1}"))
                except Exception as exc:
                    raise RuntimeError(
                        f"algo{idx + 1} failed at episode {episode + 1}, tick {engine.tick}") from exc
            events = engine.step(tuple(engine_actions))
            step_rewards = rewards_from_events(events, engine.winner)
            for idx in (0, 1):
                episode_rewards[idx] += step_rewards[idx]
            if episode == 0 and save_replay:
                frames.append(engine.snapshot())

        lives = tuple(player.lives for player in engine.players)
        winner = engine.winner if engine.winner is not None else winner_from_lives(lives)
        if winner in (0, 1):
            wins[winner] += 1
        else:
            draws += 1
        for idx in (0, 1):
            total_rewards[idx] += episode_rewards[idx]
        if episode == 0:
            replay_frames = frames
            map_payload = engine.map_payload()

    replay_name = None
    if save_replay:
        REPLAY_DIR.mkdir(exist_ok=True)
        replay_name = f"replay_contest_{int(time.time() * 1000)}.json"
        payload = {
            "created": time.time(),
            "agent": entries[0].name,
            "opponent": entries[1].name,
            "episodes": episodes,
            "wins": wins[0], "losses": wins[1], "draws": draws,
            "map": map_payload,
            "frames": replay_frames,
            "dt": C.FIXED_DT,
        }
        with (REPLAY_DIR / replay_name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    return {
        "algo1": entries[0].name,
        "algo2": entries[1].name,
        "algo1_wins": wins[0],
        "algo2_wins": wins[1],
        "draws": draws,
        "episodes": episodes,
        "algo1_avg_reward": round(total_rewards[0] / episodes, 3),
        "algo2_avg_reward": round(total_rewards[1] / episodes, 3),
        "replay_file": replay_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and battle two contest folders")
    parser.add_argument("--algo1-folder", default="algo1")
    parser.add_argument("--algo2-folder", default="algo2")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=C.MAX_EPISODE_SECONDS)
    parser.add_argument("--base-seed", type=int, default=100)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()
    reports = [
        validate_contest_folder(args.algo1_folder, 1),
        validate_contest_folder(args.algo2_folder, 2),
    ]
    if args.validate_only:
        print(json.dumps(reports, indent=2))
        return
    result = run_folder_match(
        args.algo1_folder, args.algo2_folder,
        episodes=args.episodes, max_seconds=args.max_seconds,
        base_seed=args.base_seed, save_replay=not args.no_replay)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

