"""Agent contract + loader.

A user algorithm is a Python script exposing ONE of:

    class Agent:
        def __init__(self, weights_path: str | None = None): ...
        def reset(self): ...                     # optional
        def act(self, obs: dict) -> action       # MultiDiscrete([9,16,2])

or

    def make_agent(weights_path: str | None = None): ...  # returns the above

`obs` is supplied by the selected compact or full-state environment. The
action is [move 0..8, aim 0..15, shoot 0/1]. The loader accepts a Python file
path plus an optional weights path handed to the agent unchanged.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

MAX_SCRIPT_BYTES = 512 * 1024


class LoadedAgent:
    def __init__(self, agent, name: str):
        self.agent = agent
        self.name = name

    def reset(self) -> None:
        if hasattr(self.agent, "reset"):
            self.agent.reset()

    def act(self, obs: dict) -> np.ndarray:
        a = np.asarray(self.agent.act(obs), dtype=int).ravel()
        if a.size < 3:
            raise ValueError(f"agent {self.name} returned invalid action {a}")
        return a[:3]


def _instantiate(module, weights_path: str | None, name: str) -> LoadedAgent:
    if hasattr(module, "make_agent"):
        return LoadedAgent(module.make_agent(weights_path), name)
    if hasattr(module, "Agent"):
        try:
            return LoadedAgent(module.Agent(weights_path), name)
        except TypeError:
            return LoadedAgent(module.Agent(), name)
    raise ValueError(
        f"{name}: script must define `Agent` class or `make_agent(weights_path)`"
    )


def load_agent(script_path: str | None = None, *,
               weights_path: str | None = None,
               name: str = "agent") -> LoadedAgent:
    if not script_path or not os.path.exists(script_path):
        raise ValueError("no agent script provided")
    if os.path.getsize(script_path) > MAX_SCRIPT_BYTES:
        raise ValueError("agent script too large")

    mod_name = f"user_agent_{abs(hash((script_path, os.path.getmtime(script_path))))}"
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    module = importlib.util.module_from_spec(spec)
    # Uploaded contest folders should remain unchanged after validation.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return _instantiate(module, weights_path, name)
