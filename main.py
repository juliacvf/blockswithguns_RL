#!/usr/bin/env python3
"""Blocks With Guns RL - pygame entry point.

Run from the project root (with the venv active):
    python main.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.app import App


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
