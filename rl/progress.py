"""Shared single-line training progress HUD.

Imported by rl.train_example and "Algo Test/train.py": renders an in-place
``[=====>   ]`` loading bar plus the rolling wins-per-10-episodes stat.
"""

from __future__ import annotations

import shutil
import sys


def bar(fraction: float, width: int) -> str:
    """Loading-bar body for fraction in [0, 1], e.g. ``===>    ``."""
    fraction = min(1.0, max(0.0, fraction))
    filled = min(width, int(fraction * width))
    if filled <= 0:
        return " " * width
    if filled >= width:
        return "=" * width
    return "=" * (filled - 1) + ">" + " " * (width - filled - 1)


class TrainingProgress:
    """In-place training line: bar, episode counter, wins/10ep, extra fields."""

    def __init__(self, total: int):
        self.total = max(1, total)
        self.columns = shutil.get_terminal_size((110, 24)).columns
        self.width = max(10, min(40, self.columns - 62))
        self.outcomes: list[float] = []  # 1.0 win / 0.5 draw / 0.0 loss

    def record(self, winner: int | None) -> float:
        """Log one episode result; return the mean score of the last 10."""
        self.outcomes.append(
            1.0 if winner == 0 else 0.5 if winner == -1 else 0.0)
        window = self.outcomes[-10:]
        return sum(window) / len(window)

    def show(self, episode: int, wins_per10: float, *fields: str) -> None:
        parts = [
            f"[{bar(episode / self.total, self.width)}]",
            f"ep {episode:>4d}/{self.total}",
            f"wins/10ep {wins_per10:4.1f}",
            *fields,
        ]
        line = "  ".join(parts)[: self.columns]
        sys.stdout.write("\r" + line.ljust(self.columns))
        sys.stdout.flush()

    def summary(self) -> str:
        """Overall win/draw/loss tally across recorded episodes."""
        wins = self.outcomes.count(1.0)
        draws = self.outcomes.count(0.5)
        losses = len(self.outcomes) - wins - draws
        return f"wins {wins}  draws {draws}  losses {losses}"

    def finish(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()
