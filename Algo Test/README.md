# Algo Test

This is a deliberately small, trainable slot-1 contest algorithm. A Q-table
chooses among eight tactical behaviors while simple deterministic helpers aim,
avoid blocked cells, align legal aim bins, and follow paths. It trains against
Dijkstra by default.

From the project root:

```bash
.venv/bin/python "Algo Test/train.py" --episodes 100 --seed 456
.venv/bin/python -m rl.contest --algo1-folder "Algo Test" --algo2-folder algo2 --validate-only
.venv/bin/python -m rl.visualize --player1 custom --algo1-folder "Algo Test" --player2 dijkstra
.venv/bin/python "Algo Test/train.py" --evaluate-only --opponent all
```

Use another built-in name with `--opponent` to train or evaluate against it.
The saved `weights/qtable.npz` is the playable policy used by `algo1.py`.

The included checkpoint's ten held-out 45-second Dijkstra episodes produced
9 draws and 1 loss, versus 5 draws and 5 losses from its untrained prior.
