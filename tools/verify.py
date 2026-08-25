"""Full verification suite for Blocks With Guns RL.

Run:  .venv/bin/python tools/verify.py
"""

import math
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = []


def ok(name):
    PASS.append(name)
    print(f"  PASS  {name}")


def test_engine_powerups():
    from core.engine import Engine, Action
    from core.powerups import Powerup

    # LIFE
    eng = Engine(seed=2)
    from core import constants as C
    p0 = eng.players[0]
    p0.lives = 3
    eng.powerups.items = [Powerup(p0.x, p0.y, "LIFE")]
    eng.powerups.try_pickup(eng.players)
    assert p0.lives == 4, "LIFE failed"
    p0.lives = C.PLAYER_LIVES
    eng.powerups.items = [Powerup(p0.x, p0.y, "LIFE")]
    eng.powerups.try_pickup(eng.players)
    assert p0.lives == C.PLAYER_LIVES, "LIFE cap failed"
    ok(f"powerup LIFE (+1 life, capped at {C.PLAYER_LIVES})")

    # QUICKSHOT
    eng = Engine(seed=2)
    p0 = eng.players[0]
    eng.powerups.items = [Powerup(p0.x, p0.y, "QUICKSHOT")]
    eng.powerups.try_pickup(eng.players)
    assert p0.quickshot_timer > 0 and p0.shoot_cooldown == C.QUICKSHOT_COOLDOWN, "QUICKSHOT failed"
    ok(f"powerup QUICKSHOT (cooldown {C.SHOOT_COOLDOWN} -> {C.QUICKSHOT_COOLDOWN} for 5s)")

    # SPEED
    eng = Engine(seed=2)
    p0 = eng.players[0]
    base = p0.speed
    eng.powerups.items = [Powerup(p0.x, p0.y, "SPEED")]
    eng.powerups.try_pickup(eng.players)
    assert abs(p0.speed - base * 1.5) < 1e-9, "SPEED failed"
    ok("powerup SPEED (x1.5 for 5s)")

    # SWAP
    eng = Engine(seed=2)
    p0, p1 = eng.players
    a, b = (p0.x, p0.y), (p1.x, p1.y)
    eng.powerups.items = [Powerup(p0.x, p0.y, "SWAP")]
    ev = eng.powerups.try_pickup(eng.players)
    assert (p0.x, p0.y) == b and (p1.x, p1.y) == a, "SWAP failed"
    ok("powerup SWAP (positions exchanged)")

    # Spawn at the configured first delay and interval, with no item cap.
    eng = Engine(seed=2)
    noop = (Action(), Action())
    first_steps = round(C.POWERUP_FIRST_DELAY / C.FIXED_DT)
    for _ in range(first_steps - 1):
        eng.step(noop)
    assert not eng.powerups.items, "powerup spawned before first delay"
    events = eng.step(noop)
    assert any(e[0] == "powerup_spawn" for e in events), "no powerup at first delay"
    first = eng.powerups.items[0]

    interval_steps = round(C.POWERUP_INTERVAL / C.FIXED_DT)
    for _ in range(interval_steps - 1):
        events = eng.step(noop)
        assert not any(e[0] == "powerup_spawn" for e in events), "powerup spawned before interval"
    events = eng.step(noop)
    assert any(e[0] == "powerup_spawn" for e in events), "no powerup at interval"

    for _ in range(round((C.POWERUP_TTL + C.POWERUP_INTERVAL) / C.FIXED_DT)):
        eng.step(noop)
    assert len(eng.powerups.items) > 2, "spawn capped below 25s ttl steady state"
    assert all(p is not first for p in eng.powerups.items), "powerup never expires"
    ok(f"powerup cadence {C.POWERUP_INTERVAL:.1f}s + {C.POWERUP_TTL:.0f}s expiry")


def test_engine_combat():
    from core.engine import Engine, Action
    eng = Engine(seed=1)
    p0, p1 = eng.players
    p0.x, p0.y = 15.5, 15.5
    p1.x, p1.y = 18.5, 15.5
    # being hit does NOT lock the victim's gun: its cooldown stays untouched
    hit_seen = False
    for _ in range(60):
        ev = eng.step((Action(aim=0.0, shoot=True), Action()))
        if any(e[0] == "hit" for e in ev):
            hit_seen = True
            assert p1.cooldown <= 0.0, f"hit changed cooldown: {p1.cooldown}"
            break
    assert hit_seen, "no hit registered"
    ok("hit -> no shooting lockout")
    # keep firing until p1 is out of lives (undo the low-life retreat each
    # iteration so the victim stays in the line of fire)
    from core import constants as C
    eng = Engine(seed=1)
    p0, p1 = eng.players
    p0.x, p0.y = 15.5, 15.5
    for _ in range(600):
        p1.x, p1.y = 18.5, 15.5
        eng.step((Action(aim=0.0, shoot=True), Action()))
        if eng.winner is not None:
            break
    assert eng.winner == 0 and p1.lives <= 0, "combat failed"
    ok(f"engine combat ({C.PLAYER_LIVES} hits -> game over)")


def test_bullet_range():
    from core import constants as C
    from core.engine import Engine, Action
    eng = Engine(seed=1)
    eng.grid[:] = 0                      # open field: range, not walls, must stop it
    eng.powerups.timer = 1e9             # no pickups interfering
    p0, p1 = eng.players
    p0.x, p0.y = 25.5, 50.5
    p1.x, p1.y = 25.5, 10.5              # out of the line of fire
    max_x = 0.0
    eng.step((Action(aim=0.0, shoot=True), Action()))
    for _ in range(30 * 6):
        eng.step((Action(), Action()))
        for b in eng.bullets:
            max_x = max(max_x, b.x)
    assert not eng.bullets, "bullet never died"
    traveled = max_x - 25.5
    assert C.BULLET_RANGE - 1.0 < traveled <= C.BULLET_RANGE + 0.6, f"range failed: {traveled}"
    ok(f"bullet range capped at {C.BULLET_RANGE:.0f} blocks")


def test_heat_zone():
    from core import constants as C
    from core.engine import Engine, Action
    eng = Engine(seed=1)
    eng.powerups.timer = 1e9             # no pickups interfering
    p0, p1 = eng.players
    assert eng.safe_half() == C.WORLD / 2, "zone should cover the map at t=0"
    eng.time = 60.0                      # safe half-width now 50 - 0.25*60 = 35
    assert abs(eng.safe_half() - 35.0) < 1e-6, "shrink rate failed"
    p0.x, p0.y = 10.5, 15.5              # |10.5-50| = 39.5 > 35 -> in heat
    p1.x, p1.y = 50.5, 50.5              # map center -> safe
    heat_hits = 0
    for _ in range(int(11 * 30)):
        ev = eng.step((Action(), Action()))
        heat_hits += sum(1 for e in ev if e[0] == "heat_hit" and e[1] == 0)
    assert heat_hits == 2, f"heat ticks failed: {heat_hits}"
    assert p0.lives == C.PLAYER_LIVES - 2, f"heat lives failed: {p0.lives}"
    assert p1.lives == C.PLAYER_LIVES and p1.heat_timer == 0.0, "safe player burned"
    ok(f"heat zone ({C.HEAT_SHRINK_RATE} block/s shrink, 1 life per 5 s outside)")


def test_mud_and_trees():
    from core import constants as C
    from core.engine import Engine, Action
    from core.mapgen import generate_map
    m = generate_map(seed=4)
    mud_cells = int(m.mud.sum())
    assert 0 < mud_cells < 1200, f"mud ponds missing or everywhere: {mud_cells}"
    assert m.trees, "no trees scattered"
    for tx, ty in m.trees:
        assert not m.is_wall(int(tx), int(ty)), "tree inside a wall"
        assert not m.is_mud(int(tx), int(ty)), "tree in mud"
    ok(f"mud ponds sparse ({mud_cells} cells) + trees ({len(m.trees)}) off walls")

    # trees are solid: walking straight into one is blocked like a wall
    eng = Engine(seed=4)
    pl = eng.players[0]
    for tx, ty in eng.map.trees:
        if not eng.map.is_solid(int(tx) - 1, int(ty)):
            break
    pl.x, pl.y = int(tx) - 0.5, ty
    for _ in range(60):
        eng.step((Action(move_x=1.0, move_y=0.0, aim=0.0), Action()))
    assert pl.x > int(tx) - 0.5, "player did not move toward the tree"
    assert int(pl.x) < int(tx), f"walked into a tree: {(pl.x, pl.y)}"
    ok("trees block movement like walls")

    # mud slows: one tick of movement in mud vs on clean floor
    eng = Engine(seed=4)
    eng.powerups.timer = 1e9
    mud_cell = None
    for x in range(1, 98):
        for y in range(1, 98):
            if m.is_mud(x, y) and not m.is_wall(x + 1, y):
                mud_cell = (x, y)
                break
        if mud_cell:
            break
    assert mud_cell is not None, "no usable mud cell"
    p0 = eng.players[0]
    p0.x, p0.y = mud_cell[0] + 0.5, mud_cell[1] + 0.5
    x0 = p0.x
    eng.step((Action(move_x=1.0), Action()))
    mud_dx = p0.x - x0
    p0.x, p0.y = 15.5, 15.5   # spawn area: always clean, carved open
    x0 = p0.x
    eng.step((Action(move_x=1.0), Action()))
    clean_dx = p0.x - x0
    assert abs(mud_dx - clean_dx * C.MUD_SLOW) < 1e-9, (mud_dx, clean_dx)
    ok(f"mud slows movement x{C.MUD_SLOW}")


def test_retreat():
    from core import constants as C
    from core.engine import Engine, Action
    eng = Engine(seed=1)
    eng.powerups.timer = 1e9
    p0, p1 = eng.players
    p0.x, p0.y = 15.5, 15.5
    p1.x, p1.y = 18.5, 15.5
    # retreat triggers exactly when a hit drops the victim to RETREAT_LIVES
    p1.lives = C.RETREAT_LIVES + 1
    seen = None
    for _ in range(60):
        ev = eng.step((Action(aim=0.0, shoot=True), Action()))
        got = [e for e in ev if e[0] == "retreat"]
        if got:
            seen = got[0]
            break
    assert seen is not None, "no retreat at exactly RETREAT_LIVES"
    assert p1.lives == C.RETREAT_LIVES, "hit not applied"
    assert (p1.x, p1.y) == (seen[2], seen[3]), "retreat event pos mismatch"
    dist = ((p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2) ** 0.5
    assert dist >= C.RETREAT_MIN_DIST, f"retreated too close: {dist}"
    assert not eng.in_heat(p1.x, p1.y), "retreated into the heat zone"

    # healed back above the threshold: the once-per-match retreat stays spent
    p1.x, p1.y = 18.5, 15.5
    p1.lives = C.RETREAT_LIVES + 1
    for _ in range(60):
        ev = eng.step((Action(aim=0.0, shoot=True), Action()))
        assert not any(e[0] == "retreat" for e in ev), "retreat fired twice"
        if any(e[0] == "hit" for e in ev):
            break
    assert p1.lives == C.RETREAT_LIVES, "second hit not applied"

    # hits at 2 or 1 life never retreat; the victim just dies
    for _ in range(600):
        p1.x, p1.y = 18.5, 15.5   # keep the victim in the line of fire
        ev = eng.step((Action(aim=0.0, shoot=True), Action()))
        assert not any(e[0] == "retreat" for e in ev), "retreat fired at 2/1 lives"
        if eng.winner is not None:
            break
    assert p1.lives <= 0, "victim should be eliminated without retreating again"
    ok("retreat: once per match, only at exactly 3 lives")


def test_env():
    from gymnasium.utils.env_checker import check_env
    from rl.env import BlocksWithGunsEnv
    env = BlocksWithGunsEnv(opponent="dijkstra", max_seconds=5.0)
    check_env(env.unwrapped, skip_render_check=True)
    ok("gymnasium check_env")
    env.close()


def test_full_observation_and_bot_training():
    from gymnasium.utils.env_checker import check_env
    from core.engine import Action, Engine
    from core.powerups import Powerup
    from rl.bot_training_env import BlocksWithGunsBotTrainingEnv
    from rl.full_observation import FullObservationEncoder, full_observation_space

    eng = Engine(seed=19)
    eng.powerups.items = [
        Powerup(20.5, 21.5, "LIFE"),
        Powerup(22.5, 23.5, "QUICKSHOT"),
    ]
    eng.step((Action(aim=0.0, shoot=True), Action(aim=math.pi, shoot=True)))
    encoder = FullObservationEncoder(eng)
    space = full_observation_space()
    observations = [encoder.encode(0), encoder.encode(1)]
    assert all(space.contains(obs) for obs in observations), "invalid full observation"
    assert int(observations[0]["bullet_mask"].sum()) == len(eng.bullets)
    assert int(observations[0]["powerup_mask"].sum()) == len(eng.powerups.items)
    assert observations[0]["self"][0] == observations[1]["opponent"][0]
    assert observations[0]["map"][0].shape == eng.grid.shape

    env = BlocksWithGunsBotTrainingEnv(opponent="all", max_seconds=1.0)
    check_env(env.unwrapped, skip_render_check=True)
    seen = {env.reset(seed=seed)[1]["opponent"] for seed in range(5)}
    assert seen == {"dijkstra", "astar", "strafe", "camper", "markov"}
    env.close()
    ok("full-state observations + Gymnasium training against all 5 bots")


def test_pettingzoo_env():
    from pettingzoo.test import parallel_api_test, parallel_seed_test
    from rl.pettingzoo_env import parallel_env

    game = parallel_env(max_seconds=1.0)
    parallel_api_test(game, num_cycles=40)
    game.close()
    parallel_seed_test(lambda: parallel_env(max_seconds=1.0), num_cycles=10)
    ok("PettingZoo parallel API + deterministic self-play seeding")


def test_contest_framework():
    from rl.actions import validate_action
    from rl.contest import run_folder_match, validate_contest_folder

    reports = [validate_contest_folder("algo1", 1),
               validate_contest_folder("algo2", 2)]
    assert all(report["valid"] for report in reports)
    for invalid in ([9, 0, 0], [0, 16, 0], [0, 0, 2], [0.5, 0, 0], [0, 0]):
        try:
            validate_action(invalid, "invalid-test")
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid contest action accepted: {invalid}")
    result = run_folder_match("algo1", "algo2", episodes=1,
                              max_seconds=0.5, save_replay=False)
    assert result["algo1_wins"] + result["algo2_wins"] + result["draws"] == 1
    ok("contest folders + strict actions + algo1-vs-algo2 battle")


def test_algo_test_example():
    import numpy as np
    from rl.contest import load_contest_entry, run_folder_match, validate_contest_folder

    report = validate_contest_folder("Algo Test", 1)
    assert report["valid"] and report["script"] == "algo1.py"
    entry = load_contest_entry("Algo Test", 1)
    assert entry.loaded.agent.q.shape[-1] == 8, "Algo Test Q-table not loaded"
    result = run_folder_match("Algo Test", "algo2", episodes=1,
                              max_seconds=0.2, save_replay=False)
    assert result["algo1_wins"] + result["algo2_wins"] + result["draws"] == 1

    with tempfile.TemporaryDirectory() as tmp:
        output = os.path.join(tmp, "qtable.npz")
        # one full-length training match plus one full evaluation match;
        # episodes only ever end when the engine decides the match
        subprocess.check_call([
            sys.executable, os.path.join("Algo Test", "train.py"),
            "--episodes", "1",
            "--eval-episodes", "1", "--output", output,
        ], stdout=subprocess.DEVNULL)
        q = np.load(output, allow_pickle=False)["q"]
        assert q.shape == entry.loaded.agent.q.shape
    ok("Algo Test trained weights + playable contest entry")


def test_contest_visualizer():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from rl.visualize import run_visual_match

    result = run_visual_match(player1="custom", player2="astar",
                              max_seconds=0.1, fps=240, linger=False)
    assert result["winner"] in (-1, 0, 1) and result["time"] >= 0.1
    ok("pygame custom/built-in contest visualizer")


def test_determinism():
    from core.engine import Engine
    from bots import make_bot, make_view

    def play():
        eng = Engine(seed=11)
        bots = [make_bot("markov", seed=3), make_bot("strafe", seed=4)]
        while eng.winner is None:
            eng.step(tuple(bots[i].act(make_view(eng, i)) for i in range(2)))
        return (eng.winner, round(eng.time, 3),
                [p.lives for p in eng.players], len(eng.bullets))

    r1, r2 = play(), play()
    assert r1 == r2, f"non-deterministic: {r1} vs {r2}"
    tree_code = (
        "import json; from core.mapgen import generate_map; "
        "print(json.dumps(generate_map(seed=11).trees))")
    trees1 = subprocess.check_output([sys.executable, "-c", tree_code], text=True)
    trees2 = subprocess.check_output([sys.executable, "-c", tree_code], text=True)
    assert trees1 == trees2, "tree placement changes between Python processes"
    ok(f"bot/map cross-process determinism (same seeds -> same result {r1})")


def test_tournament():
    from core.engine import Engine
    from bots import BOT_NAMES, make_bot, make_view
    for a in BOT_NAMES:
        for b in BOT_NAMES:
            if a >= b:
                continue
            eng = Engine(seed=7)
            bots = [make_bot(a, 1), make_bot(b, 2)]
            steps = 0
            while eng.winner is None and steps < 3700:
                eng.step(tuple(bots[i].act(make_view(eng, i)) for i in range(2)))
                steps += 1
            assert eng.winner is not None, f"{a} vs {b} never resolved"
    ok("bot round-robin resolves (10 pairs)")


def test_sounds():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sfx = os.path.join(base, "assets", "sfx")
    need = ["menu_theme.wav", "battle_theme.wav", "shot.wav", "hit.wav",
            "powerup.wav", "quickshot.wav", "swap.wav", "victory.wav",
            "defeat.wav", "ui_hover.wav", "heart_pickup.wav"]
    for n in need:
        p = os.path.join(sfx, n)
        assert os.path.exists(p) and os.path.getsize(p) > 1000, n
    ok("8-bit audio assets present (11 wavs)")


if __name__ == "__main__":
    test_engine_powerups()
    test_engine_combat()
    test_bullet_range()
    test_heat_zone()
    test_mud_and_trees()
    test_retreat()
    test_env()
    test_full_observation_and_bot_training()
    test_pettingzoo_env()
    test_contest_framework()
    test_algo_test_example()
    test_contest_visualizer()
    test_determinism()
    test_tournament()
    test_sounds()
    print(f"\nALL {len(PASS)} CHECKS PASSED")
