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



def test_specialist_policies():
    import unittest
    from pathlib import Path
    import numpy as np
    from core import constants as C
    from core.engine import Engine, Action
    from rl.full_observation import FullObservationEncoder
    from rl.actions import validate_action, MOVE_LUT
    from MeuAlgo.algo1 import (StrafeAgent as Agent, DijkstraAgent, STRAFE_PROFILE as PROFILE,
                              AStarAgent, MarkovAgent, ASTAR_PROFILE, MARKOV_PROFILE,
                              DIJKSTRA_PROFILE, intercept_time, BREAK_PATTERN,
                              CHASE, RETREAT, STRAFE_LEFT, EVADE, SAFE_CENTER)
    ROOT = Path(__file__).resolve().parents[1]

    class SpecialistTests(unittest.TestCase):
        def test_markov_finishes_return_before_resuming_pursuit(self):
            obs = FullObservationEncoder(Engine(seed=102000)).encode(0)
            obs["map"][:3] = 0
            obs["game"][3] = .5
            obs["self"][:2] = np.asarray([36.5, 50.5]) / 50 - 1
            previous, policy = DijkstraAgent(), MarkovAgent()
            previous._returning = policy._returning = True
            self.assertFalse(previous._return_needed(obs))
            self.assertTrue(policy._return_needed(obs))
            obs["game"][0] = 1. / C.MAX_EPISODE_SECONDS
            obs["self"][:2] = np.asarray([44.5, 50.5]) / 50 - 1
            self.assertFalse(policy._return_needed(obs))

        def test_markov_return_goal_survives_a_longer_zone_horizon(self):
            obs = FullObservationEncoder(Engine(seed=102000)).encode(0)
            obs["map"][:3] = 0
            obs["game"][3] = .5
            obs["self"][:2] = np.asarray([30.5, 50.5]) / 50 - 1
            policy = MarkovAgent()
            path, _ = policy._forecast_return(obs)
            self.assertTrue(path)
            eta = (len(path) - 1) / C.PLAYER_SPEED
            x, y = path[-1]
            future_margin = policy._safe_margin(obs, x + .5, y + .5) - C.HEAT_SHRINK_RATE * (eta + 40.)
            self.assertGreaterEqual(future_margin, 8.)

        def test_markov_can_resume_combat_near_the_time_limit(self):
            obs = FullObservationEncoder(Engine(seed=102000)).encode(0)
            obs["map"][:3] = 0
            obs["game"][0] = 100. / C.MAX_EPISODE_SECONDS
            obs["game"][3] = .5
            obs["self"][:2] = np.asarray([39.5, 50.5]) / 50 - 1
            policy = MarkovAgent()
            policy._returning = True
            self.assertFalse(policy._return_needed(obs))

        def test_markov_dodges_during_safe_return_beside_a_wall(self):
            from core.engine import Bullet
            from core.mapgen import GameMap
            from rl.actions import to_engine_action

            def scenario(dodge):
                grid = np.zeros((100, 100), dtype=np.uint8)
                grid[40:60, 51] = 1  # The downward dodge is blocked.
                engine = Engine(seed=3, game_map=GameMap(grid))
                engine.players[0].x, engine.players[0].y = 50.5, 50.5
                engine.players[1].x, engine.players[1].y = 44.5, 50.5
                engine.bullets = [Bullet(46.5, 50.5, C.BULLET_SPEED, 0., 1)]
                encoder = FullObservationEncoder(engine)
                policy = MarkovAgent()
                policy._return_needed = lambda obs: True
                policy._returning = True
                policy._return_goal = (40.5, 50.5)
                fired = False
                for _ in range(15):
                    obs = encoder.encode(0)
                    action = policy.action_for_option(obs, SAFE_CENTER)
                    fired |= bool(action[2])
                    if dodge:
                        move = to_engine_action(action)
                    else:
                        move = Action(move_x=-1.)
                    engine.step((move, Action()))
                return engine.players[0], fired

            old, _ = scenario(False)
            defended, fired = scenario(True)
            self.assertEqual(old.lives, C.PLAYER_LIVES - 1)
            self.assertEqual(defended.lives, C.PLAYER_LIVES)
            self.assertLess(defended.x, 50.5)  # Still advances toward the safe goal.
            self.assertLess(defended.y, 50.5)  # Uses the open side of the wall.
            self.assertTrue(fired)

        def test_markov_keeps_movement_for_owned_spent_and_occluded_bullets(self):
            obs = FullObservationEncoder(Engine(seed=102000)).encode(0)
            obs["map"][:3] = 0
            obs["self"][:2] = np.asarray([50.5, 50.5]) / 50 - 1
            obs["bullet_mask"][:] = 0
            obs["bullet_mask"][0] = 1
            obs["bullets"][0] = [46.5 / 50 - 1, 50.5 / 50 - 1, 1, 0, 1, 1, -1]
            policy = MarkovAgent()
            self.assertEqual(policy._defensive_move(obs, (-1, 0)), (-1, 0))
            obs["bullets"][0, 4] = -1
            obs["bullets"][0, 6] = .99  # Expires before reaching the player.
            self.assertEqual(policy._defensive_move(obs, (-1, 0)), (-1, 0))
            obs["bullets"][0, 6] = -1
            obs["map"][0, 48, 50] = 1
            self.assertEqual(policy._defensive_move(obs, (-1, 0)), (-1, 0))

        def test_markov_recognizes_straight_motion_and_forgets_teleports(self):
            encoder = FullObservationEncoder(Engine(seed=102000))
            for style, expected in (("straight", True), ("turn", False), ("still", True)):
                policy = MarkovAgent()
                for tick in range(27):
                    obs = encoder.encode(0)
                    obs["game"][0] = tick / 3600
                    obs["game"][2] = tick / 3600
                    if style == "straight":
                        x, y = 40.5 + tick * .1, 50.5
                    elif style == "turn":
                        x, y = 40.5 + min(tick, 13) * .1, 50.5 + max(0, tick - 13) * .1
                    else:
                        x, y = 40.5, 50.5
                    obs["self"][:2] = np.asarray([x, y]) / 50 - 1
                    policy.observe(obs)
                self.assertEqual(policy._predictable_motion(), expected)
                obs["game"][0] += 1 / 3600
                obs["game"][2] += 1 / 3600
                obs["self"][:2] = np.asarray([80.5, 70.5]) / 50 - 1
                policy.observe(obs)
                self.assertFalse(policy._predictable_motion())

        def test_markov_pressure_uses_live_own_projectiles_and_remaining_range(self):
            obs = FullObservationEncoder(Engine(seed=102000)).encode(0)
            obs["opponent"][:2] = np.asarray([54.5, 50.5]) / 50 - 1
            obs["bullet_mask"][:] = 0
            obs["bullet_mask"][0] = 1
            obs["bullets"][0] = [49.5 / 50 - 1, 50.5 / 50 - 1, 1, 0, 1, 1, 0]
            self.assertTrue(MarkovAgent._target_under_pressure(obs))
            obs["bullets"][0, 2] = -1
            self.assertFalse(MarkovAgent._target_under_pressure(obs))
            obs["bullets"][0, 2] = 1
            obs["bullets"][0, 6] = .9
            self.assertFalse(MarkovAgent._target_under_pressure(obs))
            obs["bullets"][0, 6] = 0
            obs["bullets"][0, 4] = -1
            self.assertFalse(MarkovAgent._target_under_pressure(obs))

        def test_markov_pattern_break_is_committed_bounded_and_yields_to_safety(self):
            obs = FullObservationEncoder(Engine(seed=102000)).encode(0)
            obs["map"][:3] = 0
            obs["self"][:2] = np.asarray([40.5, 50.5]) / 50 - 1
            obs["opponent"][:2] = np.asarray([54.5, 50.5]) / 50 - 1
            obs["bullet_mask"][:] = 0
            policy = MarkovAgent()
            self.assertIn(BREAK_PATTERN, policy.available_options(obs))
            action = policy.action_for_option(obs, BREAK_PATTERN)
            validate_action(action)
            self.assertNotEqual(int(action[0]), 4)
            vector = policy._break_vector
            self.assertEqual(policy._pattern_breaks, 1)
            obs["game"][0] = .1 / C.MAX_EPISODE_SECONDS
            obs["game"][2] = .1 / C.MAX_EPISODE_SECONDS
            validate_action(policy.action_for_option(obs, BREAK_PATTERN))
            self.assertEqual(policy._break_vector, vector)
            self.assertEqual(policy._pattern_breaks, 1)
            obs["game"][0] = .6 / C.MAX_EPISODE_SECONDS
            self.assertNotIn(BREAK_PATTERN, policy.available_options(obs))
            obs["game"][0] = 1.3 / C.MAX_EPISODE_SECONDS
            self.assertIn(BREAK_PATTERN, policy.available_options(obs))
            obs["game"][3] = .4
            obs["self"][0] = 31.5 / 50 - 1
            self.assertEqual(policy.available_options(obs).tolist(), [SAFE_CENTER])

        def test_astar_spacing_distinguishes_an_advancing_and_retreating_enemy(self):
            obs = FullObservationEncoder(Engine(seed=92000)).encode(0)
            obs["map"][:3] = 0
            obs["self"][:2] = np.asarray([40.5, 50.5]) / 50 - 1
            obs["opponent"][:2] = np.asarray([54.5, 50.5]) / 50 - 1
            obs["self"][14] = 1
            policy = AStarAgent()
            policy.observe(obs)
            policy._motion_velocity = (-3.5, 0.)
            self.assertEqual(policy.state_key(obs)[6], 0)
            action = policy.action_for_option(obs, STRAFE_LEFT)
            validate_action(action)
            self.assertLess(int(action[0]) % 3 - 1, 0)
            policy._motion_velocity = (3.5, 0.)
            self.assertEqual(policy.state_key(obs)[6], 2)
            action = policy.action_for_option(obs, STRAFE_LEFT)
            self.assertGreater(int(action[0]) % 3 - 1, 0)

        def test_astar_close_range_retreat_interrupts_for_an_incoming_bullet(self):
            obs = FullObservationEncoder(Engine(seed=92000)).encode(0)
            obs["map"][:3] = 0
            obs["self"][:2] = np.asarray([40.5, 50.5]) / 50 - 1
            obs["opponent"][:2] = np.asarray([46.5, 50.5]) / 50 - 1
            obs["self"][14] = 1
            obs["bullet_mask"][:] = 0
            policy = AStarAgent()
            self.assertEqual(policy.option(obs), RETREAT)
            obs["game"][0] += 1 / 3600
            obs["game"][2] += 1 / 3600
            obs["bullet_mask"][0] = 1
            obs["bullets"][0] = [44.5 / 50 - 1, 50.5 / 50 - 1, -1, 0, -1, 1, -1]
            self.assertEqual(policy.option(obs), EVADE)

        def test_dijkstra_return_destination_remains_safe_after_arrival(self):
            obs = FullObservationEncoder(Engine(seed=85000)).encode(0)
            obs["map"][:3] = 0
            obs["self"][:2] = np.asarray([35.5, 50.5]) / 50 - 1
            obs["game"][3] = .5
            policy = DijkstraAgent()
            path, _ = policy._forecast_return(obs)
            self.assertGreater(len(path), 1)
            eta = (len(path) - 1) / C.PLAYER_SPEED
            x, y = path[-1]
            future_margin = policy._safe_margin(obs, x + .5, y + .5) - C.HEAT_SHRINK_RATE * (eta + 20.)
            self.assertGreaterEqual(future_margin, 8.)
            # A quick return is still available, so this position may be held.
            self.assertFalse(policy._return_needed(obs))

        def test_dijkstra_long_flights_require_more_room_for_prediction_error(self):
            obs = FullObservationEncoder(Engine(seed=85017)).encode(0)
            obs["map"][:3] = 0
            obs["self"][:2] = np.asarray([40.5, 40.5]) / 50 - 1
            policy = DijkstraAgent()
            obs["opponent"][:2] = np.asarray([54.5, 40.91]) / 50 - 1
            self.assertEqual(policy._shot(obs, 40.5, 40.5)[1], 0)
            obs["opponent"][:2] = np.asarray([43.5, 40.91]) / 50 - 1
            self.assertEqual(policy._shot(obs, 40.5, 40.5)[1], 1)

        def test_dijkstra_leaves_early_for_a_detour_without_forbidding_safe_bait(self):
            obs = FullObservationEncoder(Engine(seed=85000)).encode(0)
            obs["map"][:3] = 0
            obs["self"][:2] = np.asarray([30.5, 50.5]) / 50 - 1
            obs["game"][3] = .5
            self.assertFalse(DijkstraAgent()._return_needed(obs))
            # Same distance to the zone boundary, but reaching the interior
            # requires a long detour around the wall.
            obs["map"][0, 31, 25:74] = 1
            policy = DijkstraAgent()
            self.assertGreater(policy._safe_margin(obs, 30.5, 50.5), 2.)
            self.assertTrue(policy._return_needed(obs))
            self.assertTrue(policy._return_path)
            self.assertEqual(list(policy.available_options(obs)), [SAFE_CENTER])

        def test_dijkstra_escape_forecast_accounts_for_mud(self):
            obs = FullObservationEncoder(Engine(seed=85000)).encode(0)
            obs["map"][:3] = 0
            obs["self"][:2] = np.asarray([30.5, 50.5]) / 50 - 1
            obs["game"][3] = .5
            obs["map"][0, 31, 30:70] = 1
            self.assertFalse(DijkstraAgent()._return_needed(obs))
            obs["map"][2] = 1
            self.assertTrue(DijkstraAgent()._return_needed(obs))

        def test_dijkstra_aim_uses_shooter_position_after_movement(self):
            obs = FullObservationEncoder(Engine(seed=85000)).encode(0)
            obs["map"][:3] = 0
            obs["self"][:2] = np.asarray([40.5, 40.5]) / 50 - 1
            obs["opponent"][:2] = np.asarray([43., 41.]) / 50 - 1
            policy = DijkstraAgent()
            self.assertEqual(policy._shot(obs, 40.5, 40.5)[0], 1)
            action = policy._aim_after_move(obs, 0, 1)
            self.assertEqual(int(action[1]), 0)
            self.assertEqual(int(action[2]), 1)

        def test_dijkstra_checks_quantized_bullet_lane_not_only_visibility(self):
            obs = FullObservationEncoder(Engine(seed=83005)).encode(0)
            obs["map"][:3] = 0
            obs["self"][:2] = np.asarray([40.05, 40.95]) / 50 - 1
            obs["opponent"][:2] = np.asarray([50.05, 44.95]) / 50 - 1
            obs["self"][14] = 1
            policy = DijkstraAgent()
            self.assertEqual(policy._shot(obs, 40.05, 40.95)[1], 1)
            # The coarse visibility line is clear, but the 22.5-degree bullet
            # crosses this wall before reaching the opponent.
            obs["map"][0, 45, 43] = 1
            from core.los import has_line_of_sight
            self.assertTrue(has_line_of_sight(obs["map"][0], 40.05, 40.95, 50.05, 44.95))
            self.assertEqual(policy._shot(obs, 40.05, 40.95)[1], 0)

        def test_dijkstra_does_not_extrapolate_tiny_trapped_enemy_reversals(self):
            engine = Engine(seed=83005)
            encoder = FullObservationEncoder(engine)
            policy = DijkstraAgent()
            for tick in range(18):
                obs = encoder.encode(0)
                obs["map"][:3] = 0
                obs["game"][0] = tick / 3600
                obs["game"][2] = tick / 3600
                obs["self"][:2] = np.asarray([40., 40.]) / 50 - 1
                obs["opponent"][:2] = np.asarray([50., 40. + (.08 if tick % 2 else -.08)]) / 50 - 1
                policy.observe(obs)
            self.assertGreater(policy._motion_velocity[1], .5)
            self.assertEqual(policy._shot(obs, 40., 40.)[0], 0)

        def test_dijkstra_commits_to_another_firing_position_after_failed_combat(self):
            obs = FullObservationEncoder(Engine(seed=83005)).encode(0)
            obs["map"][:3] = 0
            obs["self"][:2] = np.asarray([40., 40.]) / 50 - 1
            obs["opponent"][:2] = np.asarray([50., 42.]) / 50 - 1
            obs["self"][14] = 1
            policy = DijkstraAgent()
            policy._previous_time = 6.
            policy._combat_shots = 4
            policy._combat_history.extend([(0., (40., 40.), 0, True, (50., 42.), True),
                                           (6., (40., 40.), 4, True, (50., 42.), True)])
            move = policy._combat_move(obs)
            self.assertIsNotNone(move)
            self.assertNotEqual(move, (0, 0))
            goal = policy._reposition_goal
            self.assertIsNotNone(goal)
            self.assertGreater(math.dist((40., 40.), goal), 2.)
            policy._previous_time = 6.1
            policy._combat_move(obs)
            self.assertEqual(policy._reposition_goal, goal)
            obs["game"][3] = .1
            self.assertIsNone(policy._combat_move(obs))
            self.assertIsNone(policy._reposition_goal)

        def test_interception_meets_moving_target(self):
            for dx, dy, vx, vy in ((14, 0, 0, 3.5), (10, 8, -3, 1), (8, 0, 3.5, 0)):
                t = intercept_time(dx, dy, vx, vy)
                self.assertAlmostEqual(math.hypot(dx + vx * t, dy + vy * t), C.BULLET_SPEED * t)

        def test_motion_reversal_teleport_and_reset(self):
            eng = Engine(seed=10000)
            encoder = FullObservationEncoder(eng)
            agent = Agent()
            def observe(tick, x, y):
                obs = encoder.encode(0)
                obs["game"][0] = tick * C.FIXED_DT / C.MAX_EPISODE_SECONDS
                obs["game"][2] = tick / 3600
                obs["self"][:2] = np.asarray([40., 40.]) / 50 - 1
                obs["opponent"][:2] = np.asarray([50., y]) / 50 - 1
                obs["opponent"][0] = x / 50 - 1
                agent.observe(obs)
                return obs
            observe(0, 50, 40)
            obs = observe(1, 50, 40.1)
            self.assertEqual(agent._lateral, 2)
            velocity = agent._motion_velocity
            agent.observe(obs)
            self.assertEqual(agent._motion_velocity, velocity)
            observe(2, 50, 40)
            self.assertEqual(agent._lateral, 0)
            self.assertLess(agent._motion_velocity[1], 0)
            self.assertGreater(agent._turn_until, agent._previous_time)
            observe(3, 80, 80)
            self.assertEqual(agent._motion_velocity, (0., 0.))
            agent.reset()
            self.assertIsNone(agent._observed_tick)
            self.assertEqual(agent._motion_velocity, (0., 0.))

        def test_movement_prediction_matches_engine_with_mud_and_body_radius(self):
            eng = Engine(seed=10000)
            encoder = FullObservationEncoder(eng)
            agent = Agent()
            # Include free, muddy, and wall-adjacent starts from an actual map.
            points = [(15.5, 15.5)]
            for x in range(2, 98):
                for y in range(2, 98):
                    if not eng._circle_hits_solid(x + .5, y + .5, C.PLAYER_RADIUS):
                        if eng.map.is_mud(x, y) or eng.map.is_solid(x + 1, y):
                            points.append((x + .5, y + .5))
                    if len(points) >= 10:
                        break
                if len(points) >= 10:
                    break
            for x, y in points:
                for mx, my in MOVE_LUT:
                    player = eng.players[0]
                    player.x, player.y = x, y
                    obs = encoder.encode(0)
                    expected = agent._step_position(obs, mx, my)
                    eng._move_player(player, Action(move_x=mx, move_y=my), C.FIXED_DT)
                    self.assertAlmostEqual(player.x, expected[0], places=4)
                    self.assertAlmostEqual(player.y, expected[1], places=4)

        def test_commitment_interrupts_on_new_bullet_threat(self):
            eng = Engine(seed=10000)
            obs = FullObservationEncoder(eng).encode(0)
            agent = Agent()
            agent.q.fill(0)
            agent.q[..., CHASE] = 1
            self.assertEqual(agent.option(obs), CHASE)
            agent.q[..., EVADE] = 2
            self.assertEqual(agent.option(obs), CHASE)
            obs["game"][0] += C.FIXED_DT / C.MAX_EPISODE_SECONDS
            obs["game"][2] += 1 / 3600
            sx, sy = eng.players[0].x, eng.players[0].y
            obs["bullet_mask"][0] = 1
            obs["bullets"][0] = [(sx + 4) / 50 - 1, sy / 50 - 1, -1, 0, -1, 1, -1]
            self.assertEqual(agent.option(obs), EVADE)

        def test_checkpoint_profile_and_real_match_actions(self):
            agent = Agent()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "qtable.npz"
                np.savez_compressed(path, q=agent.q, profile=np.asarray(PROFILE))
                restored = Agent(tmp)
                self.assertTrue(np.array_equal(agent.q, restored.q))
                from rl.agent_interface import load_agent
                loaded = load_agent(str(ROOT / "MeuAlgo/algo1.py"), weights_path=str(path))
                obs = FullObservationEncoder(Engine(seed=10000)).encode(0)
                validate_action(loaded.act(obs))
                np.savez_compressed(path, q=agent.q)
                with self.assertRaises(ValueError):
                    Agent(tmp)
            from rl.bot_training_env import BlocksWithGunsBotTrainingEnv
            env = BlocksWithGunsBotTrainingEnv(opponent="strafe")
            try:
                obs, _ = env.reset(seed=10008)
                for _ in range(240):
                    action = validate_action(agent.act(obs))
                    obs, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
            finally:
                env.close()

        def test_visualizer_loads_explicit_specialist(self):
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            from rl.visualize import run_visual_match
            result = run_visual_match(player1="custom", player2="strafe",
                                      algo1_script=str(ROOT / "MeuAlgo/algo1.py"),
                                      algo1_weights=str(ROOT / "MeuAlgo/weights/especialistas/strafe.npz"),
                                      seed=30000, max_seconds=0.2, fps=240, linger=False)
            self.assertGreaterEqual(result["time"], 0.2)

        def test_evaluation_swaps_spawns_without_changing_map_or_bot_rng(self):
            from MeuAlgo.train import reset_evaluation_match as reset_match
            from rl.bot_training_env import BlocksWithGunsBotTrainingEnv
            env = BlocksWithGunsBotTrainingEnv(opponent="strafe")
            try:
                normal, _ = reset_match(env, 50000)
                bot_rng = env._opponent.rng.getstate()
                engine_rng = env.engine.rng.getstate()
                swapped, _ = reset_match(env, 50000, swapped=True)
                np.testing.assert_array_equal(normal["map"][:3], swapped["map"][:3])
                np.testing.assert_array_equal(normal["self"][:2], swapped["opponent"][:2])
                np.testing.assert_array_equal(normal["opponent"][:2], swapped["self"][:2])
                self.assertEqual(bot_rng, env._opponent.rng.getstate())
                self.assertEqual(engine_rng, env.engine.rng.getstate())
                normal_again, _ = reset_match(env, 50000)
                np.testing.assert_array_equal(normal["self"], normal_again["self"])
            finally:
                env.close()

        def test_false_obstacle_memory_cannot_disconnect_a_valid_escape(self):
            eng = Engine(seed=50037)
            player = eng.players[0]
            player.x, player.y = 58.709086698298144, 19.347529462032504
            obs = FullObservationEncoder(eng).encode(0)
            agent = Agent()
            # Actual failure observed in the swapped-spawn evaluation at 78 seconds.
            agent._blocked_cells = {(50, 13), (50, 15), (51, 14), (51, 16),
                                    (52, 15), (53, 16), (53, 17), (54, 16),
                                    (54, 18), (55, 18), (56, 17), (56, 19),
                                    (57, 18), (57, 19), (58, 20), (59, 19)}
            mx, my = agent._path_move(obs, (50., 50.), "safe")
            self.assertEqual(agent._route_recoveries, 1)
            self.assertTrue(agent._path)
            start = player.x, player.y
            eng._move_player(player, Action(move_x=mx, move_y=my), C.FIXED_DT)
            self.assertGreater(math.dist(start, (player.x, player.y)), 0.05)


        def test_unified_training_roundtrip_for_all_profiles(self):
            from contextlib import redirect_stdout
            from io import StringIO
            from MeuAlgo.train import train
            from MeuAlgo.algo1 import make_agent
            with tempfile.TemporaryDirectory() as tmp:
                for profile in ("general", PROFILE, DIJKSTRA_PROFILE, ASTAR_PROFILE, MARKOV_PROFILE):
                    path = Path(tmp) / (profile + ".npz")
                    with redirect_stdout(StringIO()):
                        opponent = ("markov" if profile == MARKOV_PROFILE else "astar" if profile == ASTAR_PROFILE
                                    else "dijkstra" if profile == DIJKSTRA_PROFILE else "strafe")
                        q = train(1, opponent,
                                  70000, path, False, 0., 0.01, profile=profile)
                    restored = make_agent(path)
                    self.assertEqual(restored.profile, profile)
                    np.testing.assert_array_equal(restored.q, q)
                    with np.load(path, allow_pickle=False) as checkpoint:
                        self.assertEqual(int(checkpoint["episodes"]), 1)
                        self.assertGreater(int(checkpoint["visits"].sum()), 0)

        def test_dijkstra_safety_mask_overrides_an_offensive_q_value(self):
            eng = Engine(seed=81000)
            obs = FullObservationEncoder(eng).encode(0)
            obs["game"][3] = 0.4
            agent = DijkstraAgent()
            agent.q.fill(0)
            agent.q[..., CHASE] = 1000
            self.assertEqual(agent.option(obs), SAFE_CENTER)
            self.assertEqual(agent.available_options(obs).tolist(), [SAFE_CENTER])
            validate_action(agent.action_for_option(obs, CHASE))

        def test_dijkstra_interrupts_a_committed_chase_before_leaving_safety(self):
            obs = FullObservationEncoder(Engine(seed=81000)).encode(0)
            obs["game"][3] = 0.4
            obs["self"][:2] = np.asarray([32.1, 50.5]) / 50 - 1
            agent = DijkstraAgent()
            agent.q.fill(0)
            agent.q[..., CHASE] = 1000
            key = agent.state_key(obs)
            self.assertEqual(agent.option(obs), CHASE)
            obs["self"][0] = 31.9 / 50 - 1
            obs["game"][0] += C.FIXED_DT / C.MAX_EPISODE_SECONDS
            obs["game"][2] += 1 / 3600
            self.assertEqual(agent.state_key(obs)[5], key[5])
            self.assertEqual(agent.option(obs), SAFE_CENTER)

        def test_dijkstra_goal_is_reachable_and_inside_the_safe_area(self):
            obs = FullObservationEncoder(Engine(seed=81000)).encode(0)
            obs["map"][:2] = 0
            # A free center cell surrounded by walls is not a reachable goal.
            obs["map"][0, 49:52, 49] = 1
            obs["map"][0, 49:52, 51] = 1
            obs["map"][0, 49, 49:52] = 1
            obs["map"][0, 51, 49:52] = 1
            obs["game"][3] = 0.4
            agent = DijkstraAgent()
            x, y = agent._safe_goal(obs, (50.5, 50.5))
            self.assertNotEqual((x, y), (50.5, 50.5))
            self.assertTrue(agent._reachable[int(x), int(y)])
            self.assertGreaterEqual(agent._safe_margin(obs, x, y), 2.)
            self.assertIsNone(DijkstraAgent()._reachable)
            agent.reset()
            self.assertIsNone(agent._reachable)

        def test_dijkstra_route_prefers_a_safe_detour(self):
            from core.pathfinding import astar
            obs = FullObservationEncoder(Engine(seed=81000)).encode(0)
            obs["map"][:3] = 0
            obs["map"][0, 50, 30:56] = 1
            obs["game"][3] = 0.4
            solid = obs["map"][0].astype(bool)
            shortcut = astar(solid, 40.5, 31.5, 60.5, 31.5)
            self.assertTrue(any(y < 30 for _, y in shortcut))
            route = DijkstraAgent()._plan_path(solid, obs, 40.5, 31.5, 60.5, 31.5)
            self.assertTrue(route)
            self.assertTrue(all(30 <= x <= 69 and 30 <= y <= 69 for x, y in route))

        def test_dijkstra_pickups_do_not_lure_it_outside_the_safe_zone(self):
            obs = FullObservationEncoder(Engine(seed=81000)).encode(0)
            obs["self"][:2] = np.asarray([40., 40.]) / 50 - 1
            obs["self"][4] = 0.
            obs["game"][3] = 0.4
            obs["powerup_mask"][0] = 1
            obs["powerups"][0] = [20.5 / 50 - 1, 40.5 / 50 - 1, 1, 1, 0, 0, 0]
            agent = DijkstraAgent()
            self.assertIsNone(agent._powerup_target(obs))
            obs["powerups"][0, 0] = 42.5 / 50 - 1
            self.assertIsNotNone(agent._powerup_target(obs))

    result = unittest.TextTestRunner(verbosity=0).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(SpecialistTests))
    assert result.wasSuccessful(), "specialist policy regressions failed"
    ok(f"unified specialist profiles and training ({result.testsRun} regressions)")


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
    test_specialist_policies()
    print(f"\nALL {len(PASS)} CHECKS PASSED")
