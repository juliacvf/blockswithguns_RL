"""Evaluate a frozen generalist against bots and specialists on new seeds."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT), str(ROOT/'.venv/Lib/site-packages')])
from bots import make_bot, make_view
from core.engine import Engine
from rl.actions import to_engine_action
from rl.full_observation import FullObservationEncoder

NAMES = ('camper', 'strafe', 'dijkstra', 'astar', 'markov')
POLICY = None


def policy():
    global POLICY
    if POLICY is None:
        spec = importlib.util.spec_from_file_location('frozen_generalista', ROOT/'MeuAlgo/algo1.py')
        POLICY = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = POLICY
        spec.loader.exec_module(POLICY)
    return POLICY


def duel(job):
    group, name, seed, swapped = job
    module = policy()
    agent = module.make_agent(ROOT/'MeuAlgo/weights/generalista/qtable.npz')
    opponent = (make_bot(name, seed=seed+1) if group == 'bots' else
                module.make_agent(ROOT/f'MeuAlgo/weights/especialistas/{name}.npz'))
    agent.reset()
    opponent.reset()
    engine = Engine(seed=seed, map_seed=seed)
    if swapped:
        engine.reset(swap_spawns=True)
    encoder = FullObservationEncoder(engine, 120.)
    shots, hits, heat = [0, 0], [0, 0], [0, 0]
    while engine.winner is None:
        first = to_engine_action(agent.act(encoder.encode(0)))
        second = (opponent.act(make_view(engine, 1)) if group == 'bots' else
                  to_engine_action(opponent.act(encoder.encode(1))))
        for event in engine.step((first, second)):
            if event[0] == 'shot':
                shots[event[1]] += 1
            elif event[0] == 'hit':
                hits[event[2]] += 1
            elif event[0] == 'heat_hit':
                heat[event[1]] += 1
    return dict(group=group, opponent=name, seed=seed, bot_seed=seed+1 if group == 'bots' else None,
                swapped=swapped, winner=engine.winner, lives=[p.lives for p in engine.players],
                seconds=engine.time, ticks=engine.tick, shots=shots, hits=hits, heat=heat)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(rows):
    wins = sum(r['winner'] == 0 for r in rows)
    draws = sum(r['winner'] == -1 for r in rows)
    return dict(matches=len(rows), wins=wins, draws=draws, losses=len(rows)-wins-draws,
                win_rate=wins/len(rows) if rows else None,
                hits_dealt=sum(r['hits'][0] for r in rows), hits_received=sum(r['hits'][1] for r in rows),
                heat_damage=sum(r['heat'][0] for r in rows))


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def main(args):
    rng = random.Random(args.seed)
    seeds = dict(bots=rng.sample(range(1_000_000_000, 1_400_000_000), 50),
                 specialists=rng.sample(range(1_500_000_000, 1_900_000_000), 50))
    prior = json.loads((ROOT/'MeuAlgo/experiments/generalista_75_each_results.json').read_text(encoding='utf-8'))
    previous_seeds = {r['seed'] for rows in prior['matches'].values() for r in rows}
    assert not previous_seeds & {s for values in seeds.values() for s in values}
    paths = [ROOT/'MeuAlgo/algo1.py', *(ROOT/'core').glob('*.py'), *(ROOT/'bots').glob('*.py'),
             ROOT/'rl/actions.py', ROOT/'rl/full_observation.py', *(ROOT/'MeuAlgo/weights').rglob('*.npz')]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    manifest = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    seed_generator=args.seed, map_seeds=seeds, inputs=hashes,
                    max_seconds=120, matches_per_opponent=100, both_spawn_sides=True,
                    policy_frozen=True, disjoint_from_prior_report=True)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if output.exists():
        existing = json.loads(output.read_text(encoding='utf-8'))
        assert existing['manifest'] == manifest, 'Frozen inputs or protocol differ from saved run'
        rows = existing['matches']
    key = lambda r: (r['group'], r['opponent'], r['seed'], r['swapped'])
    done = {key(r) for r in rows}
    assert len(done) == len(rows)
    jobs = [(group, name, seed, swapped) for i in range(50) for group in seeds
            for name in NAMES for seed in [seeds[group][i]] for swapped in (False, True)]
    assert done <= set(jobs)
    start = time.monotonic()

    def report(complete=False):
        return dict(manifest=manifest, complete=complete,
                    summary={g: dict(total=stats([r for r in rows if r['group'] == g]),
                        by_opponent={n: stats([r for r in rows if r['group'] == g and r['opponent'] == n]) for n in NAMES}) for g in seeds},
                    combined=stats(rows), matches=sorted(rows, key=key))
    save(output, report())
    print(json.dumps(dict(remaining=1000-len(rows), workers=args.workers, output=str(output))), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(duel, j) for j in jobs if j not in done]
        for future in as_completed(futures):
            rows.append(future.result())
            save(output, report())
            if len(rows) % 10 == 0:
                print(json.dumps(dict(completed=len(rows), elapsed=round(time.monotonic()-start),
                                      groups={g:stats([r for r in rows if r['group'] == g]) for g in seeds})), flush=True)
    assert len(rows) == len({key(r) for r in rows}) == 1000
    assert all(sha(ROOT/p) == expected for p, expected in hashes.items()), 'Input changed during evaluation'
    assert all(v['matches'] == 100 for g in report()['summary'].values() for v in g['by_opponent'].values())
    save(output, report(complete=True))
    print(json.dumps(report()['summary']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=202611101000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, default=ROOT/'MeuAlgo/experiments/generalista_1000_new_seeds_results.json')
    main(parser.parse_args())
