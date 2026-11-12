"""Compare both standalone submissions with development on complete matches."""
import os,sys,importlib.util,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT),str(ROOT/'.venv/Lib/site-packages')])
from core.engine import Engine
from rl.full_observation import FullObservationEncoder
from rl.actions import to_engine_action

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def compare(job):
    opponent,seed,swap=job
    dev=load('equivalence_development',ROOT/'MeuAlgo/algo1.py')
    slot1=load('equivalence_slot1',ROOT/'Concurso/slot1/algo1.py')
    slot2=load('equivalence_slot2',ROOT/'Concurso/slot2/algo2.py')
    frozen=load('equivalence_frozen',ROOT/'MeuAlgo/experiments/pocket_escape_20261111/official_policy_frozen.py')
    candidate=load('equivalence_candidate',ROOT/'MeuAlgo/experiments/pocket_escape_20261111/pocket_escape_candidate.py')
    candidate.install(frozen)
    agents=[dev.make_agent(ROOT/'MeuAlgo/weights/generalista/qtable.npz'),slot1.make_agent(ROOT/'Concurso/slot1/weights'),slot2.make_agent(ROOT/'Concurso/slot2/weights'),frozen.make_agent(ROOT/'MeuAlgo/weights/generalista/qtable.npz')]
    enemy=dev.make_agent(ROOT/f'MeuAlgo/weights/especialistas/{opponent}.npz')
    e=Engine(seed=seed,map_seed=seed)
    if swap:e.reset(swap_spawns=True)
    enc=FullObservationEncoder(e,120.)
    while e.winner is None:
        obs=enc.encode(0);actions=[a.act(obs) for a in agents]
        assert np.array_equal(actions[0],actions[1]),(job,e.tick,'slot1',actions)
        assert np.array_equal(actions[0],actions[2]),(job,e.tick,'slot2',actions)
        assert np.array_equal(actions[0],actions[3]),(job,e.tick,'fixed winning candidate',actions)
        e.step((to_engine_action(actions[0]),to_engine_action(enemy.act(enc.encode(1)))))
    return dict(opponent=opponent,seed=seed,swapped=swap,ticks=e.tick,winner=e.winner)

if __name__=='__main__':
    cases=[(n,s,swap) for n,s in [('camper',7193382),('strafe',1310003),('dijkstra',1310004),('astar',1120000),('markov',1310001),('markov',7193382),('dijkstra',1526107030),('markov',517358468),('dijkstra',1716643869),('strafe',2131977416),('dijkstra',1570262884)] for swap in (False,True)]
    rows=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for f in as_completed([pool.submit(compare,c) for c in cases]):
            rows.append(f.result());print(json.dumps(rows[-1]),flush=True)
    print(json.dumps(dict(matches=len(rows),all_actions_identical=True,total_ticks=sum(r['ticks'] for r in rows))),flush=True)
