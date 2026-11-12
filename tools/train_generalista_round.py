"""Continue official weights in isolation using the existing Q-learning loop."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import os

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'MeuAlgo'),str(ROOT),str(ROOT/'.venv/Lib/site-packages')]
import train as training
from algo1 import make_agent
from bots import make_bot
from rl.actions import to_engine_action

NAMES=('camper','strafe','dijkstra','astar','markov')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


class SpecialistOpponent:
    def __init__(self,env,name):
        self.env=env
        self.policy=make_agent(ROOT/f'MeuAlgo/weights/especialistas/{name}.npz')
        self.policy.reset()
    def act(self,view):
        return to_engine_action(self.policy.act(self.env._encoder.encode(1)))


class CurriculumEnv(training.BlocksWithGunsBotTrainingEnv):
    records=[]
    def reset(self,*,seed=None,options=None):
        index=self._episode
        pair=index//2
        cycle,slot=divmod(pair,6)
        group='specialists' if slot<5 else 'bots'
        name=NAMES[slot] if slot<5 else NAMES[cycle%5]
        obs,info=super().reset(seed=seed,options=options)
        self._opponent_name=name
        self._opponent=SpecialistOpponent(self,name) if group=='specialists' else make_bot(name,seed=seed+1)
        if group=='bots':self._opponent.reset()
        self._training_group=group
        self._match_seed=seed
        info['opponent']=group+':'+name
        return obs,info
    def step(self,action):
        obs,reward,terminated,truncated,info=super().step(action)
        if terminated or truncated:
            assert not truncated and self.engine.winner is not None
            self.records.append(dict(group=self._training_group,opponent=self._opponent_name,
                seed=self._match_seed,swapped=bool((self._episode-1)%2),winner=info['winner'],
                lives=list(info['lives']),seconds=info['time'],ticks=self.engine.tick))
        return obs,reward,terminated,truncated,info


def main(args):
    folder=args.folder;folder.mkdir(parents=True,exist_ok=True)
    output=folder/'candidate.npz'
    assert not output.exists(),'Use a new run folder; this driver does not silently restart training'
    source=ROOT/'MeuAlgo/weights/generalista/qtable.npz'
    paths=[ROOT/'MeuAlgo/algo1.py',ROOT/'MeuAlgo/train.py',*(ROOT/'core').glob('*.py'),
           *(ROOT/'bots').glob('*.py'),*(ROOT/'rl').glob('*.py'),*(ROOT/'MeuAlgo/weights').rglob('*.npz'),
           ROOT/'Concurso/slot1/algo1.py',ROOT/'Concurso/slot2/algo2.py',
           ROOT/'Concurso/slot1/weights/qtable.npz',ROOT/'Concurso/slot2/weights/qtable.npz']
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths}
    shutil.copyfile(source,folder/'baseline.npz')
    shutil.copyfile(source,output)
    with np.load(source,allow_pickle=False) as old:
        original_q=old['q'].copy()
        original_visits=old['visits'].copy()
        old_episodes=int(old['episodes'])
    manifest=dict(algorithm='Existing MeuAlgo/train.py Q-learning loop, unchanged rewards and policy',
        input_hashes=inputs,episodes_before=old_episodes,additional_episodes=120,
        seed=args.seed,actual_train_map_seeds=list(range(args.seed+old_episodes//2,args.seed+old_episodes//2+60)),
        epsilon_start=.12,alpha_start=.01,gamma=args.gamma,groups=dict(specialists=100,bots=20),
        evaluation_protocol=dict(
            validation=dict(generator_seed=2026111101,offset=410000000),
            holdout=dict(generator_seed=args.holdout_seed,offset=args.holdout_offset)),
        maps_per_specialist=10,maps_per_bot=2,both_spawn_sides=True,maximum_seconds=120,
        official_weights_unchanged=True)
    save(folder/'manifest.json',manifest)
    original_save=np.savez_compressed
    start=time.monotonic()
    def save_checkpoint(file,*a,**kw):
        original_save(file,*a,**kw)
        if Path(file)==output:
            completed=int(kw['episodes'])-old_episodes
            if completed in (40,80,120):shutil.copyfile(output,folder/f'checkpoint_{completed:03d}.npz')
            q=kw['q']
            save(folder/'progress.json',dict(complete=False,training_matches=completed,
                elapsed_seconds=round(time.monotonic()-start),changed_q_entries=int(np.count_nonzero(q!=original_q)),
                q_updates=int(np.asarray(kw['visits'],dtype=np.int64).sum()-np.asarray(original_visits,dtype=np.int64).sum()),
                results=dict(wins=sum(r['winner']==0 for r in CurriculumEnv.records),
                    draws=sum(r['winner']==-1 for r in CurriculumEnv.records),losses=sum(r['winner']==1 for r in CurriculumEnv.records)),
                matches=CurriculumEnv.records))
    training.BlocksWithGunsBotTrainingEnv=CurriculumEnv
    training.np.savez_compressed=save_checkpoint
    try:
        training.train(120,'all',args.seed,output,True,.12,.01,profile='generalista_v1',gamma=args.gamma)
    finally:
        training.np.savez_compressed=original_save
    assert len(CurriculumEnv.records)==120
    assert Counter(r['group'] for r in CurriculumEnv.records)==dict(specialists=100,bots=20)
    for group,count in [('specialists',20),('bots',4)]:
        assert all(sum(r['group']==group and r['opponent']==n for r in CurriculumEnv.records)==count for n in NAMES)
    assert all(sha(ROOT/p)==h for p,h in inputs.items()),'Frozen official input changed'
    progress=json.loads((folder/'progress.json').read_text())
    assert progress['q_updates']>0 and progress['changed_q_entries']>0
    progress.update(complete=True,candidate_sha256=sha(output),official_inputs_verified=True)
    save(folder/'progress.json',progress)
    print(json.dumps({k:v for k,v in progress.items() if k!='matches'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    parser.add_argument('--seed',type=int,default=310000000)
    parser.add_argument('--gamma',type=float,default=.99)
    parser.add_argument('--holdout-seed',type=int,default=2026111102)
    parser.add_argument('--holdout-offset',type=int,default=510000000)
    main(parser.parse_args())
