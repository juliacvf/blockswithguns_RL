"""Reproduce official-policy matches, including wins as diagnostic controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import json
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from review_generalista_matches import review
from evaluate_generalista_1000 import ROOT,sha,save

def main(folder,workers):
    folder.mkdir(parents=True,exist_ok=True)
    source=ROOT/'MeuAlgo/experiments/training_20261111_round2'
    selected=[];sources=[];frozen={}
    for mode in ['validation','holdout']:
        path=source/f'{mode}_baseline.json'
        data=json.loads(path.read_text(encoding='utf-8'))
        assert data['complete'] and len(data['matches'])==150
        inputs=data['manifest']['official_inputs']
        inference={p:h for p,h in inputs.items() if Path(p)!=Path('MeuAlgo/train.py')}
        assert all(sha(ROOT/p)==h for p,h in inference.items()),'Official inference inputs changed'
        assert not frozen or frozen==inference
        frozen=inference
        assert data['manifest']['weight_sha256']==sha(ROOT/'MeuAlgo/weights/generalista/qtable.npz')
        sources.append(dict(path=str(path),sha256=sha(path),training_source_excluded='Training parameterization does not execute in frozen replay'))
        selected.extend(dict(r,cohort=mode) for r in data['matches'] if r['group']=='specialists' and r['opponent'] in ['markov','dijkstra'])
    assert len(selected)==80
    selected.sort(key=lambda r:(r['winner']!=1,r['winner']!=-1,r['opponent'],r['seed'],r['swapped']))
    manifest=dict(policy='Official generalista, no training or policy change',inputs=frozen,sources=sources,selection='All 80 Markov/Dijkstra games in the two existing evaluation cohorts; wins included as controls',sample_seconds=.2)
    save(folder/'manifest.json',manifest)
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(review,(r,str(folder))) for r in selected]):
            rows.append(f.result())
            save(folder/'diagnosis.json',dict(complete=False,manifest=manifest,matches=rows))
            print(json.dumps(dict(completed=len(rows),total=len(selected),outcomes=dict(Counter(r['winner'] for r in rows)))),flush=True)
    assert all(sha(ROOT/p)==h for p,h in frozen.items())
    assert len(rows)==80
    save(folder/'diagnosis.json',dict(complete=True,manifest=manifest,matches=rows,exact_winner_lives_shots_hits_heat_verified=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();main(args.folder,args.workers)
