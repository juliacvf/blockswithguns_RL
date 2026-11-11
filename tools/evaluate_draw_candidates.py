"""Matched evaluations of isolated generalist candidates, with frozen inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
import random
import time
import evaluate_generalista_1000 as benchmark

ROOT=benchmark.ROOT
CACHE={}


def run(job):
    path,expected=job
    if path not in CACHE:
        spec=importlib.util.spec_from_file_location('candidate_'+Path(path).stem,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);CACHE[path]=module
    benchmark.POLICY=CACHE[path]
    result=benchmark.duel((expected['group'],expected['opponent'],expected['seed'],expected['swapped']))
    result['candidate']=Path(path).stem
    result['before']=expected.get('winner')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('mode',choices=['draws','controls','specialists','bots','holdout','holdout_second','holdout_third','bot_control'])
    parser.add_argument('candidates',nargs='+')
    args=parser.parse_args()
    baseline=json.loads((ROOT/'MeuAlgo/experiments/generalista_1000_new_seeds_results.json').read_text())
    all_rows=baseline['matches']
    if args.mode=='draws':selected=[r for r in all_rows if r['winner']==-1]
    elif args.mode=='controls':
        losses=[r for r in all_rows if r['winner']==1]
        wins=[r for r in all_rows if r['group']=='specialists' and r['winner']==0]
        rng=random.Random(2026111011)
        selected=losses+sum((rng.sample([r for r in wins if r['opponent']==n],10) for n in benchmark.NAMES),[])
    elif args.mode in ['specialists','bots']:selected=[r for r in all_rows if r['group']==args.mode]
    elif args.mode=='bot_control':
        previous=json.loads((args.folder/'holdout_reference_aim_reset.json').read_text())
        assert previous['complete']
        selected=[r for r in previous['matches'] if r['candidate']=='reference' and r['group']=='bots']
    else:
        seeds=(random.Random(2026111025).sample(range(2_130_000_000,2_140_000_000),5) if args.mode=='holdout_third' else random.Random(2026111024).sample(range(2_110_000_000,2_120_000_000),10) if args.mode=='holdout_second' else random.Random(2026111023).sample(range(2_000_000_000,2_100_000_000),10))
        groups=['specialists'] if args.mode=='holdout_second' else ['specialists','bots']
        selected=[dict(group=g,opponent=n,seed=s,swapped=swap) for s in seeds for g in groups for n in benchmark.NAMES for swap in [False,True]]
    paths=[args.folder/(n+'.py') for n in args.candidates]
    inputs={str(p):benchmark.sha(p) for p in paths}
    inputs.update({str(ROOT/k):v for k,v in baseline['manifest']['inputs'].items()})
    assert all(benchmark.sha(Path(p))==h for p,h in inputs.items())
    output=args.folder/(args.mode+'_'+'_'.join(args.candidates)+'.json')
    manifest=dict(inputs=inputs,mode=args.mode,jobs=len(selected)*len(paths))
    rows=[]
    if output.exists():
        old=json.loads(output.read_text());assert old['manifest']==manifest
        rows=old['matches']
    elif args.mode=='specialists':
        allowed={(r['group'],r['opponent'],r['seed'],r['swapped']) for r in selected}
        reused={}
        for old_path in args.folder.glob('*.json'):
            old=json.loads(old_path.read_text())
            if not old.get('complete') or 'manifest' not in old:continue
            old_inputs=old['manifest'].get('inputs',{})
            if not all(old_inputs.get(str(ROOT/k))==v for k,v in baseline['manifest']['inputs'].items()):continue
            for p in paths:
                if old_inputs.get(str(p))!=inputs[str(p)]:continue
                for r in old['matches']:
                    k=(r['group'],r['opponent'],r['seed'],r['swapped'])
                    if r['candidate']==p.stem and k in allowed:
                        reused[(p.stem,*k)]=dict(r,reused_from=old_path.name)
        rows=list(reused.values())
    key=lambda r:(r['candidate'],r['group'],r['opponent'],r['seed'],r['swapped'])
    done={key(r) for r in rows}
    start=time.monotonic()
    def summary():
        result={}
        for n in args.candidates:
            subset=[r for r in rows if r['candidate']==n]
            result[n]=dict(total=benchmark.stats(subset),
                transitions={str(before):{str(after):sum(r.get('before')==before and r['winner']==after for r in subset) for after in [0,-1,1]} for before in [0,-1,1]},
                by_opponent={name:benchmark.stats([r for r in subset if r['opponent']==name]) for name in benchmark.NAMES})
        return result
    def save(complete=False):benchmark.save(output,dict(complete=complete,manifest=manifest,summary=summary(),matches=rows))
    save()
    jobs=[(str(p),r) for r in selected for p in paths if (p.stem,r['group'],r['opponent'],r['seed'],r['swapped']) not in done]
    print(json.dumps(dict(remaining=len(jobs))),flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):
            rows.append(f.result());save()
            if len(rows)%10==0:print(json.dumps(dict(completed=len(rows),elapsed=round(time.monotonic()-start),totals={n:v['total'] for n,v in summary().items()})),flush=True)
    assert all(benchmark.sha(Path(p))==h for p,h in inputs.items())
    assert len(rows)==len(paths)*len(selected)
    save(True);print(json.dumps(summary()),flush=True)
