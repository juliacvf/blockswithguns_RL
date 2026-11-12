"""Confirm the fixed escape candidate against the official policy on untouched maps."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
from pathlib import Path
import random
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import evaluate_generalista_1000 as benchmark
from pocket_escape_candidate import install

TYPES=None
def run(job):
    global TYPES
    variant,row=job
    module=benchmark.policy()
    if TYPES is None:
        official=module.GeneralistAgent;install(module)
        TYPES=dict(official=official,candidate=module.GeneralistAgent)
    module.GeneralistAgent=TYPES[variant]
    result=benchmark.duel((row['group'],row['opponent'],row['seed'],row['swapped']))
    return dict(result,variant=variant)

def main(folder,workers):
    diagnosis=json.loads((folder/'diagnosis.json').read_text(encoding='utf-8'))
    fixed=json.loads((folder/'escape_candidate_all.json').read_text(encoding='utf-8'))
    assert fixed['complete']
    assert fixed['summary']['specialists']['after']['wins']>fixed['summary']['specialists']['before']['wins']
    assert fixed['summary']['bots']['after']['wins']>=fixed['summary']['bots']['before']['wins']
    assert fixed['summary']['specialists']['transitions']['0']['-1']==fixed['summary']['specialists']['transitions']['0']['1']==0
    frozen=diagnosis['manifest']['inputs']
    candidate=Path(__file__).with_name('pocket_escape_candidate.py')
    assert benchmark.sha(candidate)==fixed['manifest']['candidate_sha256']
    rng=random.Random(2026111103)
    seeds=dict(specialists=rng.sample(range(610000000,620000000),10),bots=rng.sample(range(630000000,640000000),5))
    assert not {s for v in seeds.values() for s in v} & {r['seed'] for r in fixed['matches']}
    assert all(benchmark.sha(benchmark.ROOT/p)==h for p,h in frozen.items())
    manifest=dict(inputs=frozen,candidate_sha256=benchmark.sha(candidate),generator_seed=2026111103,map_seeds=seeds,
        models=['official','candidate'],specialist_games_per_model=100,bot_games_per_model=50,both_spawn_sides=True,
        candidate_fixed_before_confirmation=True,official_unchanged=True)
    output=folder/'escape_candidate_fresh.json';results=[]
    if output.exists():
        old=json.loads(output.read_text());assert old['manifest']==manifest;results=old['matches']
    key=lambda r:(r['variant'],r['group'],r['opponent'],r['seed'],r['swapped'])
    done={key(r) for r in results}
    selected=[dict(group=g,opponent=n,seed=s,swapped=swap) for g,values in seeds.items() for s in values for n in benchmark.NAMES for swap in [False,True]]
    def save(complete=False):
        summary={variant:{g:dict(total=benchmark.stats([r for r in results if r['variant']==variant and r['group']==g]),
            by_opponent={n:benchmark.stats([r for r in results if r['variant']==variant and r['group']==g and r['opponent']==n]) for n in benchmark.NAMES}) for g in seeds} for variant in manifest['models']}
        benchmark.save(output,dict(complete=complete,manifest=manifest,summary=summary,matches=results))
    save()
    jobs=[(variant,row) for row in selected for variant in manifest['models'] if (variant,row['group'],row['opponent'],row['seed'],row['swapped']) not in done]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):
            results.append(f.result());save()
            if len(results)%20==0:print(json.dumps(dict(completed=len(results),total=300)),flush=True)
    assert len(results)==len({key(r) for r in results})==300
    assert all(benchmark.sha(benchmark.ROOT/p)==h for p,h in frozen.items())
    assert benchmark.sha(candidate)==manifest['candidate_sha256']
    save(True);print('FRESH_CONFIRMATION_FINISHED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();main(args.folder,args.workers)
