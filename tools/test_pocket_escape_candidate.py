"""Paired, frozen, resumable evaluation of a navigation-only policy variant."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import evaluate_generalista_1000 as benchmark
from pocket_escape_candidate import install

INSTALLED=False
def run(row):
    global INSTALLED
    if not INSTALLED:install(benchmark.policy());INSTALLED=True
    result=benchmark.duel((row['group'],row['opponent'],row['seed'],row['swapped']))
    result['before']=row['winner']
    return result

def main(folder,scope,workers):
    diagnosis=json.loads((folder/'diagnosis.json').read_text(encoding='utf-8'))
    assert diagnosis['complete']
    paths=[Path(p['path']) for p in diagnosis['manifest']['sources']]
    rows=[r for p in paths for r in json.loads(p.read_text(encoding='utf-8'))['matches']]
    if scope=='pocket':selected=[r for r in rows if r['group']=='specialists' and r['opponent'] in ['markov','dijkstra'] and r['seed']==517358468 and r['swapped']]
    elif scope=='targets':selected=[r for r in rows if r['group']=='specialists' and r['opponent'] in ['markov','dijkstra']]
    else:selected=rows
    frozen=diagnosis['manifest']['inputs']
    assert all(benchmark.sha(benchmark.ROOT/p)==h for p,h in frozen.items())
    manifest=dict(inputs=frozen,candidate_sha256=benchmark.sha(Path(__file__).with_name('pocket_escape_candidate.py')),scope=scope,matches=len(selected),official_unchanged=True)
    output=folder/f'escape_candidate_{scope}.json'
    key=lambda r:(r['group'],r['opponent'],r['seed'],r['swapped'])
    results=[]
    if output.exists():
        old=json.loads(output.read_text());assert old['manifest']==manifest;results=old['matches']
    elif scope!='pocket':
        for previous_scope in ['pocket','targets']:
            previous=folder/f'escape_candidate_{previous_scope}.json'
            if previous.exists():
                old=json.loads(previous.read_text());assert old['manifest']['candidate_sha256']==manifest['candidate_sha256']
                allowed={key(r) for r in selected};found={key(r):r for r in results}
                found.update({key(r):r for r in old['matches'] if key(r) in allowed});results=list(found.values())
    done={key(r) for r in results}
    def save(complete=False):
        summary={}
        for group in ['specialists','bots']:
            subset=[r for r in results if r['group']==group]
            original={key(r):r for r in selected}
            summary[group]=dict(before=benchmark.stats([original[key(r)] for r in subset]),after=benchmark.stats(subset),
                by_opponent={n:benchmark.stats([r for r in subset if r['opponent']==n]) for n in benchmark.NAMES},
                transitions={str(a):{str(b):sum(r['before']==a and r['winner']==b for r in subset) for b in [0,-1,1]} for a in [0,-1,1]})
        benchmark.save(output,dict(complete=complete,manifest=manifest,summary=summary,matches=results))
    save()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(run,r) for r in selected if key(r) not in done]):
            results.append(f.result());save()
            if len(results)%10==0:print(json.dumps(dict(completed=len(results),total=len(selected))),flush=True)
    assert len(results)==len(selected) and len({key(r) for r in results})==len(selected)
    assert all(benchmark.sha(benchmark.ROOT/p)==h for p,h in frozen.items())
    assert benchmark.sha(Path(__file__).with_name('pocket_escape_candidate.py'))==manifest['candidate_sha256']
    save(True)
    print(output.read_text() if scope=='pocket' else json.dumps(dict(complete=True,matches=len(results))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('scope',choices=['pocket','targets','all']);parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();main(args.folder,args.scope,args.workers)
