"""Replay the last November 10 cohorts with the fixed escape candidate."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
from pathlib import Path
import shutil
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import evaluate_generalista_1000 as benchmark
from pocket_escape_candidate import install

INSTALLED=False
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def key(r):return r['group'],r['opponent'],r['seed'],r['swapped']
def run(row):
    global INSTALLED
    if not INSTALLED:install(benchmark.policy());INSTALLED=True
    result=benchmark.duel(key(row))
    return dict(result,cohort=row['cohort'],before=row['winner'])

def main(folder,source,workers):
    folder.mkdir(parents=True,exist_ok=True)
    root=benchmark.ROOT
    candidate=root/'tools/pocket_escape_candidate.py'
    previous=read(root/'MeuAlgo/experiments/generalista_pocket_escape_20261111_results.json')
    assert benchmark.sha(candidate)==previous['candidate_sha256']
    baseline_source=source/'deny_full_only.py'
    assert benchmark.sha(baseline_source)==benchmark.sha(root/'MeuAlgo/algo1.py')
    frozen={str(root/p):h for p,h in previous['diagnosis']['source_manifest']['inputs'].items()}
    frozen[str(candidate)]=benchmark.sha(candidate)
    frozen[str(baseline_source)]=benchmark.sha(baseline_source)
    assert all(benchmark.sha(Path(p))==h for p,h in frozen.items())
    selected=[];provenance=[]
    for cohort,filename,count in [('last_new_maps','holdout_third_reference_deny_full_only.json',100),('last_large_specialists','specialists_deny_full_only.json',500)]:
        path=source/filename;data=read(path)
        assert data['complete']
        rows=[dict(r,cohort=cohort) for r in data['matches'] if r['candidate']=='deny_full_only']
        assert len(rows)==count and len({key(r) for r in rows})==count
        assert data['manifest']['inputs'][str(baseline_source)]==benchmark.sha(baseline_source)
        old_script=root/'MeuAlgo/algo1.py'
        for p,h in data['manifest']['inputs'].items():
            if Path(p)==old_script:continue
            assert benchmark.sha(Path(p))==h,p
        provenance.append(dict(cohort=cohort,source=str(path),sha256=benchmark.sha(path),matches=count,
            baseline_candidate='deny_full_only',baseline_candidate_policy_sha256=benchmark.sha(baseline_source),
            excluded_reference_script_entry=dict(path=str(old_script),historical_sha256=data['manifest']['inputs'][str(old_script)],
                reason='Rows selected are deny_full_only, whose source exactly equals current official policy; the historical main-script entry describes the earlier reference policy')))
        selected.extend(rows)
    assert len({key(r) for r in selected})==600
    original={key(r):r for r in selected}
    manifest=dict(inputs=frozen,candidate_sha256=benchmark.sha(candidate),sources=provenance,
        selection='Last new-map cohort and last completed large specialist cohort from November 10',
        identical_maps_opponents_spawns=True,maximum_seconds=120,training=False,official_policy_preserved=True)
    output=folder/'results.json';results=[]
    if output.exists():
        old=read(output);assert old['manifest']==manifest;results=old['matches']
    done={key(r) for r in results}
    def report(complete=False):
        summary={}
        for cohort in ['last_new_maps','last_large_specialists']:
            summary[cohort]={}
            for group in ['specialists','bots']:
                rows=[r for r in results if r['cohort']==cohort and r['group']==group]
                before=[original[key(r)] for r in rows]
                summary[cohort][group]=dict(before=benchmark.stats(before),after=benchmark.stats(rows),
                    by_opponent={n:dict(before=benchmark.stats([r for r in before if r['opponent']==n]),after=benchmark.stats([r for r in rows if r['opponent']==n])) for n in benchmark.NAMES},
                    transitions={str(a):{str(b):sum(r['before']==a and r['winner']==b for r in rows) for b in [0,-1,1]} for a in [0,-1,1]})
        return dict(complete=complete,manifest=manifest,summary=summary,matches=results,
            changed_outcomes=[dict(before=original[key(r)],after=r) for r in results if r['before']!=r['winner']])
    benchmark.save(output,report())
    for cohort in ['last_new_maps','last_large_specialists']:
        jobs=[r for r in selected if r['cohort']==cohort and key(r) not in done]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for f in as_completed([pool.submit(run,r) for r in jobs]):
                results.append(f.result());benchmark.save(output,report())
                if len(results)%20==0:print(json.dumps(dict(completed=len(results),total=600,cohort=cohort)),flush=True)
        print(json.dumps(dict(cohort_finished=cohort,summary=report()['summary'][cohort])),flush=True)
    assert len(results)==len({key(r) for r in results})==600
    assert all(benchmark.sha(Path(p))==h for p,h in frozen.items())
    assert all(benchmark.sha(Path(p['source']))==p['sha256'] for p in provenance)
    final=report(True);benchmark.save(output,final)
    destination=root/'MeuAlgo/experiments/pocket_escape_yesterday_seeds_20261111';destination.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(output,destination/'results.json')
    shutil.copyfile(candidate,destination/candidate.name)
    shutil.copyfile(Path(__file__),destination/Path(__file__).name)
    report_path=root/'MeuAlgo/experiments/generalista_pocket_escape_yesterday_seeds_20261111_results.json'
    benchmark.save(report_path,dict(final,artifact_directory=str(folder),archive_directory=str(destination),limitations='Previously tested seeds; a deterministic replay comparison, not a new independent confirmation. Candidate and official policy unchanged.'))
    print(json.dumps(dict(complete=True,report=str(report_path),summary=final['summary'])),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('source',type=Path);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();main(args.folder,args.source,args.workers)
