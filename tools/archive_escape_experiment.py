"""Archive a diagnosed policy experiment, keeping historical and fresh scores separate."""
import argparse
import json
from pathlib import Path
import shutil
from evaluate_generalista_1000 import ROOT,sha,save,stats,NAMES

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main(folder):
    diagnosis=read(folder/'summary.json')
    known=read(folder/'escape_candidate_all.json');fresh=read(folder/'escape_candidate_fresh.json')
    assert known['complete'] and fresh['complete']
    assert len(known['matches'])==len(fresh['matches'])==300
    candidate=ROOT/'tools/pocket_escape_candidate.py'
    assert known['manifest']['candidate_sha256']==fresh['manifest']['candidate_sha256']==sha(candidate)
    assert known['manifest']['inputs']==fresh['manifest']['inputs']
    assert all(sha(ROOT/p)==h for p,h in known['manifest']['inputs'].items())
    key=lambda r:(r['group'],r['opponent'],r['seed'],r['swapped'])
    historical_original={key(r):r for source in diagnosis['source_manifest']['sources'] for r in read(Path(source['path']))['matches']}
    assert len(historical_original)==300
    original={key(r):r for r in fresh['matches'] if r['variant']=='official'}
    changed={key(r):r for r in fresh['matches'] if r['variant']=='candidate'}
    assert len(original)==len(changed)==150 and original.keys()==changed.keys()
    assert not {r['seed'] for r in known['matches']} & {r['seed'] for r in fresh['matches']}
    transitions={g:{str(a):{str(b):sum(k[0]==g and original[k]['winner']==a and changed[k]['winner']==b for k in original) for b in [0,-1,1]} for a in [0,-1,1]} for g in ['specialists','bots']}
    changes=[dict(group=k[0],opponent=k[1],seed=k[2],swapped=k[3],before=original[k],after=changed[k]) for k in original if original[k]['winner']!=changed[k]['winner']]
    pooled={}
    for g in ['specialists','bots']:
        pooled[g]={}
        for label,role in [('before','official'),('after','candidate')]:
            parts=[known['summary'][g][label],fresh['summary'][role][g]['total']]
            totals={k:sum(p[k] for p in parts) for k in ['matches','wins','draws','losses']}
            totals['win_rate']=totals['wins']/totals['matches']
            pooled[g][label]=totals
    per_opponent={}
    for name in NAMES:
        old_rows=[r for r in known['matches'] if r['group']=='specialists' and r['opponent']==name]
        new_rows=[r for r in changed.values() if r['group']=='specialists' and r['opponent']==name]
        old_before=[historical_original[key(r)] for r in old_rows]
        new_before=[original[key(r)] for r in new_rows]
        per_opponent[name]=dict(historical_before=stats(old_before),historical_after=stats(old_rows),fresh_before=stats(new_before),fresh_after=stats(new_rows))
        per_opponent[name]['pooled_before_win_rate']=(sum(r['winner']==0 for r in old_before+new_before))/60
        per_opponent[name]['pooled_after_win_rate']=(sum(r['winner']==0 for r in old_rows+new_rows))/60
    destination=ROOT/'MeuAlgo/experiments/pocket_escape_20261111';destination.mkdir(parents=True,exist_ok=True)
    for filename in ['summary.json','manifest.json','diagnosis.json','early_route_probe.json','escape_candidate_pocket.json','escape_candidate_all.json','escape_candidate_fresh.json']:
        shutil.copyfile(folder/filename,destination/filename)
    for filename in ['pocket_escape_candidate.py','test_pocket_escape_policy.py','test_pocket_escape_candidate.py','evaluate_escape_fresh.py','diagnose_specialist_gaps.py','summarize_specialist_gaps.py','probe_early_pocket_route.py','archive_escape_experiment.py']:
        shutil.copyfile(ROOT/'tools'/filename,destination/filename)
    if (folder/'candidate_regressions.log').exists():shutil.copyfile(folder/'candidate_regressions.log',destination/'candidate_regressions.log')
    shutil.copyfile(ROOT/'MeuAlgo/algo1.py',destination/'official_policy_frozen.py')
    reference=fresh['summary']['official']['specialists']['total'];result=fresh['summary']['candidate']['specialists']['total']
    report=dict(status='diagnosis_and_isolated_candidate_evaluation_completed',diagnosis=diagnosis,
        change='Earlier escape when no heat-free route to an end-safe area exists; unchanged weights, shooting and combat controllers',
        historical_evaluation=known['summary'],fresh_evaluation=fresh['summary'],fresh_outcome_transitions=transitions,fresh_changed_outcomes=changes,
        pooled_evaluation=pooled,per_specialist=per_opponent,
        fresh_specialist_wins_improved=result['wins']>reference['wins'],
        official_policy_and_weights_preserved=True,candidate_applied_to_official_policy=False,
        candidate_sha256=sha(candidate),artifact_directory=str(folder),archive_directory=str(destination),
        verification=dict(existing_navigation_and_defense_tests_passed=20,targeted_escape_tests_passed=3,paired_historical_games=300,paired_fresh_games=150),
        limitations=['Historical cohorts were used to diagnose and select this candidate; keep their gain separate from fresh confirmation',
            'One fixed candidate and one fresh cohort, twenty games per specialist; rates are sample results rather than guarantees',
            'No claim that all remaining draws or close-range hits have been resolved'])
    output=ROOT/'MeuAlgo/experiments/generalista_pocket_escape_20261111_results.json'
    save(output,report)
    print(json.dumps(dict(report=str(output),historical=known['summary'],fresh=fresh['summary'],transitions=transitions,pooled=pooled),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    main(parser.parse_args().folder)
