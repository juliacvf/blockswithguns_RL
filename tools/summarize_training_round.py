"""Archive actual learned checkpoints and paired evaluations without promotion."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def paired_summary(rows):
    def stats(items):
        n=len(items);wins=sum(r['winner']==0 for r in items);draws=sum(r['winner']==-1 for r in items)
        return dict(matches=n,wins=wins,draws=draws,losses=n-wins-draws,win_rate=wins/n if n else None)
    return {g:stats([r for r in rows if r['group']==g]) for g in ['specialists','bots']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    parser.add_argument('--name',default='training_20261111')
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--checkpoint',choices=['checkpoint_040','checkpoint_080','checkpoint_120'],default='checkpoint_120')
    args=parser.parse_args()
    assert Path(args.name).name==args.name and args.name not in ('.','..')
    folder=args.folder
    manifest=read(folder/'manifest.json');progress=read(folder/'progress.json')
    before=read(folder/'validation_baseline.json');after=read(folder/f'validation_{args.checkpoint}.json')
    assert progress['complete'] and progress['training_matches']==120
    rejection_path=folder/(args.checkpoint+'_rejection.json')
    final_rejection=read(rejection_path) if rejection_path.exists() and not after['complete'] else None
    assert before['complete'] and (after['complete'] or final_rejection)
    assert all(sha(ROOT/p)==h for p,h in manifest['input_hashes'].items()),'Official input changed'
    key=lambda r:(r['group'],r['opponent'],r['seed'],r['swapped'])
    assert len(before['matches'])==150
    originals={key(r):r for r in before['matches']}
    assert {key(r) for r in after['matches']}<=originals.keys()
    assert len(after['matches'])==len({key(r) for r in after['matches']})
    if after['complete']:assert len(after['matches'])==150
    transitions={g:{str(a):{str(b):sum(r['group']==g and originals[key(r)]['winner']==a and r['winner']==b for r in after['matches']) for b in [0,-1,1]} for a in [0,-1,1]} for g in ['specialists','bots']}
    new_score=after['summary']['specialists']['total'];old_score=before['summary']['specialists']['total']
    new_min=min((v['win_rate'] for v in after['summary']['specialists']['by_opponent'].values() if v['win_rate'] is not None),default=0.)
    old_min=min(v['win_rate'] for v in before['summary']['specialists']['by_opponent'].values())
    quality_gain=(new_score['wins']>old_score['wins'] or
                  (new_score['wins']==old_score['wins'] and new_score['losses']<old_score['losses']) or
                  (new_score['wins']==old_score['wins'] and new_score['losses']==old_score['losses'] and new_min>old_min))
    constraints_met=(after['complete'] and
        after['summary']['bots']['total']['wins']>=before['summary']['bots']['total']['wins'] and
        all(v['win_rate'] is not None and v['win_rate']>=.75 for v in after['summary']['specialists']['by_opponent'].values()))
    validation_improved=bool(after['complete'] and quality_gain and constraints_met)
    holdout=None
    if validation_improved:
        h1=read(folder/'holdout_baseline.json');h2=read(folder/f'holdout_{args.checkpoint}.json')
        assert h1['complete'] and h2['complete'],'Independent confirmation is required before reporting improvement'
        assert {key(r) for r in h1['matches']}=={key(r) for r in h2['matches']}
        assert len(h1['matches'])==len(h2['matches'])==150
        assert h1['manifest']['map_seeds']==h2['manifest']['map_seeds']
        independent_seeds={s for values in h1['manifest']['map_seeds'].values() for s in values}
        validation_seeds={s for values in before['manifest']['map_seeds'].values() for s in values}
        assert not independent_seeds & (validation_seeds | set(manifest['actual_train_map_seeds']))
        holdout=dict(before=h1['summary'],after=h2['summary'],before_matches=h1['matches'],after_matches=h2['matches'])
    pooled=None
    if holdout is not None:
        pooled={}
        for role,first in [('before',before),('after',after)]:
            pooled[role]={}
            for group in ['specialists','bots']:
                parts=[first['summary'][group]['total'],holdout[role][group]['total']]
                counts={k:sum(part[k] for part in parts) for k in ['matches','wins','draws','losses']}
                counts['win_rate']=counts['wins']/counts['matches']
                pooled[role][group]=counts
    checkpoint_info={}
    with np.load(folder/'baseline.npz',allow_pickle=False) as data:
        q0=data['q'].copy();visits0=data['visits'].copy()
    for n in [40,80,120]:
        p=folder/f'checkpoint_{n:03d}.npz'
        with np.load(p,allow_pickle=False) as data:
            q=data['q'];visits=data['visits']
            assert q.shape==q0.shape and np.isfinite(q).all()
            assert str(data['profile'].item())=='generalista_v1'
            assert int(data['episodes'])==manifest['episodes_before']+n
            assert np.all(visits>=visits0)
            checkpoint_info[str(n)]=dict(sha256=sha(p),changed_q_entries=int(np.count_nonzero(q!=q0)),
                maximum_absolute_q_change=float(np.max(np.abs(q-q0))),mean_absolute_q_change=float(np.mean(np.abs(q-q0))),
                episodes=int(data['episodes']))
    destination=ROOT/'MeuAlgo/experiments'/args.name
    if (destination/'manifest.json').exists():
        assert read(destination/'manifest.json')==manifest,'Destination belongs to a different training run'
    destination.mkdir(parents=True,exist_ok=True)
    for p in folder.glob('*.json'):shutil.copyfile(p,destination/p.name)
    for name in ['baseline.npz','checkpoint_040.npz','checkpoint_080.npz','checkpoint_120.npz']:
        shutil.copyfile(folder/name,destination/name)
    for name in ['control_gamma099_120.npz','train_before_gamma_parameter.py']:
        if (folder/name).exists():shutil.copyfile(folder/name,destination/name)
    shutil.copyfile(ROOT/'MeuAlgo/algo1.py',destination/'policy_frozen.py')
    rejected={n:read(folder/f'checkpoint_{n:03d}_rejection.json') for n in [40,80,120] if (folder/f'checkpoint_{n:03d}_rejection.json').exists()}
    evaluations={}
    for mode in ['validation','holdout']:
        for label in ['baseline','checkpoint_040','checkpoint_080','checkpoint_120']:
            path=folder/f'{mode}_{label}.json'
            if path.exists():
                data=read(path)
                evaluations[f'{mode}_{label}']=dict(complete=data['complete'],stopped_early=data.get('stopped_early',False),
                    matches=len(data['matches']),summary=data['summary'],file=path.name,sha256=sha(path))
    previous_comparison=None
    if args.previous is not None:
        previous=read(args.previous)
        prior_inputs=previous['training']['manifest']['input_hashes']
        assert prior_inputs.keys()==manifest['input_hashes'].keys()
        training_source_changes={p:dict(previous=prior_inputs[p],current=h) for p,h in manifest['input_hashes'].items() if prior_inputs[p]!=h}
        assert all(Path(p)==Path('MeuAlgo/train.py') for p in training_source_changes),'Inference input changed'
        old={key(r):r for r in previous['validation']['after_matches']}
        new={key(r):r for r in after['matches']}
        shared=old.keys() & new.keys()
        previous_comparison=dict(previous_report=str(args.previous),common_matches=len(shared),training_source_changes=training_source_changes,
            previous_checkpoint=previous['validation'].get('selected_checkpoint','checkpoint_120'),current_checkpoint=args.checkpoint,
            reference=paired_summary([originals[k] for k in shared]),
            previous_training=paired_summary([old[k] for k in shared]),
            current_training=paired_summary([new[k] for k in shared]))
    report=dict(date='2026-11-11',status='actual_training_completed_official_preserved',
        training=dict(manifest=manifest,actual_result=progress,checkpoint_info=checkpoint_info,
                      gamma_per_tick=manifest.get('gamma',.99),
                      effective_epsilon_first_last=[max(.04,manifest['epsilon_start']*.97**manifest['episodes_before']),max(.04,manifest['epsilon_start']*.97**(manifest['episodes_before']+119))],
                      by_group={g:dict(Counter('win' if r['winner']==0 else 'draw' if r['winner']==-1 else 'loss' for r in progress['matches'] if r['group']==g)) for g in ['specialists','bots']},
                      trainers=dict(existing=sha(ROOT/'MeuAlgo/train.py'),curriculum=sha(ROOT/'tools/train_generalista_round.py'))),
        validation=dict(before=before['summary'],after=after['summary'],transitions=transitions,
                        selected_checkpoint=args.checkpoint,
                        qualification_constraints_met=constraints_met,
                        minimum_specialist_win_rate=dict(before=old_min,after=new_min),
                        paired_before=paired_summary([originals[key(r)] for r in after['matches']]),
                        paired_after=paired_summary(after['matches']),
                        selected_comparison_complete=after['complete'],early_rejection=final_rejection,
                        selected_weights_improved=validation_improved,before_matches=before['matches'],after_matches=after['matches']),
        checkpoint_rejections=rejected,checkpoint_evaluations=evaluations,
        holdout=holdout,confirmation_status='completed' if holdout is not None else 'not_run_validation_did_not_improve',
        pooled_evaluation=pooled,previous_training_comparison=previous_comparison,
        official_weights_unchanged=True,official_policy_and_weights_preserved=True,
        frozen_inputs_verified_unchanged_during_run=True,
        model_directory=str(destination),
        limitations='A pilot with 120 additional learning games; training win rates are not frozen evaluation scores. Evaluations stopped early only after their best possible result could no longer beat the reference (maximize specialist wins, then minimize losses). Partial counts must not be treated as completed 100-game scores. No policy or reward changes.',
        artifact_folder=str(folder))
    output=ROOT/'MeuAlgo/experiments'/f'generalista_{args.name}_results.json'
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(training_matches=120,validation_improved=validation_improved,
        before=before['summary'],after=after['summary'],official_preserved=True,report=str(output))))
