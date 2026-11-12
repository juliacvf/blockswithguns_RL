"""Summarize observed failure mechanisms without attributing untested causality."""
import argparse
import json
from pathlib import Path
import statistics
from build_match_review import sheet
from evaluate_generalista_1000 import ROOT,sha,save

def main(folder):
    data=json.loads((folder/'diagnosis.json').read_text(encoding='utf-8'))
    assert data['complete'] and len(data['matches'])==80
    rows=data['matches'];losses=[r for r in rows if r['winner']==1];draws=[r for r in rows if r['winner']==-1]
    assert all(sha(ROOT/p)==h for p,h in data['manifest']['inputs'].items())
    all_reference=[]
    for source in data['manifest']['sources']:
        path=Path(source['path']);assert sha(path)==source['sha256']
        all_reference.extend(json.loads(path.read_text(encoding='utf-8'))['matches'])
    reference={}
    for group in ['specialists','bots']:
        items=[r for r in all_reference if r['group']==group]
        reference[group]=dict(matches=len(items),wins=sum(r['winner']==0 for r in items),draws=sum(r['winner']==-1 for r in items),losses=sum(r['winner']==1 for r in items))
        reference[group]['win_rate']=reference[group]['wins']/len(items)
    def stats(items):
        shots=sum(r['shots'][0] for r in items);hits=sum(r['hits'][0] for r in items)
        return dict(matches=len(items),wins=sum(r['winner']==0 for r in items),draws=sum(r['winner']==-1 for r in items),losses=sum(r['winner']==1 for r in items),
            own_shots=shots,own_hits=hits,hits_per_shot=hits/shots if shots else None,
            mean_visible_in_range_seconds=statistics.mean(r['metrics']['in_range_visible_seconds'] for r in items),
            mean_pressure_seconds=statistics.mean(r['metrics']['pressure_seconds'] for r in items),
            mean_over_range_seconds=statistics.mean(r['metrics']['over_20_distance_seconds'] for r in items))
    by_opponent={name:stats([r for r in rows if r['opponent']==name]) for name in ['markov','dijkstra']}
    outcome_stats={name:stats([r for r in rows if r['winner']==value]) for name,value in [('wins',0),('draws',-1),('losses',1)]}
    own_received=[e for r in losses for e in r['events'] if e['event'][0]=='hit' and e['event'][1]==0]
    heat_cases=[dict(opponent=r['opponent'],seed=r['seed'],swapped=r['swapped'],heat=r['heat'],lives=r['lives']) for r in losses if r['heat'][0]]
    summary=dict(status='diagnosis_only_official_policy_preserved',source_manifest=data['manifest'],
        exact_outcomes_verified=True,total_matches=80,reference_overall_by_group=reference,by_opponent=by_opponent,by_outcome=outcome_stats,
        loss_observations=dict(matches=len(losses),all_at_time_limit=all(abs(r['seconds']-120)<.01 for r in losses),
            had_lead=sum(r['max_lead']>0 for r in losses),one_life_margin=sum(r['lives'][1]-r['lives'][0]==1 for r in losses),
            permanent_deficit_last_15_seconds=sum(r['permanent_deficit_since'] is not None and r['permanent_deficit_since']>=105 for r in losses),
            own_heat_cases=heat_cases,received_hit_events=len(own_received),
            received_hit_events_with_post_step_distance_under_4=sum(e['distance']<4 for e in own_received),
            distance_caveat='Post-step distance can include automatic teleport; descriptive, not a projectile counterfactual'),
        draw_observations=dict(matches=len(draws),minimum_visible_in_range_seconds=min(r['metrics']['in_range_visible_seconds'] for r in draws),
            had_lead=sum(r['max_lead']>0 for r in draws),zero_hits_both=sum(r['hits']==[0,0] for r in draws),own_heat_cases=sum(r['heat'][0]>0 for r in draws)),
        interpretation_limits=['Previously evaluated seeds, used for diagnosis rather than new confirmation','In-range line of sight is geometric visibility, not a guaranteed hit opportunity','No candidate policy or training change has been validated by these replays'])
    selected=[('dijkstra',517358468,True,[0,20,40,49.6,59.6,70]),
              ('dijkstra',510396388,False,[35,90,95.8,98.2,99.6,120]),
              ('markov',517438955,True,[57,63.4,76.8,112.8,118.8,120]),
              ('markov',411510783,False,[16,40,70,89,110,120])]
    sheets=[]
    for name,seed,swapped,times in selected:
        label=f'{name}_{seed}_{int(swapped)}'
        replay=json.loads((folder/(label+'.json')).read_text(encoding='utf-8'))
        sheet(replay,times,folder/(label+'.png'))
        sheets.append(dict(opponent=name,seed=seed,swapped=swapped,path=str(folder/(label+'.png'))))
    summary['visual_review_sheets']=sheets
    summary['visually_inspected_cases']=[dict(opponent=name,seed=seed,swapped=swapped) for name,seed,swapped,_ in selected]
    probe=folder/'early_route_probe.json'
    if probe.exists():
        summary['navigation_only_probe']=json.loads(probe.read_text(encoding='utf-8'))
        summary['navigation_only_probe'].pop('frames',None)
    summary['prioritized_hypotheses']=[
        dict(priority=1,issue='Long escape delayed when no heat-free end-safe route exists',
             evidence='Two losses share map 517358468, swapped spawn and three own heat hits; at spawn end-safe forecast returns no path while ordinary return says not needed',
             proposed_experiment='Trigger a separately tested earliest low-damage escape only in this failure condition; retain current behavior when a heat-free route exists',
             validation_needed='Full combat replays of these cases, then existing wins and all opponents, followed by untouched seeds before promotion'),
        dict(priority=2,issue='Close exchanges and escape under fire',
             evidence='32 of 36 received hit events in losses have post-step distance below four blocks; one lost lead occurs during a forced escape',
             limitation='Proximity alone does not prove an avoidable hit; compare movement alternatives with cooldowns, bullets and wall constraints before changing aggression'),
        dict(priority=3,issue='Draws despite prolonged contact',
             evidence='All ten draws have at least 80.6 seconds of geometric visibility within range; 889 own shots produced twelve hits, and two matches had zero hits for either side',
             limitation='Geometric visibility does not imply a valid firing solution; inspect aim and firing-position alternatives rather than increase shooting indiscriminately')]
    summary['official_policy_and_weights_preserved']=True
    save(folder/'summary.json',summary)
    report=ROOT/'MeuAlgo/experiments/generalista_markov_dijkstra_diagnosis_20261111.json'
    save(report,summary)
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    main(parser.parse_args().folder)
