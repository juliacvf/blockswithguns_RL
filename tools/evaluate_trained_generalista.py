"""Compare frozen training checkpoints on fixed validation/holdout maps."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import random
import time
import numpy as np
from evaluate_generalista_1000 import ROOT, NAMES, policy, sha, stats, save
from core.engine import Engine
from bots import make_bot, make_view
from rl.actions import to_engine_action
from rl.full_observation import FullObservationEncoder


def duel(job):
    weight,group,name,seed,swapped=job
    module=policy()
    agent=module.make_agent(Path(weight))
    before=agent.q.copy()
    opponent=(make_bot(name,seed=seed+1) if group=='bots' else module.make_agent(ROOT/f'MeuAlgo/weights/especialistas/{name}.npz'))
    agent.reset();opponent.reset()
    engine=Engine(seed=seed,map_seed=seed)
    if swapped:engine.reset(swap_spawns=True)
    encoder=FullObservationEncoder(engine,120.)
    shots,hits,heat=[0,0],[0,0],[0,0]
    while engine.winner is None:
        own=to_engine_action(agent.act(encoder.encode(0)))
        enemy=opponent.act(make_view(engine,1)) if group=='bots' else to_engine_action(opponent.act(encoder.encode(1)))
        for event in engine.step((own,enemy)):
            if event[0]=='shot':shots[event[1]]+=1
            elif event[0]=='hit':hits[event[2]]+=1
            elif event[0]=='heat_hit':heat[event[1]]+=1
    assert np.array_equal(before,agent.q),'Evaluation updated Q-table'
    return dict(checkpoint=Path(weight).stem,group=group,opponent=name,seed=seed,swapped=swapped,
                winner=engine.winner,lives=[p.lives for p in engine.players],seconds=engine.time,
                ticks=engine.tick,shots=shots,hits=hits,heat=heat)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    parser.add_argument('mode',choices=['validation','holdout']);parser.add_argument('checkpoint')
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--stop-if-dominated',action='store_true',help='Stop scheduling games once the reference cannot be beaten')
    parser.add_argument('--finish-rejected',action='store_true',help='Finish a previously pruned evaluation without discarding its recorded games')
    args=parser.parse_args()
    assert not (args.stop_if_dominated and args.finish_rejected)
    weight=args.folder/(args.checkpoint+'.npz');assert weight.is_file()
    training=json.loads((args.folder/'manifest.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in training['input_hashes'].items())
    protocol=training.get('evaluation_protocol',{}).get(args.mode,{})
    rng=random.Random(protocol.get('generator_seed',2026111101 if args.mode=='validation' else 2026111102))
    offset=protocol.get('offset',410000000 if args.mode=='validation' else 510000000)
    seeds=dict(specialists=rng.sample(range(offset,offset+10000000),10),bots=rng.sample(range(offset+20000000,offset+30000000),5))
    assert not set(training['actual_train_map_seeds']) & {s for values in seeds.values() for s in values}
    manifest=dict(checkpoint=str(weight),weight_sha256=sha(weight),official_inputs=training['input_hashes'],
                  mode=args.mode,map_seeds=seeds,both_spawn_sides=True,maximum_seconds=120,weights_frozen=True)
    reference=None
    if args.stop_if_dominated:
        assert args.checkpoint!='baseline'
        reference=json.loads((args.folder/(args.mode+'_baseline.json')).read_text())
        assert reference['complete']
        manifest['stop_rule']='Maximize specialist wins, then minimize losses; stop only when the reference is unreachable'
    output=args.folder/(args.mode+'_'+args.checkpoint+'.json')
    rows=[]
    if output.exists():
        old=json.loads(output.read_text())
        old_manifest=dict(old['manifest'])
        if args.finish_rejected:
            old_manifest.pop('stop_rule',None)
        assert old_manifest==manifest
        if old.get('stopped_early') and not args.finish_rejected:
            print(json.dumps(old['rejection']),flush=True)
            raise SystemExit(0)
        rows=old['matches']
    key=lambda r:(r['group'],r['opponent'],r['seed'],r['swapped'])
    done={key(r) for r in rows}
    jobs=[(str(weight),g,n,s,swap) for g,values in seeds.items() for s in values for n in NAMES for swap in (False,True) if (g,n,s,swap) not in done]
    rejection=None
    def report(complete=False):
        return dict(complete=complete,manifest=manifest,summary={g:dict(total=stats([r for r in rows if r['group']==g]),
            by_opponent={n:stats([r for r in rows if r['group']==g and r['opponent']==n]) for n in NAMES}) for g in seeds},
            stopped_early=rejection is not None,rejection=rejection,matches=sorted(rows,key=key))
    save(output,report());start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        waiting=iter(jobs)
        pending=set()
        def submit_next():
            job=next(waiting,None)
            if job is not None:pending.add(pool.submit(duel,job))
        for _ in range(args.workers):submit_next()
        while pending:
            finished,_=wait(pending,return_when=FIRST_COMPLETED)
            for f in finished:
                pending.remove(f);rows.append(f.result())
                if reference is not None and rejection is None:
                    target=reference['summary']['specialists']['total']
                    current=stats([r for r in rows if r['group']=='specialists'])
                    maximum=target['matches']-current['draws']-current['losses']
                    if maximum<target['wins'] or (maximum==target['wins'] and current['losses']>target['losses']):
                        rejection=dict(reason='Cannot beat reference specialist wins, then losses',
                            maximum_possible_wins=maximum,baseline_wins=target['wins'],baseline_losses=target['losses'],partial=current)
                        save(args.folder/(args.checkpoint+'_rejection.json'),rejection)
                        print(json.dumps(rejection),flush=True)
                save(output,report())
                if len(rows)%10==0:print(json.dumps(dict(checkpoint=args.checkpoint,completed=len(rows),elapsed=round(time.monotonic()-start))),flush=True)
                if rejection is None:submit_next()
    assert len(rows)==len({key(r) for r in rows})
    assert len(rows)==150 or rejection is not None
    assert sha(weight)==manifest['weight_sha256']
    assert all(sha(ROOT/p)==h for p,h in training['input_hashes'].items())
    save(output,report(rejection is None));print(json.dumps(report()['summary']),flush=True)
