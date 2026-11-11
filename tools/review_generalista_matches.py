"""Reproduce losses/draws without altering the frozen policy; export replay data."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import json
import math
from pathlib import Path
import sys

from evaluate_generalista_1000 import ROOT, policy, sha, save
from core.engine import Engine
from core.los import has_line_of_sight
from rl.actions import to_engine_action
from rl.full_observation import FullObservationEncoder


def review(job):
    expected, folder = job
    module = policy()
    agent = module.make_agent(ROOT/'MeuAlgo/weights/generalista/qtable.npz')
    enemy = module.make_agent(ROOT/f"MeuAlgo/weights/especialistas/{expected['opponent']}.npz")
    agent.reset(); enemy.reset()
    engine = Engine(seed=expected['seed'], map_seed=expected['seed'])
    if expected['swapped']:
        engine.reset(swap_spawns=True)
    encoder = FullObservationEncoder(engine, 120.)
    frames, events, gaps = [], [], []
    metrics = Counter()
    damage = []
    def frame():
        players = [[round(p.x,3),round(p.y,3),p.lives,round(p.aim,3),round(p.quickshot_timer,2),round(p.speed_timer,2)] for p in engine.players]
        return dict(t=round(engine.time,3), p=players,
            b=[[round(b.x,2),round(b.y,2),b.owner] for b in engine.bullets if b.alive],
            u=[[p.x,p.y,p.kind] for p in engine.powerups.items if p.alive],
            phase=getattr(agent,'_tactic_phase',''), mode=getattr(agent,'_response_mode',''),
            goal=getattr(agent,'_tactic_goal',None), returning=bool(getattr(agent,'_returning',False)),
            escape=bool(getattr(agent,'_escape_active',False)),
            pickup=getattr(agent,'_pickup_choice',None))
    frames.append(frame())
    shots, hits, heat = [0,0], [0,0], [0,0]
    while engine.winner is None:
        before = [(p.x,p.y) for p in engine.players]
        actions = (to_engine_action(agent.act(encoder.encode(0))), to_engine_action(enemy.act(encoder.encode(1))))
        tick_events = engine.step(actions)
        a,b = engine.players
        dt=1/30
        gap=a.lives-b.lives
        gaps.append((engine.time,gap))
        metrics['ahead_seconds' if gap>0 else 'behind_seconds' if gap<0 else 'tied_seconds']+=dt
        distance=math.hypot(a.x-b.x,a.y-b.y)
        visible=has_line_of_sight(engine.grid,a.x,a.y,b.x,b.y)
        metrics['visible_seconds']+=dt*visible
        metrics['in_range_visible_seconds']+=dt*(visible and distance<=20)
        metrics['heat_seconds']+=dt*engine.in_heat(a.x,a.y)
        metrics['over_20_distance_seconds']+=dt*(distance>20)
        metrics['pressure_seconds']+=dt*(getattr(agent,'_tactic_phase','')=='pressure')
        metrics['boost_seconds']+=dt*(a.quickshot_timer>0 or a.speed_timer>0)
        for ev in tick_events:
            if ev[0]=='shot':shots[ev[1]]+=1
            elif ev[0]=='hit':hits[ev[2]]+=1
            elif ev[0]=='heat_hit':heat[ev[1]]+=1
            if ev[0] in ('hit','heat_hit','pickup','retreat','gameover'):
                item=dict(t=round(engine.time,3),event=list(ev),lives=[a.lives,b.lives],
                          positions=[[round(p.x,3),round(p.y,3)] for p in engine.players],
                          distance=round(distance,3),phase=getattr(agent,'_tactic_phase',''),
                          mode=getattr(agent,'_response_mode',''),
                          returning=bool(getattr(agent,'_returning',False)))
                events.append(item)
                if ev[0] in ('hit','heat_hit'):damage.append(item)
        if engine.tick%6==0 or engine.winner is not None:frames.append(frame())
    assert engine.winner==expected['winner'] and [a.lives,b.lives]==expected['lives']
    assert shots==expected['shots'] and hits==expected['hits'] and heat==expected['heat']
    permanent_deficit=next((round(t,3) for t,g in gaps if g<0 and t>max((s for s,h in gaps if h>=0),default=0)),None)
    label=f"{expected['opponent']}_{expected['seed']}_{int(expected['swapped'])}"
    summary=dict(**expected,metrics={k:round(v,3) for k,v in metrics.items()},
                 max_lead=max(g for _,g in gaps),max_deficit=min(g for _,g in gaps),
                 permanent_deficit_since=permanent_deficit,events=events,
                 tactic_events=getattr(agent,'_tactic_events',[]),
                 response_events=getattr(agent,'_response_events',[]),
                 escape_events=getattr(agent,'_escape_events',[]),replay=label+'.json')
    replay=dict(summary=summary,grid=engine.grid.tolist(),mud=engine.map.mud.tolist(),frames=frames)
    save(Path(folder)/(label+'.json'),replay)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    source=json.loads((ROOT/'MeuAlgo/experiments/generalista_1000_new_seeds_results.json').read_text())
    assert source['complete']
    assert all(sha(ROOT/p)==h for p,h in source['manifest']['inputs'].items())
    args.output.mkdir(parents=True,exist_ok=True)
    selected=sorted([r for r in source['matches'] if r['winner']!=0],key=lambda r:(r['winner']!=1,r['opponent'],r['seed']))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(review,(r,str(args.output))) for r in selected]):
            rows.append(f.result())
            save(args.output/'diagnosis.json',dict(complete=False,matches=rows))
            print(json.dumps(dict(completed=len(rows),total=len(selected))),flush=True)
    assert all(sha(ROOT/p)==h for p,h in source['manifest']['inputs'].items())
    save(args.output/'diagnosis.json',dict(complete=True,inputs=source['manifest'],matches=rows))
