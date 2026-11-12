"""Navigation-only counterfactual: a feasibility probe, not a win-rate test."""
import argparse
import math
from pathlib import Path
from evaluate_generalista_1000 import ROOT,policy,save
from core.engine import Engine,Action
from rl.full_observation import FullObservationEncoder

def main(folder):
    m=policy();engine=Engine(seed=517358468,map_seed=517358468);engine.reset(swap_spawns=True)
    agent=m.make_agent(ROOT/'MeuAlgo/weights/generalista/qtable.npz')
    encoder=FullObservationEncoder(engine,120.)
    obs=encoder.encode(0)
    end_path,end_slack=agent._end_safe_route(obs)
    regular_path,regular_slack=agent._forecast_return(obs)
    default_needed=agent._return_needed(obs)
    agent.return_lookahead_seconds=120.;agent.return_min_margin=.75
    path,slack=m.DijkstraAgent._forecast_return(agent,obs,allow_heat=True)
    assert path and not end_path and not default_needed
    node=1;events=[];frames=[];stuck=0
    while node<len(path) and engine.time<120 and engine.winner is None:
        p=engine.players[0];target=(path[node][0]+.5,path[node][1]+.5)
        if math.dist((p.x,p.y),target)<.18:
            node+=1;continue
        obs=encoder.encode(0)
        options=[]
        for mx,my in m.MOVE_LUT if hasattr(m,'MOVE_LUT') else [(x,y) for x in (-1,0,1) for y in (-1,0,1)]:
            x,y=agent._step_position(obs,mx,my)
            options.append((math.dist((x,y),target),mx,my))
        _,mx,my=min(options)
        before=(p.x,p.y)
        for e in engine.step((Action(mx,my),Action())):
            if e[0] in ['heat_hit','gameover']:events.append(dict(t=engine.time,event=list(e)))
        stuck=stuck+1 if math.dist(before,(p.x,p.y))<1e-6 else 0
        if engine.tick%6==0:frames.append(dict(t=engine.time,position=[p.x,p.y],lives=p.lives))
        if stuck>60:break
    report=dict(kind='Navigation-only feasibility probe; stationary unarmed opponent, no combat or win-rate inference',
        seed=517358468,swapped=True,default_return_needed_at_start=default_needed,
        heat_free_end_route_at_start=len(end_path),ordinary_comfort_route_cells=len(regular_path),ordinary_slack=regular_slack,
        earliest_allow_heat_route_cells=len(path),earliest_allow_heat_slack=slack,
        completed_route=node==len(path),elapsed=engine.time,own_heat_hits=sum(e['event']==['heat_hit',0] for e in events),
        endpoint=[engine.players[0].x,engine.players[0].y],events=events,frames=frames)
    save(folder/'early_route_probe.json',report)
    print({k:v for k,v in report.items() if k not in ['events','frames']})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    main(parser.parse_args().folder)
