"""Regression tests from an actual Markov defeat, without replaying the match."""
import importlib.util,json,os,sys,unittest
from pathlib import Path
import numpy
ROOT=Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT),str(ROOT/'.venv/Lib/site-packages')])
from core.engine import Engine,PlayerState,Bullet,Action
from rl.full_observation import FullObservationEncoder
from rl.actions import to_engine_action,MOVE_LUT

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

base=module('unchanged_markov_base',ROOT/'MeuAlgo/experiments/generalista_base.py')
candidate=module('markov_candidate',Path(os.environ.get('MARKOV_DEFENSE_SCRIPT',str(ROOT/'MeuAlgo/algo1.py'))))

class MarkovDefenseTests(unittest.TestCase):
    def state(self,policy):
        f=json.loads((ROOT/'tools/fixtures/markov_immediate_hit.json').read_text())
        e=Engine(seed=f['map_seed'],map_seed=f['map_seed'])
        e.time=f['time'];e.tick=f['tick']
        e.players=[PlayerState(**p) for p in f['players']]
        e.bullets=[Bullet(**b) for b in f['bullets']]
        a=policy.GeneralistAgent()
        obs=FullObservationEncoder(e,120.).encode(0)
        return e,a,obs,f

    def test_original_move_receives_immediate_hit(self):
        e,a,obs,f=self.state(base)
        movement=MOVE_LUT[f['own_action'][0]]
        move=a._defensive_move(obs,movement)
        action=to_engine_action(f['own_action'])
        events=e.step((Action(*move,action.aim,action.shoot),to_engine_action(f['enemy_action'])))
        self.assertTrue(any(ev[0]=='hit' and ev[1]==0 for ev in events))

    def test_tie_break_avoids_hit_on_next_step(self):
        e,a,obs,f=self.state(candidate)
        movement=MOVE_LUT[f['own_action'][0]]
        move=a._defensive_move(obs,movement)
        action=to_engine_action(f['own_action'])
        events=e.step((Action(*move,action.aim,action.shoot),to_engine_action(f['enemy_action'])))
        self.assertFalse(any(ev[0]=='hit' and ev[1]==0 for ev in events))
        self.assertEqual(e.players[0].lives,8)

    def test_no_risk_keeps_original_movement(self):
        e,a,obs,f=self.state(candidate)
        e.bullets=[]
        obs=FullObservationEncoder(e,120.).encode(0)
        self.assertEqual(a._defensive_move(obs,(0,1)),(0,1))

if __name__=='__main__':unittest.main()
