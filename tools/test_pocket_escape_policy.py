"""Check the final escape trigger against the preserved pre-promotion policy."""
import importlib.util
import unittest
import numpy as np
from evaluate_generalista_1000 import ROOT,policy
from core.engine import Engine
from rl.full_observation import FullObservationEncoder

m=policy();Markov=m.MarkovAgent
spec=importlib.util.spec_from_file_location('before_pocket_promotion',ROOT/'MeuAlgo/experiments/pocket_escape_20261111/official_policy_frozen.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
Official=previous.GeneralistAgent
WEIGHTS=ROOT/'MeuAlgo/weights/generalista/qtable.npz'

class EscapePolicyTests(unittest.TestCase):
    def test_pocket_triggers_before_first_heat_damage(self):
        engine=Engine(seed=517358468,map_seed=517358468);engine.reset(swap_spawns=True)
        obs=FullObservationEncoder(engine,120.).encode(0)
        official=Official(WEIGHTS);candidate=m.make_agent(WEIGHTS)
        self.assertFalse(official._return_needed(obs))
        self.assertFalse(candidate._end_safe_route(obs)[0])
        self.assertTrue(candidate._return_needed(obs))
        self.assertTrue(candidate._pocket_active)
        self.assertGreater(len(candidate._return_path),1)
        self.assertEqual(candidate._pocket_events[0]['t'],0.)

    def test_heat_free_route_keeps_official_action(self):
        engine=Engine(seed=1250021,map_seed=1250021)
        engine.time=70.;engine.tick=2100
        engine.players[0].x,engine.players[0].y=34.18,36.57
        obs=FullObservationEncoder(engine,120.).encode(0)
        official=Official(WEIGHTS);candidate=m.make_agent(WEIGHTS)
        self.assertTrue(candidate._end_safe_route(obs)[0])
        np.testing.assert_array_equal(official.act(obs),candidate.act(obs))
        self.assertFalse(candidate._pocket_active)

    def test_reset_clears_escape_and_specialists_keep_their_class(self):
        engine=Engine(seed=517358468,map_seed=517358468);engine.reset(swap_spawns=True)
        candidate=m.make_agent(WEIGHTS)
        candidate._return_needed(FullObservationEncoder(engine,120.).encode(0))
        candidate.reset()
        self.assertFalse(candidate._pocket_active)
        self.assertEqual(candidate._pocket_events,[])
        self.assertIs(m.policy_class('markov_v1'),Markov)

if __name__=='__main__':unittest.main()
