"""Check opportunistic collection and use of short-lived combat bonuses."""
import importlib.util,os,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT),str(ROOT/'.venv/Lib/site-packages')])
from core.engine import Engine
from core.mapgen import GameMap
from core.powerups import Powerup
from rl.full_observation import FullObservationEncoder
path=Path(os.environ.get('GENERALISTA_SCRIPT',str(ROOT/'MeuAlgo/algo1.py')))
spec=importlib.util.spec_from_file_location('bonus_candidate',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class PowerupPolicyTests(unittest.TestCase):
    def scenario(self):
        grid=np.zeros((100,100),dtype=np.uint8)
        e=Engine(seed=0,game_map=GameMap(grid=grid,mud=np.zeros_like(grid)))
        e.time=50.;e.tick=1500
        e.players[0].x=e.players[0].y=50.
        e.players[1].x=65.;e.players[1].y=50.
        a=m.GeneralistAgent()
        return e,a

    def observe(self,e,a):
        obs=FullObservationEncoder(e,120.).encode(0);a.observe(obs);return obs

    def test_collect_nearby_quickshot_when_tied(self):
        e,a=self.scenario();e.powerups.items=[Powerup(52.5,50.5,'QUICKSHOT')]
        np.testing.assert_allclose(a._nearby_advantage(self.observe(e,a)),(52.5,50.5),atol=1e-4)

    def test_do_not_abandon_health_lead_for_bonus(self):
        e,a=self.scenario();e.players[1].lives=7
        e.powerups.items=[Powerup(52.5,50.5,'QUICKSHOT')]
        self.assertIsNone(a._nearby_advantage(self.observe(e,a)))

    def test_ignore_swap_and_unneeded_life(self):
        e,a=self.scenario()
        e.powerups.items=[Powerup(51.5,50.5,'SWAP'),Powerup(52.5,50.5,'LIFE')]
        self.assertIsNone(a._nearby_advantage(self.observe(e,a)))

    def test_prioritize_reachable_healing_when_behind(self):
        e,a=self.scenario();e.players[0].lives=7
        e.powerups.items=[Powerup(52.5,50.5,'LIFE'),Powerup(53.5,50.5,'QUICKSHOT')]
        np.testing.assert_allclose(a._nearby_advantage(self.observe(e,a)),(52.5,50.5),atol=1e-4)

    def test_ignore_expiring_bonus(self):
        e,a=self.scenario();e.powerups.items=[Powerup(54.5,50.5,'QUICKSHOT',ttl=.1)]
        self.assertIsNone(a._nearby_advantage(self.observe(e,a)))

    def test_ignore_bonus_near_closing_zone(self):
        e,a=self.scenario();e.time=100.;e.tick=3000
        e.players[0].x=68.;e.players[1].x=56.
        e.powerups.items=[Powerup(74.5,50.5,'QUICKSHOT')]
        self.assertIsNone(a._nearby_advantage(self.observe(e,a)))

    def test_quickshot_opens_attack_without_waiting_for_stall(self):
        e,a=self.scenario();e.players[0].quickshot_timer=4.
        obs=self.observe(e,a);a._tactic_last=a._previous_time
        self.assertEqual(a._boost_window(obs),2)
        self.assertEqual(a._tactic_trigger(obs),'temporary_power_advantage')

    def test_no_bonus_advantage_when_opponent_has_same_bonus(self):
        e,a=self.scenario();e.players[0].quickshot_timer=e.players[1].quickshot_timer=4.
        self.assertEqual(a._boost_window(self.observe(e,a)),0)

    def test_quickshot_advantage_allows_contact_before_stall(self):
        e,a=self.scenario();e.players[1].x=60.;e.players[0].quickshot_timer=4.
        obs=self.observe(e,a)
        self.assertTrue(a._close_pressure_needed(obs))

    def test_do_not_force_contact_when_already_winning(self):
        e,a=self.scenario();e.players[1].x=60.;e.players[1].lives=7
        e.players[0].quickshot_timer=4.
        obs=self.observe(e,a)
        self.assertEqual(a._boost_window(obs),0)
        self.assertFalse(a._close_pressure_needed(obs))

    def test_exit_deadline_takes_priority_over_nearby_healing(self):
        e=Engine(seed=7193382,map_seed=7193382)
        e.players[0].lives=7
        e.powerups.items=[Powerup(17.5,15.5,'LIFE')]
        a=m.GeneralistAgent();obs=self.observe(e,a)
        self.assertIsNone(a._nearby_advantage(obs))
        self.assertTrue(a._escape_active)

if __name__=='__main__':unittest.main()
