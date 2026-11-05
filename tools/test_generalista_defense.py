import os,sys,unittest,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(os.environ.get('BLOCKS_PROJECT_ROOT',str(Path(__file__).resolve().parents[1])))
sys.dont_write_bytecode=True
sys.path.extend([str(ROOT),str(ROOT/'.venv/Lib/site-packages')])
script=Path(os.environ.get('GENERALISTA_SCRIPT',str(ROOT/'MeuAlgo/algo1.py')))
spec=importlib.util.spec_from_file_location('tested_generalist',script)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from core.engine import Engine,Action,Bullet
from core.mapgen import GameMap
from rl.full_observation import FullObservationEncoder

class DefenseTests(unittest.TestCase):
    def scenario(self):
        grid=np.zeros((100,100),dtype=np.uint8)
        grid[0,:]=grid[-1,:]=grid[:,0]=grid[:,-1]=1
        eng=Engine(seed=0,game_map=GameMap(grid=grid,mud=np.zeros_like(grid)))
        eng.players[0].x=50.;eng.players[0].y=50.
        eng.players[1].x=65.;eng.players[1].y=50.
        agent=m.GeneralistAgent()
        return eng,agent

    def encode(self,eng):return FullObservationEncoder(eng).encode(0)

    def test_last_live_substep_hits_body_before_reaching_center(self):
        eng,agent=self.scenario()
        eng.bullets=[Bullet(49.3,50.,14.,0.,1,dist=19.6)]
        obs=self.encode(eng)
        self.assertEqual(m._bullet_threat(obs),0,'reproduce legacy false negative')
        self.assertEqual(agent._bullet_threat(obs),2)
        self.assertEqual(agent._movement_risk(obs,(0,0))[0],1)
        eng.step((Action(),Action()))
        self.assertEqual(eng.players[0].lives,7)

    def test_dodge_avoids_that_same_engine_hit(self):
        eng,agent=self.scenario()
        eng.bullets=[Bullet(49.3,50.,14.,0.,1,dist=19.6)]
        obs=self.encode(eng)
        move=agent._defensive_move(obs,(-1,0))
        self.assertEqual(agent._movement_risk(obs,move)[0],0)
        eng.step((Action(move_x=move[0],move_y=move[1]),Action()))
        self.assertEqual(eng.players[0].lives,8)

    def test_range_and_ttl_expiration_precede_body_collision(self):
        for distance,ttl in [(19.9,6.),(0.,.01)]:
            with self.subTest(distance=distance,ttl=ttl):
                eng,agent=self.scenario()
                eng.bullets=[Bullet(49.3,50.,14.,0.,1,dist=distance,ttl=ttl)]
                obs=self.encode(eng)
                self.assertEqual(agent._bullet_threat(obs),0)
                eng.step((Action(),Action()))
                self.assertEqual(eng.players[0].lives,8)

    def test_wall_stops_bullet_but_tree_does_not(self):
        for wall in [True,False]:
            with self.subTest(wall=wall):
                eng,agent=self.scenario()
                eng.players[0].x=51.
                if wall:eng.map.grid[49,50]=1
                else:eng.map.solid[49,50]=1
                eng.bullets=[Bullet(48.9,50.,14.,0.,1)]
                obs=self.encode(eng)
                self.assertEqual(agent._bullet_threat(obs),0 if wall else 2)
                for _ in range(6):eng.step((Action(),Action()))
                self.assertEqual(eng.players[0].lives,8 if wall else 7)

    def test_lateral_move_keeps_legal_shot(self):
        eng,agent=self.scenario()
        eng.players[1].last_shot_time=0.
        obs=self.encode(eng)
        action=agent.action_for_option(obs,m.CHASE)
        self.assertNotEqual(int(action[0])//3-1,0,'move sideways against observed fire')
        self.assertEqual(int(action[2]),1,'shoot while moving when aim is valid')

    def test_own_bullets_do_not_trigger_dodge(self):
        eng,agent=self.scenario()
        eng.bullets=[Bullet(49.3,50.,14.,0.,0,dist=19.6)]
        self.assertEqual(agent._bullet_threat(self.encode(eng)),0)

    def test_future_body_path_matches_engine_mud_walls_and_speed_expiry(self):
        eng,agent=self.scenario()
        eng.players[0].x=50.5;eng.players[0].y=50.5
        eng.players[0].speed_timer=.13
        eng.map.mud[50:53,50:53]=1
        eng.map.grid[51,51]=eng.map.solid[51,51]=1
        obs=self.encode(eng)
        agent._incoming_flights(obs)
        positions,_=agent._future_positions(obs,(1,1))
        for expected in positions:
            eng.step((Action(move_x=1,move_y=1),Action()))
            np.testing.assert_allclose(expected,[eng.players[0].x,eng.players[0].y],atol=1e-4,rtol=0)

    def test_safe_return_is_not_replaced_by_optional_circling(self):
        eng,agent=self.scenario()
        eng.players[0].x=80.
        eng.players[1].last_shot_time=0.
        eng.time=100.
        obs=self.encode(eng)
        agent._returning=True
        self.assertEqual(agent._defensive_move(obs,(-1,0)),(-1,0))

if __name__=='__main__':unittest.main()
