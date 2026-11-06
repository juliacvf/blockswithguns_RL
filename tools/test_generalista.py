import sys, os, importlib.util, unittest
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT
sys.dont_write_bytecode=True
sys.path.extend([str(ROOT),str(ROOT/'.venv/Lib/site-packages')])
spec=importlib.util.spec_from_file_location('generalist_module',ROOT/'MeuAlgo/algo1.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from rl.bot_training_env import BlocksWithGunsBotTrainingEnv
from rl.actions import validate_action

class GeneralistTests(unittest.TestCase):
    def test_safe_shelter_does_not_require_a_hot_detour_for_comfort(self):
        from core.engine import Engine
        from rl.full_observation import FullObservationEncoder
        engine=Engine(seed=1250021)
        engine.time=70.;engine.tick=2100
        engine.players[0].x,engine.players[0].y=34.18,36.57
        obs=FullObservationEncoder(engine).encode(0)
        previous=m.MarkovAgent();previous._returning=True
        self.assertTrue(previous._return_needed(obs))
        agent=m.GeneralistAgent();agent._returning=True
        self.assertFalse(agent._return_needed(obs))
        self.assertTrue(agent._return_path)
        self.assertEqual(agent.return_lookahead_seconds,40.)
        self.assertEqual(agent.return_min_margin,8.)

    def test_shelter_margin_matches_the_remaining_match_time(self):
        from core.engine import Engine
        from rl.full_observation import FullObservationEncoder
        engine=Engine(seed=1250021)
        engine.time=110.;engine.tick=3300
        engine.players[0].x,engine.players[0].y=34.18,36.57
        obs=FullObservationEncoder(engine).encode(0)
        agent=m.GeneralistAgent()
        self.assertFalse(agent._return_needed(obs))
        x,y=agent._return_path[-1]
        self.assertGreater(agent._safe_margin(obs,x+.5,y+.5)-.25*10.,.35)
        self.assertLess(agent.return_release_margin,8.)
        agent.reset()
        self.assertEqual(agent.return_release_margin,18.)

    def test_emergency_escape_remains_available_from_outside(self):
        from core.engine import Engine
        from rl.full_observation import FullObservationEncoder
        engine=Engine(seed=1250021)
        engine.time=50.;engine.tick=1500
        engine.players[0].x,engine.players[0].y=7.56,14.16
        obs=FullObservationEncoder(engine).encode(0)
        agent=m.GeneralistAgent()
        self.assertTrue(agent._return_needed(obs))
        self.assertTrue(agent._return_path)
        self.assertGreater(agent._safe_margin(obs,*(v+.5 for v in agent._return_path[-1])),0.)

    def test_local_search_finds_a_shooting_position_between_sampled_rings(self):
        from core.engine import Engine
        from rl.full_observation import FullObservationEncoder
        engine=Engine(seed=1170003)
        engine.players[0].x,engine.players[0].y=35.85955,35.521984
        engine.players[1].x,engine.players[1].y=29.743043,23.655185
        obs=FullObservationEncoder(engine).encode(0)
        previous=m.DijkstraAgent();previous.preferred_range=11.
        agent=m.GeneralistAgent()
        self.assertIsNone(previous._firing_position(obs))
        goal=agent._firing_position(obs)
        self.assertIsNotNone(goal)
        self.assertEqual(agent._shot(obs,*goal)[1],1)
        solid=np.logical_or(obs['map'][0],obs['map'][1])
        path=agent._plan_path(solid,obs,35.85955,35.521984,*goal)
        self.assertEqual(path[-1],(int(goal[0]),int(goal[1])))

    def test_early_reposition_also_recovers_without_line_of_sight(self):
        from core.engine import Engine
        from rl.full_observation import FullObservationEncoder
        engine=Engine(seed=1020008)
        engine.players[0].x,engine.players[0].y=58.933753,36.57521
        engine.players[1].x,engine.players[1].y=57.055569,42.346728
        encoder=FullObservationEncoder(engine)
        agent=m.GeneralistAgent()
        recovered=None
        for tick in range(100):
            engine.time=tick/30;engine.tick=tick
            obs=encoder.encode(0)
            self.assertLess(obs['self'][14],0)
            agent.observe(obs)
            move=agent._combat_move(obs)
            if move is not None:
                recovered=move
                break
        self.assertIsNotNone(recovered)
        self.assertNotEqual(recovered,(0,0))
        self.assertLess(engine.time,4.)

    def test_optional_pursuit_does_not_require_crossing_heat(self):
        from core.engine import Engine
        from core.mapgen import GameMap
        from rl.full_observation import FullObservationEncoder
        grid=np.zeros((100,100),dtype=np.uint8)
        grid[50,20:81]=1
        engine=Engine(seed=0,game_map=GameMap(grid=grid))
        engine.time=80.
        engine.players[0].x,engine.players[0].y=40.5,50.5
        engine.players[1].x,engine.players[1].y=60.5,50.5
        obs=FullObservationEncoder(engine).encode(0)
        previous=m.DijkstraAgent()
        old=previous._plan_path(grid,obs,40.5,50.5,60.5,50.5)
        self.assertTrue(any(previous._safe_margin(obs,x+.5,y+.5)<0 for x,y in old))
        agent=m.GeneralistAgent()
        route=agent._plan_path(grid,obs,40.5,50.5,60.5,50.5)
        self.assertTrue(route)
        self.assertTrue(all(agent._safe_margin(obs,x+.5,y+.5)>0 for x,y in route))
        # Emergency routes remain available when leaving an unsafe pocket
        # really requires a temporary traversal outside the safe zone.
        agent._returning=True
        agent._return_path=old
        self.assertEqual(agent._plan_path(grid,obs,40.5,50.5,60.5,50.5),old)

    def test_transfer_preserves_every_learned_markov_value(self):
        with np.load(ROOT/'MeuAlgo/weights/especialistas/markov.npz') as checkpoint:
            q=checkpoint['q']
            extended=m.generalist_q_from_markov(q)
            self.assertEqual(extended.shape,m.GENERALIST_Q_SHAPE)
            np.testing.assert_array_equal(extended[...,:m.MARKOV_NUM_OPTIONS],q)
            self.assertTrue(np.isfinite(extended).all())
        with self.assertRaises(ValueError): m.generalist_q_from_markov(np.zeros((1,)))

    def test_approach_retreat_and_heat_masks(self):
        env=BlocksWithGunsBotTrainingEnv(opponent='astar')
        try: obs,_=env.reset(seed=420001)
        finally: env.close()
        obs['map'][:]=0
        obs['self'][:2]=[0,0]
        obs['opponent'][:2]=[.22,0]
        obs['self'][14]=1
        obs['game'][3]=1
        obs['bullet_mask'][:]=0
        agent=m.GeneralistAgent()
        agent._motion_velocity=(-3.,0.)
        agent._previous_time=10.
        with patch.object(agent,'observe'),patch.object(agent,'_return_needed',return_value=False),patch.object(agent,'_combat_move',return_value=None):
            self.assertIn(m.CONTROL_RANGE,agent.available_options(obs))
            action=agent._pattern_action(obs,m.CONTROL_RANGE)
            validate_action(action,'range control')
            self.assertEqual(int(action[0])%3-1,-1,'retreat radially against approach')
            self.assertEqual(agent._range_control_ticks,1)
            agent._motion_velocity=(3.,0.)
            self.assertNotIn(m.CONTROL_RANGE,agent.available_options(obs))
            agent._motion_velocity=(-3.,0.)
            with patch.object(agent,'_return_needed',return_value=True):
                np.testing.assert_array_equal(agent.available_options(obs),[m.SAFE_CENTER])

    def test_profiles_and_checkpoint_contract(self):
        self.assertIs(m.policy_class('generalista'),m.GeneralistAgent)
        self.assertIs(m.policy_class(m.GENERALIST_PROFILE),m.GeneralistAgent)
        for bot in ['markov','dijkstra','astar','strafe','camper']:
            agent=m.make_agent(ROOT/'MeuAlgo/weights/especialistas'/f'{bot}.npz')
            self.assertNotIsInstance(agent,m.GeneralistAgent)
        with self.assertRaises(ValueError):
            m.GeneralistAgent(ROOT/'MeuAlgo/weights/especialistas/markov.npz')

if __name__=='__main__': unittest.main()
