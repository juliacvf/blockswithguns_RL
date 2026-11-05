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
