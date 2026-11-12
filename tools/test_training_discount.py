"""Check that the configurable discount reaches TD updates and checkpoints."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'MeuAlgo'),str(ROOT/'.venv/Lib/site-packages')]
spec=importlib.util.spec_from_file_location('discount_training',ROOT/'MeuAlgo/train.py')
training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)


class ToyPolicy:
    profile='generalista_v1'
    learned_tactics=True
    decision_ticks=2
    def __init__(self):self.q=np.zeros((1,1),dtype=np.float32)
    def reset(self):pass
    def state_key(self,obs):return (0,)
    def available_options(self,obs):return np.array([0])
    def best_option(self,obs,key):return 0
    def learning_context(self,key):return None
    def learning_potential(self,obs):return 0.
    def learning_extra_reward(self,events):return 0.
    def action_for_option(self,obs,option):return np.array([4,0,0])


class DelayedRewardEnv:
    def __init__(self,**kwargs):self.tick=0
    def reset(self,**kwargs):self.tick=0;return {},{'opponent':'toy'}
    def step(self,action):
        self.tick+=1
        ended=self.tick==2
        return {},float(ended),ended,False,{'events':[],'winner':0 if ended else None}
    def close(self):pass


class TrainingDiscountTests(unittest.TestCase):
    def run_episode(self,folder,name,**kwargs):
        output=folder/(name+'.npz')
        with patch.object(training,'policy_class',return_value=ToyPolicy),patch.object(training,'BlocksWithGunsBotTrainingEnv',DelayedRewardEnv),contextlib.redirect_stdout(io.StringIO()):
            q=training.train(1,'all',7,output,False,0.,1.,profile='generalista_v1',**kwargs)
        with np.load(output,allow_pickle=False) as data:
            metadata=dict(gamma=float(data['gamma']),episodes=int(data['episodes']))
        return q,metadata

    def test_default_preserves_explicit_old_discount(self):
        with tempfile.TemporaryDirectory(prefix='generalista_gamma_') as folder:
            implicit,meta=self.run_episode(Path(folder),'implicit')
            explicit,_=self.run_episode(Path(folder),'explicit',gamma=.99)
            np.testing.assert_array_equal(implicit,explicit)
            self.assertEqual(meta,dict(gamma=.99,episodes=1))

    def test_later_reward_is_discounted_per_tick_and_saved(self):
        with tempfile.TemporaryDirectory(prefix='generalista_gamma_') as folder:
            folder=Path(folder)
            low,_=self.run_episode(folder,'low',gamma=.99)
            high,meta=self.run_episode(folder,'high',gamma=.999)
            half,_=self.run_episode(folder,'half',gamma=.5)
            whole,_=self.run_episode(folder,'whole',gamma=1.)
            self.assertGreater(float(high[0,0]),float(low[0,0]))
            self.assertAlmostEqual(float(half[0,0]/whole[0,0]),.5,places=6)
            self.assertEqual(meta['gamma'],.999)

    def test_invalid_discounts_are_rejected_before_training(self):
        for gamma in [-.1,1.1,float('nan'),float('inf')]:
            with self.subTest(gamma=gamma),self.assertRaises(ValueError):
                training.train(1,'all',7,Path('unused.npz'),False,.1,.01,gamma=gamma)

    def test_cli_passes_discount_to_training(self):
        args=['train.py','--profile','generalista','--output','unused.npz','--episodes','1','--gamma','0.999']
        with patch.object(sys,'argv',args),patch.object(training,'policy_class',return_value=ToyPolicy),patch.object(training,'train',return_value=np.zeros((1,1))) as run,patch.object(training,'evaluate',return_value={}),contextlib.redirect_stdout(io.StringIO()):
            training.main()
        self.assertEqual(run.call_args.kwargs['gamma'],.999)


if __name__=='__main__':unittest.main()
