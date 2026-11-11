"""Check future exit deadlines, including the pocket in Markov map 7193382."""
import importlib.util,os,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT),str(ROOT/'.venv/Lib/site-packages')])
from core.engine import Engine
from core.mapgen import GameMap
from rl.full_observation import FullObservationEncoder

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

old=load('timing_only',ROOT/'MeuAlgo/experiments/markov_defense.py')
m=load('deadline_return',Path(os.environ.get('GENERALISTA_SCRIPT',str(ROOT/'MeuAlgo/algo1.py'))))

class FutureReturnTests(unittest.TestCase):
    def test_real_map_requires_early_escape(self):
        e=Engine(seed=7193382,map_seed=7193382)
        obs=FullObservationEncoder(e,120.).encode(0)
        self.assertFalse(old.GeneralistAgent()._return_needed(obs))
        a=m.GeneralistAgent()
        self.assertTrue(a._return_needed(obs))
        self.assertTrue(a._escape_active)
        self.assertGreater(len(a._return_path),1)
        self.assertLess(a._escape_events[0]['slack'],a.return_safety_seconds+.5)

    def test_escape_route_respects_every_cell_deadline(self):
        e=Engine(seed=7193382,map_seed=7193382)
        obs=FullObservationEncoder(e,120.).encode(0)
        a=m.GeneralistAgent();path,slack=a._end_safe_route(obs)
        eta=0.
        for (x,y),(nx,ny) in zip(path,path[1:]):
            cost=(.5/(m.C.MUD_SLOW if obs['map'][2,x,y] else 1.)+.5/(m.C.MUD_SLOW if obs['map'][2,nx,ny] else 1.))/m.C.PLAYER_SPEED
            eta+=cost
            self.assertGreaterEqual(a._safe_margin(obs,nx+.5,ny+.5)-m.C.HEAT_SHRINK_RATE*eta,.5)

    def open_state(self,position):
        grid=np.zeros((100,100),dtype=np.uint8)
        e=Engine(seed=0,game_map=GameMap(grid=grid,mud=np.zeros_like(grid)))
        e.time=10.;e.tick=300
        e.players[0].x,e.players[0].y=position
        return FullObservationEncoder(e,120.).encode(0)

    def test_active_escape_does_not_stop_on_local_comfort(self):
        obs=self.open_state((25.5,25.5))
        a=m.GeneralistAgent();a._escape_active=True
        self.assertGreater(a._safe_margin(obs,25.5,25.5),16.)
        self.assertTrue(a._return_needed(obs))
        self.assertTrue(a._escape_active)

    def test_escape_releases_inside_final_safe_area(self):
        obs=self.open_state((50.5,50.5))
        a=m.GeneralistAgent();a._escape_active=True
        self.assertFalse(a._return_needed(obs))
        self.assertFalse(a._escape_active)

    def test_reset_clears_escape_commitment(self):
        a=m.GeneralistAgent();a._escape_active=True;a._escape_checked=10.
        a.reset()
        self.assertFalse(a._escape_active)
        self.assertLess(a._escape_checked,0.)

    def test_teleport_invalidates_escape_cache_immediately(self):
        grid=np.zeros((100,100),dtype=np.uint8)
        e=Engine(seed=0,game_map=GameMap(grid=grid,mud=np.zeros_like(grid)))
        e.time=10.;e.tick=300
        e.players[0].x=e.players[0].y=25.5
        enc=FullObservationEncoder(e,120.)
        a=m.GeneralistAgent();a.observe(enc.encode(0))
        a._escape_active=True;a._escape_checked=10.;a._return_checked=10.
        e.players[0].x=e.players[0].y=50.5
        e.time+=m.C.FIXED_DT;e.tick+=1
        obs=enc.encode(0);a.observe(obs)
        self.assertFalse(a._return_needed(obs))
        self.assertFalse(a._escape_active)

if __name__=='__main__':unittest.main()
