"""Isolated experiment. Never installed into the official policy automatically."""
def install(module):
    base=module.GeneralistAgent
    class EarlyPocketEscape(base):
        def reset(self):
            super().reset()
            self._pocket_checked=-1e9
            self._pocket_active=False
            self._pocket_events=[]

        def _pocket_route(self,obs):
            now=float(obs['game'][0])*module.C.MAX_EPISODE_SECONDS
            horizon,minimum=self.return_lookahead_seconds,self.return_min_margin
            try:
                self.return_lookahead_seconds=max(0.,module.C.MAX_EPISODE_SECONDS-now)
                self.return_min_margin=.75
                return module.DijkstraAgent._forecast_return(self,obs,allow_heat=True)
            finally:
                self.return_lookahead_seconds,self.return_min_margin=horizon,minimum

        def _return_needed(self,obs):
            now=float(obs['game'][0])*module.C.MAX_EPISODE_SECONDS
            invalidated=self._return_checked<self._pocket_checked-1e-6
            if now-self._pocket_checked<.75 and not invalidated:
                return True if self._pocket_active else super()._return_needed(obs)
            if invalidated:self._pocket_active=False
            self._pocket_checked=now
            sx,sy=module._world_xy(obs['self'])
            margin=self._safe_margin(obs,sx,sy)
            remaining=max(0.,module.C.MAX_EPISODE_SECONDS-now)
            if margin>=.75+module.C.HEAT_SHRINK_RATE*remaining:
                self._pocket_active=False
                return super()._return_needed(obs)
            if not self._pocket_active:
                regular=super()._return_needed(obs)
                if regular or margin<.5:return regular
                free_path,_=self._end_safe_route(obs)
                if free_path:return regular
            else:
                free_path,_=self._end_safe_route(obs)
            if free_path:
                path=free_path;slack=0.
            else:
                path,slack=self._pocket_route(obs)
            if not path or len(path)<2 or (not self._pocket_active and slack>=self.return_safety_seconds):
                self._pocket_active=False
                return super()._return_needed(obs)
            if not self._pocket_active:self._pocket_events.append(dict(t=now,event='early_escape',slack=slack))
            self._pocket_active=True
            self._return_checked=now
            self._returning=True
            self._return_path=path
            self._return_goal=path[-1][0]+.5,path[-1][1]+.5
            return True
    module.GeneralistAgent=EarlyPocketEscape
    return module
