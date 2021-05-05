import numpy as np
from rl.torch.constants import params
import torch
from simulations.ws.src.quadruped.scripts.quadruped_torch import Quadruped

class Env:
    def __init__(self,
            params,
            experiment,
        ):
        self.params = params
        self.quadruped = Quadruped(params, experiment)

    def reset(self):
        print('[RDDPG] Resetting Environment')
        self._state, self._reward = self.quadruped.reset()
        self._state = self._state[:-1]
        return self._state

    def set_motion_state(self, desired_motion):
        self.quadruped.set_motion_state(desired_motion)

    def set_osc_state(self, osc_state):
        self.quadruped.set_osc_state = osc_state

    def step(
        self, action, desired_motion, first_step = False, last_step = False
    ):
        observation = self.quadruped.get_state()
        reward = 0.0
        self.quadruped.set_last_pos()
        self.COT = 0.0
        self.r_motion = 0.0
        self.stability = 0.0
        _action = [action, torch.zeros(1, 2 * self.params['units_osc'])]
        observation =  self.quadruped.step(
            _action,
            desired_motion
        )
        observation = observation[:-1]
        self.COT +=  0.005 * self.quadruped.get_COT()
        self.r_motion += self.quadruped.get_motion_reward()
        self.quadruped.set_support_lines()
        self.stability += self.quadruped.get_stability_reward()
        reward += self.quadruped.reward
        reward += self.COT + self.r_motion + self.stability
        self._state = observation
        done = False
        self._episode_ended = last_step
        if last_step:
            print('[RDDPG] Last Step of episode')
            done = True
        if not self.quadruped.upright:
            print('[RDDPG] Quadruped Not Upright')
            done = True

        return self._state, reward, done, None

