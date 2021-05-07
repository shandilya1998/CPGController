import numpy as np
from rl.torch.constants import params
import torch
from simulations.ws.src.quadruped.scripts.quadruped_torch import Quadruped


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count

class Env:
    def __init__(self,
            params,
            experiment,
        ):
        self.params = params
        self.quadruped = Quadruped(params, experiment)
        self.mean = np.zeros(
            (self.params['robot_state_size'],),
            dtype = np.float32
        )
        self.var = np.zeros(
            (self.params['robot_state_size'],), 
            dtype = np.float32
        )
        self.count = 0.0
        self.epsilon = 1e-8

    def update(self, x):
        batch_mean = np.mean([x], axis=0)
        batch_var = np.var([x], axis=0)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def reset(self, version = 1):
        if version == 1:
            return self._reset_v1()
        elif version == 2:
            return self._reset_v2()
        else:
            raise ValueError

    def _reset_v2(self):
        self._state, self._reward = self.quadruped.reset()
        self._state = self._state[:-1]
        self.update(self._state[1])
        self._state[1] = \
            (self._state[1] - self.mean) / np.sqrt(self.var + self.epsilon)
        return self._state

    def _reset_v1(self):
        print('[RDDPG] Resetting Environment')
        self._state, self._reward = self.quadruped.reset()
        self._state = self._state[:-1]
        return self._state

    def set_motion_state(self, desired_motion):
        self.quadruped.set_motion_state(desired_motion)

    def set_osc_state(self, osc_state):
        self.quadruped.set_osc_state = osc_state

    def step(self,
        action, desired_motion, first_step = False, last_step = False, version = 1,
    ):
        if version == 1:
            return self._step_v1(
                action, desired_motion, first_step, last_step
            )
        elif version == 2:
            return self._step_v2(
                action, desired_motion, first_step, last_step
            )
        else:
            raise ValueError

    def _step_v2(
        self, action, desired_motion, first_step = False, last_step = False
    ):
        observation = self.quadruped.get_state()
        self.update(observation[1])
        observation[1] = \
            (observation[1] - self.mean) / np.sqrt(self.var + self.epsilon)
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
        self.r_motion += self.quadruped.get_motion_reward_v3()
        self.quadruped.set_support_lines()
        self.stability += self.quadruped.get_stability_reward()
        reward += self.quadruped.reward
        reward += self.r_motion + self.stability
        self._state = observation
        done = False
        self._episode_ended = last_step
        if last_step:
            print('[RDDPG] Last Step of episode')
            done = True
        if not self.quadruped.upright:
            #print('[RDDPG] Quadruped Not Upright')
            done = True

        return self._state, reward, done, None

    def _step_v1(
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
        self.r_motion += self.quadruped.get_motion_reward_v2()
        self.quadruped.set_support_lines()
        self.stability += 0.05 * self.quadruped.get_stability_reward()
        reward += self.quadruped.reward
        reward += self.r_motion + self.stability
        self._state = observation
        done = False
        self._episode_ended = last_step
        if last_step:
            print('[RDDPG] Last Step of episode')
            done = True
        if not self.quadruped.upright:
            #print('[RDDPG] Quadruped Not Upright')
            done = True

        return self._state, reward, done, None

