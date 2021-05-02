
import numpy as np

from rl.torch.util import *

class Evaluator(object):
    def __init__(self, num_episodes, interval, max_episode_length=None):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.results = np.array([]).reshape(num_episodes,0)

    def __call__(self,
        env, policy, debug=False, visualize=False, save=True
    ):
        self.is_training = False
        observation = None
        result = []

        for episode in range(self.num_episodes):
            # reset at the start of episode
            observation = env.reset()
            episode_steps = 0
            episode_reward = 0.
            assert observation is not None

            # start episode
            done = False
            first_step = True
            last_step = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = policy(observation)

                observation, reward, done, info = env.step(
                    action,
                    observation[0],
                    first_step,
                    last_step
                )
                if self.max_episode_length and \
                    episode_steps >= self.max_episode_length -1:
                    done = True
                    last_step = True
                # update
                episode_reward += reward
                episode_steps += 1
                if first_step:
                    first_step = False

            if debug:
                prYellow('[RDDPG] #Episode{}: episode_reward:{}'.format(
                    episode,episode_reward
                ))
            result.append(episode_reward)

        result = np.array(result).reshape(-1,1)
        self.results = np.hstack([self.results, result])

        return np.mean(result)
