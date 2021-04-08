import numpy as np
import tensorflow as tf
from simulations.ws.src.quadruped.scripts.quadruped import Quadruped
from rl.env import Env
from rl.constants import params, params_ars
from rl.net import ActorNetwork
import tf_agents as tfa
import time
import argparse
import os
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import matplotlib

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class HP:
    def __init__(self, params, params_ars):
        self.params = params
        self.params.update(params_ars)

class Policy:
    def __init__(self, params):
        self.params = params
        self.net = ActorNetwork(params, create_target = False).model

    def evaluate(self, states, delta = None, direction = None):
        weights = self.net.trainable_variables
        if direction is None:
            return self.net(states)
        elif direction == 'positive':
            for i in range(len(weights)):
                weights[i] = weights[i] + self.params['noise'] * delta[i]
            self.net.set_weights(weights)
            return self.net(states)
        else:
            for i in range(len(weights)):
                weights[i] = weights[i] - self.params['noise'] * delta[i]
            self.net.set_weights(weights)
            return self.net(states)

    def load_weights(self, path):
        self.net.load_weights(path)

    def sample_deltas(self):
        deltas = []
        for _ in range(self.params['nb_directions']):
            deltas.append([
                tf.random.normal(weight.shape) \
                    for weight in self.net.trainable_variables
            ])
        return deltas

    def update(self, rollouts, sigma_r):
        scale = self.params['learning_rate'] / (
            self.params['nb_best_directions'] * sigma_r
        )
        weights = self.net.trainable_variables
        for r_pos, r_neg, d in rollouts:
            for i in range(len(weights)):
                weights[i] = weights[i] + scale * (r_pos - r_neg) * d[i]
        self.net.set_weights(weights)

class Learner:
    def __init__(self, params, params_ars, experiment, \
            desired_motion_path = 'data/pretrain/X_0.npy'):
        matplotlib.use('Agg')
        self.experiment = experiment
        self.params = HP(params, params_ars).params
        self.policy = Policy(self.params)
        self.time_step_spec = tfa.trajectories.time_step.time_step_spec(
            observation_spec = self.params['observation_spec'],
            reward_spec = self.params['reward_spec']
        )
        self.env = Env(
            self.time_step_spec,
            self.params,
            self.experiment
        )
        x = self.get_desired_motion(desired_motion_path)
        self.desired_motion = tf.repeat(
            tf.expand_dims(x[0], 0),
            self.params['episode_length'] + 1,
            0
        ).numpy()
        self.rewards = []
        self.total_reward = []
        self.COT = []
        self.stability = []
        self.d1 = []
        self.d2 = []
        self.d3 = []
        self.motion = []

    def get_desired_motion(self, path = 'data/pretrain/X_0.npy'):
        X_0 = np.load(
            path,
            allow_pickle = True,
            fix_imports=True
        )
        num_data = X_0.shape[0]
        X_0 = tf.data.Dataset.from_tensor_slices(X_0)
        X_0 = X_0.shuffle(num_data).batch(1)
        i, x = next(enumerate(X_0))
        return x

    def test_policy(self):
        self.current_time_step = self.env.reset()
        self._state = self.current_time_step.observation
        step = 0
        break_loop = False
        while(step < 2 and not break_loop):
            start = time.perf_counter()
            [out, osc], [omega, mu, mean, state] = self.policy.net(self._state)
            self._params = [mu, mean]
            self._action = [out, osc]
            steps = self._action[0].shape[1]
            action = self._action[0] * tf.repeat(
                tf.expand_dims(self._params[0], 1),
                steps,
                axis = 1
            ) + tf.repeat(
                tf.expand_dims(self._params[1], 1),
                steps,
                axis = 1
            )
            try:
                self.current_time_step = self.env.step(
                    [action, self._action[1]],
                    self.desired_motion[step + 1]
                )
            except FloatingPointError:
                break_loop = True
                continue
            self._state = self.current_time_step.observation
            print('[ARS] Step {step} Reward {reward:.5f} Time {t:.5f}'.format(
                step = step,
                reward = self.current_time_step.reward.numpy(),
                t = time.perf_counter() - start
            ))
            step += 1

    def explore(self, delta = None, direction = None):
        self.current_time_step = self.env.reset()
        num_plays = 0
        sum_rewards = 0.0
        step = 0
        break_loop = False
        while(step < self.params['episode_length'] and not break_loop):
            [out, osc], [omega, mu, mean, state] = self.policy.evaluate(
                self.current_time_step.observation, delta, direction
            )
            self._action = [out, osc]
            self._params = [mu, mean]
            steps = self._action[0].shape[1]
            action =  self._action[0] * tf.repeat(
                tf.expand_dims(self._params[0], 1),
                steps,
                axis = 1
            ) + tf.repeat(
                tf.expand_dims(self._params[1], 1),
                steps,
                axis = 1
            )
            try:
                self.current_time_step = self.env.step(
                    [action, self._action[1]],
                    self.desired_motion[step + 1]
                )
            except FloatingPointError:
                print('[DDPG] Floating Point Error in reward computation')
                break_loop = True
                continue
            if direction is None:
                self.rewards.append(self.current_time_step.reward.numpy())
                self.COT.append(self.env.quadruped.COT)
                self.stability.append(self.env.quadruped.stability)
                self.d1.append(self.env.quadruped.d1)
                self.d2.append(self.env.quadruped.d2)
                self.d3.append(self.env.quadruped.d2)
                self.motion.append(self.env.quadruped.r_motion)
            sum_rewards += self.current_time_step.reward.numpy()
            num_plays += 1
            step += 1
            if self.current_time_step.step_type == \
                tfa.trajectories.time_step.StepType.LAST:
                break_loop = True
        return sum_rewards

    def learn(self, model_dir, experiment):
        self.test_policy()
        for ep in range(self.params['nb_steps']):
            start = time.perf_counter()
            deltas = self.policy.sample_deltas()
            positive_rewards = [0.0] * self.params['nb_directions']
            negative_rewards = [0.0] * self.params['nb_directions']

            for k in range(self.params['nb_directions']):
                positive_rewards[k] = self.explore(
                    delta = deltas[k],
                    direction = 'positive'
                )

            for k in range(self.params['nb_directions']):
                negative_rewards[k] = self.explore(
                    delta = deltas[k],
                    direction = 'negative'
                )

            all_rewards = np.array(positive_rewards + negative_rewards)
            sigma_r = all_rewards.std()

            scores = {k:max(r_pos, r_neg) \
                for k,(r_pos,r_neg) in enumerate(zip(
                    positive_rewards, negative_rewards
                ))
            }
            order = sorted(
                scores.keys(), key = lambda x:scores[x], reverse = True
            )[:self.params['nb_best_directions']]
            rollouts = [
                (positive_rewards[k], negative_rewards[k], deltas[k]) \
                    for k in order
            ]

            self.policy.update(rollouts, sigma_r)

            reward_evaluation = self.explore()
            self.total_reward.append(reward_evaluation)
            print('[ARS] Step {step} Reward {reward:.5f} Time {t:.5f}'.format(
                step = ep,
                reward = reward_evaluation,
                t = time.perf_counter() - start
            ))
            if ep % self.params['TEST_AFTER_N_EPISODES'] == 0:
                self.save(model_dir, ep, self.rewards, self.total_reward, \
                    self.COT, self.motion, self.stability, \
                    self.d1, self.d2, self.d3)

    def save(self, model_dir, ep, rewards, total_reward, COT, motion, \
            d1, d2, d3, stability):
        print('[ARS] Saving Model')
        self.policy.net.save_weights(
            os.path.join(
                model_dir,
                'policy',
                'policy_ep{ep}.ckpt'.format(
                    ep = ep,
                )
            )
        )

        pkl = open(os.path.join(
            model_dir,
            'rewards_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(rewards, pkl)
        pkl.close()
        fig1, ax1 = plt.subplots(1,1,figsize = (5,5))
        ax1.plot(rewards)
        ax1.set_ylabel('reward')
        ax1.set_xlabel('steps')
        fig1.savefig(os.path.join(
            model_dir,
            'rewards_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        pkl = open(os.path.join(
            model_dir,
           'total_reward_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(total_reward, pkl)
        pkl.close()
        fig2, ax2 = plt.subplots(1,1,figsize = (5,5))
        ax2.plot(total_reward)
        ax2.set_ylabel('total reward')
        ax2.set_xlabel('episodes')
        fig2.savefig(os.path.join(
            model_dir,
            'total_reward_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        pkl = open(os.path.join(
            model_dir,
            'COT_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(COT, pkl)
        pkl.close()
        fig9, ax9 = plt.subplots(1,1,figsize = (5,5))
        ax9.plot(COT)
        ax9.set_ylabel('COT')
        ax9.set_xlabel('steps')
        fig9.savefig(os.path.join(
            model_dir,
            'COT_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        pkl = open(os.path.join(
            model_dir,
            'motion_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(motion, pkl)
        pkl.close()
        fig10, ax10 = plt.subplots(1,1,figsize = (5,5))
        ax10.plot(motion)
        ax10.set_ylabel('motion deviation')
        ax10.set_xlabel('steps')
        fig10.savefig(os.path.join(
            model_dir,
            'motion_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        pkl = open(os.path.join(
            model_dir,
            'stability_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(stability, pkl)
        pkl.close()
        fig11, ax11 = plt.subplots(1,1,figsize = (5,5))
        ax11.plot(stability)
        ax11.set_ylabel('stability')
        ax11.set_xlabel('steps')
        fig11.savefig(os.path.join(
            model_dir,
            'stability_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        pkl = open(os.path.join(
            model_dir,
            'd1_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(d1, pkl)
        pkl.close()
        fig12, ax12 = plt.subplots(1,1,figsize = (5,5))
        ax12.plot(d1)
        ax12.set_ylabel('d1')
        ax12.set_xlabel('steps')
        fig12.savefig(os.path.join(
            model_dir,
            'd1_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        pkl = open(os.path.join(
            model_dir,
            'd2_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(d1, pkl)
        pkl.close()
        fig13, ax13 = plt.subplots(1,1,figsize = (5,5))
        ax13.plot(d2)
        ax13.set_ylabel('d2')
        ax13.set_xlabel('steps')
        fig13.savefig(os.path.join(
            model_dir,
            'd2_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        pkl = open(os.path.join(
            model_dir,
            'd3_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(d3, pkl)
        pkl.close()
        fig14, ax14 = plt.subplots(1,1,figsize = (5,5))
        ax14.plot(d3)
        ax14.set_ylabel('d3')
        ax14.set_xlabel('steps')
        fig14.savefig(os.path.join(
            model_dir,
            'd3_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type = int,
        help = 'ID of experiment being performaed'
    )
    parser.add_argument(
        '--out_path',
        type = str,
        help = 'Path to output directory'
    )
    args = parser.parse_args()
    learner = Learner(params, params_ars, args.experiment)

    learner.policy.load_weights('weights/actor_pretrain/exp26/pretrain_actor/actor_pretrained_pretrain_actor_26_42.ckpt')

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    path = os.path.join(args.out_path, 'exp{exp}'.format(
        exp=args.experiment
    ))
    if not os.path.exists(path):
        os.mkdir(path)

    policy_path = os.path.join(path, 'policy')
    if not os.path.exists(policy_path):
        os.mkdir(policy_path)

    learner.learn(path, experiment = args.experiment)
