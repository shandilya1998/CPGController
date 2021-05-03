import numpy as np
import argparse
from copy import deepcopy
import torch
import pickle
from rl.torch.normalized_env import NormalizedEnv
from rl.torch.evaluator import Evaluator
from rl.torch.memory import EpisodicMemory
from rl.torch.agent import Agent
from rl.torch.util import *

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

class RDPG(object):
    def __init__(self, env, params, steps = None, cell = None, \
        window_length = None
    ):
        self.params = params
        if params['seed'] > 0:
            self.seed(params['seed'])
        self.window_length = window_length
        self.steps = steps
        if steps is None:
            self.steps = self.params['rnn_steps'] * self.params['max_steps']
        if window_length is None:
            self.window_length = self.params['window_length']

        self.env = env

        self.nb_actions = self.params['action_dim']
        self.agent = Agent(
            self.params,
            steps = steps,
            cell = cell,
            USE_CUDA = USE_CUDA
        )
        self.memory = EpisodicMemory(
            capacity = self.params['BUFFER_SIZE'],
            max_episode_length = self.steps,
            window_length = self.window_length
        )

        self.evaluate = Evaluator(
            self.params['num_validation_episodes'],
            self.steps,
            max_episode_length = self.steps)

        self.critic_optim  = torch.optim.Adam(
            self.agent.critic.parameters(),
            lr = self.params['LRC']
        )
        self.actor_optim  = torch.optim.Adam(
            self.agent.actor.parameters(),
            lr = self.params['LRA'])

        # Hyper-parameters
        self.batch_size = self.params['BATCH_SIZE']
        self.trajectory_length = self.params['trajectory_length']
        self.max_episode_length = self.steps
        self.tau = self.params['TAU']
        self.discount = self.params['GAMMA']
        self.depsilon = 1.0 / self.params['EXPLORE']
        self.warmup = self.params['WARMUP']
        self.validate_steps = self.params['validate_interval']
        self.epsilon = self.params['epsilon']
        self.is_training = True

        if USE_CUDA: self.cuda()

    def train(self, num_iterations, checkpoint_path, debug):
        self.agent.is_training = True
        step = episode = episode_steps = trajectory_steps = 0
        episode_reward = 0.
        state0 = None
        last_step = False
        first_step = False
        self.rewards = []
        self.total_rewards = []
        self.policy_loss = []
        self.critic_loss = []
        self.save(checkpoint_path, step)
        while step < num_iterations:
            episode_steps = 0
            total_reward = 0.0
            while episode_steps < self.max_episode_length:
                if episode_steps == self.max_episode_length - 1:
                    last_step = True
                else:
                    last_step = False
                if episode_steps == 0:
                    first_step = True
                else:
                    first_step = False
                # reset if it is the start of episode
                if state0 is None:
                    state0 = deepcopy(self.env.reset())
                    state0 = [
                        to_tensor(state0[0]),
                        to_tensor(state0[1])
                    ]
                    self.agent.reset()

                # agent pick action ...
                if step <= self.warmup:
                    action = self.agent.random_action()
                else:
                    action = self.agent.select_action([
                        state0[0], state0[1]
                    ])

                # env response with next_observation, reward, terminate_info
                state, reward, done, info = self.env.step(
                    action,
                    state0[0].numpy(),
                    first_step,
                    last_step
                )
                total_reward += reward
                self.rewards.append(reward)
                state = [
                    to_tensor(state[0]),
                    to_tensor(state[1])
                ]
                state = deepcopy(state)
                # agent observe and update policy
                self.memory.append(state0, torch.squeeze(action, 0), reward, done)
                # update
                step += 1
                episode_steps += 1
                trajectory_steps += 1
                episode_reward += reward
                state0 = deepcopy(state)

                if trajectory_steps >= self.trajectory_length:
                    self.agent.reset_gru_hidden_state(done=False)
                    trajectory_steps = 0
                    if step > self.warmup:
                        self.update_policy()

                # [optional] save intermideate model
                if step % int(num_iterations/3) == 0:
                    print('[RDDPG] Saving Model')
                    self.agent.save_model(checkpoint_path)
                    self.save(checkpoint_path, step)

                if done: # end of episode
                    if debug:
                        prGreen(
                            '[RDDPG] {}: episode_reward:{} steps:{}'.format(
                                episode,
                                episode_reward,
                                step
                            )
                        )

                    # reset
                    state0 = None
                    episode_reward = 0.0
                    episode += 1
                    self.total_rewards.append(total_reward)
                    self.agent.reset_gru_hidden_state(done=True)
                    print('[RDDPG] Episode Done')
                    break

            # [optional] evaluate
            if self.evaluate is not None and \
                self.validate_steps > 0 and \
                step % self.validate_steps == 0:
                print('[RDDPG] Start Evaluation')
                policy = lambda x: self.agent.select_action(
                    x,
                    decay_epsilon=False
                )
                validate_reward = self.evaluate(
                    self.env,
                    policy,
                    debug=False,
                    visualize=False
                )
                if debug:
                    prYellow(
                        '[RDDPG] Step_{:07d}: mean_reward:{}'.format(
                            step,
                            validate_reward
                        )
                    )

    def save(self, checkpoint_path, step):
        pkl = open(os.path.join(
            checkpoint_path,
            'rewards_{st}.pickle'.format(st = step)), 'wb'
        )
        pickle.dump(self.rewards, pkl)
        pkl.close()

        pkl = open(os.path.join(
            checkpoint_path,
            'total_rewards_{st}.pickle'.format(st = step)), 'wb'
        )
        pickle.dump(self.total_rewards, pkl)
        pkl.close()

        pkl = open(os.path.join(
            checkpoint_path,
            'policy_loss_{st}.pickle'.format(st = step)), 'wb'
        )
        pickle.dump(self.policy_loss, pkl)
        pkl.close()

        pkl = open(os.path.join(
            checkpoint_path,
            'critic_loss_{st}.pickle'.format(st = step)), 'wb'
        )
        pickle.dump(self.critic_loss, pkl)
        pkl.close()

    def update_policy(self):
        # Sample batch
        print('[RDDPG] Updating Policy')
        experiences = self.memory.sample(self.batch_size, self.max_episode_length)
        if len(experiences) == 0: # not enough samples
            return

        policy_loss_total = 0
        value_loss_total = 0
        for t in range(len(experiences) - 1): # iterate over episodes
            target_robot_enc_state = torch.autograd.Variable(
                torch.zeros(
                    self.batch_size,
                    self.params['units_robot_state'][0]
                )
            ).type(FLOAT)
            target_z = torch.autograd.Variable(
                torch.zeros(self.batch_size, 2 * self.params['units_osc']
                )
            ).type(FLOAT)

            robot_enc_state = torch.autograd.Variable(
                torch.zeros(
                    self.batch_size,
                    self.params['units_robot_state'][0]
                )
            ).type(FLOAT)
            z = torch.autograd.Variable(
                torch.zeros(self.batch_size, 2 * self.params['units_osc']
                )
            ).type(FLOAT)

            # we first get the data out of the sampled experience
            state0 = [trajectory.state0 for trajectory in experiences[t]]
            state0_0 = [state[0] for state in state0]
            state0_1 = [state[1] for state in state0]
            state0_0 = np.stack(state0_0)
            state0_1 = np.stack(state0_1)
            state0 = [state0_0, state0_1]
            action = np.stack(
                (trajectory.action for trajectory in experiences[t])
            )
            reward = np.expand_dims(
                np.stack(
                    (trajectory.reward for trajectory in experiences[t])
                ), axis=1
            )
            state1 = [trajectory.state0 for trajectory in experiences[t+1]]
            state1_0 = [state[0] for state in state1]
            state1_1 = [state[1] for state in state1]
            state1_0 = np.stack(state1_0)
            state1_1 = np.stack(state1_1)
            state1 = [state1_0, state1_1]

            target_action, (target_robot_state_enc, target_z) = \
                self.agent.actor_target(
                    to_tensor(state1[0], volatile=True),
                    to_tensor(state1[1], volatile=True),
                    (target_robot_enc_state, target_z)
                )
            next_q_value = self.agent.critic_target(
                to_tensor(state1[0], volatile=True),
                to_tensor(state1[1], volatile=True),
                target_action
            )

            self.agent.critic.zero_grad()
            target_q = to_tensor(reward) + self.discount*next_q_value
            # Critic update
            current_q = self.agent.critic(
                to_tensor(state0[0], volatile=True),
                to_tensor(state0[1], volatile=True),
                to_tensor(action)
            )

            # value_loss = criterion(q_batch, target_q_batch)
            value_loss = torch.nn.functional.smooth_l1_loss(
                current_q, target_q
            )
            value_loss /= len(experiences) # divide by trajectory length
            value_loss_total += value_loss

            self.agent.critic.zero_grad()
            value_loss.backward()
            self.critic_optim.step()

            # Actor update
            self.agent.actor.zero_grad()
            action, (robot_enc_state, z) = self.agent.actor(
                to_tensor(state0[0], volatile=True),
                to_tensor(state0[1], volatile=True),
                (robot_enc_state, z)
            )
            policy_loss = -self.agent.critic(
                to_tensor(state0[0], volatile=True),
                to_tensor(state0[1], volatile=True),
                action
            )
            policy_loss /= len(experiences) # divide by trajectory length
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.actor_optim.step()
            policy_loss_total += policy_loss.mean()
        self.policy_loss.append(policy_loss_total)
        self.critic_loss.append(value_loss_total)
        soft_update(self.agent.actor_target, self.agent.actor, self.tau)
        soft_update(self.agent.critic_target, self.agent.critic, self.tau)

    def test(self, num_episodes, model_path, visualize=False, debug=False):
        if self.agent.load_weights(model_path) == False:
            prRed("[RDDPG] model path not found")
            return

        self.agent.is_training = False
        self.agent.eval()
        policy = lambda x: self.agent.select_action(
            x, noise_enable=False, decay_epsilon=False
        )

        for i in range(num_episodes):
            validate_reward = self.evaluate(
                self.env, policy, debug=debug, visualize=visualize, save=False
            )
            if debug:
                prYellow('[RDDPG] #{}: mean_reward:{}'.format(
                    i, validate_reward
                ))

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
