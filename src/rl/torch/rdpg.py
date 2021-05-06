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
import time
from plot import plot_fit_curve_polymonial_5
import matplotlib.pyplot as plt
from gait_generation.gait_generator import Signal
from tqdm import tqdm
from rl.torch.pretrain import pretrain, GaitDataset

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
DEVICE = 'cpu'
if USE_CUDA:
    DEVICE = 'gpu'

class RDPG(object):
    def __init__(self, env, params, steps = None, cell = None, \
        window_length = None, experiment = 0
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

        self.signal_gen = Signal(
            self.trajectory_length + 1,
            self.params['dt']
        )
        self.create_desired_motion_lst()

        self.experiment = experiment
        if USE_CUDA: self.agent.cuda()

    def create_desired_motion_lst(self):
        self.Tst = self.params['Tst']
        self.Tsw = self.params['Tsw']
        self.theta_h = self.params['theta_h']
        self.theta_k = self.params['theta_k']
        self.desired_motion = []
        for tst, tsw, theta_h, theta_k in tqdm(zip(
            self.Tst,
            self.Tsw,
            self.theta_h,
            self.theta_k
        )):
            self.signal_gen.build(tsw, tst, theta_h, theta_k)
            signal, _ = self.signal_gen.get_signal()
            signal = signal[:, 1:].astype(np.float32)
            v = self.signal_gen.compute_v((0.1+0.015)*2.2)
            motion = np.array([0, -1, 0, 0, -1 * v ,0], dtype = np.float32)
            self.desired_motion.append(motion)
        self.desired_motion.append(np.zeros((6,), dtype = np.float32))

    def train(self, num_iterations, checkpoint_path, debug):
        self.agent.is_training = True
        step = episode = episode_steps = trajectory_steps = val_step = 0
        episode_reward = 0.
        state0 = None
        last_step = False
        first_step = False
        self.rewards = []
        self.total_rewards = []
        self.policy_loss = []
        self.critic_loss = []
        self.d1 = []
        self.d2 = []
        self.d3 = []
        self.stability = []
        self.COT = []
        self.motion = []
        self.val_reward = []
        #self.save(checkpoint_path, step)
        self.agent.save_model(checkpoint_path)
        goal_id = np.random.randint(0, len(self.desired_motion))
        desired_motion = self.desired_motion[goal_id]
        self.env.quadruped.set_motion_state(desired_motion)
        dataset = GaitDataset(self.params['pretrain_ds_path'], self.params['pretrain_bs'])
        pretrain(
            self.params['pretrain_epochs'],
            self.params['pretrain_bs'],
            checkpoint_path,
            self.experiment,
            dataset,
            self.agent,
            self.actor_optim,
            self.env,
            self.memory
        )
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
        pkl = open(os.path.join(
            checkpoint_path,
            'robot_state_mean_var.pickle'), 'wb'
        )
        pickle.dump({
            'mean' : self.env.mean,
            'var' : self.env.var
        }, pkl)
        pkl.close()
        while step < num_iterations:
            episode_steps = 0
            total_reward = 0.0
            while episode_steps < self.max_episode_length:
                penalty = 0.0
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
                    state0 = deepcopy(self.env.reset(
                        version = self.params['step_version']
                    ))
                    state0 = [
                        to_tensor(state0[0]),
                        to_tensor(state0[1])
                    ]
                    self.agent.reset()

                action, hs = self.agent.select_action([
                    state0[0], state0[1]
                ], train = True)

                if torch.isnan(action).any():
                    action = torch.zeros_like()
                    penalty = -1.0

                # env response with next_observation, reward, terminate_info
                state, reward, done, info = self.env.step(
                    action,
                    state0[0].cpu().numpy(),
                    first_step,
                    last_step,
                    version = self.params['step_version']
                )

                if np.isnan(reward):
                    reward = -1.0
                reward += penalty

                self.d1.append(self.env.quadruped.d1)
                self.d2.append(self.env.quadruped.d2)
                self.d3.append(self.env.quadruped.d3)
                self.stability.append(self.env.quadruped.stability)
                self.motion.append(self.env.quadruped.r_motion)
                self.COT.append(self.env.quadruped.COT)

                total_reward += reward
                self.rewards.append(reward)
                state = [
                    to_tensor(state[0]),
                    to_tensor(state[1])
                ]
                state = deepcopy(state)
                # agent observe and update policy
                self.memory.append(
                    state0, 
                    torch.squeeze(action, 0), 
                    [torch.squeeze(h, 0) for h in hs],
                    reward,
                    done
                )
                # update
                step += 1
                val_step += 1
                episode_steps += 1
                trajectory_steps += 1
                episode_reward += reward
                state0 = deepcopy(state)

                if trajectory_steps >= self.trajectory_length:
                    self.agent.reset_gru_hidden_state(done=False)
                    trajectory_steps = 0
                    if step > self.warmup:
                        try:
                            self.update_policy()
                        except RuntimeError:
                            pass

                # [optional] evaluate
                if self.evaluate is not None and \
                    self.validate_steps > 0 and \
                    val_step % self.validate_steps == 0:
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
                    self.val_reward.append(validate_reward)
                    if debug:
                        prYellow(
                            '[RDDPG] Step_{:07d}: mean_reward:{}'.format(
                                step,
                                validate_reward
                            )
                        )
                    if step > 0:
                        print('[RDDPG] Saving Model')
                        pkl = open(os.path.join(
                            checkpoint_path,
                            'robot_state_mean_var.pickle'), 'wb'
                        )
                        pickle.dump({
                            'mean' : self.env.mean,
                            'var' : self.env.var
                        }, pkl)
                        pkl.close()
                        self.save(checkpoint_path, step)
                        self.agent.save_model(checkpoint_path)

                if done: # end of episode
                    print('[RDDPG] Episode Done')
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
                    goal_id = np.random.randint(0, len(self.desired_motion))
                    desired_motion = self.desired_motion[goal_id]
                    self.env.quadruped.set_motion_state(desired_motion)
                    break

    def plot(self, checkpoint_path, file_name, y, x_label, y_label, title):
        x = list(range(len(y)))
        fig, ax = plt.subplots(1,1, figsize = (7.5, 7.5))
        ax.plot(x, y)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        fig.savefig(os.path.join(checkpoint_path, file_name))

    def save(self, checkpoint_path, step):
        pkl = open(os.path.join(
            checkpoint_path,
            'rewards_{st}.pickle'.format(st = step)), 'wb'
        )
        pickle.dump(self.rewards, pkl)
        pkl.close()
        self.plot(checkpoint_path, 'rewards_{st}.png'.format(
            st = step
        ), self.rewards, 'Steps', 'Reward', 'Trends in Reward')

        pkl = open(os.path.join(
            checkpoint_path,
            'total_rewards_{st}.pickle'.format(st = step)), 'wb'
        )
        pickle.dump(self.total_rewards, pkl)
        pkl.close()
        self.plot(checkpoint_path, 'total_rewards_{st}.png'.format(
            st = step
        ), self.total_rewards, 'Steps', 'Total Reward', \
            'Trends in Total Reward')

        pkl = open(os.path.join(
            checkpoint_path,
            'policy_loss_{st}.pickle'.format(st = step)), 'wb'
        )
        pickle.dump(self.policy_loss, pkl)
        pkl.close()
        self.plot(checkpoint_path, 'policy_loss_{st}.png'.format(
            st = step
        ), self.policy_loss, 'Steps', 'Policy Loss', 'Trends in Policy Loss')

        pkl = open(os.path.join(
            checkpoint_path,
            'critic_loss_{st}.pickle'.format(st = step)), 'wb'
        )
        pickle.dump(self.critic_loss, pkl)
        pkl.close()
        self.plot(checkpoint_path, 'value_loss_{st}.png'.format(
            st = step
        ), self.critic_loss, 'Steps', 'Critic Loss', 'Trends in Critic Loss')

        pkl = open(os.path.join(
            checkpoint_path,
            'd1_{st}.pickle'.format(st = step)), 'wb'
        )
        pickle.dump(self.d1, pkl)
        pkl.close()
        self.plot(checkpoint_path, 'd1_{st}.png'.format(
            st = step
        ), self.d1, 'Steps', 'Policy D1', 'Trends in D1')

        pkl = open(os.path.join(
            checkpoint_path,
            'd2_{st}.pickle'.format(st = step)), 'wb'
        )
        pickle.dump(self.d2, pkl)
        pkl.close()
        self.plot(checkpoint_path, 'd2_{st}.png'.format(
            st = step
        ), self.d2, 'Steps', 'Policy D2', 'Trends in D2')

        pkl = open(os.path.join(
            checkpoint_path,
            'd3_{st}.pickle'.format(st = step)), 'wb'
        )
        pickle.dump(self.d3, pkl)
        pkl.close()
        self.plot(checkpoint_path, 'd3_{st}.png'.format(
            st = step
        ), self.d3, 'Steps', 'Policy D3', 'Trends in D3')

        pkl = open(os.path.join(
            checkpoint_path,
            'stability_{st}.pickle'.format(st = step)), 'wb'
        )
        pickle.dump(self.stability, pkl)
        pkl.close()
        self.plot(checkpoint_path, 'stability_{st}.png'.format(
            st = step
        ), self.stability, 'Steps', 'Policy Stability', 'Trends in Stability')

        pkl = open(os.path.join(
            checkpoint_path,
            'COT_{st}.pickle'.format(st = step)), 'wb'
        )
        pickle.dump(self.COT, pkl)
        pkl.close()
        self.plot(checkpoint_path, 'COT_{st}.png'.format(
            st = step
        ), self.COT, 'Steps', 'Policy COT', 'Trends in COT')

        pkl = open(os.path.join(
            checkpoint_path,
            'motion_{st}.pickle'.format(st = step)), 'wb'
        )
        pickle.dump(self.motion, pkl)
        pkl.close()
        self.plot(checkpoint_path, 'motion_{st}.png'.format(
            st = step
        ), self.motion, 'Steps', 'Policy Motion', 'Trends in Motion')

    def update_policy(self):
        # Sample batch
        print('[RDDPG] Updating Policy')
        start = time.perf_counter()
        experiences = self.memory.sample(self.batch_size, self.trajectory_length)
        if len(experiences) == 0: # not enough samples
            return

        policy_loss_total = 0
        value_loss_total = 0

        self.agent.critic.zero_grad()
        self.agent.actor.zero_grad()

        hs = [trajectory.hs for trajectory in experiences[0]]
        robot_enc_state = [h[0].cpu() for h in hs]
        z = [h[1].cpu() for h in hs]
        robot_enc_state = to_tensor(np.stack(robot_enc_state), volatile = True)
        z = to_tensor(np.stack(z), volatile = True)
        t_hs = [trajectory.hs for trajectory in experiences[0]]
        target_robot_enc_state = [h[0].cpu() for h in t_hs]
        target_z = [h[1].cpu() for h in t_hs]
        target_robot_enc_state = to_tensor(np.stack(target_robot_enc_state), volatile = True)
        target_z = to_tensor(np.stack(target_z), volatile = True)

        h = torch.autograd.Variable(
            torch.zeros(self.batch_size, self.params['units_gru_rddpg'])
        ).type(FLOAT)

        h_ = torch.autograd.Variable(
            torch.zeros(self.batch_size, self.params['units_gru_rddpg'])
        ).type(FLOAT)

        target_h = torch.autograd.Variable(
            torch.zeros(self.batch_size, self.params['units_gru_rddpg'])
        ).type(FLOAT)

        for t in range(len(experiences) - 1): # iterate over episodes
            # we first get the data out of the sampled experience
            state0 = [trajectory.state0 for trajectory in experiences[t]]
            state0_0 = [state[0].cpu() for state in state0]
            state0_1 = [state[1].cpu() for state in state0]
            state0_0 = np.stack(state0_0)
            state0_1 = np.stack(state0_1)
            state0 = [state0_0, state0_1]
            action = np.stack(
                (trajectory.action.cpu() for trajectory in experiences[t])
            )
            reward = np.expand_dims(
                np.stack(
                    (trajectory.reward for trajectory in experiences[t])
                ), axis=1
            )
            state1 = [trajectory.state0 for trajectory in experiences[t+1]]
            state1_0 = [state[0].cpu() for state in state1]
            state1_1 = [state[1].cpu() for state in state1]
            state1_0 = np.stack(state1_0)
            state1_1 = np.stack(state1_1)
            state1 = [state1_0, state1_1]

            target_action, (target_robot_state_enc, target_z) = \
                self.agent.actor_target(
                    to_tensor(state1[0], volatile=True),
                    to_tensor(state1[1], volatile=True),
                    (target_robot_enc_state, target_z)
                )
            next_q_value, target_h = self.agent.critic_target(
                to_tensor(state1[0], volatile=True),
                to_tensor(state1[1], volatile=True),
                target_action,
                hidden_state = target_h
            )

            self.agent.critic.zero_grad()
            target_q = to_tensor(reward) + self.discount*next_q_value
            # Critic update
            current_q, h = self.agent.critic(
                to_tensor(state0[0], volatile=True),
                to_tensor(state0[1], volatile=True),
                to_tensor(action),
                hidden_state = h
            )

            # Actor update
            action, (robot_enc_state, z) = self.agent.actor(
                to_tensor(state0[0], volatile=True),
                to_tensor(state0[1], volatile=True),
                (robot_enc_state, z)
            )
            q_val, h_ = self.agent.critic(
                to_tensor(state0[0], volatile=True),
                to_tensor(state0[1], volatile=True),
                action,
                hidden_state = h_
            )
            # value_loss = criterion(q_batch, target_q_batch)
            value_loss_total += torch.nn.functional.smooth_l1_loss(
                current_q, target_q.detach()
            ) / len(experiences)
            #self.agent.critic.zero_grad()
            #self.critic_optim.zero_grad()
            policy_loss_total += -torch.mean(q_val) / len(experiences)
            #self.agent.actor.zero_grad()
            #self.actor_optim.zero_grad()
            #value_loss.backward(retain_graph = True)
            #policy_loss.backward(retain_graph = True)
            #self.critic_optim.step()
            #self.actor_optim.step()

        value_loss_total.backward()
        policy_loss_total.backward()
        self.critic_optim.step()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()

        self.policy_loss.append(policy_loss_total.detach().cpu().numpy())
        self.critic_loss.append(value_loss_total.detach().cpu().numpy())
        soft_update(self.agent.actor_target, self.agent.actor, self.tau)
        soft_update(self.agent.critic_target, self.agent.critic, self.tau)
        del value_loss_total
        del policy_loss_total
        print('[RDDPG] Update Time: {t:.5f}'.format(t = time.perf_counter() - start))

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
