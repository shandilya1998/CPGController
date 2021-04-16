import numpy as np
import tensorflow as tf
import os
import pickle
import time
import math
from rl.net import ActorNetwork, CriticNetwork
from rl.env import Env
from rl.constants import params

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class Learner:
    def __init__(self, params, experiment):
        self.params = params
        self.experiment = experiment
        self.actor = ActorNetwork(self.params)
        self.critic = CriticNetwork(self.params)
        self.replay_buffer = ReplayBuffer(self.params)
        self.time_step_spec = tfa.trajectories.time_step.time_step_spec(
            observation_spec = self.params['observation_spec'],
            reward_spec = self.params['reward_spec']
        )
        self.env = Env(
            self.time_step_spec,
            self.params,
            experiment
        )
        self.current_time_step = None
        self._action = self.env._action_init
        self._noise_init = [
            tf.expand_dims(tf.zeros(
                spec.shape,
                spec.dtype
            ), 0) for spec in self.env.action_spec()
        ]
        self._noise = self._noise_init
        self.OU = OU()
        self.dt = self.params['dt']
        self.desired_motion = []
        count = 0
        for i, (x, y) in enumerate(self.pretrain_dataset):
            self.desired_motion.append(np.repeat(
                np.expand_dims(x[0][0], 0),
                self.params['max_steps'] + 1,
                0
            ))
            if count > 10:
                break
            count += 1
        count = 0
        for i, (x, y) in enumerate(self.pretrain_dataset):
            x_ = np.zeros(x[0][0].shape, dtype = np.float32)
            x_[0] = -1 / np.sqrt(2, dtype = np.float32)
            x_[1] = -1 / np.sqrt(2, dtype = np.float32)
            x_[3] = x[0][0][4] / np.sqrt(2, dtype = np.float32)
            x_[4] = x[0][0][4] / np.sqrt(2, dtype = np.float32)
            self.desired_motion.append(np.repeat(
                np.expand_dims(x_, 0),
                self.params['max_steps'] + 1,
                0
            ))
            if count > 10:
                break
            count += 1
        count = 0
        for i, (x, y) in enumerate(self.pretrain_dataset):
            x_ = np.zeros(x[0][0].shape, dtype = np.float32)
            x_[0] = 1 / np.sqrt(2, dtype = np.float32)
            x_[1] = -1 / np.sqrt(2, dtype = np.float32)
            x_[3] = -x[0][0][4] / np.sqrt(2, dtype = np.float32)
            x_[4] = x[0][0][4] / np.sqrt(2, dtype = np.float32)
            self.desired_motion.append(np.repeat(
                np.expand_dims(x_, 0),
                self.params['max_steps'] + 1,
                0
            ))
            if count > 10:
                break
            count += 1
        self._state = [
            self.env.quadruped.motion_state,
            self.env.quadruped.robot_state,
            self.env.quadruped.osc_state
        ]
        matplotlib.use('Agg')
        physical_devices = tf.config.list_physical_devices('GPU')
        np.seterr(all='raise')
        print('[Actor] GPU>>>>>>>>>>>>')
        print('[Actor] {lst}'.format(lst = physical_devices))
        self.p1 = 1.0
        self.epsilon = 1

    def load_actor(self, path, path_target):
        print('[DDPG] Loading Actor Weights')
        self.actor.model.load_weights(path)
        self.actor.target_model.load_weights(path_target)

    def load_critic(self, path, path_target):
        print('[DDPG] Loading Actor Weights')
        self.critic.model.load_weights(path)
        self.critic.target_model.load_weights(path_target)

    def plot_y(self, y, name):
        time = np.arange(y.shape[0])
        fig, axes = plt.subplots(4,1, figsize = (5,20))
        for i in range(4):
            axes[i].plot(time, y[:, 3*i], color = 'r', label = 'ankle')
            axes[i].plot(time, y[:, 3*i + 1], color = 'g', label = 'knee')
            axes[i].plot(time, y[:, 3*i + 2], color = 'b', label = 'hip')
            axes[i].legend()
        fig.savefig(name)
        plt.close()

    def _add_noise(self, action):
        # noise theta and sigma scaled by 0.1 for exp5
        self._noise[0] = max(self.epsilon, 0) * self.OU.function(
            action[0],
            0.0,
            0.15,
            0.2
        )
        self._noise[1] = max(self.epsilon, 0) * self.OU.function(
            action[1],
            0.0,
            0.15,
            0.2
        )
        self._action[0] = action[0] + self._noise[0]
        self._action[1] = action[1] + self._noise[1]

    def get_batch(self):
        batch = self.replay_buffer.get_next(
            self.params['BATCH_SIZE']
        )
        batch_size = len(batch_size)
        states = [[] for i in range(
            len(
                self.params['observation_spec']
            )
        )]
        actions = [[] for i in range(
            len(
                self.params['action_spec']
            )
        )]
        actor_recurrent_states = [[] for i in range(len(
            self._actor_recurrent_state
        ))]
        params = [[] for i in range(len(self._params))]
        next_states = [[] for i in range(
            len(
                self.params['observation_spec']
            )
        )]
        rewards = []
        step_types = []
        for b in batch:
            state = [[] for i in range(len(states))]
            action = [[] for i in range(len(actions))]
            reward = []
            param = [[] for i in range(len(params))]
            next_state = [[] for i in range(len(next_states))]
            step_type = []
            for i, item in enumerate(b):
                for j, s in enumerate(item[0]):
                    state[j].append(s)
                for j, r in enumerate(item[1]):
                    if i == 0:
                        actor_recurrent_states[j].append(r)
                for j, a in enumerate(item[2]):
                    action[j].append(a)
                for j, p in enumerate(item[3]):
                    param[j].append(p)
                reward.append(tf.expand_dims(tf.expand_dims(
                    item[4], 0
                ), 0))
                for j, s in enumerate(item[5]):
                    next_state[j].append(s)
                step_type.append(item[6])
            state = [tf.concat(tf.expand_dims(s, 1), 1) for s in state]
            action = [tf.concat(tf.expand_dims(a, 1), 1) for a in action]
            param = [tf.concat(tf.expand_dims(p, 1), 1) for p in param]
            reward = tf.expand_dims(tf.concat(reward, 0), 0)
            next_state = [tf.concat(tf.expand_dims(s, 1), 1) \
                for s in next_state]

            states = [st.append(s) \
                for s, st in zip(state, states)]
            actions = [ac.append(a) \
                for a, ac in zip(action, actions)]
            params = [pm.append(p) \
                for p, pm in zip(param, params)]
            next_states = [nst.append(ns) \
                for ns, nst in zip(next_state, next_states)]
            rewards.append(reward)

        states = [tf.concat(state, 0) for state in states]
        actor_recurrent_states = [tf.concat(ars, 0) \
            for ars in actor_recurrent_states]
        actions = [tf.concat(action, 0) for action in actions]
        params = [tf.concat(param, 0) for param in params]
        rewards = tf.concat(rewards, 0)
        next_states = [tf.concat(state, 0) for state in next_states]
        return states, actor_recurrent_states, actions \
            params, rewards, next_states, batch_size

    def learn(self, model_dir, experiment, start_epoch = 0, per = False \
            her = False):
        if per:
            print('[DDPG] Initializing PER')
            self.replay_buffer = PER(self.params)
            raise NotImplementedError
        if her:
            print('[DDPG] Initializing HER')
            raise NotImplementedError
        print('[DDPG] Training Start')
        critic_loss = []
        total_critic_loss = []
        hist_rewards = []
        total_reward = []
        _steps_ = []
        COT = []
        stability = []
        d1 = []
        d2 = []
        d3 = []
        motion = []

        self.env.set_motion_state(self.desired_motion[0][0])
        self.current_time_step = self.env.reset()
        self._state = self.current_time_step.observation
        print('[DDPG] Starting Pretraining Test')
        self.total_reward = 0.0
        step = 0
        tot_loss = 0.0
        break_loop = False
        self.epsilon -= 1/self.params['EXPLORE']
        start = None
        self._actor_recurrent_state = self.actor.recurrent_state_init
        while(step < 5 and not break_loop):
            start = time.perf_counter()
            out, osc, omega, mu, mean, state, new_state, new_m_state = \
                self.actor.moodel.layers[-1].rnn_cell(
                    self._state + self._actor_recurrent_state
                )
            self._params = [mu, mean]
            action_original = [out, osc]
            self._add_noise(action_original)
            if math.isnan(np.sum(self._action[0].numpy())):
                print('[DDPG] Action value NaN. Ending Episode')
                break_loop = True

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
                    self.desired_motion[0][step + 1]
                )
            except FloatingPointError:
                print('[DDPG] Floating Point Error in reward computation')
                break_loop = True
                continue
            self._state = self.current_time_step.observation
            self.actor_recurrent_state = [new_state, new_m_state]
             print('[DDPG] Step {step} Reward {reward:.5f} Time {time:.5f}'.format(
                step = step,
                reward = self.current_time_step.reward.numpy(),
                time = time.perf_counter() - start
            ))
            step += 1
        enc_goals = []
        ep = start_epoch
        if ep != 0:
            if her:
                path = os.path.join(
                    model_dir,
                    'enc_goals.pickle'
                )
                pkl = open(path, 'rb')
                enc_goals = pickle.load(pkl)
                pkl.close()
            if per:
                path = os.path.join(
                    model_dir,
                    'per_tree.pickle'
                )
                pkl = open(path, 'rb')
                tree = pickle.load(pkl)
                pkl.close()
                self.replay_buffer.set_priority_tree(tree)
            data_path = os.path.join(
                model_dir,
                'data.pickle'
            )
            pkl = open(data_path, 'rb')
            buff = pickle.load(pkl)
            pkl.close()
            self.replay_buffer.set_buffer(buff)
            self.actor.model.load_weights(
                os.path.join(
                    model_dir,
                    'actor',
                    'model',
                    'model_ep{ep}.ckpt'.format(
                        ep = ep,
                    )
                )
            )
            self.actor.target_model.load_weights(
                os.path.join(
                    model_dir,
                    'actor',
                    'target',
                    'target_model_ep{ep}.ckpt'.format(
                        ep = ep,
                    )
                )
            )
            self.critic.model.load_weights(
                os.path.join(
                    model_dir,
                    'critic',
                    'model',
                    'model_ep{ep}.ckpt'.format(
                        ep = ep,
                    )
                )
            )
            self.critic.target_model.load_weights(
                os.path.join(
                    model_dir,
                    'critic',
                    'target',
                    'target_model_ep{ep}.ckpt'.format(
                        ep = ep,
                    )
                )
            )
            pkl = open(os.path.join(
                model_dir,
                'rewards_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            hist_rewards = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'total_reward_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            total_reward = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'critic_loss_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            critic_loss = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'total_critic_loss_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            total_critic_loss = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'COT_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            COT = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'motion_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            motion = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'stability_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            stability = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'd1_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            d1 = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'd2_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            d2 = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'd3_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            d3 = pickle.load(pkl)
            pkl.close()
        while ep < self.params['train_episode_count']:
            goal_id = np.random.randint(0, len(self.desired_motion))
            desired_motion = self.desired_motion[goal_id]
            enc_goals.append(desired_motion[0])
            self._action = self.env._action_init
            self._noise = self._noise_init
            self.env.set_motion_state(desired_motion[0])
            self.current_time_step = self.env.reset()
            self._state = self.current_time_step.observation
            self._actor_recurrent_state = self.actor.recurrent_state_init
            print('[DDPG] Starting Episode {i}'.format(i = ep))
            self._state = self.current_time_step.observation
            self.total_reward = 0.0
            step = 0
            tot_loss = 0.0
            break_loop = False
            self.epsilon -= 1/self.params['EXPLORE']
            start = None
            experience = []
            while(step < self.params['max_steps'] and not break_loop):
                start = time.perf_counter()
                out, osc, omega, mu, mean, state, new_state, new_m_state = \ 
                    self.actor.moodel.layers[-1].rnn_cell(
                        self._state + self._actor_recurrent_state
                    )
                self._params = [mu, mean]
                action_original = [out, osc]
                self._add_noise(action_original)
                if math.isnan(np.sum(self._action[0].numpy())):
                    print('[DDPG] Action value NaN. Ending Episode')
                    break_loop = True
                    continue
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
                        desired_motion[step + 1]
                    )
                except FloatingPointError:
                    print('[DDPG] Floating Point Error in reward computation')
                    break_loop = True
                    continue
                motion.append(self.env.quadruped.r_motion)
                COT.append(self.env.quadruped.COT)
                experience.append([
                    self._state,
                    self._actor_recurrent_state
                    self._action,
                    self._params,
                    self.current_time_step.reward,
                    self.current_time_step.observation,
                    self.current_time_step.step_type
                ])
                stability.append(self.env.quadruped.stability)
                d1.append(self.env.quadruped.d1):
                d2.append(self.env.quadruped.d2)
                d3.append(self.env.quadruped.d3)
                hist_rewards.append(self.current_time_step.reward.numpy())
                self.total_reward += self.current_time_step.reward.numpy()
                self._actor_recurrent_state = [new_state, new_m_state]
                self._state = self.current_time_step.observation
                print('[DDPG] Episode {ep} Step {step} Reward {reward:.5f} Time {time:.5f}'.format(
                    ep = ep,
                    step = step,
                    reward = self.current_time_step.reward.numpy(),
                    time = time.perf_counter() - start
                ))
                start = None
                step += 1
                if self.current_time_step.step_type == \
                    tfa.trajectories.time_step.StepType.LAST:
                    print('[DDPG] Starting Next Episode')
                    break_loop = True
            self.replay_buffer.add_batch(experience)
            start = time.perf_counter()
            states, actor_recurrent_states, actions \
                params, rewards, next_states, batch_size = self.get_batch()
            out, osc, omega, mu, mean, state, new_state, new_m_state = \
                self.actor.target_model(next_states + recurrent_states)
            ac = [out, osc]
            recurrent_state = tf.repeat(
                self.recurrent_state_init,
                batch_size,
                0
            )
            inputs = next_states + ac + [mu, mean, recurrent_state]
            target_q_values = self.critic.target_model(inputs)
            y = rewards + self.params['GAMMA'] * target_q_values
            loss = self.critic.train(states, actions, params[0], \
                params[1], recurrent_state,  y)
            critic_loss.append(loss.numpy())
            tot_loss += loss.numpy()

            out, osc, omega, mu, mean, state, new_state, new_m_state = \
                self.actor.model(states + recurrent_states)
            a_for_grad = [out, osc]


    def save(self, model_dir, ep, rewards, total_reward, total_critic_loss, \
            critic_loss, COT, motion, stability, d1, d2, d3, tree = None, enc_goals = None):
        print('[DDPG] Saving Data')
        data_path = os.path.join(
            model_dir,
            'data.pickle'
        )
        if os.path.exists(data_path):
            os.remove(data_path)
        pkl = open(data_path, 'wb')
        pickle.dump(self.replay_buffer.buffer, pkl)
        pkl.close()
        if tree is not None:
            print('[DDPG] Saving PER priorities')
            path = os.path.join(
                model_dir,
                'per_tree.pickle'
            )
            if os.path.exists(path):
                os.remove(path)
            pkl = open(path, 'wb')
            pickle.dump(tree, pkl)
            pkl.close()
        if enc_goals is not None:
            print('[DDPG] Saving HER goals')
            path = os.path.join(
                model_dir,
                'enc_goals.pickle'
            )
            pkl = open(path, 'wb')
            pickle.dump(enc_goals, pkl)
            pkl.close()
        print('[DDPG] Saving Model')
        self.actor.model.save_weights(
            os.path.join(
                model_dir,
                'actor',
                'model',
                'model_ep{ep}.ckpt'.format(
                    ep = ep,
                )
            )
        )

        self.actor.target_model.save_weights(
            os.path.join(
                model_dir,
                'actor',
                'target',
                'target_model_ep{ep}.ckpt'.format(
                    ep = ep,
                )
            )
        )

        self.critic.model.save_weights(
            os.path.join(
                model_dir,
                'critic',
                'model',
                'model_ep{ep}.ckpt'.format(
                    ep = ep,
                )
            )
        )

        self.critic.model.save_weights(
            os.path.join(
                model_dir,
                'critic',
                'target',
                'target_model_ep{ep}.ckpt'.format(
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
            'critic_loss_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(critic_loss, pkl)
        pkl.close()
        fig3, ax3 = plt.subplots(1,1,figsize = (5,5))
        ax3.plot(critic_loss)
        ax3.set_ylabel('critic loss')
        ax3.set_xlabel('steps')
        fig3.savefig(os.path.join(
            model_dir,
            'critic_loss_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        pkl = open(os.path.join(
            model_dir,
            'total_critic_loss_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(total_critic_loss, pkl)
        pkl.close()
        fig4, ax4 = plt.subplots(1,1,figsize = (5,5))
        ax4.plot(total_critic_loss)
        ax4.set_ylabel('total critic loss')
        ax4.set_xlabel('episodes')
        fig4.savefig(os.path.join(
            model_dir,
            'total_critic_loss_ep{ep}.png'.format(
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
        ax10.set_ylabel('motion')
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
                    ep = ep
                )
            )
        )

        pkl = open(os.path.join(
            model_dir,
            'd2_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(d2, pkl)
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
