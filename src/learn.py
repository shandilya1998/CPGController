from rl.constants import params
#import rospy
from rl.net import ActorNetwork, CriticNetwork
from rl.env import Env
from rl.replay_buffer import ReplayBuffer, OU
import tf_agents as tfa
import tensorflow as tf
import numpy as np
from gait_generation.gait_generator import Signal
from tf_agents.trajectories.time_step import TimeStep, time_step_spec
from tqdm import tqdm
import pickle
import os
from frequency_analysis import frequency_estimator
import time

class SignalDataGen:
    def __init__(self, params, create_data = False):
        self.Tst = params['Tst']
        self.Tsw = params['Tsw']
        self.theta_h = params['theta_h']
        self.theta_k = params['theta_k']
        self.N = params['rnn_steps']
        self.params = params
        self.signal_gen = Signal(
            self.N + 1,
            self.params['dt']
        )
        self.dt = self.params['dt']
        self.data = []
        self.num_data = 0
        if create_data:
            self._create_data()
        else:
            self.num_data = 896*5

    def get_ff(self, signal, ff_type = 'fft'):
        if ff_type == 'fft':
            return frequency_estimator.freq_from_fft(signal, 1/self.dt)
        elif ff_type == 'autocorr':
            return frequency_estimator.freq_from_autocorr(signal, 1/self.dt)

    def _create_data(self):
        """
            Turning Behaviour is to be learnt by RL
        """
        print('[Actor] Creating Data.')
        self.data = []
        deltas = [0]#, 3]#, -3]
        delta = []
        for i in range(len(deltas)):
            for j in range(len(deltas)):
                delta.append([
                    deltas[i],
                    deltas[j],
                ])

        """
            Data for straight ahead
        """
        for d in tqdm(delta):
            for tst, tsw, theta_h, theta_k in zip(
                self.Tst,
                self.Tsw,
                self.theta_h,
                self.theta_k
            ):
                tsw = tsw + d[0]
                tst = tst + d[1]
                self.signal_gen.build(tsw, tst, theta_h, theta_k)
                signal, _ = self.signal_gen.get_signal()
                signal = signal[:, 1:].astype(np.float32)
                v = self.signal_gen.compute_v((0.1+0.015)*2.2)
                motion = np.array([0, 1, 0, 0, v ,0], dtype = np.float32)
                mu = np.array([theta_k, theta_k / 5, theta_h])
                mu = [mu for i in range(4)]
                mu =  np.concatenate(mu, 0)
                freq = self.get_ff(signal[:, 0], 'autocorr')
                self.data.append(
                    [signal, motion, freq, mu]
                )
        """
            Data for straight back
        """
        for d in tqdm(delta):
            for tst, tsw, theta_h, theta_k in zip(
                self.Tst,
                self.Tsw,
                self.theta_h,
                self.theta_k
            ):  
                tsw = tsw + d[0]
                tst = tst + d[1]
                self.signal_gen.build(tsw, tst, theta_h, theta_k)
                signal, _ = self.signal_gen.get_signal()
                signal = signal[:, 1:].astype(np.float32)
                signal[:, 2] = -signal[:, 2]
                signal[:, 5] = -signal[:, 5]
                signal[:, 8] = -signal[:, 8]
                signal[:, 11] = -signal[:, 11]
                v = self.signal_gen.compute_v((0.1+0.015)*2.2)
                motion = np.array([0, -1, 0, 0, v ,0], dtype = np.float32)
                mu = np.array([theta_k, theta_k / 5, theta_h])
                mu = [mu for i in range(4)]
                mu =  np.concatenate(mu, 0)
                freq = self.get_ff(signal[:, 0], 'autocorr')
                self.data.append(
                    [signal, motion, freq, mu] 
                )
        """
            Data for straight right
        """
        for d in tqdm(delta):
            for tst, tsw, theta_h, theta_k in zip(
                self.Tst,
                self.Tsw,
                self.theta_h,
                self.theta_k
            ):
                tsw = tsw + d[0]
                tst = tst + d[1]
                self.signal_gen.build(tsw, tst, theta_h, theta_k)
                signal, _ = self.signal_gen.get_signal()
                signal = signal[:, 1:].astype(np.float32)
                signal_ = np.zeros(signal.shape)
                signal_[:, 0:3] = signal[:, 3:6]
                signal_[:, 3:6] = signal[:, 9:12]
                signal_[:, 6:9] = signal[:, 0:3]
                signal_[:, 9:12] = signal[:, 6:9]
                v = self.signal_gen.compute_v((0.1+0.015)*2.2)
                motion = np.array([1, 0, 0, v, 0 ,0], dtype = np.float32)
                mu = np.array([theta_k, theta_k / 5, theta_h])
                mu = [mu for i in range(4)]
                mu =  np.concatenate(mu, 0)
                freq = self.get_ff(signal_[:, 0], 'autocorr')
                self.data.append(
                    [signal_, motion, freq, mu]
                )
        """
            Data for straight left
        """
        for d in tqdm(delta):
            for tst, tsw, theta_h, theta_k in zip(
                self.Tst,
                self.Tsw,
                self.theta_h,
                self.theta_k
            ):
                tsw = tsw + d[0]
                tst = tst + d[1]
                self.signal_gen.build(tsw, tst, theta_h, theta_k)
                signal, _ = self.signal_gen.get_signal()
                signal = signal[:, 1:].astype(np.float32)
                signal[:, 2] = -signal[:, 2]
                signal[:, 5] = -signal[:, 5]
                signal[:, 8] = -signal[:, 8]
                signal[:, 11] = -signal[:, 11]
                signal_ = np.zeros(signal.shape)
                signal_[:, 0:3] = signal[:, 3:6]
                signal_[:, 3:6] = signal[:, 9:12]
                signal_[:, 6:9] = signal[:, 0:3]
                signal_[:, 9:12] = signal[:, 6:9]
                v = self.signal_gen.compute_v((0.1+0.015)*2.2)
                motion = np.array([-1, 0, 0, v, 0 ,0], dtype = np.float32)
                mu = np.array([theta_k, theta_k / 5, theta_h])
                mu = [mu for i in range(4)]
                mu =  np.concatenate(mu, 0)
                freq = self.get_ff(signal_[:, 0], 'autocorr')
                self.data.append(
                    [signal_, motion, freq, mu]
                )
        self.num_data = len(self.data)
        print('[Actor] Number of Data Points: {num}'.format(
            num = self.num_data)
        )

    def preprocess(self, signal):
        signal = signal-np.mean(signal, axis = 0)
        signal = signal/(np.abs(signal.max(axis = 0)))
        return signal

    def generator(self):
        for batch in range(self.num_data):
            y, x, f, mu = self.data[batch]
            mu = np.expand_dims(mu, 0)
            yield y, x, f, mu

class Learner():
    def __init__(self, params, create_data = False):
        tf.config.run_functions_eagerly(False)
        self.params = params
        self.actor = ActorNetwork(params)
        self.critic = CriticNetwork(params)
        self.replay_buffer = ReplayBuffer(params)
        self.time_step_spec = tfa.trajectories.time_step.time_step_spec(
            observation_spec = self.params['observation_spec'],
            reward_spec = self.params['reward_spec']
        )
        """
        self.env = Env(
            self.time_step_spec,
            self.params,
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
        """
        self.OU = OU()
        self.desired_motion = np.zeros((
            self.params['max_steps'], 6
        ), dtype = np.float32)
        self.desired_motion[:, 3] = 0.05
        self.signal_gen = SignalDataGen(params, create_data)
        self.pretrain_osc_mu = np.ones((
            1,
            self.params['units_osc']
        ), dtype = np.float32)
        self.mse_mu = tf.keras.losses.MeanSquaredError()
        self.mse_omega = tf.keras.losses.MeanSquaredError()
        self.dt = self.params['dt']
        if create_data:
            self.create_dataset()
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.1,
            decay_steps=20,
            decay_rate=0.75,
            staircase=True
        )
        self.pretrain_actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate = lr_schedule
        )
        """
        self._state = [
            self.env.quadruped.motion_state,
            self.env.quadruped.robot_state,
            self.env.quadruped.osc_state
        ]
        """
        self._action = None
        physical_devices = tf.config.list_physical_devices('GPU')
        print('[Actor] GPU>>>>>>>>>>>>')
        print('[Actor] {lst}'.format(lst = physical_devices))
        try:
            print('[Actor] Memory Growth Allowed')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
             # Invalid device or cannot modify virtual 
             # devices once initialized.
            pass

    def set_desired_motion(self, motion):
        self.desired_motion = motion

    def print_grads(self, name, grads):
        print('[Actor] {name} >>>>>>>>>>>>'.format(name=name))
        print(grads)

    def _pretrain_actor(self, x, y):
        with tf.GradientTape(persistent=False) as tape:
            _action, [omega, mu] = self.actor.model(x)
            y_pred = _action[0]
            loss_mu = self.mse_mu(y[2], mu)
            loss_action = self.actor._pretrain_loss(y[0], y_pred)
            loss_omega = self.mse_omega(y[1], omega)
            loss = loss_mu + loss_action + loss_omega

        grads = tape.gradient(
            loss,
            self.actor.model.trainable_variables
        )
        #self.print_grads(self.actor.model.trainable_variables, grads_action)
        self.pretrain_actor_optimizer.apply_gradients(
            zip(
                grads,
                self.actor.model.trainable_variables
            )
        )
        return loss, [loss_action, loss_omega, loss_mu]

    def _pretrain_actor_segments(self, x, y):
        with tf.GradientTape(persistent=False) as tape:
            _action, [omega, mu] = self.actor.model(x)
            y_pred = _action[0]
            loss_mu = self.mse_mu(y[2], mu)
            loss_action = self.actor._pretrain_loss(y[0], y_pred)
            loss_omega = self.mse_omega(y[1], omega)
            loss = loss_mu + loss_action + loss_omega

        vars_action = []
        for var in self.actor.model.trainable_variables:
            if 'complex_dense' in var.name:
                vars_action.append(var)
        grads_action = tape.gradient(
            loss_action,
            vars_action
        )
        #self.print_grads(self.actor.model.trainable_variables, grads_action)
        self.pretrain_actor_optimizer.apply_gradients(
            zip(
                grads_action,
                vars_action
            )
        )

        vars_omega = []
        for var in self.actor.model.trainable_variables:
            if 'state_encoder' in var.name:
                if 'mu_dense' in var.name or 'b_dense' in var.name:
                    continue
                else:
                    vars_omega.append(var)
        grads_omega = tape.gradient(
            loss_omega,
            vars_omega
        )
        #self.print_grads(vars_omega, grads_omega)
        self.pretrain_actor_optimizer.apply_gradients(
            zip(
                grads_omega,
                vars_omega
            )
        )

        vars_mu = []
        for var in self.actor.model.trainable_variables:
            if 'state_encoder' in var.name:
                if 'omega_dense' in var.name or 'b_dense' in var.name:
                    continue
                else:
                    vars_mu.append(var)
        grads_mu = tape.gradient(
            loss_mu,
            vars_mu
        )
        #self.print_grads(vars_mu, grads_mu)
        self.pretrain_actor_optimizer.apply_gradients(
            zip(
                grads_mu,
                vars_mu
            )
        )
        return loss, [loss_action, loss_omega, loss_mu]

    def _hopf_oscillator(self, omega, mu, b, z):
        rng = np.arange(1, self.params['units_osc'] + 1)
        x, y = z[:self.params['units_osc']], z[self.params['units_osc']:]
        x = x + ((mu - (x*x + y*y)) * x - omega * rng * y) * self.dt + b
        y = y + ((mu - (x*x + y*y)) * y + omega * rng * x) * self.dt + b
        return np.concatenate([x, y], -1)

    def create_dataset(self):
        self.num_batches = self.signal_gen.num_data//self.params['pretrain_bs']
        self.env.quadruped.reset()
        motion_state, robot_state, osc_state = \
            self.env.quadruped.get_state_tensor()
        F = []
        MU = []
        Y = []
        X = [[] for j in range(len(self.params['observation_spec']))]
        for y, x, f, mu in tqdm(self.signal_gen.generator()):
            mu = (mu * np.pi / 180)/(np.pi/3)
            y = y * np.pi / 180
            ac = y[0]
            y = y[1:]
            y = np.expand_dims(self.signal_gen.preprocess(y), 0)
            f = f * 2 * np.pi
            osc = self._hopf_oscillator(
                f,
                np.ones((self.params['units_osc'],)),
                np.zeros((self.params['units_osc'],)),
                osc_state[0]
            )
            self.env.quadruped.all_legs.move(ac)
            rospy.sleep(0.3)
            self.env.quadruped.set_initial_motion_state(x)
            self.env.quadruped.set_osc_state(osc)
            diff_joint = self.env.quadruped.joint_position - \
                self.env.quadruped.last_joint
            _state = [
                np.expand_dims(
                    self.env.quadruped.motion_state, 0
                ),
                np.expand_dims(
                    np.concatenate(
                        [
                            np.sin(self.env.quadruped.joint_position),
                            np.sin(diff_joint),
                            self.env.quadruped.orientation,
                            self.env.quadruped.angular_vel,
                            self.env.quadruped.linear_acc
                        ]
                    ), 0
                ),
                np.expand_dims(self.env.quadruped.osc_state, 0)
            ]
            #_state = self.env.quadruped.get_state_tensor()
            for j, s in enumerate(_state):
                X[j].append(s)
            Y.append(y)
            F.append(np.array([[f]], dtype = np.float32))
            MU.append(mu)
        for j in range(len(X)):
            X[j] = np.concatenate(X[j], axis = 0)
        Y = np.concatenate(Y, axis = 0)
        F = np.concatenate(F, axis = 0)
        MU = np.concatenate(MU, axis = 0)
        print('[Actor] Y Shape : {sh}'.format(sh=Y.shape))
        np.save('data/pretrain/Y.npy', Y, allow_pickle = True, fix_imports=True)
        time.sleep(3)
        np.save('data/pretrain/F.npy', F, allow_pickle = True, fix_imports=True)
        time.sleep(3)
        np.save('data/pretrain/MU.npy', MU,allow_pickle = True,fix_imports=True)
        time.sleep(3)
        for j in range(len(X)):
            time.sleep(3)
            np.save('data/pretrain/X_{j}.npy'.format(j=j), X[j], allow_pickle = True, fix_imports=True)

    def load_dataset(self):
        Y =np.load('data/pretrain/Y.npy', allow_pickle = True, fix_imports=True)
        time.sleep(3)
        num_data = Y.shape[0]
        Y = tf.convert_to_tensor(Y)
        F = tf.convert_to_tensor(np.load('data/pretrain/F.npy', allow_pickle = True, fix_imports=True))
        time.sleep(3)
        MU = tf.convert_to_tensor(np.load('data/pretrain/MU.npy', allow_pickle = True, fix_imports=True))
        X = []
        for j in range(len(self.params['observation_spec'])):
            X.append(
                tf.data.Dataset.from_tensor_slices(
                    tf.convert_to_tensor(
                        np.load('data/pretrain/X_{j}.npy'.format(j=j), allow_pickle = True, fix_imports=True)
                    )
                )
            )
        self.num_batches = num_data//self.params['pretrain_bs']
        Y = tf.data.Dataset.from_tensor_slices(Y)
        F = tf.data.Dataset.from_tensor_slices(F)
        MU = tf.data.Dataset.from_tensor_slices(MU)
        Y = tf.data.Dataset.zip((Y, F, MU))
        X = tf.data.Dataset.zip(tuple(X))
        dataset = tf.data.Dataset.zip((X, Y))
        dataset = dataset.shuffle(num_data).batch(
            self.params['pretrain_bs'],
            drop_remainder=True
        )
        return dataset

    def _pretrain_loop(self, grad_update, experiment, checkpoint_dir, name):
        total_loss = 0.0
        avg_loss = 0.0
        prev_loss = 1e10
        history_loss = []
        """
        dataset = tf.data.Dataset.from_generator(
            self._dataset,
            output_types = [
                [spec.dtype for spec in self.params['observation_spec']],
                self.params['action_spec'][0].dtype
            ]
        )
        """
        dataset = self.load_dataset()
        print('[Actor] Dataset {ds}'.format(ds = dataset))
        print('[Actor] Starting Actor Pretraining')
        for episode in range(self.params['train_episode_count']):
            print('[Actor] Starting Episode {ep}'.format(ep = episode))
            total_loss = 0.0
            start = time.time()
            for step, (x, y) in enumerate(dataset):
                loss, [loss_action, loss_omega, loss_mu] = \
                    grad_update(x, y)
                loss = loss.numpy()
                print('[Actor] Episode {ep} Step {st} Loss: {loss}'.format(
                    ep = episode,
                    st = step,
                    loss = loss
                ))
                total_loss += loss
                if step >= 25:
                    break
            end = time.time()
            avg_loss = total_loss / (self.num_batches)
            print('-------------------------------------------------')
            print('[Actor] Episode {ep} Average Loss: {l}'.format(
                ep = episode,
                l = avg_loss
            ))
            print('[Actor] Learning Rate: {lr}'.format(
                lr = self.pretrain_actor_optimizer.lr((episode + 1) * 5))
            )
            print('[Actor] Epoch Time: {time}s'.format(time = end - start))
            print('-------------------------------------------------')
            history_loss.append(avg_loss)
            if episode % 5 == 0:
                if prev_loss < avg_loss:
                    break
                else:
                    self.actor.model.save_weights(
                        os.path.join(
                            checkpoint_dir,
                            'actor_pretrained_{name}_{ex}_{ep}.ckpt'.format(
                                name = name,
                                ep=episode,
                                ex = experiment
                            )
                        )
                    )
                prev_loss = avg_loss
        pkl = open(os.path.join(checkpoint_dir, 'loss_{ex}_{name}.pickle'.format(
            name = name,
            ex = experiment
        )), 'wb')
        pickle.dump(history_loss, pkl)
        pkl.close()

    def _pretrain_encoder(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            _action, [omega, mu] = self.actor.model(x)
            y_pred = _action[0]
            loss_mu = self.mse_mu(y[2], mu)
            loss_omega = self.mse_omega(y[1], omega)
            loss_action = self.actor._pretrain_loss(y[0], y_pred)
            loss = loss_omega + loss_mu

        vars_encoder = []
        for var in self.actor.model.trainable_variables:
            if 'motion_state_encoder' in var.name:
                vars_encoder.append(var)
        grads = tape.gradient(
            loss,
            vars_encoder
        )
        #self.print_grads(self.actor.model.trainable_variables, grads_action)
        self.pretrain_actor_optimizer.apply_gradients(
            zip(
                grads,
                vars_encoder
            )
        )
        return loss, [loss_action, loss_omega, loss_mu]

    def pretrain_actor(self, experiment, checkpoint_dir = 'weights/actor_pretrain'):
        self._pretrain_loop(
            self._pretrain_encoder, experiment, checkpoint_dir, 'pretrain_enc'
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.01,
            decay_steps=50,
            decay_rate=0.95,
            staircase=True
        )
        self.pretrain_actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate = lr_schedule
        )

        self._pretrain_loop(
            self._pretrain_actor, experiment, checkpoint_dir, 'pretrain_actor'
        )

    def load_actor(self, path):
        self.actor.model.load(path)

    def learn(self, model_dir, identifier=''):
        i = 0
        epsilon = 1
        self._noise_init = [
                    tf.expand_dims(tf.zeros(
                        spec.shape,
                        spec.dtype
                    ), 0) for spec in self.env.action_spec()
                ]
        print('[DDPG] Training Start')
        while i < self.params['train_episode_count']:
            self.current_time_step = self.env.reset()
            print('[DDPG] Starting Episode {i}'.format(i = i), end = '')
            self._state = self.current_time_step.observation
            self.total_reward = 0.0
            step = 0
            for j in range(self.params['max_steps']):
                loss = 0.0
                epsilon -= 1/self.params['EXPLORE']
                self._action = self.env._action_init
                self._noise = self._noise_init
                action_original, [omega, mu] = self.actor.model(self._state)
                self._noise[0] = max(epsilon, 0) * self.OU.function(
                    action_original[0],
                    0.0,
                    0.15,
                    0.2
                )
                self._action[0] = action_original[0] + self._noise[0]
                self._action[1] = action_original[1] + self._noise[1]
                self.current_time_step = self.env.step(
                    self._action,
                    self.desired_motion[j]
                )
                experience = [
                    self._state,
                    self._action,
                    self.current_time_step.reward,
                    self.current_time_step.observation,
                    self.current_time_step.step_type
                ]
                self.replay_buffer.add_batch(experience)
                batch = self.replay_buffer.get_next(
                    self.params['BATCH_SIZE']
                )
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
                next_states = [[] for i in range(
                    len(
                        self.params['observation_spec']
                    )
                )]
                rewards = []
                step_types = []
                for item in batch:
                    state = item[0]
                    action = item[1]
                    rewards.append(item[2])
                    next_state = item[3]
                    step_types.append(item[4])
                    for i, s in enumerate(state):
                        states[i].append(s)
                    for i, a in enumerate(action):
                        actions[i].append(a)
                    for i, s in enumerate(next_state):
                        next_states[i].append(s)
                states = [tf.concat(state, 0) for state in states]
                actions = [tf.concat(action, 0) for action in actions]
                rewards = tf.concat(rewards, 0)
                next_states = [tf.concat(state, 0) for state in next_states]

                actions, [o, m, b] = self.actor.target_model(next_states)
                inputs = next_states + actions
                target_q_values = self.critic.target_model(inputs)

                y = [tf.repeat(reward, self.params['action_dim']) \
                        for reward in rewards]
                y = tf.stack([
                    y[k] + self.params['GAMMA'] * target_q_values[k] \
                    if step_types[k] != \
                        tfa.trajectories.time_step.StepType.LAST \
                    else y[k] for k in range(len(y))
                ])

                loss += self.critic.train(states, actions, y)
                a_for_grad, [omega_, mu_] = self.actor.model(states)
                q_grads = self.critic.q_grads(states, a_for_grad)
                self.actor.train(states, q_grads)
                self.actor.target_train()
                self.critic.target_train()

                self.total_reward += self.current_time_step.reward
                self._state = self.current_time_step.observation
                print('.', end = '')
                step += 1
                if self.current_time_step.step_type == \
                    tfa.trajectories.time_step.StepType.LAST:
                    break

                if not self.quadruped.upright:
                    break
                # Save the model after every n episodes
                if i > 0 and \
                    np.mod(i, self.params['TEST_AFTER_N_EPISODES']) == 0:
                    actor.model.save_weights(
                        os.path.join(
                            model_dir, 
                            'actormodel_'+identifier+'_{}'.format(i)+'.h5'
                        ), 
                        overwrite=True)
                    with open(
                        os.path.join(
                            model_dir, 
                            'actormodel_'+identifier+'_{}'.format(i)+'.json'
                        ), "w") as outfile:
                        json.dump(actor.model.to_json(), outfile)

                    critic.model.save_weights(
                        os.path.join(
                            model_dir, 
                            'criticmodel_'+identifier+'_{}'.format(i)+'.h5'
                        ), overwrite=True)
                    with open(
                        os.path.join(
                            model_dir,
                            'criticmodel_'+identifier+'_{}'.format(
                                i
                            )+'.json'
                        ), "w") as outfile:
                        json.dump(critic.model.to_json(), outfile)

            i += 1
            print('\n')

if __name__ == '__main__':
    learner = Learner(params, True)
    experiment = 10
    learner.pretrain_actor(experiment)
    learner.learn('rl/out_dir/models')
