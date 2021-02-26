from rl.constants import params
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

class SignalDataGen:
    def __init__(self, params):
        self.Tst = params['Tst']
        self.Tsw = params['Tsw']
        self.theta_h = params['theta_h']
        self.theta_k = params['theta_k']
        self.N = params['rnn_steps']
        self.params = params
        self.signal_gen = Signal(
            self.N,
            self.params['dt']
        )
        self.dt = self.params['dt']
        self.data = []
        self.num_data = 0
        self._create_data()

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
        deltas = [-3, 0, 3]
        delta = []
        for i in range(len(deltas)):
            for j in range(len(deltas)):
                delta.append([
                    deltas[i], 
                    deltas[j], 
                ])
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
                motion = np.array([1, 0, 0, v, 0 ,0], dtype = np.float32)
                freq = self.get_ff(signal, 'autocorr')
                self.data.append(
                    [signal, motion, freq]
                )
        self.num_data = len(self.data)
        print('[Actor] Number of Data Points: {num}'.format(
            num = self.num_data)
        )    

    def generator(self):
        for batch in range(self.num_data):
            y, x = self.data[batch]
            y = np.expand_dims(y, 0)
            x = x
            yield y, x

class Learner():
    def __init__(self, params):
        self.params = params
        self.actor = ActorNetwork(params)
        self.critic = CriticNetwork(params)
        self.replay_buffer = ReplayBuffer(params)
        self.time_step_spec = tfa.trajectories.time_step.time_step_spec(
            observation_spec = self.params['observation_spec'],
            reward_spec = self.params['reward_spec']
        )
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
        self.OU = OU()
        self.desired_motion = np.zeros((
            self.params['max_steps'], 6
        ), dtype = np.float32)
        self.desired_motion[:, 3] = 0.05
        self.signal_gen = SignalDataGen(params)
        self.pretrain_actor_optimizer = tf.keras.optimizers.SGD(
            learning_rate = self.params['LRA']
        )
        self._state = [
            self.env.quadruped.motion_state,
            self.env.quadruped.robot_state,
            self.env.quadruped.osc_state
        ]
        self._action = None
        physical_devices = tf.config.list_physical_devices('GPU')
        print('[Actor] GPU>>>>>>>>>>>>')
        print('[Actor] {lst}'.format(lst = physical_devices))
        try:
            print('[Actor] Memory Growth Allowed')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
             # Invalid device or cannot modify virtual devices once initialized.
            pass

    def set_desired_motion(self, motion):
        self.desired_motion = motion

    def _pretrain_actor(self, x, y):
        with tf.GradientTape() as tape:
            _action = self.actor.model(x)
            y_pred = _action[0]
            loss = self.actor._pretrain_loss(y, y_pred)
        grads = tape.gradient(
            loss,
            self.actor.model.trainable_variables
        )
        self.pretrain_actor_optimizer.apply_gradients(
            zip(
                grads,
                self.actor.model.trainable_variables
            )
        )
        return loss

    def create_dataset(self):
        self.num_batches = self.signal_gen.num_data//self.params['pretrain_bs']
        self.env.quadruped.reset()
        _state = self.env.quadruped.get_state_tensor()
        F = []
        Y = []
        X = [[] for j in range(len(self.params['observation_spec']))]
        for y, x, f in self.signal_gen.generator():
            _state[0] = np.expand_dims(x, 0)
            for j, s in enumerate(_state):
                X[j].append(s)
            Y.append(y)
            F.append(np.array([f]))
        for j in range(len(X)):
            X[j] = np.concatenate(X[j], axis = 0)
        Y = np.concatenate(Y, axis = 0)
        F = np.concatenate(F, axis = 0)
        print(Y.shape)
        np.save('data/pretrain/Y.npy', Y)
        np.save('data/pretrain/F.npy', F)
        for j in range(len(X)):
            np.save('data/pretrain/X_{j}.npy'.format(j=j), X[j])
    
    def load_dataset(self):
        Y = tf.convert_to_tensor(np.load('data/pretrain/Y.npy'))
        #F = tf.convert_to_tensor(np.load('data/pretrain/F.npy'))
        X = []
        for j in range(len(self.params['observation_spec'])):
            X.append(
                tf.data.Dataset.from_tensor_slices(
                    tf.convert_to_tensor(
                        np.load('data/pretrain/X_{j}.npy'.format(j=j))
                    )
                )
            )
        Y = tf.data.Dataset.from_tensor_slices(Y)
        X = tf.data.Dataset.zip(tuple(X))
        dataset = tf.data.Dataset.zip((X, Y))
        dataset = dataset.batch(self.params['pretrain_bs']).prefetch(2)
        self.num_batches = self.signal_gen.num_data//self.params['pretrain_bs']
        return dataset

    def pretrain_actor(self, experiment, checkpoint_dir = 'weights/actor_pretrain'):
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
        print('[Actor] Starting Actor Pretraining')
        for episode in range(self.params['train_episode_count']):
            print('[Actor] Starting Episode {ep}'.format(ep = episode))
            for step, (x, y) in tqdm(enumerate(dataset)):
                loss = self._pretrain_actor(x, y)
                total_loss += loss.numpy()
            avg_loss = total_loss / self.num_batches
            print('[Actor] Episode {ep} Average Loss: {l}'.format(
                ep = episode,
                l = avg_loss
            ))
            history_loss.append(avg_loss)
            if episode % 5 == 0:
                if prev_loss < avg_loss:
                    break
                else:
                    self.actor.model.save_weights(
                        os.path.join(
                            checkpoint_dir,
                            'actor_pretrained_{ex}_{ep}.ckpt'.format(ep=episode, ex = experiment)
                        ),
                    )
                prev_loss = avg_loss
        pkl = open(os.path.join(checkpoint_dir, 'loss.pickle'), 'wb')
        pickle.dump(history_loss, pkl)
        pkl.close()

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
                action_original = self.actor.model(self._state)
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

                inputs = next_states + self.actor.target_model(next_states)
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
                a_for_grad = self.actor.model(states)
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
    learner = Learner(params)
    experiment = 1
    learner.pretrain_actor(experiment)
    learner.learn('rl/out_dir/models')
