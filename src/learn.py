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

class SignalDataGen:
    def __init__(self, params):
        self.Tst = params['Tst']
        self.Tsw = params['Tsw']
        self.theta_h = params['theta_h']
        self.theta_k = params['theta_k']
        self.params = params
        self.signal_gen = Signal(
            self.params['rnn_steps'],
            self.params['dt']
        )
        self.data = []
        self.num_batches = 0
        self._create_data()

    def _create_data(self):
        """
            Turning Behaviour is to be learnt by RL
        """
        self.data = []
        for tst, tsw, theta_h, theta_k in zip(self.Tst, self.Tsw, self.theta_h, self.theta_k):
            self.signal_gen.build(tst, tst, theta_h, theta_k)
            signal, _ = self.signal_gen.get_signal()
            v = self.signal_gen.compute_v((0.1+0.015)*2.2)
            motion = np.stack([
                np.array([1, 0, 0, v, 0 ,0]) \
                for i in range(self.params['rnn_steps'])
            ])
            self.data.append(
                [signal, motion]
            )
        self.num_batches = len(self.data)

    def generator(self):
        for batch in range(self.num_batches):
            y, x = self.data[batch]
            y = tf.convert_to_tensor(np.expand_dims(y, 0))
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
        ))
        self.desired_motion[:, 3] = 0.05
        self.signal_gen = SignalDataGen(params)
        self.pretrain_actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate = self.params['LRA']
        )
        self._state = [
            self.env.quadruped.motion_state,
            self.env.quadruped.robot_state,
            self.env.quadruped.osc_state
        ]
        self._action = None

    def set_desired_motion(self, motion):
        self.desired_motion = motion

    def _pretrain_actor(self, x, y):
        self._state = [
            self.env.quadruped.motion_state,
            self.env.quadruped.robot_state,
            self.env.quadruped.osc_state
        ]
        with tf.GradientTape() as tape:
            self._action = self.actor.model(self._state)
            y_pred = self._action[0][:, 0, :]
            y_true = y[:, i, :]
        loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        self._action = [
            tf.make_ndarray(
                tf.make_tensor_proto(a)
            ) for a in self._action
        ]
        self.env.quadruped.set_observation([
            self._action[0][0][0],
            self._action[1][0]
        ], x[i, :])
        grads = tape.gradient(
            loss,
            self.model.trainable_variables
        )
        self.pretrain_actor_optimizer.apply_gradients(
            zip(
                grads,
                self.model.trainable_variables
            )
        )
        return loss

    def pretrain_actor(self, checkpoint_path = 'weights/actor_pretrain'):
        total_loss = 0.0
        avg_loss = 0.0
        prev_loss = 1e10
        history_loss = []
        print('[Actor] Starting Actor Pretraining')
        for episode in range(self.params['train_episode_count']):
            print('[Actor] Starting Episode {ep}'.format(ep = episode))
            for y, x in tqdm(self.signal_gen.generator()):
                self.env.quadruped.reset()
                for i in range(self.params['rnn_steps']):
                    self._pretrain_actor(x, y)
                    total_loss += loss
                total_loss = total_loss / self.params['rnn_steps']
                avg_loss += total_loss
            avg_loss = avg_loss / self.signal_gen.num_batches
            print('[Actor] Episode {ep} Average Loss: {l}'.format(
                ep = episode,
                l = avg_loss
            ))
            history_loss.append(avg_loss)
            if episode % 5 == 0:
                if prev_loss < avg_loss:
                    break
                else:
                    self.actor.model.save(
                        os.path.join(
                            checkpoint_path,
                            'actor_pretrained.h5'
                        ),
                        overwrite = True
                    )
                prev_loss = avg_loss


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
    learner.pretrain_actor()
    learner.learn('rl/out_dir/models')
