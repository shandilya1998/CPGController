from rl.constants import params
from rl.net import ActorNetwork, CriticNetwork
from rl.env import Env
from rl.replay_buffer import ReplayBuffer, OU
import tf_agents as tfa
import tensorflow as tf
import numpy as np

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

    def set_desired_motion(self, motion):
        self.desired_motion = motion

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
            self._action = self.env._action_init
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
                if self.current_time_step.step_type == tfa.trajectories.time_step.StepType.LAST:
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
    learner.learn('rl/out_dir/models')
