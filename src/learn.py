from simulations.ws.src.quadruped.scripts.quadruped import Quadruped
from rl.constants import params
from rl.net import ActorNetwork, CriticNetwork
from rl.env import Env
import tf_agents as tfa

class Learner():
    def __init__(self):
        self.params = params
        self.quadruped = Quadruped(params)
        self.actor = ActorNetwork(params)
        self.critic = CrticNetwork(params)
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

    def learn(self, model_dir, identifier=''):
        i = 0
        epsilon = 1
        self._noise_init = [•
                    tf.expand_dims(tf.zeros(
                        spec.shape,
                        spec.dtype
                    ), 0) for spec in self.env.action_spec()
                ]
        print('DDPG training start')
        while i < self.params['train_episode_count']:
            self.current_time_step = self.env.reset()
            self._state = self.current_time_step.observation
            self.total_reward = 0.0
            self._action = self.env._action_init
            self._noise = self._noise_init
            for j in range(self.parmas['max_steps']):
                loss = 0.0
                epsilon -= 1/self.params['EXPLORE']
                self._action = self.env._action_init
                self._noise = self._noise_init
                action_original = self.actor.model.predict(self._state)
                self._noise[0] = max(epsilon, 0) * OU.function(
                    action_original[0],
                    0.0,
                    0.15,
                    0.2
                )
                self._action[0] = action_original + self._noise
                self._action[1] = action_original
                self.current_time_step = self.env.step(self._action)
                experience = [
                    self._state,
                    self._action,
                    self.current_time_step.reward,
                    self.current_time_step.observation,
                    self.current_time_step.step_type
                ]
                self.replay_buffer.add(experience)
                batch = self.replay_buffer.get_next(self.params['BATCH_SIZE'])
                states = tf.concat([e[0] for e in batch], 0)
                actions = tf.concat([e[1] for e in batch], 0)
                rewards = tf.concat([e[2] for e in batch], 0)
                next_states = tf.concat([e[3] for e in batch], 0)
                step_types = [e[4] for e in batch]
                target_q_values = self.critic.target_model.predict([
                    new_states,
                    self.actor.target_model.predict(new_states)
                ])
                y = tf.concat([e[2] for e in batch], 0)

                for k in range(len(batch)):
                    if step_types[k]==tfa.trajectories.time_step.StepType.LAST:
                        y[k] = rewards[k]
                    else:
                        y[k] = rewards[k] + \
                            self.params['GAMMA'] * target_q_values[k]

                with tf.GradientTape() as tape:
                    loss += self.critic.model.train_on_batch(
                        [states, actions],
                        y
                    )
                    a_for_grad = self.actor.model.predict(states)
                """
                    Need to implement gradient calculation and update code here
                """
                self.actor.target_train()
                self.critic.target_train()

                self.total_reward += self.current_time_step.reward
                self._state = self.current_time_step.observation


                step += 1
                if done:
                    break

                # Save the model after every n episodes
                if i > 0 and np.mod(i, TEST_AFTER_N_EPISODES) == 0:
                    actor.model.save_weights(os.path.join(model_dir, 'actormodel_'+identifier+'_{}'.format(i)+'.h5'), overwrite=True)
                    with open(os.path.join(model_dir, 'actormodel_'+identifier+'_{}'.format(i)+'.json'), "w") as outfile:
                        json.dump(actor.model.to_json(), outfile)

                    critic.model.save_weights(os.path.join(model_dir, 'criticmodel_'+identifier+'_{}'.format(i)+'.h5'), overwrite=True)
                    with open(os.path.join(model_dir, 'criticmodel_'+identifier+'_{}'.format(i)+'.json'), "w") as outfile:
                        json.dump(critic.model.to_json(), outfile)

            step = 0
            i += 1

