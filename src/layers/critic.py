import tensorflow as tf
from layers.complex import ComplexDense
import sys

def swap_batch_timestep(input_t):
    # Swap the batch and timestep dim for the incoming tensor.
    axes = list(range(len(input_t.shape)))
    axes[0], axes[1] = 1, 0
    return tf.transpose(input_t, axes)

class Critic(tf.keras.Model):
    def __init__(
        self,
        steps,
        units_combine,
        units_robot_state,
        units_motion_state,
        units_action_input,
        units_history,
        units_osc,
        units_lstm,
        units_out,
        activation_combine = 'tanh',
        activation_robot_state = 'tanh',
        activation_motion_state = 'tanh',
        activation_action_input = 'tanh',
        activation_osc = 'tanh',
        activation_lstm = 'tanh',
        recurrent_activation_lstm = 'sigmoid',
        activation_out = 'relu',
        training = True
    ):
        super(Critic, self).__init__()

        self.steps = steps
        self.combine_dense = tf.keras.layers.Dense(
            units = units_combine,
            activation = activation_combine,
            name = 'combine_dense'
        )
        self.combine_osc_dense = tf.keras.layers.Dense(
            units = units_osc,
            activation = activation_combine,
            name = 'combine_osc_dense'
        )
        self.robot_state_dense = tf.keras.layers.Dense(
            units = units_robot_state,
            activation = activation_robot_state,
            name = 'robot_state_dense'
        )
        self.motion_state_dense = tf.keras.layers.Dense(
            units = units_motion_state,
            activation = activation_motion_state,
            name = 'motion_state_dense'
        )
        self.osc_state_dense = tf.keras.layers.Dense(
            units = units_osc,
            activation = activation_osc,
            name = 'osc_state_dense'
        )
        self.osc_dense = tf.keras.layers.Dense(
            units = units_osc,
            activation = activation_osc,
            name = 'osc_dense'
        )
        self.lstm = tf.keras.layers.LSTMCell(
            units = units_lstm,
            activation = activation_lstm,
            recurrent_activation = recurrent_activation_lstm
        )
        self.out_dense = tf.keras.layers.Dense(
            units = units_out,
            activation = activation_out,
            name = 'out_dense'
        )
        self.training = training

    def call(self, inputs):
        motion_state, robot_state, osc_state, action, osc, history = inputs
        
        osc_state = self.osc_state_dense(osc_state)
        osc = self.osc_dense(osc)
        osc = tf.concat([osc_state, osc], -1)
        osc = self.combine_osc_dense(osc)
        motion_state = self.motion_state_dense(motion_state)
        robot_state = self.robot_state_dense(robot_state)

        state = tf.concat([
            osc,
            motion_state,
            robot_state
        ], -1)
        state = self.combine_dense(state)

        return state
    
def get_critic(params):
    critic = Critic(
        steps = params['rnn_steps'],
        units_combine = params['action_dim'],
        units_robot_state = params['units_robot_state_critic'],
        units_motion_state = params['units_motion_state_critic'],
        units_action_input = params['units_action_input'],
        units_history = params['units_history'],
        units_osc = params['units_osc'],
        units_lstm = params['action_dim'],
        units_out = params['action_dim'],
    )
    return critic
