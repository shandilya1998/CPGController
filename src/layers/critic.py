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
        activation_combine = 'elu',
        activation_robot_state = 'elu',
        activation_motion_state = 'elu',
        activation_action_input = 'elu',
        activation_osc = 'elu',
        activation_lstm = 'tanh',
        recurrent_activation_lstm = 'sigmoid',
        activation_out = 'elu',
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
        self.osc_dense_real = tf.keras.layers.Dense(
            units = units_osc//2,
            activation = activation_osc,
            name = 'osc_dense_real'
        )
        self.osc_dense_imag = tf.keras.layers.Dense(
            units = units_osc//2,
            activation = activation_osc,
            name = 'osc_dense_imag'
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
        motion_state, robot_state, osc_state, \
            action, osc, history, history_osc = inputs
        motion_state = self.motion_state_dense(motion_state)
        robot_state = self.robot_state_dense(robot_state)
        osc_size = osc.shape[-1] // 2
        state = tf.concat([
            motion_state,
            robot_state
        ], -1)
        state = self.combine_dense(state)

        ta_history = tf.TensorArray(
            tf.dtypes.float32, size = 0, dynamic_size = True
        )
        history = swap_batch_timestep(history)
        history = history[-1 * self.steps + 1:]
        ta_history.unstack(history)

        ta_action = tf.TensorArray(
            tf.dtypes.float32,
            size = 0,
            dynamic_size = True
        )
        action = swap_batch_timestep(action)
        ta_action.unstack(action)

        ta_osc = tf.TensorArray(
            tf.dtypes.float32,
            size = 0,
            dynamic_size = True
        )
        osc = swap_batch_timestep(osc)
        ta_osc.unstack(osc)

        ta_history_osc = tf.TensorArray(
            tf.dtypes.float32,
            size = 0,
            dynamic_size = True
        )
        history_osc = swap_batch_timestep(history_osc)
        history_osc = history_osc[-1 * self.steps + 1:, :, :]
        ta_history_osc.unstack(history_osc)

        step = tf.constant(0, dtype = tf.dtypes.int32)
        def cond(out, h, c, step):
            return tf.math.less(
                step,
                self.steps-1
            )

        def body(out, h, c, step):
            inp1 = ta_history.read(step)
            inp2 = ta_history_osc.read(step)
            inp2 = self.osc_dense_real(inp2[:, :osc_size])
            inp3 = self.osc_dense_imag(inp2[:, osc_size:])
            inp = tf.concat([inp1, inp2, inp3], -1)
            out, [h, c] = self.lstm(inp, [h, c])
            step = tf.math.add(step, tf.constant(1, tf.dtypes.int32))
            return out, h, c, step

        out, h, c, step = tf.while_loop(cond, body, [state, state, state, step])

        step = tf.constant(0, dtype = tf.dtypes.int32)
        def cond(out, h, c, step):
            return tf.math.less(
                step,
                self.steps
            )

        def body(out, h, c, step):
            inp1 = ta_action.read(step)
            inp2 = ta_osc.read(step)
            inp2 = self.osc_dense_real(inp2[:, :osc_size])
            inp3 = self.osc_dense_imag(inp2[:, osc_size:])
            inp = tf.concat([inp1, inp2, inp3], -1)
            out, [h, c] = self.lstm(inp, [h, c])
            step = tf.math.add(step, tf.constant(1, tf.dtypes.int32))
            return out, h, c, step

        out, h, c, step = tf.while_loop(cond, body, [out, h, c, step])

        out = self.out_dense(out)
        return out

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
