import tensorflow as tf
from layers.complex import ComplexDense, relu

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
        activation_combine = relu,
        activation_robot_state = relu,
        activation_motion_state = relu,
        activation_action_input = relu,
        activation_osc = relu,
        activation_lstm = 'tanh',
        recurrent_activation_lstm = 'sigmoid',
        activation_out = relu
    ):
        super(Critic, self).__init__()

        self.steps = steps
        self.combine_dense = ComplexDense(
            units = units_combine,
            activation = activation_combine,
            name = 'combine_dense'
        )
        self.robot_state_dense = ComplexDense(
            units = units_robot_state,
            activation = activation_robot_state,
            name = 'robot_state_dense'
        )
        self.action_input_dense = ComplexDense(
            units = units_action_input,
            activation = activation_action_input,
            name = 'action_input_dense'
        )
        self.motion_state_dense = ComplexDense(
            units = units_motion_state,
            activation = activation_motion_state,
            name = 'motion_state_dense'
        )
        self.real_osc_dense = ComplexDense(
            units = units_osc,
            activation = activation_osc,
            name = 'real_osc_dense'
        )
        self.imag_osc_dense = ComplexDense(
            units = units_osc,
            activation = activation_osc,
            name = 'imag_osc_dense'
        )
        self.lstm = ComplexLSTMCell(
            units = units_lstm,
            activation = activation_lstm,
            recurrent_activation = recurrent_activation_lstm
        )
        self.out_dense = ComplexDense(
            units = units_out,
            activation = activation_out
        )

    def call(self, inputs):
        S, A, history = inputs
        motion_state, robot_state, osc_state = S
        action, osc = A

        osc_state = tf.concat([osc_state, osc], axis = -1)

        motion_state = self.motion_state_dense(motion_state)
        robot_state = self.robot_state_dense(robot_state)

        state = tf.concat([
            motion_state,
            robot_state,
            osc_state
        ], axis = -1)
        state = self.combine_dense(state)

        ta_history = tf.TensorArray('complex64', size = 0, dynamic_size = True)
        history = swap_batch_timestep(history)
        ta_history.unstack(history)

        ta_action = tf.TensorArray(
            'complex64',
            size = 0,
            dynamic_size = True
        )
        action = swap_batch_timestep(action)
        ta_action.unstack(action)

        step = tf.constant(0, dtype = tf.dtypes.int32)
        def cond(out, h, c, step):
            return tf.math.less(
                step,
                self.steps-1
            )

        def body(out, h, c, step):
            inp = ta_history.read(step)
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
            inp = ta_action.read(step)
            out, [h, c] = self.lstm(inp, [h, c])
            step = tf.math.add(step, tf.constant(1, tf.dtypes.int32))
            return out, h, c, step

        out, h, c, step = tf.while_loop(cond, body, [out, h, c, step])

        out = self.out_dense(out)
`
        return out


def get_critic(params):
    critic = Critic(
        steps = params['rnn_steps'],
        units_combine = params['units_combine'],
        units_robot_state = params['units_robot_state'],
        units_motion_state = params['units_motion_state'],
        units_action_input = params['units_action_input'],
        units_history = params['units_history'],
        units_osc = params['units_osc'],
        units_lstm = params['units_combine'],
        units_out = params['action_dim'],
    )
    return critic

