import tensorflow as tf
from layers.oscillator import HopfOscillator
from layers.complex import ComplexDense, relu

class MotionStateEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        action_dim,
        units_osc,
        units_mu,
        units_mean,
        units_combine,
        units_motion_state,
        activation_output_mlp = 'tanh',
        activation_combine = 'tanh',
        activation_motion_state = 'tanh',
        activation_mu = 'tanh',
        activation_omega = 'relu',
    ):
        super(MotionStateEncoder, self).__init__()
        self.motion_state_dense = [tf.keras.layers.Dense(
            units = units,
            activation = activation_motion_state,
            name = 'motion_state_dense'
        ) for units in units_motion_state]

        self.mu_dense = tf.keras.layers.Dense(
            units = units_mu,
            activation = activation_mu,
            name = 'mu_dense'
        )

        self.mu_dense_2 = tf.keras.layers.Dense(
            units = action_dim,
            activation = tf.keras.activations.linear,
            name = 'mu_dense'
        )

        self.mean_dense = tf.keras.layers.Dense(
            units = units_mean,
            activation = activation_mu,
            name = 'mean_dense'
        )

        self.mean_dense_2 = tf.keras.layers.Dense(
            units = action_dim,
            activation = tf.keras.activations.linear,
            name = 'mu_dense'
        )

        self.omega_dense = tf.keras.layers.Dense(
            units = 1,
            activation = activation_omega,
            name = 'omega_dense'
        )

    def call(self, motion_state):
        for layer in self.motion_state_dense:
            motion_state = layer(motion_state)

        omega = self.omega_dense(motion_state)
        mu = self.mu_dense_2(self.mu_dense(motion_state))
        mean = self.mean_dense_2(self.mean_dense(motion_state))
        return [omega, mu, mean, motion_state]


class RobotStateEncoder(tf.keras.layers.Layer):
    def __init__(self, 
        units_robot_state,
        activation_robot_state = 'tanh'
    ):
        super(RobotStateEncoder, self).__init__()
        self.robot_state_dense = [tf.keras.layers.Dense(
            units = units,
            activation = activation_robot_state,
            name = 'motion_state_dense'
        ) for units in units_robot_state]

    def call(self, robot_state):
        for layer in self.robot_state_dense:
            robot_state = layer(robot_state)
        return robot_state

class RNNCell(tf.keras.layers.Layer):
    def __init__(self,
        dt,
        units_osc,
        activation_osc_state_dense = 'tanh',
        activation_robot_state_dense = 'tanh',
        activation_motion_state_dense = 'tanh'
    ):
        super(RNNCell, self).__init__()
        self.dt = dt
        self.osc = HopfOscillator(
            units = units_osc,
            dt = dt
        )

        self.osc_state_dense = ComplexDense(
            units = units_osc,
            activation = activation_osc_state_dense
        )

        self.state_dense = ComplexDense(
            units = units_osc,
            activation = activation_robot_state_dense
        )

    def build(self, input_shapes):
        self.z_shape, self.robot_shape = input_shapes
        self.built = True

    def call(self, inputs):
        z, state = inputs
        z = self.osc_state_dense(z)
        state = self.state_dense(state)
        out = tf.math.add(z, state)
        return out, out


def swap_batch_timestep(input_t):
    # Swap the batch and timestep dim for the incoming tensor.
    axes = list(range(len(input_t.shape)))
    axes[0], axes[1] = 1, 0
    return tf.transpose(input_t, axes)

class ComplexMLP(tf.keras.layers.Layer):
    def __init__(
        self,
        dt,
        steps,
        action_dim,
        units_output_mlp, # a list of units in all layers in output MLP
        units_osc,
        units_combine,
        units_robot_state,
        units_motion_state,
        activation_output_mlp = 'tanh',
        activation_combine = 'tanh',
        activation_robot_state = 'tanh',
        activation_motion_state = 'tanh',
        activation_omega = 'relu',
        name = 'TimeDistributedActor'
    ):
        super(ComplexMLP, self).__init__(name = name)
        self.steps = steps
        self.out_dim = action_dim

        self.output_mlp = tf.keras.Sequential()
        if isinstance(units_output_mlp, list):
            for i, num in enumerate(units_output_mlp):
                self.output_mlp.add(
                    ComplexDense(
                        units = num,
                        activation = activation_output_mlp,
                        name = 'complex_dense{i}'.format(i=i)
                    )
                )
        else:
            raise ValueError(
                'Expected units_output_mlp to be of type `list`, \
                    got typr `{t}`'.format(
                        type(t = units_output_mlp)
                )
            )

        self.osc = HopfOscillator(
            units = units_osc,
            dt = dt
        )

        self.rnn_cell = RNNCell(
            dt = dt,
            units_osc = units_osc
        )

        self.combine_dense = ComplexDense(
            units = units_osc,
            activation = activation_combine
        )

    def build(self, input_shapes):
        self.z_shape, self.omega_shape, self.robot_state_shape, \
            self.motion_state_shape = input_shapes
        self.osc.build([self.z_shape, self.omega_shape])
        self.rnn_cell.build([self.z_shape, self.robot_state_shape])
        self.built = True

    def call(self, inputs):
        z, omega, robot_state, motion_state = inputs
        state = tf.concat([robot_state, motion_state], -1)
        zeros1 = robot_state - robot_state
        zeros2 = motion_state - motion_state
        zeros = tf.concat([zeros1, zeros2], -1)
        state = tf.concat([state, zeros], -1)
        state = self.combine_dense(state)
        out = tf.TensorArray(tf.dtypes.float32, size = 0, dynamic_size=True)
        step = tf.constant(0)
        z_out = self.osc([z, omega])
        o, robot_state = self.rnn_cell([z_out, state])
        o = self.output_mlp(o)
        out = out.write(
            step,
            o
        )
        step = tf.math.add(step, tf.constant(1))

        def cond(out, step, z, state):
            return tf.math.less(
                step,
                tf.constant(
                    self.steps,
                    dtype = tf.int32
                )
            )

        def body(out, step, z, state):
            inputs = [
                z,
                omega,
            ]

            z = self.osc(inputs)
            o, robot_state = self.rnn_cell([z, state])
            o = self.output_mlp(z)

            out = out.write(
                step,
                o
            )

            step = tf.math.add(step, tf.constant(1))
            return out, step, z, robot_state

        out, step, _, _ = tf.while_loop(cond, body,[out,step,z_out,state])

        out = out.stack()
        out = swap_batch_timestep(out)
        out = tf.ensure_shape(
            out,
            tf.TensorShape(
                (None, self.steps, 2 * self.out_dim)
            ),
            name='ensure_shape_critic_time_distributed_out'
        )
        out = out[:, :, :self.out_dim]
        return [out, z_out]

def get_encoders(params):
    motion_encoder = MotionStateEncoder(
        action_dim = params['action_dim'],
        units_osc = params['units_osc'],
        units_mu = params['units_mu'],
        units_mean = params['units_mean'],
        units_combine = params['units_combine'],
        units_motion_state = params['units_motion_state'],
    )

    robot_encoder = RobotStateEncoder(
        units_robot_state = params['units_robot_state']
    )

    return motion_encoder, robot_encoder

def get_complex_mlp(params):
    cell = ComplexMLP(
        dt = params['dt'],
        steps = params['rnn_steps'],
        action_dim = params['action_dim'],
        units_output_mlp = params['units_output_mlp'],
        units_osc = params['units_osc'],
        units_combine = params['units_combine'],
        units_robot_state = params['units_robot_state'],
        units_motion_state = params['units_motion_state']
    )
    return cell
