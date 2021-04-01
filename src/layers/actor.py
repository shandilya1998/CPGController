import tensorflow as tf
from layers.oscillator import HopfOscillator
from layers.complex import ComplexDense, ComplexGRUCell, relu

class MotionStateEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        action_dim,
        units_osc,
        units_mu,
        units_mean,
        units_motion_state,
        activation_output_mlp = 'relu',
        activation_combine = 'relu',
        activation_motion_state = 'relu',
        activation_mu = 'relu',
        activation_omega = 'relu',
        name = 'motion_state_encoder',
        trainable = True
    ):
        super(MotionStateEncoder, self).__init__(trainable, name)
        self.motion_state_dense = tf.keras.Sequential(name='motion_state_dense')
        for i, units in enumerate(units_motion_state):
            self.motion_state_dense.add(
                tf.keras.layers.Dense(
                    units = units,
                    activation = activation_motion_state,
                    name = 'motion_state_dense_{i}'.format(i = i),
                    kernel_regularizer = tf.keras.regularizers.l2(1e-3)
                )
            )

    def call(self, motion_state):
        return self.motion_state_dense(motion_state)


class RobotStateEncoder(tf.keras.layers.Layer):
    def __init__(self, 
        units_robot_state,
        activation_robot_state = 'relu',
        name = 'robot_state_encoder',
        trainable = True
    ):
        super(RobotStateEncoder, self).__init__(trainable, name)
        self.robot_state_dense = tf.keras.Sequential(name='robot_state_dense')
        for i, units in enumerate(units_robot_state):
            self.robot_state_dense.add(
                tf.keras.layers.Dense(
                    units = units,
                    activation = activation_robot_state,
                    name = 'robot_state_dense_{i}'.format(i = i),
                    kernel_regularizer = tf.keras.regularizers.l2(1e-3)
                )
            )

    def call(self, robot_state):
        return self.robot_state_dense(robot_state)


class ParamNet(tf.keras.layers.Layer):
    def __init__(
            self,
            units_osc,
            action_dim,
            units_combine,
            units_mu,
            units_omega,
            units_mean,
            activation_combine = 'relu',
            activation_mu = 'relu',
            activation_mean = 'relu',
            activation_omega = 'relu',
            name = 'param_net',
            trainable = True
        ):
        super(ParamNet, self).__init__(trainable, name)

        self.combine_dense = tf.keras.Sequential(name = 'combine_dense')
        for units in units_combine:
            self.combine_dense.add(
                tf.keras.layers.Dense(
                    units = units,
                    activation = activation_combine,
                    kernel_regularizer = tf.keras.regularizers.l2(1e-3)
                )
            )

        self.mu_layers = tf.keras.Sequential(name = 'mu_dense')
        for units in units_mu:
            self.mu_layers.add(
                tf.keras.layers.Dense(
                    units = units,
                    activation = activation_mu,
                    kernel_regularizer = tf.keras.regularizers.l2(1e-3)
                )
            )
        self.mu_layers.add(
            tf.keras.layers.Dense(
                units = action_dim,
                activation = tf.keras.activations.linear,
                name = 'mu_dense_out',
                kernel_regularizer = tf.keras.regularizers.l2(1e-3)
            )
        )

        self.mean_layers = tf.keras.Sequential(name = 'mean_dense')
        for units in units_mean:
            self.mean_layers.add(
                tf.keras.layers.Dense(
                    units = units,
                    activation = activation_mean,
                    kernel_regularizer = tf.keras.regularizers.l2(1e-3)
                )
            )
        self.mean_layers.add(
            tf.keras.layers.Dense(
                units = action_dim,
                activation = tf.keras.activations.linear,
                name = 'mean_dense_out',
                kernel_regularizer = tf.keras.regularizers.l2(1e-3)
            )
        )

        self.omega_layers = tf.keras.Sequential(name = 'omega_dense')
        for units in units_omega:
            self.omega_layers.add(
                tf.keras.layers.Dense(
                    units = units,
                    activation = activation_omega,
                    kernel_regularizer = tf.keras.regularizers.l2(1e-3)
                )
            )
        self.omega_layers.add(
            tf.keras.layers.Dense(
                units = 1,
                activation = tf.keras.activations.linear,
                name = 'omega_dense_out',
                kernel_regularizer = tf.keras.regularizers.l2(1e-3)
            )
        )

    def call(self, inputs):
        state = tf.concat(inputs, -1)
        state = self.combine_dense(state)
        mu = self.mu_layers(state)
        mean = self.mean_layers(state)
        omega = self.omega_layers(state)
        return [state, omega, mu, mean]

def swap_batch_timestep(input_t):
    # Swap the batch and timestep dim for the incoming tensor.
    axes = list(range(len(input_t.shape)))
    axes[0], axes[1] = 1, 0
    return tf.transpose(input_t, axes)

class ComplexRNN(tf.keras.Model):
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
        activation_combine = 'relu',
        activation_robot_state = 'relu',
        activation_motion_state = 'relu',
        activation_omega = 'relu',
        name = 'TimeDistributedActor'
    ):
        super(ComplexRNN, self).__init__(name = name)
        self.steps = steps
        self.out_dim = action_dim

        self.output_mlp = tf.keras.Sequential(name = 'output_mlp')
        if isinstance(units_output_mlp, list):
            for i, num in enumerate(units_output_mlp):
                self.output_mlp.add(
                    ComplexDense(
                        units = num,
                        activation = activation_output_mlp,
                        name = 'complex_dense{i}'.format(i=i),
                        kernel_regularizer = tf.keras.regularizers.l2(1e-3)
                    )
                )
        else:
            raise ValueError(
                'Expected units_output_mlp to be of type `list`, \
                    got typr `{t}`'.format(
                        type(t = units_output_mlp)
                )
            )

        self.gru_cell = ComplexGRUCell(
            units = units_combine[-1],
            kernel_regularizer = tf.keras.regularizers.l2(1e-3)
        )

        self.osc = HopfOscillator(
            units = units_osc,
            dt = dt
        )

    def build(self, input_shapes):
        self.z_shape, self.state_shape, self.omega_shape = input_shapes
        self.osc.build([self.z_shape, self.omega_shape])
        self.gru_cell.build(self.z_shape)
        self.built = True

    def call(self, inputs):
        z, state, omega = inputs
        out = tf.TensorArray(tf.dtypes.float32, size = 0, dynamic_size=True)
        step = tf.constant(0)
        z_out = self.osc([z, omega])
        o, state = self.gru_cell(z_out, state)
        o = self.output_mlp(z_out)
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
            o, state = self.gru_cell(z, state)
            o = self.output_mlp(o)

            out = out.write(
                step,
                o
            )

            step = tf.math.add(step, tf.constant(1))
            return out, step, z, state

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
        out = 2 * out[:, :, :self.out_dim]
        return [out, z_out]

def get_encoders(params):
    motion_encoder = MotionStateEncoder(
        action_dim = params['action_dim'],
        units_osc = params['units_osc'],
        units_mu = params['units_mu'],
        units_mean = params['units_mean'],
        units_motion_state = params['units_motion_state'],
    )

    robot_encoder = RobotStateEncoder(
        units_robot_state = params['units_robot_state']
    )

    return motion_encoder, robot_encoder

def get_param_net(params):
    param_net = ParamNet(
        units_osc = params['units_osc'],
        action_dim = params['action_dim'],
        units_combine = params['units_combine'],
        units_mu = params['units_mu'],
        units_omega = params['units_omega'],
        units_mean = params['units_mean']
    )
    return param_net

def get_complex_mlp(params):
    cell = ComplexRNN(
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
