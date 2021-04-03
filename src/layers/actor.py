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
        activation_output_mlp = 'elu',
        activation_combine = 'elu',
        activation_motion_state = 'elu',
        activation_mu = 'elu',
        activation_omega = 'elu',
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
        activation_robot_state = 'elu',
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
            activation_combine = 'elu',
            activation_mu = 'elu',
            activation_mean = 'elu',
            activation_omega = 'elu',
            name = 'param_net',
            trainable = True
        ):
        super(ParamNet, self).__init__(trainable, name)

        self.combine_dense = tf.keras.Sequential(name = 'combine_dense')
        for i, units in enumerate(units_combine):
            self.combine_dense.add(
                tf.keras.layers.Dense(
                    units = units,
                    activation = activation_combine,
                    kernel_regularizer = tf.keras.regularizers.l2(1e-3),
                    name = 'param_net_combine_dense_{i}'.format(i = i)
                )
            )

        self.mu_layers = tf.keras.Sequential(name = 'mu_dense')
        for i, units in enumerate(units_mu):
            self.mu_layers.add(
                tf.keras.layers.Dense(
                    units = units,
                    activation = activation_mu,
                    kernel_regularizer = tf.keras.regularizers.l2(1e-3),
                    name = 'mu_dense_{i}'.format(i = i)
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
        """
        self.A_layers = tf.keras.Sequential(name = 'mu_dense')
        for i, units in enumerate(units_mu):
            self.A_layers.add(
                tf.keras.layers.Dense(
                    units = units,
                    activation = activation_mu,
                    kernel_regularizer = tf.keras.regularizers.l2(1e-3),
                    name = 'A_dense_{i}'.format(i = i)
                )
            )
        self.A_layers.add(
            tf.keras.layers.Dense(
                units = units_osc,
                activation = tf.keras.activations.linear,
                name = 'A_dense_out',
                kernel_regularizer = tf.keras.regularizers.l2(1e-3)
            )
        )
        """

        self.mean_layers = tf.keras.Sequential(name = 'mean_dense')
        for i, units in enumerate(units_mean):
            self.mean_layers.add(
                tf.keras.layers.Dense(
                    units = units,
                    activation = activation_mean,
                    kernel_regularizer = tf.keras.regularizers.l2(1e-3),
                    name = 'mean_dense_{i}'.format(i = i)
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
        for i, units in enumerate(units_omega):
            self.omega_layers.add(
                tf.keras.layers.Dense(
                    units = units,
                    activation = activation_omega,
                    kernel_regularizer = tf.keras.regularizers.l2(1e-3),
                    name = 'omega_dense_{i}'.format(i = i)
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
        state = tf.concat(inputs[1:], -1)
        state = self.combine_dense(state)
        mu = self.mu_layers(state)
        mean = self.mean_layers(state)
        omega = self.omega_layers(inputs[0])
        return [state, omega, mu, mean]

class Encoder(tf.keras.Model):
    def __init__(self, 
            params, 
            name = 'state_encoder',
            trainable = True,
        ):
        super(Encoder, self).__init__(
            name = name,
            trainable = trainable
        )
        self.motion_encoder, self.robot_encoder = get_encoders(
            params, 
            trainable
        )
        self.param_net = get_param_net(params, trainable)

    def build(self, input_shapes):
        self.motion_state_shape = input_shapes[0]
        self.robot_state_shape = input_shapes[1]
        self.motion_encoder.build(self.motion_state_shape)
        self.robot_encoder.build(self.robot_state_shape)
        self.param_net.build(input_shapes)
        self.build = True

    def call(self, inputs):
        motion_state = self.motion_encoder(inputs[0])
        robot_state = self.robot_encoder(inputs[1])
        [state, omega, mu, mean] = self.param_net([motion_state, robot_state])
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
        units_mu,
        activation_output_mlp = 'tanh',
        activation_combine = 'elu',
        activation_robot_state = 'elu',
        activation_mu = 'elu',
        activation_motion_state = 'elu',
        activation_omega = 'elu',
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

        self.units_osc = units_osc

        self.A_layers = tf.keras.Sequential(name = 'mu_dense')
        for i, units in enumerate(units_mu):
            self.A_layers.add(
                tf.keras.layers.Dense(
                    units = units,
                    activation = activation_mu,
                    kernel_regularizer = tf.keras.regularizers.l2(1e-3),
                    name = 'A_dense_{i}'.format(i = i)
                )
            )
        self.A_layers.add(
            tf.keras.layers.Dense(
                units = units_osc,
                activation = 'relu',
                name = 'A_dense_out',
                kernel_regularizer = tf.keras.regularizers.l2(1e-3)
            )
        )

        self.osc = HopfOscillator(
            units = units_osc,
            dt = dt
        )

    def build(self, input_shapes):
        self.z_shape, self.state_shape, self.omega_shape = input_shapes
        self.A_layers.build(self.state_shape)
        self.A_shape = self.A_layers.compute_output_shape(self.state_shape)
        self.osc.build([self.z_shape, self.omega_shape, self.A_shape])
        self.built = True

    def call(self, inputs):
        z, state, omega = inputs
        A = self.A_layers(state)
        out = tf.TensorArray(tf.dtypes.float32, size = 0, dynamic_size=True)
        Z = tf.TensorArray(tf.dtypes.float32, size = 0, dynamic_size=True)
        step = tf.constant(0)
        z_out = self.osc([z, omega, A])
        o = self.output_mlp(z_out)
        out = out.write(
            step,
            o
        )
        Z = Z.write(
            step,
            z
        )
        step = tf.math.add(step, tf.constant(1))

        def cond(out, Z, step, z):
            return tf.math.less(
                step,
                tf.constant(
                    self.steps,
                    dtype = tf.int32
                )
            )

        def body(out, Z, step, z):
            inputs = [
                z,
                omega,
                A
            ]

            z = self.osc(inputs)
            o = self.output_mlp(z)

            out = out.write(
                step,
                o
            )
            Z = Z.write(
                step,
                z
            )

            step = tf.math.add(step, tf.constant(1))
            return out, Z, step, z

        out, Z, step, _ = tf.while_loop(cond, body,[out,Z,step,z_out])

        out = out.stack()
        Z = Z.stack()
        out = swap_batch_timestep(out)
        Z = swap_batch_timestep(Z)
        out = tf.ensure_shape(
            out,
            tf.TensorShape(
                (None, self.steps, 2 * self.out_dim)
            ),
            name='ensure_shape_critic_time_distributed_out'
        )

        Z = tf.ensure_shape(
            Z,
            tf.TensorShape(
                (None, self.steps, 2 * self.units_osc)
            ),
            name='ensure_shape_critic_time_distributed_out'
        )

        out = 2 * out[:, :, :self.out_dim]
        return [out, Z]

def get_encoders(params, trainable = True):
    motion_encoder = MotionStateEncoder(
        action_dim = params['action_dim'],
        units_osc = params['units_osc'],
        units_mu = params['units_mu'],
        units_mean = params['units_mean'],
        units_motion_state = params['units_motion_state'],
        trainable = trainable
    )

    robot_encoder = RobotStateEncoder(
        units_robot_state = params['units_robot_state'],
        trainable = trainable
    )

    return motion_encoder, robot_encoder

def get_state_encoder(params, trainable = True):
    return Encoder(params, trainable = trainable)

def get_param_net(params, trainable = True):
    param_net = ParamNet(
        units_osc = params['units_osc'],
        action_dim = params['action_dim'],
        units_combine = params['units_combine'],
        units_mu = params['units_mu'],
        units_omega = params['units_omega'],
        units_mean = params['units_mean'],
        trainable = trainable
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
        units_motion_state = params['units_motion_state'],
        units_mu = params['units_mu']
    )
    return cell
