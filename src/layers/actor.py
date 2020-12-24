import tensorflow as tf
from layers.oscillator import HopfOscillator
from layers.complex import ComplexDense, relu

class Actor(tf.keras.Model):
    def __init__(
        self, 
        dt,
        units_output_mlp, # a list of units in all layers in output MLP
        units_osc,
        units_combine,
        units_robot_state,
        units_motion_state,
        activation_output_mlp = relu,
        activation_combine = 'relu',
        activation_robot_state = 'relu',
        activation_motion_state = 'relu',
        activation_mu = 'relu',
        activation_omega = 'relu',
        activation_b = 'relu',
    ):
        super(Actor, self).__init__()

        self.output_mlp = tf.keras.Sequential()
        if isinstance(units_output_mlp, list):
            for num in units_output_mlp:
                self.output_mlp.add(
                    ComplexDense(
                        units = num,
                        activation = activation_output_mlp
                    )
                )
        else:
            raise ValueError(
                'Expected units_output_mlp to be of type `list`, got typr `{t}`'.format(
                    type(t = units_output_mlp)
                )
            )

        self.osc = HopfOscillator(
            units = units_osc, 
            dt = dt
        )

        self.combine_dense = tf.keras.layers.Dense(
            units = units_combine,
            activation = activation_combine,
            dtype = 'float32',
            name = 'combine_dense'
        )
        self.robot_state_dense = tf.keras.layers.Dense(
            units = units_robot_state,
            activation = activation_robot_state,
            dtype = 'float32',
            name = 'robot_state_dense'
        )
        self.motion_state_dense = tf.keras.layers.Dense(
            units = units_motion_state,
            activation = activation_motion_state,
            dtype = 'float32',
            name = 'motion_state_dense'
        )
        self.mu_dense = tf.keras.layers.Dense(
            units = units_osc,
            activation = activation_mu,
            dtype = 'float32',
            name = 'mu_dense'
        )
        self.omega_dense = tf.keras.layers.Dense(
            units = 1,
            activation = activation_omega,
            dtype = 'float32',
            name = 'omega_dense'
        )
        self.b_dense = tf.keras.layers.Dense(
            units = units_osc,
            activation = activation_b,
            dtype = 'float32',
            name = 'b_dense'
        )

    def call(self, inputs):
        x1, x2, z = inputs
        x1 = self.motion_state_dense(x1)
        x2 = self.robot_state_dense(x2)
        x = tf.concat([x1, x2], axis = -1)
        x = self.combine_dense(x)
        omega = self.omega_dense(x)
        b = self.b_dense(x)
        mu = self.mu_dense(x)
        inputs = [z, omega, mu, b]
        z = self.osc(inputs)
        out = self.output_mlp(z)
        return [out, z]


def get_actor_net(params):
    inp1 = tf.keras.Input((params['motion_state_size']), dtype = 'float32')
    inp2 = tf.keras.Input((params['robot_state_size']), dtype = 'float32')
    inp3 = tf.keras.Input((params['units_osc']), dtype = 'complex64' )
    out, z = Actor(
        dt = params['dt'],
        units_output_mlp = params['units_output_mlp'],
        units_osc = params['units_osc'],
        units_combine = params['units_combine'],
        units_robot_state = params['units_robot_state'],
        units_motion_state = params['units_motion_state']
    )([inp1, inp2, inp3])
    model = tf.keras.Model(inputs = [inp1, inp2, inp3], outputs = [out, z])
    return model

def swap_batch_timestep(input_t):
    # Swap the batch and timestep dim for the incoming tensor.
    axes = list(range(len(input_t.shape)))
    axes[0], axes[1] = 1, 0
    return tf.transpose(input_t, axes)

class TimeDistributed(tf.keras.Model):
    def __init__(self, layer, params, name = 'TimeDistributedActor'):
        super(TimeDistributed, self).__init__(name = name)
        self.layer = layer
        self.steps = params['rnn_steps']
        self.out_dim = params['action_dim']

    def call(self, inputs):
        ta_inp1 = tf.TensorArray('float32', size = 0, dynamic_size = True)
        ta_inp2 = tf.TensorArray('float32', size = 0, dynamic_size = True)
        out = tf.TensorArray('complex64', size = 0, dynamic_size=True)

        inp1 = swap_batch_timestep(inputs[0])
        inp2 = swap_batch_timestep(inputs[1])
        z = inputs[2]

        ta_inp1.unstack(inp1)
        ta_inp2.unstack(inp2)

        step = tf.constant(0)

        def cond(out, step, z):
            return tf.math.less(
                step,
                tf.constant(
                    self.steps,
                    dtype = tf.int32
                )
            )

        def body(out, step, z):
            inputs = [
                ta_inp1.read(step),
                ta_inp2.read(step),
                z
            ]

            o, z = self.layer(inputs)

            out = out.write(
                step,
                o
            )

            step = tf.math.add(step, tf.constant(1))
            return out, step, z

        out, step, z = tf.while_loop(cond, body, [out, step, z])

        out = out.stack()
        out = swap_batch_timestep(out)

        out = tf.ensure_shape(out, tf.TensorShape((None, self.steps, self.out_dim)), name='ensure_shape_critic_time_distributed_out')
        return [out, z]
