import tensorflow as tf
from layers.oscillator import HopfOscillator
from layers.complex import ComplexDense, relu

class StateEncoder(tf.keras.Model):
    def __init__(
        self,
        units_osc,
        units_combine,
        units_robot_state,
        units_motion_state,
        activation_output_mlp = 'relu',
        activation_combine = 'relu',
        activation_robot_state = 'relu',
        activation_motion_state = 'relu',
        activation_mu = 'relu',
        activation_omega = 'relu',
        activation_b = 'relu',
    ):
        super(StateEncoder, self).__init__()
        self.combine_dense = tf.keras.layers.Dense(
            units = units_combine,
            activation = activation_combine,
            name = 'combine_dense'
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
        self.mu_dense = tf.keras.layers.Dense(
            units = units_osc,
            activation = activation_mu,
            name = 'mu_dense'
        )
        self.omega_dense = tf.keras.layers.Dense(
            units = 1,
            activation = activation_omega,
            name = 'omega_dense'
        )
        self.b_dense = tf.keras.layers.Dense(
            units = units_osc,
            activation = activation_b,
            name = 'b_dense'
        )

    def call(self, S):
        motion_state, robot_state, _ = S
        motion_state = self.motion_state_dense(motion_state)
        robot_state = self.robot_state_dense(robot_state)
        state = tf.concat([motion_state, robot_state], axis = -1)
        state = self.combine_dense(state)
        omega = tf.math.abs(self.omega_dense(state))
        b = tf.math.abs(self.b_dense(state))
        mu = tf.math.abs(self.mu_dense(state))
        return [omega, mu, b]


def swap_batch_timestep(input_t):
    # Swap the batch and timestep dim for the incoming tensor.
    axes = list(range(len(input_t.shape)))
    axes[0], axes[1] = 1, 0
    return tf.transpose(input_t, axes)

class Actor(tf.keras.Model):
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
        activation_output_mlp = 'relu',
        activation_combine = 'relu',
        activation_robot_state = 'relu',
        activation_motion_state = 'relu',
        activation_mu = 'relu',
        activation_omega = 'relu',
        activation_b = 'relu',
        name = 'TimeDistributedActor'
    ):
        super(Actor, self).__init__(name = name)
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
                'Expected units_output_mlp to be of type `list`, got typr `{t}`'.format(
                    type(t = units_output_mlp)
                )
            )

        self.osc = HopfOscillator(
            units = units_osc,
            dt = dt
        )

        self.encoder = StateEncoder(
            units_osc,
            units_combine,
            units_robot_state,
            units_motion_state,
        )


    def call(self, S):
        omega, mu, b = self.encoder(S)
        _, _, z = S

        out = tf.TensorArray(tf.dtypes.float32, size = 0, dynamic_size=True)

        step = tf.constant(0)
        z_out = self.osc([z, omega, mu, b])
        o = self.output_mlp(z_out)
        out = out.write(
            step,
            o
        )
        step = tf.math.add(step, tf.constant(1))

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
                z,
                omega,
                mu,
                b
            ]

            z = self.osc(inputs)
            o = self.output_mlp(z)

            out = out.write(
                step,
                o
            )

            step = tf.math.add(step, tf.constant(1))
            return out, step, z

        out, step, _ = tf.while_loop(cond, body, [out, step, z_out])

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
        return [out, z_out], [omega, mu, b]

def get_actor(params):
    cell = Actor(
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
