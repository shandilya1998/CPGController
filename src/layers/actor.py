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
            activation = activation_combine
        )
        self.robot_state_dense = tf.keras.layers.Dense(
            units = units_robot_state,
            activation = activation_robot_state
        )
        self.motion_state_dense = tf.keras.layers.Dense(
            units = units_motion_state,
            activation = activation_motion_state
        )
        self.mu_dense = tf.keras.layers.Dense(
            units = units_osc,
            activation = activation_mu
        )
        self.omega_dense = tf.keras.layers.Dense(
            units = 1,
            activation = activation_omega
        )
        self.b_dense = tf.keras.layers.Dense(
            units = units_osc,
            activation = activation_b
        )

    def call(self, x1, x2, z):
        x1 = self.motion_state_dense(x1)
        x2 = self.robot_state_dense(x2)
        x = tf.concat([x1, x2], saxis = -1)
        x = self.combine_dense(x)
        omega = self.omega_dense(x)
        b = self.b_dense(x)
        mu = self.mu_dense(x)
        z = self.osc([z, omega, mu, b])
        out = self.output_mlp(z)
        return [out, z]
