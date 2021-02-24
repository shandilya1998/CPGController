import tensorflow as tf
from tensorflow.keras import activations

class ComplexInitializer(tf.keras.initializers.Initializer):
    def __init__(self, initializer, name = 'complex_initializer'):
        self.initializer = initializer
        self.name = name

    def __call__(self, shape, dtype=tf.dtypes.complex64):
        return tf.complex(self.initializer(shape), self.initializer(shape))

    def get_config(self):  # To support serialization
        return {
            "initializer": tf.keras.initializer.serialize(self.initializer),
            "name": self.name
        }

def relu(x):
    return tf.complex(tf.nn.relu(tf.math.real(x)), tf.nn.relu(tf.math.imag(x)))

class ComplexDense(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 init_criterion='he',
                 kernel_initializer='complex',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ComplexDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.init_criterion = init_criterion
        if kernel_initializer in {'complex'}:
            self.kernel_initializer = kernel_initializer
        else:
            self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        if seed is None:
            self.seed = np.random.randint(1, 10e6)
        else:
            self.seed = seed
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[-1] % 2 == 0
        input_dim = input_shape[ -1] // 2
        data_format = tf.keras.backend.image_data_format()
        kernel_shape = (input_dim, self.units)
        fan_in, fan_out = tf.keras.initializers._compute_fans(
            kernel_shape,
            data_format=data_format
        )
        if self.init_criterion == 'he':
            s = tf.math.sqrt(1. / fan_in)
        elif self.init_criterion == 'glorot':
            s = tf.math.sqrt(1. / (fan_in + fan_out))
        rng = RandomStreams(seed=self.seed)

        # Equivalent initialization using amplitude phase representation:
        """modulus = rng.rayleigh(scale=s, size=kernel_shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
        def init_w_real(shape, dtype=None):
            return modulus * tf.math.cos(phase)
        def init_w_imag(shape, dtype=None):
            return modulus * tf.math.sin(phase)"""

        # Initialization using euclidean representation:
        def init_w_real(shape, dtype=None):
            return rng.normal(
                size=kernel_shape,
                avg=0,
                std=s,
                dtype=dtype
            )
        def init_w_imag(shape, dtype=None):
            return rng.normal(
                size=kernel_shape,
                avg=0,
                std=s,
                dtype=dtype
            )
        if self.kernel_initializer in {'complex'}:
            real_init = init_w_real
            imag_init = init_w_imag
        else:
            real_init = self.kernel_initializer
            imag_init = self.kernel_initializer

        self.real_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=real_init,
            name='real_kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        self.imag_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=imag_init,
            name='imag_kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(2 * self.units,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=2, axes={-1: 2 * input_dim})
        self.built = True

    def call(self, inputs):
        input_shape = inputs.shape
        input_dim = input_shape[-1] // 2
        real_input = inputs[:, :input_dim]
        imag_input = inputs[:, input_dim:]

        cat_kernels_4_real = tf.concat(
            [self.real_kernel, -self.imag_kernel],
            axis=-1
        )
        cat_kernels_4_imag = tf.concat(
            [self.imag_kernel, self.real_kernel],
            axis=-1
        )
        cat_kernels_4_complex = tf.concat(
            [cat_kernels_4_real, cat_kernels_4_imag],
            axis=0
        )

        output = tf.keras.backend.dot(inputs, cat_kernels_4_complex)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = 2 * self.units
        return tuple(output_shape)

    def get_config(self):
        if self.kernel_initializer in {'complex'}:
            ki = self.kernel_initializer
        else:
            ki = initializers.serialize(self.kernel_initializer)
        config = {
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'init_criterion': self.init_criterion,
            'kernel_initializer': ki,
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
            'seed': self.seed,
        }
        base_config = super(ComplexDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ComplexLSTMCell(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        name = 'complex_lstmcell',
        activation = 'tanh',
        init_criterion='he',
        recurrent_activation = 'sigmoid',
        use_bias = True,
        kernel_initializer='glorot_uniform',
        recurrent_initializer = 'orthogonal',
        bias_initializer = 'zeros',
        kernel_regularizer = None,
        bias_regularizer = None,
        activity_regularizer = None,
        kernel_constraint = None,
        bias_constraint = None,
        recurrent_regularizer = None,
        recurrent_constraint = None,
        unit_forget_bias = True,
        trainable = True,
        **kwargs
    ):
        super(ComplexLSTMCell, self).__init__(
            trainable = trainable,
            name = name,
            **kwargs
        )
        self.init_criterion = init_criterion
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.init_criterion = init_criterion
        if kernel_initializer in {'complex'}:
            self.kernel_initializer = kernel_initializer
        else:
            self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        if seed is None:
            self.seed = np.random.randint(1, 10e6)
        else:
            self.seed = seed

        self.unit_forget_bias = unit_forget_bias
        self.trainable = trainable

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[-1] % 2 == 0
        input_dim = input_shape[-1] // 2
        data_format = tf.keras.backend.image_data_format()
        kernel_shape = (input_dim, self.units * 4)
        fan_in, fan_out = initializers._compute_fans(
            kernel_shape,
            data_format=data_format
        )
        if self.init_criterion == 'he':
            s = tf.math.sqrt(1. / fan_in)
        elif self.init_criterion == 'glorot':
            s = tf.math.sqrt(1. / (fan_in + fan_out))
        rng = RandomStreams(seed=self.seed)

        # Equivalent initialization using amplitude phase representation:
        """modulus = rng.rayleigh(scale=s, size=kernel_shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
        def init_w_real(shape, dtype=None):
            return modulus * tf.math.cos(phase)
        def init_w_imag(shape, dtype=None):
            return modulus * tf.math.sin(phase)"""

        # Initialization using euclidean representation:
        def init_w_real(shape, dtype=None):
            return rng.normal(
                size=kernel_shape,
                avg=0,
                std=s,
                dtype=dtype
            )
        def init_w_imag(shape, dtype=None):
            return rng.normal(
                size=kernel_shape,
                avg=0,
                std=s,
                dtype=dtype
            )
        if self.kernel_initializer in {'complex'}:
            real_init = init_w_real
            imag_init = init_w_imag
        else:
            real_init = self.kernel_initializer
            imag_init = self.kernel_initializer

        self.real_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=real_init,
            name='real_kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        self.imag_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=imag_init,
            name='imag_kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        if self.recurrent_initializer in {'complex'}:
            real_init = init_w_real
            imag_init = init_w_imag
        else:
            real_init = self.recurrent_initializer
            imag_init = self.recurrent_initializer

        self.real_recurrent_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=real_init,
            name='real_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint
        )
        self.imag_recurrent_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=imag_init,
            name='imag_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(2 * self.units * 4,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=2, axes={-1: 2 * input_dim})
        self.built = True


    def _compute_carry_and_output_fused(self, z, c_tm1):
        """Computes carry and output using fused kernels."""
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        return c, o

    def _dot_kernel(self, inputs):
        cat_kernels_4_real = tf.concat(
            [self.real_kernel, -self.imag_kernel],
            axis=-1
        )
        cat_kernels_4_imag = tf.concat(
            [self.imag_kernel, self.real_kernel],
            axis=-1
        )
        cat_kernels_4_complex = tf.concat(
            [cat_kernels_4_real, cat_kernels_4_imag],
            axis=0
        )

        return tf.keras.backend.dot(inputs, cat_kernels_4_complex)

    def _dot_recurrent_kernel(self, inputs):
        cat_kernels_4_real = tf.concat(
            [self.real_recurrent_kernel, -self.imag_recurrent_kernel],
            axis=-1
        )
        cat_kernels_4_imag = tf.concat(
            [self.imag_recurrent_kernel, self.real_recurrent_kernel],
            axis=-1
        )
        cat_kernels_4_complex = tf.concat(
            [cat_kernels_4_real, cat_kernels_4_imag],
            axis=0
        )

        output = tf.keras.backend.dot(inputs, cat_kernels_4_complex)

    def call(self, inputs, states):
        input_shape = inputs.shape
        input_dim = input_shape[-1] // 2

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        z = self._dot_kernel(inputs, self.real_kernel)
        z += self.dot_recurrent_kernel(h_tm1, self.real_recurrent_kernel)
        if self.use_bias:
            z = tf.nn.bias_add(z, self.bias)
        z = tf.split(z, num_or_size_splits=8, axis=1)
        real_z = z[:4]
        imag_z = z[4:]
        z = [tf.concat([real, imag], -1) for real, imag, in zip(real_z, imag_z)]
        c, o = self._compute_carry_and_output_fused(z, c_tm1)
        h = o * self.activation(c)
        return h, [h, c]
