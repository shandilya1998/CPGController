import tensorflow as tf

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

class ComplexDense(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        name = 'complex_dense',
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        dtype = 'complex64',
        trainable = True,
        **kwargs
    ):
        super(
            ComplexDense,
            self
        ).__init__(
            name = name,
            dtype = dtype,
            **kwargs
        )

        self.units = int(units) if not isinstance(units, int) else units

        if isinstance(activation, str):
            self.activation = activations.get(activation)
        else:
            self.activation = activation

        self.use_bias = use_bias

        if isinstance(kernel_initializer, str):
            self.kernel_initializer = ComplexInitializer(
                tf.keras.initializers.get(kernel_initializer),
                kernel_initializer
            )
        else:
            self.kernel_initializer = ComplexInitializer(
                kernel_initializer
            )

        if isinstance(bias_initializer, str):
            self.bias_initializer = ComplexInitializer(
                tf.keras.initializers.get(kernel_initializer),
                bias_initializer
            )
        else:
            self.bias_initializer = ComplexInitializer(
                bias_initializer
            )

        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
        self.supports_masking = True
        self.trainable = trainable

    def build(self, input_shape):
        dtype = tf.dtypes.as_dtype(self.dtype)
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                'should be defined. Found `None`.')
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            self.name + '_kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable = self.trainable
        )
        if self.use_bias:
            self.bias = self.add_weight(
                self.name + '_bias',
                shape=[self.units,],
                initializer=self.bias_initializer,
                dtype=self.dtype,
                trainable = self.trainable
            )
        else:
            self.bias = tf.zeros(shape = [self.units,], dtype = self.dtype)
        self.built = True

    def call(self, inputs):
        return tf.add(tf.matmul(inputs, self.kernel), self.bias) 

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = super(Dense, self).get_config()
        config.update({
            'units':
                self.units,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                tf.keras.initializers.serialize(self.bias_initializer)
        })
        return config

class ComplexLSTMCell(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        name = 'complex_dense',
        activation = 'tanh',
        recurrent_activation = 'sigmoid',
        use_bias = True,
        kernel_initializer='glorot_uniform',
        recurrent_initializer = 'orthogonal',
        bias_initializer = 'zeros',
        unit_forget_bias = True,
        dtype = 'complex64',
        trainable = True,
        **kwargs
    ):
        self.units = units
        self.activation = activations.get(activation),
        self.recurrent_activation = activations.get(recurrent_activation),
        self.use_bias = use_bias

        self.kernel_initializer = ComplexInitializer(
            initializers.get(kernel_initializer),
            'complex_kernel_initializer'
        )
        self.recurrent_initializer = ComplexInitializer(
            initializers.get(recurrent_initializer),
            'complex_recurrent_initializer'
        )
        self.bias_initializer = ComplexInitializer(
            initializers.get(bias_initializer),
            'complex_bias_initializer'
        )
        self.unit_forget_bias = unit_forget_bias
        self.trainable = trainable

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer,
            dtype = self.dtype,
            trainable = self.trainable
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            dtype = self.dtype,
            trainable = self.trainable
        )
        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.get('ones')((self.units,),*args, **kwargs),
                        self.bias_initializer((self.units * 2,),*args,**kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name='bias',
                initializer=bias_initializer,
                dtype = self.dtype,
                trainable = self.trainable
            )
        else:
            self.bias = None

    def _compute_carry_and_output_fused(self, z, c_tm1):
        """Computes carry and output using fused kernels."""
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        return c, o

    def call(self, inputs, states):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        z = K.dot(inputs, self.kernel)
        z += K.dot(h_tm1, self.recurrent_kernel)
        if self.use_bias:
            z = K.bias_add(z, self.bias)

        z = tf.split(z, num_or_size_splits=4, axis=1)
        c, o = self._compute_carry_and_output_fused(z, c_tm1)
        h = o * self.activation(c)
        return h, [h, c]
