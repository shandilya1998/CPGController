import tensorflow as tf


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
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        dtype = 'complex64',
        **kwargs
    ):
        super(
            ComplexDense, 
            self
        ).__init__(
            name = name,
            activity_regularizer=activity_regularizer, 
            **kwargs
        )
        self.name = name
        self.dtype = dtype
        self.units = int(units) if not isinstance(units, int) else units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
        self.supports_masking = True

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
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True
        )
        if self.use_bias:
            self.bias = self.add_weight(
                self.name + '_bias',
                shape=[self.units,],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        if self.use_bias:
            return tf.add(tf.matmul(self.kernel, inputs), self.bias) 
        else: 
            return tf.matmul(self.kernel, inputs)

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
                tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                tf.keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                tf.keras.constraints.serialize(self.bias_constraint)
        })
        return config

