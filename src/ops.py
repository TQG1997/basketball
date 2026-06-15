"""Custom Keras layers: Conv1D with spectral normalization + residual blocks."""

import tensorflow as tf


class Conv1D_SN(tf.keras.layers.Layer):
    """1D convolution with spectral normalization (single power iteration).

    Matches the original spectral_norm + conv1d_sn ops exactly.
    """

    def __init__(self, filters, kernel_size=5, stride=1, padding='VALID',
                 power_iterations=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding.upper()
        self.power_iterations = power_iterations

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.kernel_size, input_shape[-1], self.filters),
            initializer=tf.keras.initializers.GlorotNormal())
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer='zeros')
        self.u = self.add_weight(
            name='u',
            shape=(1, self.filters),
            initializer='random_normal',
            trainable=False)

    def call(self, inputs):
        w_shape = tf.shape(self.kernel)
        w = tf.reshape(self.kernel, [-1, w_shape[-1]])

        u_hat = self.u
        for _ in range(self.power_iterations):
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_)
            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        w_norm = tf.reshape(w / sigma, w_shape)

        self.u.assign(u_hat)

        outputs = tf.nn.conv1d(
            inputs, w_norm,
            stride=self.stride, padding=self.padding)
        outputs = tf.nn.bias_add(outputs, self.bias)
        return outputs


class ResidualBlock(tf.keras.layers.Layer):
    """Residual block with manual edge-padding and leaky ReLU.

    Architecture: 2x (LeakyReLU → pad edges → Conv1D_SN) + skip connection.
    """

    def __init__(self, n_filters, n_layers=2, residual_alpha=1.0,
                 leaky_relu_alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.residual_alpha = residual_alpha
        self.leaky_relu_alpha = leaky_relu_alpha

    def build(self, input_shape):
        self.convs = [
            Conv1D_SN(filters=self.n_filters, name=f'conv{i}')
            for i in range(self.n_layers)
        ]

    def call(self, inputs):
        next_input = inputs
        for conv in self.convs:
            nonlinear = tf.nn.leaky_relu(next_input, alpha=self.leaky_relu_alpha)
            # Manual "same" padding: prepend/append first/last 2 frames
            padded = tf.concat([
                nonlinear[:, 0:2],
                nonlinear,
                nonlinear[:, -2:]
            ], axis=1)
            next_input = conv(padded)
        return next_input * self.residual_alpha + inputs
