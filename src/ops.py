"""Custom Keras layers: Conv1D with spectral normalization + residual blocks + attention."""

import tensorflow as tf


# ---------------------------------------------------------------------------
#   Attention blocks
# ---------------------------------------------------------------------------


class SelfAttentionBlock(tf.keras.layers.Layer):
    """Multi-head self-attention with residual connection + layer norm.

    Adds global temporal reasoning on top of the local conv features.
    """

    def __init__(self, num_heads=4, key_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attn = None  # built in build()
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        kd = self.key_dim or input_shape[-1] // self.num_heads
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=kd,
            dropout=0.0, name='mha')

    def call(self, x, mask=None, training=None):
        attn_out = self.attn(query=x, key=x, value=x, attention_mask=mask)
        return self.layernorm(x + attn_out)


class CrossAttentionBlock(tf.keras.layers.Layer):
    """Cross-attention: query attends to key/value with residual + layer norm."""

    def __init__(self, num_heads=4, key_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attn = None
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        # input_shape is ignored; we handle two inputs in call()
        kd = self.key_dim or 64
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=kd,
            dropout=0.0, name='cross_mha')

    def call(self, query, key_value, mask=None, training=None):
        attn_out = self.attn(query=query, key=key_value, value=key_value,
                             attention_mask=mask)
        return self.layernorm(query + attn_out)


# ---------------------------------------------------------------------------
#   Spectral norm conv + residual blocks
# ---------------------------------------------------------------------------


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
