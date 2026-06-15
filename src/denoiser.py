"""Denoiser network for conditional basketball trajectory diffusion.

Architecture: Time embedding + ResBlocks + Self-Attention + Cross-Attention.
Predicts the noise added to a trajectory given conditioning (offensive play).
"""

import tensorflow as tf
from ops import SelfAttentionBlock, CrossAttentionBlock
from diffusion import sinusoidal_embedding


# ---------------------------------------------------------------------------
#   Simple residual block (no spectral norm — diffusion models don't need it)
# ---------------------------------------------------------------------------

class ResBlock1D(tf.keras.layers.Layer):
    """Conv1D residual block with LayerNorm + SiLU, no spectral norm."""

    def __init__(self, n_filters, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.conv1 = tf.keras.layers.Conv1D(
            self.n_filters, self.kernel_size, padding='same',
            kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.conv2 = tf.keras.layers.Conv1D(
            self.n_filters, self.kernel_size, padding='same',
            kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.proj = (tf.keras.layers.Conv1D(
            self.n_filters, 1, kernel_initializer=tf.keras.initializers.GlorotNormal())
                     if input_shape[-1] != self.n_filters else None)

    def call(self, x, training=None):
        shortcut = x
        if self.proj is not None:
            shortcut = self.proj(shortcut)

        h = self.norm1(x)
        h = tf.nn.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = tf.nn.silu(h)
        h = self.conv2(h)

        return h + shortcut


# ---------------------------------------------------------------------------
#   Denoiser Network
# ---------------------------------------------------------------------------

class DenoiserNet(tf.keras.Model):
    """Predicts noise for diffusion denoising, conditioned on offensive play.

    Input:
        xt:           noisy trajectory    [B, T, 16]  (defence + ball features)
        t:            diffusion timestep  [B]
        conditioning: offence sequence    [B, T, 18]  (offence + seq_feat)

    Output:
        predicted noise                  [B, T, 16]
    """

    def __init__(self, n_filters=256, n_resblock=4, num_heads=4,
                 T=1000, time_emb_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.n_resblock = n_resblock
        self.num_heads = num_heads
        self.T = T
        time_emb_dim = time_emb_dim or n_filters

        # Time embedding MLP
        self.time_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(time_emb_dim, activation='swish'),
            tf.keras.layers.Dense(time_emb_dim, activation='swish'),
        ], name='time_mlp')

        # Input projections
        self.input_proj = tf.keras.layers.Conv1D(
            n_filters, 1, kernel_initializer=tf.keras.initializers.GlorotNormal(),
            name='input_proj')
        self.cond_proj = tf.keras.layers.Conv1D(
            n_filters, 1, kernel_initializer=tf.keras.initializers.GlorotNormal(),
            name='cond_proj')

        # Residual blocks
        self.res_blocks = [
            ResBlock1D(n_filters, name=f'resblock_{i}')
            for i in range(n_resblock)
        ]

        # Attention (interleaved)
        self.attn_blocks = [
            SelfAttentionBlock(num_heads=num_heads, key_dim=n_filters // num_heads,
                               name=f'self_attn_{i}')
            for i in range(n_resblock // 2)
        ]
        self.cross_attn = CrossAttentionBlock(
            num_heads=num_heads, key_dim=n_filters // num_heads,
            name='cross_attn')

        # Output projection
        self.output_proj = tf.keras.layers.Conv1D(
            16, 1, dtype='float32',
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            name='output_proj')

    def call(self, xt, t, conditioning, mask=None, training=None):
        # --- Time embedding ---
        t_emb = sinusoidal_embedding(t, self.n_filters)
        t_emb = self.time_mlp(t_emb, training=training)       # [B, time_dim]
        t_emb = tf.expand_dims(t_emb, axis=1)                  # [B, 1, time_dim]

        # --- Project inputs ---
        x = self.input_proj(xt)                                # [B, T, n_filters]
        c = self.cond_proj(conditioning)                       # [B, T, n_filters]

        # Combine: noisy input + conditioning + time
        x = x + c + t_emb

        # --- ResBlocks + Attention ---
        attn_idx = 0
        for i, block in enumerate(self.res_blocks):
            x = block(x, training=training)
            # Self-attention after every other ResBlock
            if i % 2 == 0 and attn_idx < len(self.attn_blocks):
                x = self.attn_blocks[attn_idx](x, mask=mask, training=training)
                attn_idx += 1

        # Cross-attention: attend to conditioning
        x = self.cross_attn(x, c, mask=mask, training=training)

        # --- Output ---
        return self.output_proj(x)                             # [B, T, 16]
