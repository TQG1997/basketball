"""DDPM / DDIM diffusion for conditional basketball trajectory generation.

Diffuses: defence positions (10 dim) + ball status features (6 dim) = 16 dims
Conditioning: offence trajectory (12 dim) + ball-possession features (6 dim) = 18 dims
"""

import tensorflow as tf
import numpy as np


# ---------------------------------------------------------------------------
#   Timestep embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t, dim, max_period=10000):
    """Transformer-style sinusoidal timestep encoding."""
    half = dim // 2
    freqs = tf.exp(
        -tf.math.log(tf.cast(max_period, tf.float32))
        * tf.range(half, dtype=tf.float32) / half)
    args = tf.cast(t, tf.float32)[:, None] * freqs[None, :]
    return tf.concat([tf.sin(args), tf.cos(args)], axis=-1)


# ---------------------------------------------------------------------------
#   Diffusion process
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    """DDPM forward process + DDIM reverse sampling.

    Args:
        T: total diffusion timesteps (default 1000)
        beta_start, beta_end: linear noise schedule endpoints
    """

    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        betas = tf.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = tf.math.cumprod(alphas, axis=0)

        self.betas = tf.cast(betas, tf.float32)
        self.alphas = tf.cast(alphas, tf.float32)
        self.alpha_bars = tf.cast(alpha_bars, tf.float32)

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise to clean data at timestep t.

        Returns (noisy_sample, applied_noise).
        """
        if noise is None:
            noise = tf.random.normal(tf.shape(x0))
        alpha_bar = tf.gather(self.alpha_bars, t)
        alpha_bar = tf.reshape(alpha_bar, [-1, 1, 1])
        xt = tf.sqrt(alpha_bar) * x0 + tf.sqrt(1.0 - alpha_bar) * noise
        return xt, noise

    def ddim_step(self, denoiser, xt, t, conditioning, mask=None):
        """Single DDIM reverse step (deterministic, eta=0)."""
        B = tf.shape(xt)[0]
        pred_noise = denoiser(xt, t, conditioning, mask=mask, training=False)

        alpha_bar_t = tf.gather(self.alpha_bars, t)
        alpha_bar_t = tf.reshape(alpha_bar_t, [-1, 1, 1])

        # Predict x0
        pred_x0 = (xt - tf.sqrt(1.0 - alpha_bar_t) * pred_noise) / tf.sqrt(alpha_bar_t)

        if tf.shape(t)[0] > 0 and t[0] > 0:
            alpha_bar_prev = tf.gather(self.alpha_bars, t - 1)
            alpha_bar_prev = tf.reshape(alpha_bar_prev, [-1, 1, 1])
            direction = tf.sqrt(1.0 - alpha_bar_prev) * pred_noise
            xt_prev = tf.sqrt(alpha_bar_prev) * pred_x0 + direction
        else:
            xt_prev = pred_x0

        return xt_prev

    @tf.function
    def sample(self, denoiser, conditioning, shape, steps=50, mask=None):
        """Generate a sample via DDIM with `steps` denoising steps."""
        B, T, D = shape
        xt = tf.random.normal([B, T, D])

        stride = max(1, self.T // steps)
        timesteps = tf.range(self.T - 1, -1, -stride, dtype=tf.int32)

        for t_val in timesteps:
            t_batch = tf.fill([B], t_val)
            xt = self.ddim_step(denoiser, xt, t_batch, conditioning, mask=mask)

        return xt
