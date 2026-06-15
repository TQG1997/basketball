"""BasketballGAN VAE-GAN model — Keras implementation for TF 2.16+.

Architecture:
    Encoder:         (conds, real_play) → z_mean, z_log_var
    Generator:       (seq, seq_feat, z) → fake_play [batch, seq_len, 28]
    Three Discriminators: O_disc, D_disc, P_disc (WGAN-GP, independent scopes)
"""

import os
import tensorflow as tf
import numpy as np
from ops import Conv1D_SN, ResidualBlock, SelfAttentionBlock, CrossAttentionBlock


# ---------------------------------------------------------------------------
#   Reparameterization
# ---------------------------------------------------------------------------

def reparameterize(z_mean, z_log_var):
    """Reparameterization trick: z = mu + sigma * epsilon."""
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ---------------------------------------------------------------------------
#   Encoder
# ---------------------------------------------------------------------------

class Encoder(tf.keras.Model):
    """Encodes (conds, real_play) into latent distribution parameters.

    Includes self-attention for global temporal + player-interaction modeling.
    """

    def __init__(self, latent_dims, n_filters, n_resblock,
                 num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.latent_dims = latent_dims
        self.n_filters = n_filters
        self.n_resblock = n_resblock

        self.conv_input = Conv1D_SN(n_filters, kernel_size=1, name='enc_conv_input')
        self.res_blocks = [
            ResidualBlock(n_filters, name=f'enc_Res{i}')
            for i in range(n_resblock)
        ]
        self.self_attn = SelfAttentionBlock(
            num_heads=num_heads, key_dim=n_filters // num_heads, name='enc_attn')
        self.dense_mean = tf.keras.layers.Dense(
            latent_dims, activation=None, dtype='float32',
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            bias_initializer='zeros', name='z_mean')
        self.dense_log_var = tf.keras.layers.Dense(
            latent_dims, activation=None, dtype='float32',
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            bias_initializer='zeros', name='z_log_var')

    def call(self, conds, target, mask=None, training=None):
        x = tf.concat([conds, target], axis=-1)
        x = self.conv_input(x)
        for block in self.res_blocks:
            x = block(x, training=training)
        x = self.self_attn(x, mask=mask, training=training)  # global temporal attention
        x = tf.reduce_mean(x, axis=1)                         # global temporal pooling
        return self.dense_mean(x), self.dense_log_var(x)


# ---------------------------------------------------------------------------
#   Generator / Decoder
# ---------------------------------------------------------------------------

class Generator(tf.keras.Model):
    """Decodes (seq, seq_feat, z) into a full play [B, T, 28].

    Includes self-attention for temporal coherence and cross-attention
    to let generated features attend to offensive conditioning.

    Output channels 0-21: player positions (ball xy + 10 players xy)
    Output channels 22-27: ball-status features (sigmoid)
    """

    def __init__(self, n_filters, n_resblock, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.n_resblock = n_resblock

        self.conds_dense = tf.keras.layers.Dense(
            n_filters, activation=None,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            bias_initializer='zeros', name='conds_linear')
        self.latent_dense = tf.keras.layers.Dense(
            n_filters, activation=None,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            bias_initializer='zeros', name='latents_linear')
        self.res_blocks = [
            ResidualBlock(n_filters, name=f'G_Res{i}')
            for i in range(n_resblock)
        ]
        self.self_attn = SelfAttentionBlock(
            num_heads=num_heads, key_dim=n_filters // num_heads, name='G_self_attn')
        self.cross_attn = CrossAttentionBlock(
            num_heads=num_heads, key_dim=n_filters // num_heads, name='G_cross_attn')
        self.output_conv = Conv1D_SN(28, kernel_size=1, dtype='float32', name='conv_result')

    def call(self, seq, seq_feat, z, mask=None, training=None):
        conds_full = tf.concat([seq, seq_feat], axis=-1)
        conds_linear = self.conds_dense(conds_full)            # [B, T, n_filters]

        latents_linear = self.latent_dense(z)
        latents_linear = tf.reshape(latents_linear, [-1, 1, self.n_filters])

        x = conds_linear + latents_linear                      # [B, T, n_filters]
        for block in self.res_blocks:
            x = block(x, training=training)
        x = self.self_attn(x, mask=mask, training=training)    # temporal self-attention
        x = self.cross_attn(x, conds_linear, mask=mask,        # attend to offence conditioning
                            training=training)

        nonlinear = tf.nn.leaky_relu(x)
        padded = tf.concat([nonlinear[:, 0:2], nonlinear, nonlinear[:, -2:]], axis=1)
        out = self.output_conv(padded)                         # [B, T, 28]

        seq_part = out[:, :, :22]                              # player positions
        feat_part = tf.math.sigmoid(out[:, :, 22:])            # ball-status probs
        return tf.concat([seq_part, feat_part], axis=-1)


# ---------------------------------------------------------------------------
#   Discriminator
# ---------------------------------------------------------------------------

class Discriminator(tf.keras.Model):
    """Critic: (conds, x) → (global_score, per_frame_score).

    Includes self-attention for global temporal consistency judgment.
    Architecture shared by all three discriminators (O_disc, D_disc, P_disc).
    """

    def __init__(self, n_filters, n_resblock, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.n_resblock = n_resblock

        self.conv_input = Conv1D_SN(n_filters, kernel_size=1, name='conv_input')
        self.res_blocks = [
            ResidualBlock(n_filters, name=f'disc_Res{i}')
            for i in range(n_resblock)
        ]
        self.self_attn = SelfAttentionBlock(
            num_heads=num_heads, key_dim=n_filters // num_heads, name='disc_attn')
        self.conv_output = Conv1D_SN(1, kernel_size=1, dtype='float32', name='conv_output')

    def call(self, conds, x, mask=None, training=None):
        inp = tf.concat([conds, x], axis=-1)
        h = self.conv_input(inp)
        for block in self.res_blocks:
            h = block(h, training=training)
        h = self.self_attn(h, mask=mask, training=training)    # temporal self-attention
        nonlinear = tf.nn.leaky_relu(h)
        score = self.conv_output(nonlinear)                    # [B, T, 1]
        global_score = tf.reduce_mean(score, axis=1)           # [B, 1]
        global_score = tf.reshape(global_score, [-1])          # [B]
        return global_score, score


# ---------------------------------------------------------------------------
#   VAE-GAN Orchestrator
# ---------------------------------------------------------------------------

class VAEGAN_Model:
    """Training orchestrator — holds sub-models, optimizers, and loss functions.

    Usage:
        model = VAEGAN_Model(config)
        model.train_D(real, real_d, seq, seq_feat, real_feat)
        model.train_G(real, real_d, seq, seq_feat, real_feat)
        fake = model.reconstruct(seq, seq_feat, z)
    """

    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        self.latent_dims = config.latent_dims
        self.n_filters = config.n_filters
        self.n_resblock = config.n_resblock
        self.lr_ = config.lr_
        self.beta = getattr(config, 'beta', 0.001)
        self.recon_weight = getattr(config, 'recon_weight', 1.0)
        self.features_ = config.features_
        self.features_d = config.features_d
        self.num_heads = getattr(config, 'num_heads', 4)

        # Sub-models
        self.encoder = Encoder(self.latent_dims, self.n_filters, self.n_resblock,
                               num_heads=self.num_heads, name='E_')
        self.generator = Generator(self.n_filters, self.n_resblock,
                                   num_heads=self.num_heads, name='G_')

        # Three discriminators (independent copies — different variable scopes)
        self.disc_O = Discriminator(self.n_filters, self.n_resblock,
                                    num_heads=self.num_heads, name='O_disc')
        self.disc_D = Discriminator(self.n_filters, self.n_resblock,
                                    num_heads=self.num_heads, name='D_disc')
        self.disc_P = Discriminator(self.n_filters, self.n_resblock,
                                    num_heads=self.num_heads, name='P_disc')

        # Optimizers
        self.gen_optimizer = tf.keras.optimizers.Adam(
            self.lr_, beta_1=0.5, beta_2=0.9, name='Adam_G')
        self.o_optimizer = tf.keras.optimizers.Adam(
            self.lr_, beta_1=0.5, beta_2=0.9, name='Adam_O')
        self.d_optimizer = tf.keras.optimizers.Adam(
            self.lr_, beta_1=0.5, beta_2=0.9, name='Adam_D')
        self.p_optimizer = tf.keras.optimizers.Adam(
            self.lr_, beta_1=0.5, beta_2=0.9, name='Adam_P')

        # Checkpoint
        self.checkpoint = tf.train.Checkpoint(
            encoder=self.encoder,
            generator=self.generator,
            disc_O=self.disc_O,
            disc_D=self.disc_D,
            disc_P=self.disc_P,
            gen_optimizer=self.gen_optimizer,
            o_optimizer=self.o_optimizer,
            d_optimizer=self.d_optimizer,
            p_optimizer=self.p_optimizer,
        )

        # Summary writers
        log_dir = os.path.join(config.folder_path, 'Log')
        self.G_summary_writer = tf.summary.create_file_writer(
            os.path.join(log_dir, 'G'))
        self.D_summary_writer = tf.summary.create_file_writer(
            os.path.join(log_dir, 'D'))
        self.D_valid_summary_writer = tf.summary.create_file_writer(
            os.path.join(log_dir, 'D_valid'))

        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

        # Warm models to build variables
        self._build()

    def _build(self):
        """Run a dummy forward pass to create all variables."""
        B, T = self.batch_size, self.seq_length
        dummy_conds = tf.zeros([B, T, self.features_ + 6])   # seq + seq_feat
        dummy_play = tf.zeros([B, T, self.features_ + self.features_d + 6])
        dummy_seq = tf.zeros([B, T, self.features_])
        dummy_feat = tf.zeros([B, T, 6])
        dummy_z = tf.zeros([B, self.latent_dims])

        z_mean, z_log_var = self.encoder(dummy_conds, dummy_play)
        _ = self.generator(dummy_seq, dummy_feat, dummy_z)
        # O_disc: conds=defence, x=offence+feat
        _ = self.disc_O(dummy_play[:, :, self.features_:self.features_ + self.features_d],
                        tf.concat([dummy_play[:, :, :self.features_], dummy_play[:, :, -6:]], axis=-1))
        _ = self.disc_D(dummy_seq, dummy_play[:, :, self.features_:self.features_ + self.features_d])
        _ = self.disc_P(dummy_conds, dummy_play)

    # -----------------------------------------------------------------------
    #   WGAN-GP gradient penalty
    # -----------------------------------------------------------------------

    def _gradient_penalty(self, disc_fn, conds, real_sample, fake_sample):
        """WGAN-GP gradient penalty (λ=10)."""
        epsilon = tf.random.uniform([tf.shape(real_sample)[0], 1, 1], 0.0, 1.0)
        x_inter = epsilon * real_sample + (1.0 - epsilon) * fake_sample

        with tf.GradientTape() as tape:
            tape.watch(x_inter)
            _, score = disc_fn(conds, x_inter, training=True)

        grad = tape.gradient(score, x_inter)
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2]))
        return 10.0 * tf.reduce_mean(tf.square(grad_norm - 1.0))

    # -----------------------------------------------------------------------
    #   Domain-specific penalty losses
    # -----------------------------------------------------------------------
    #   (Identical math to original — seq_feat / real_feat passed as argument)

    def _pass_ball_penalty(self, fake, seq_feat):
        ball_pos = fake[:, :, 0:2]
        ball_status = seq_feat
        ballpass_frames = tf.equal(tf.reduce_sum(ball_status, axis=-1), 0)[:, 1:-1]
        vel_1 = ball_pos[:, 1:-1] - ball_pos[:, 0:-2]
        vel_2 = ball_pos[:, 2:] - ball_pos[:, 1:-1]
        dot_p = vel_1[:, :, 0] * vel_2[:, :, 0] + vel_1[:, :, 1] * vel_2[:, :, 1]
        vel_1_norm = tf.sqrt(vel_1[:, :, 0] ** 2 + vel_1[:, :, 1] ** 2 + 1e-10)
        vel_2_norm = tf.sqrt(vel_2[:, :, 0] ** 2 + vel_2[:, :, 1] ** 2 + 1e-10)
        v = dot_p / (vel_1_norm * vel_2_norm)
        clip = tf.clip_by_value(v, -1.0 + 1e-5, 1.0 - 1e-5)
        theta = tf.math.acos(clip)
        pass_theta = tf.cast(ballpass_frames, tf.float32) * theta
        frames = tf.cast(tf.math.count_nonzero(ballpass_frames), tf.float32)
        return tf.div_no_nan(tf.reduce_sum(pass_theta), frames)

    def _dribbler_score(self, inputs, seq_feat, basket_right_x, basket_right_y):
        B = tf.shape(inputs)[0]
        T = tf.shape(inputs)[1]
        basket_right_x_t = tf.fill([B, T, 1, 1], basket_right_x)
        basket_right_y_t = tf.fill([B, T, 1, 1], basket_right_y)
        basket_pos = tf.concat([basket_right_x_t, basket_right_y_t], axis=-1)

        ball_pos = tf.reshape(inputs[:, :, :2], [B, T, 1, 2])
        teamB_pos = tf.reshape(inputs[:, :, 2:12], [B, T, 5, 2])
        teamB_pos = tf.concat([teamB_pos, basket_pos], axis=2)

        vec_ball = ball_pos - teamB_pos
        dist_ = tf.norm(vec_ball, ord='euclidean', axis=-1)
        dist_f = tf.multiply(dist_, tf.round(seq_feat))
        dribbler_scMin = tf.reduce_max(dist_f, axis=-1)
        return tf.reduce_mean(dribbler_scMin)

    def _dribbler_penalty(self, fake, real, seq_feat, basket_right):
        fake_score = self._dribbler_score(
            fake, seq_feat, basket_right[0], basket_right[1])
        real_score = self._dribbler_score(
            real, seq_feat, basket_right[0], basket_right[1])
        return tf.abs(real_score - fake_score)

    def _acc_penalty(self, real, fake):
        def _acc(data):
            speed = data[:, 1:, 2:22] - data[:, :-1, 2:22]
            acc = speed[:, 1:] - speed[:, :-1]
            B = tf.shape(data)[0]
            T = tf.shape(data)[1]
            acc = tf.reshape(acc, [B, T - 2, 10, 2])
            dist_ = tf.norm(acc, ord='euclidean', axis=-1)
            return tf.reduce_mean(dist_)
        return tf.abs(_acc(real) - _acc(fake))

    def _open_shot_score(self, offence_, defence_, seq_feat, basket_right):
        B = tf.shape(offence_)[0]
        T = tf.shape(offence_)[1]
        ball_pos = tf.reshape(offence_[:, :, :2], [B, T, 1, 2])
        teamB_pos = tf.reshape(defence_, [B, T, 5, 2])
        basket_right_x_t = tf.fill([B, T, 1, 1], basket_right[0])
        basket_right_y_t = tf.fill([B, T, 1, 1], basket_right[1])
        basket_pos = tf.concat([basket_right_x_t, basket_right_y_t], axis=-1)

        vec_ball_2_team = ball_pos - teamB_pos
        vec_ball_2_basket = ball_pos - basket_pos
        b2t_dot_b2b = tf.matmul(vec_ball_2_team, vec_ball_2_basket, transpose_b=True)
        b2t_dot_b2b = tf.reshape(b2t_dot_b2b, [B, T, 5])

        dist_teamB = tf.norm(vec_ball_2_team, ord='euclidean', axis=-1)
        dist_basket = tf.norm(vec_ball_2_basket, ord='euclidean', axis=-1)

        theta = tf.acos(b2t_dot_b2b / (dist_teamB * dist_basket + 1e-3))
        open_score = (theta + 1.0) * (dist_teamB + 1.0)
        open_score_min = tf.reduce_min(open_score, axis=-1)

        dribble_frames = tf.equal(tf.reduce_sum(seq_feat[:, :, :5], axis=-1), 1)
        frames = tf.cast(tf.math.count_nonzero(dribble_frames), tf.float32)
        return tf.div_no_nan(
            tf.reduce_sum(tf.multiply(
                open_score_min, tf.cast(dribble_frames, tf.float32))), frames)

    def _open_shot_penalty(self, real, fake, seq_feat, basket_right):
        real_o = tf.reshape(real[:, :, :12], [-1, self.seq_length, 12])
        real_d = tf.reshape(real[:, :, 12:22], [-1, self.seq_length, 10])
        fake_o = tf.reshape(fake[:, :, :12], [-1, self.seq_length, 12])
        fake_d = tf.reshape(fake[:, :, 12:22], [-1, self.seq_length, 10])
        return tf.abs(
            self._open_shot_score(real_o, real_d, seq_feat, basket_right) -
            self._open_shot_score(fake_o, fake_d, seq_feat, basket_right))

    # -----------------------------------------------------------------------
    #   Training steps
    # -----------------------------------------------------------------------

    def train_D(self, real, real_d, seq, seq_feat, real_feat, df_basket_right):
        """One D-update step — updates all three discriminators."""
        B = tf.shape(real)[0]
        real_play = tf.concat([real, real_d, real_feat], axis=-1)
        conds = tf.concat([seq, seq_feat], axis=-1)

        # ---- Forward through encoder + generator ----
        z_mean, z_log_var = self.encoder(conds, real_play, training=False)
        z_enc = reparameterize(z_mean, z_log_var)
        fake_play = self.generator(seq, seq_feat, z_enc, training=False)

        fake_offence = tf.concat(
            [fake_play[:, :, :12], fake_play[:, :, 22:]], axis=-1)
        fake_defence = fake_play[:, :, 12:22]
        real_offence_full = tf.concat([real, real_feat], axis=-1)

        # ---- O_disc: offence discriminator (conditioned on defence) ----
        with tf.GradientTape() as tape_o:
            real_o, _ = self.disc_O(real_d, real_offence_full, training=True)
            fake_o, _ = self.disc_O(real_d, fake_offence, training=True)
            gp_o = self._gradient_penalty(
                lambda c, x: self.disc_O(c, x, training=True),
                real_d, real_offence_full, fake_offence)
            d_o_cost = tf.reduce_mean(fake_o) - tf.reduce_mean(real_o) + gp_o
        grads_o = tape_o.gradient(d_o_cost, self.disc_O.trainable_variables)
        self.o_optimizer.apply_gradients(zip(grads_o, self.disc_O.trainable_variables))

        # ---- D_disc: defence discriminator (conditioned on offence) ----
        with tf.GradientTape() as tape_d:
            real_d_out, _ = self.disc_D(real_offence_full, real_d, training=True)
            fake_d_out, _ = self.disc_D(real_offence_full, fake_defence, training=True)
            gp_d = self._gradient_penalty(
                lambda c, x: self.disc_D(c, x, training=True),
                real_offence_full, real_d, fake_defence)
            d_d_cost = tf.reduce_mean(fake_d_out) - tf.reduce_mean(real_d_out) + gp_d
        grads_d = tape_d.gradient(d_d_cost, self.disc_D.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads_d, self.disc_D.trainable_variables))

        # ---- P_disc: full-play discriminator ----
        with tf.GradientTape() as tape_p:
            real_p, _ = self.disc_P(conds, real_play, training=True)
            fake_p, _ = self.disc_P(conds, fake_play, training=True)
            gp_p = self._gradient_penalty(
                lambda c, x: self.disc_P(c, x, training=True),
                conds, real_play, fake_play)
            d_p_cost = tf.reduce_mean(fake_p) - tf.reduce_mean(real_p) + gp_p
        grads_p = tape_p.gradient(d_p_cost, self.disc_P.trainable_variables)
        self.p_optimizer.apply_gradients(zip(grads_p, self.disc_P.trainable_variables))

        # ---- Summaries ----
        em_dist = (tf.reduce_mean(real_o) - tf.reduce_mean(fake_o) +
                   tf.reduce_mean(real_d_out) - tf.reduce_mean(fake_d_out) +
                   tf.reduce_mean(real_p) - tf.reduce_mean(fake_p)) / 3.0
        d_cost_all = (d_o_cost + d_d_cost + d_p_cost) / 3.0
        gp_all = (gp_o + gp_d + gp_p) / 3.0

        with self.D_summary_writer.as_default():
            tf.summary.scalar('loss_D_ALL', d_cost_all, step=self.global_step)
            tf.summary.scalar('grad_penalty_ALL', gp_all, step=self.global_step)
            tf.summary.scalar('EM_Dist_ALL', em_dist, step=self.global_step)

        return {'d_cost': d_cost_all, 'grad_pen': gp_all, 'em_dist': em_dist}

    def train_G(self, real, real_d, seq, seq_feat, real_feat, df_basket_right,
                beta=None):
        """One G-update step — updates generator + encoder jointly.

        Args:
            beta: KL weight (if None, uses self.beta). Allows annealing schedule.
        """
        B = tf.shape(real)[0]
        _beta = self.beta if beta is None else beta
        real_play = tf.concat([real, real_d, real_feat], axis=-1)
        conds = tf.concat([seq, seq_feat], axis=-1)

        gen_vars = self.generator.trainable_variables + self.encoder.trainable_variables

        with tf.GradientTape() as tape:
            # ---- Forward ----
            z_mean, z_log_var = self.encoder(conds, real_play, training=True)
            z_enc = reparameterize(z_mean, z_log_var)
            fake_play = self.generator(seq, seq_feat, z_enc, training=True)

            fake_offence = tf.concat(
                [fake_play[:, :, :12], fake_play[:, :, 22:]], axis=-1)
            fake_defence = fake_play[:, :, 12:22]
            real_offence_full = tf.concat([real, real_feat], axis=-1)

            # ---- Adversarial losses ----
            fake_o, _ = self.disc_O(real_d, fake_offence, training=False)
            fake_d_out, _ = self.disc_D(real_offence_full, fake_defence, training=False)
            fake_p, _ = self.disc_P(conds, fake_play, training=False)

            g_o_cost = -tf.reduce_mean(fake_o)
            g_d_cost = -tf.reduce_mean(fake_d_out)
            g_p_cost = -tf.reduce_mean(fake_p)
            g_mean_cost = (g_o_cost + g_d_cost + g_p_cost) / 3.0
            scale = tf.stop_gradient(tf.abs(g_mean_cost))

            # ---- Domain penalties ----
            penalty = self._dribbler_penalty(
                fake_play, real_play, seq_feat, df_basket_right)
            open_pen = self._open_shot_penalty(
                fake_play, real_play, seq_feat, df_basket_right)
            pass_pen = self._pass_ball_penalty(fake_play, seq_feat)
            acc_pen = self._acc_penalty(real_play, fake_play)

            # ---- VAE losses ----
            recon_loss = tf.reduce_mean(tf.abs(real_play - fake_play))
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=-1))

            # ---- Combined generator loss ----
            gen_cost = (g_mean_cost
                        + scale * penalty
                        + scale * open_pen
                        + scale * pass_pen
                        + scale * acc_pen
                        + self.recon_weight * recon_loss
                        + _beta * kl_loss)

        grads = tape.gradient(gen_cost, gen_vars)
        self.gen_optimizer.apply_gradients(zip(grads, gen_vars))

        self.global_step.assign_add(1)

        # ---- Summaries ----
        with self.G_summary_writer.as_default():
            tf.summary.scalar('loss_G_ALL', gen_cost, step=self.global_step)
            tf.summary.scalar('loss_G_mean', g_mean_cost, step=self.global_step)
            tf.summary.scalar('recon_loss', recon_loss, step=self.global_step)
            tf.summary.scalar('kl_loss', kl_loss, step=self.global_step)
            tf.summary.scalar('dribble_penalty', penalty, step=self.global_step)
            tf.summary.scalar('open_penalty', open_pen, step=self.global_step)
            tf.summary.scalar('pass_penalty', pass_pen, step=self.global_step)
            tf.summary.scalar('acc_penalty', acc_pen, step=self.global_step)

        return {
            'gen_cost': gen_cost, 'g_mean': g_mean_cost,
            'recon': recon_loss, 'kl': kl_loss,
            'penalty': penalty, 'open_pen': open_pen,
            'pass_pen': pass_pen, 'acc_pen': acc_pen
        }

    def valid_loss(self, real, real_d, seq, seq_feat, real_feat, df_basket_right):
        """Run validation — compute losses without updating weights."""
        real_play = tf.concat([real, real_d, real_feat], axis=-1)
        conds = tf.concat([seq, seq_feat], axis=-1)

        z_mean, z_log_var = self.encoder(conds, real_play, training=False)
        z_enc = reparameterize(z_mean, z_log_var)
        fake_play = self.generator(seq, seq_feat, z_enc, training=False)

        recon_loss = tf.reduce_mean(tf.abs(real_play - fake_play))
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1))

        with self.D_valid_summary_writer.as_default():
            tf.summary.scalar('valid_recon_loss', recon_loss, step=self.global_step)
            tf.summary.scalar('valid_kl_loss', kl_loss, step=self.global_step)

        return {'recon': recon_loss, 'kl': kl_loss}

    # -----------------------------------------------------------------------
    #   Inference
    # -----------------------------------------------------------------------

    def reconstruct(self, seq, seq_feat, z):
        """Generate play from random noise z (inference, no encoder)."""
        return self.generator(seq, seq_feat, z, training=False)

    def encode_latent(self, real, real_d, seq, seq_feat, real_feat):
        """Encode a real play into its latent code."""
        real_play = tf.concat([real, real_d, real_feat], axis=-1)
        conds = tf.concat([seq, seq_feat], axis=-1)
        z_mean, z_log_var = self.encoder(conds, real_play, training=False)
        return reparameterize(z_mean, z_log_var)

    # -----------------------------------------------------------------------
    #   Save / Load
    # -----------------------------------------------------------------------

    def save_model(self, checkpoint_path):
        """Save all model weights and optimizer states."""
        self.checkpoint.write(checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')

    def load_model(self, checkpoint_path):
        """Restore all model weights and optimizer states."""
        status = self.checkpoint.read(checkpoint_path)
        status.expect_partial()  # Allow partial restore
        print(f'Restored checkpoint from {checkpoint_path}')
