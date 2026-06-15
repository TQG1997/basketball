"""BasketballGAN training entry — Diffusion model (TF 2.16+ / Keras 3 / Hydra / WandB).

Usage:
    python Train_Triple.py                                        # default config
    python Train_Triple.py model.n_filters=128 training.batch_size=32
    python Train_Triple.py training.max_epochs=500 wandb.enabled=true

Model types:
    training.model_type=diffusion   (default) — DDPM + DDIM sampling
    training.model_type=vaegan      — VAE-GAN with 3 discriminators
"""

import os
import shutil
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')

from utils import DataFactory
from diffusion import GaussianDiffusion
from denoiser import DenoiserNet
import game_visualizer


# ---------------------------------------------------------------------------
#   Diffusion Trainer
# ---------------------------------------------------------------------------

class DiffusionTrainer:
    """Simple diffusion training: forward process → predict noise → MSE loss."""

    def __init__(self, data_factory, cfg, checkpoint_path, sample_path):
        self.data_factory = data_factory
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.sample_path = sample_path

        m = cfg.model
        t = cfg.training
        d = cfg.diffusion

        self.bs = t.batch_size
        self.seq_len = t.seq_length

        # Diffusion process
        self.diffusion = GaussianDiffusion(
            T=d.T, beta_start=d.beta_start, beta_end=d.beta_end)

        # Denoiser network
        self.denoiser = DenoiserNet(
            n_filters=m.n_filters,
            n_resblock=m.n_resblock,
            num_heads=m.num_heads,
            T=d.T)

        # Optimizer (AdamW-style — standard for diffusion)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=t.lr_, beta_1=0.9, beta_2=0.999)

        # EMA for better sampling quality
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        self.ema_initialized = False

        # Checkpoint
        self.checkpoint = tf.train.Checkpoint(
            denoiser=self.denoiser,
            optimizer=self.optimizer,
        )

        self.num_data = data_factory.train_data['A'].shape[0]
        self.num_batch = self.num_data // self.bs
        self.epoch_id = 0
        self.batch_id = 0

        # WandB
        self.use_wandb = cfg.wandb.enabled
        if self.use_wandb:
            import wandb
            self.wandb = wandb
            self.wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                config=OmegaConf.to_container(cfg, resolve=True))

        print(f'Samples: {self.num_data}  Batches/epoch: {self.num_batch}')
        print(f'Diffusion steps: {d.T}  DDIM sampling steps: {d.ddim_steps}')

    # ------------------------------------------------------------------
    #   Data helpers
    # ------------------------------------------------------------------

    def _get_batch(self, idx):
        """Return (target, conditioning) tensors.

        target:       defence(10) + ball_features(6) = 16 dims
        conditioning: offence(12) + seq_feat(6) = 18 dims
        """
        end = idx + self.bs
        # Offence (drop ball z): [B, T, 12]
        offence = self.data_factory.train_data['A'][idx:end]
        offence = offence[:, :, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        # Defence: [B, T, 10]
        defence = self.data_factory.train_data['B'][idx:end]
        # Features: [B, T, 6] each
        seq_feat = self.data_factory.f_train[idx:end]
        real_feat = self.data_factory.rf_train[idx:end]

        target = tf.concat([tf.constant(defence, dtype=tf.float32),
                            tf.constant(real_feat, dtype=tf.float32)], axis=-1)
        conds = tf.concat([tf.constant(offence, dtype=tf.float32),
                           tf.constant(seq_feat, dtype=tf.float32)], axis=-1)
        return target, conds

    def _get_sample_batch(self):
        """Get conditioning for visualization."""
        idx = self.batch_id * self.bs
        end = idx + self.bs
        offence = self.data_factory.train_data['A'][idx:end]
        offence = offence[:, :, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        seq_feat = self.data_factory.f_train[idx:end]
        return (tf.constant(offence, dtype=tf.float32),
                tf.constant(seq_feat, dtype=tf.float32))

    # ------------------------------------------------------------------
    #   Training step
    # ------------------------------------------------------------------

    @tf.function
    def _train_step(self, target, conds):
        """One diffusion training step."""
        B = tf.shape(target)[0]
        t = tf.random.uniform([B], 0, self.cfg.diffusion.T, dtype=tf.int32)

        # Forward diffusion: add noise
        xt, noise = self.diffusion.q_sample(target, t)

        with tf.GradientTape() as tape:
            pred_noise = self.denoiser(xt, t, conds, training=True)
            loss = tf.reduce_mean(tf.square(pred_noise - noise))

        grads = tape.gradient(loss, self.denoiser.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.denoiser.trainable_variables))

        # Update EMA
        if not self.ema_initialized:
            self.ema.apply(self.denoiser.trainable_variables)
            self.ema_initialized = True
        else:
            self.ema.apply(self.denoiser.trainable_variables)

        return loss

    # ------------------------------------------------------------------
    #   Main loop
    # ------------------------------------------------------------------

    def __call__(self):
        max_epochs = self.cfg.training.max_epochs
        vis_freq = self.cfg.training.vis_freq
        ckpt_step = self.cfg.training.checkpoint_step

        while max_epochs is None or self.epoch_id < max_epochs:
            target, conds = self._get_batch(self.batch_id * self.bs)
            loss = self._train_step(target, conds)

            self.batch_id += 1
            if self.batch_id >= self.num_batch:
                self.epoch_id += 1
                self.batch_id = 0
                self.data_factory.shuffle_train()

                print(f'Epoch {self.epoch_id:4d}  loss={loss.numpy():.6f}')

                # WandB log
                if self.use_wandb:
                    self.wandb.log({'epoch': self.epoch_id, 'loss': loss.numpy()})

                # Checkpoint
                if self.epoch_id % ckpt_step == 0:
                    ckpt = os.path.join(self.checkpoint_path, 'model.ckpt')
                    self.checkpoint.write(ckpt)
                    print(f'  Saved checkpoint: {ckpt}')

                # Visualization sample
                if self.epoch_id % vis_freq == 0:
                    self._visualize()

    # ------------------------------------------------------------------
    #   Sampling + Visualization
    # ------------------------------------------------------------------

    @tf.function
    def generate(self, conds, steps=None):
        """Generate a play via DDIM sampling."""
        if steps is None:
            steps = self.cfg.diffusion.ddim_steps
        B = tf.shape(conds)[0]
        T = tf.shape(conds)[1]
        return self.diffusion.sample(self.denoiser, conds,
                                     [B, T, 16], steps=steps)

    def _visualize(self):
        """Generate and save a sample play animation."""
        offence, seq_feat = self._get_sample_batch()

        conds = tf.concat([offence, seq_feat], axis=-1)
        generated = self.generate(conds)                          # [B, T, 16]
        gen_np = generated.numpy()

        defence_gen = gen_np[:, :, :10]                           # defence positions
        # Reconstruct full play for visualization
        off_np = offence.numpy()
        sample = np.concatenate([off_np, defence_gen], axis=-1)   # [B, T, 22]

        samples = self.data_factory.recover_BALL_and_A(sample)
        samples = self.data_factory.recover_B(samples)
        fname = os.path.join(self.sample_path, f'reconstruct{self.epoch_id}.mp4')
        game_visualizer.plot_data(
            samples[0], self.seq_len, file_path=fname, if_save=True)

        if self.use_wandb:
            self.wandb.log({'sample': self.wandb.Video(fname)})

    # ------------------------------------------------------------------
    #   Save / Load
    # ------------------------------------------------------------------

    def save_model(self, path):
        self.checkpoint.write(path)

    def load_model(self, path):
        self.checkpoint.read(path).expect_partial()
        print(f'Restored checkpoint from {path}')


# ---------------------------------------------------------------------------
#   Main entry
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    folder_path = os.path.abspath(cfg.paths.folder_path)
    data_path = os.path.abspath(cfg.paths.data_path)

    # Output directory
    if os.path.exists(folder_path):
        ans = input(f'"{folder_path}" will be removed!! are you sure (y/N)? ')
        if ans.lower() == 'y':
            shutil.rmtree(folder_path)
        else:
            exit(0)
    checkpoint_path = os.path.join(folder_path, 'Checkpoints')
    sample_path = os.path.join(folder_path, 'Samples')
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)

    print(OmegaConf.to_yaml(cfg))

    # Mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f'Compute dtype: {tf.keras.mixed_precision.global_policy().compute_dtype}')

    # Load data
    real_data = np.load(os.path.join(data_path, '50Real.npy'))[:, :cfg.training.seq_length, :, :]
    seq_data = np.load(os.path.join(data_path, '50Seq.npy'))
    features_ = np.load(os.path.join(data_path, 'SeqCond.npy'))
    real_feat = np.load(os.path.join(data_path, 'RealCond.npy'))

    data_factory = DataFactory(real_data=real_data, seq_data=seq_data,
                               features_=features_, real_feat=real_feat)

    # Train
    model_type = cfg.training.get('model_type', 'diffusion')
    if model_type == 'vaegan':
        from ThreeDiscrim import VAEGAN_Model
        # ... (VAE-GAN path kept for comparison)
        raise NotImplementedError(
            "VAE-GAN training via Hydra: use the old Train_Triple.py. "
            "Set training.model_type=diffusion (default).")
    else:
        trainer = DiffusionTrainer(data_factory, cfg, checkpoint_path, sample_path)

    trainer()

    if cfg.wandb.enabled:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
