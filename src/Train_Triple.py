"""BasketballGAN training entry point (TF 2.16+ / Keras 3 / Hydra / WandB).

Usage:
    python Train_Triple.py                                    # default config
    python Train_Triple.py model.latent_dims=64 training.batch_size=32
    python Train_Triple.py training.max_epochs=500 wandb.enabled=true
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
from ThreeDiscrim import VAEGAN_Model
import game_visualizer


def z_samples(batch_size, latent_dims):
    return np.random.normal(0., 1., size=[batch_size, latent_dims]).astype(np.float32)


class Trainer:
    def __init__(self, data_factory, cfg, checkpoint_path, sample_path):
        self.data_factory = data_factory
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.sample_path = sample_path

        # Build config object for model
        model_cfg = self._make_model_config(cfg)
        self.model = VAEGAN_Model(model_cfg)

        self.bs = cfg.training.batch_size
        self.num_data = data_factory.train_data['A'].shape[0]
        self.num_batch = self.num_data // self.bs
        self.num_batch_valid = data_factory.valid_data['A'].shape[0] // self.bs
        self.epoch_id = 0
        self.batch_id = 0
        self.batch_id_valid = 0

        # WandB
        self.use_wandb = cfg.wandb.enabled
        if self.use_wandb:
            import wandb
            self.wandb = wandb
            self.wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                config=OmegaConf.to_container(cfg, resolve=True))
            # Watch model gradients
            self.wandb_run = self.wandb

        print(f'Batches per epoch: {self.num_batch}')
        print(f'Validation batches: {self.num_batch_valid}')

    @staticmethod
    def _make_model_config(cfg):
        """Build a simple config object for VAEGAN_Model."""
        class MC:
            pass
        mc = MC()
        m = cfg.model
        t = cfg.training
        p = cfg.paths
        mc.batch_size = t.batch_size
        mc.seq_length = t.seq_length
        mc.latent_dims = m.latent_dims
        mc.n_filters = m.n_filters
        mc.n_resblock = m.n_resblock
        mc.lr_ = t.lr_
        mc.beta = t.beta
        mc.recon_weight = t.recon_weight
        mc.features_ = m.features_
        mc.features_d = m.features_d
        mc.keep_prob = 1.0
        mc.folder_path = p.folder_path
        return mc

    def _get_batch(self, data_dict, seq_data, feat_data, real_feat_data, idx):
        batch = data_dict['A'][idx:idx + self.bs]
        batch_d = data_dict['B'][idx:idx + self.bs]
        seq = seq_data[idx:idx + self.bs]
        feat = feat_data[idx:idx + self.bs]
        real_feat = real_feat_data[idx:idx + self.bs]
        # Drop ball z (index 2): [B,T,13] → [B,T,12]
        batch = batch[:, :, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        return (tf.constant(batch, dtype=tf.float32),
                tf.constant(batch_d, dtype=tf.float32),
                tf.constant(seq, dtype=tf.float32),
                tf.constant(feat[:, :, :], dtype=tf.float32),
                tf.constant(real_feat[:, :, :], dtype=tf.float32))

    def _prep_basket_right(self):
        br = tf.constant(
            [self.data_factory.BASKET_RIGHT[0],
             self.data_factory.BASKET_RIGHT[1]], dtype=tf.float32)
        return br

    def __call__(self):
        br = self._prep_basket_right()
        max_epochs = self.cfg.training.max_epochs
        pretrain_d = self.cfg.training.pretrain_D
        train_d = self.cfg.training.train_D
        ckpt_step = self.cfg.training.checkpoint_step
        vis_freq = self.cfg.training.vis_freq

        while max_epochs is None or self.epoch_id < max_epochs:
            num_d = 10 if self.epoch_id < pretrain_d else train_d

            # ---- Train D ----
            for _ in range(num_d):
                real, real_d, seq, seq_feat, real_feat = self._get_batch(
                    self.data_factory.train_data,
                    self.data_factory.seq_train,
                    self.data_factory.f_train,
                    self.data_factory.rf_train,
                    self.batch_id * self.bs)
                self.model.train_D(real, real_d, seq, seq_feat, real_feat, br)

            # ---- Train G ----
            real, real_d, seq, seq_feat, real_feat = self._get_batch(
                self.data_factory.train_data,
                self.data_factory.seq_train,
                self.data_factory.f_train,
                self.data_factory.rf_train,
                self.batch_id * self.bs)
            g_losses = self.model.train_G(real, real_d, seq, seq_feat, real_feat, br)

            self._update_batch_id_and_shuffle(ckpt_step, vis_freq)

            # ---- Validation ----
            vidx = self.batch_id_valid * self.bs
            vreal, vreal_d, vseq, vseq_feat, vreal_feat = self._get_batch(
                self.data_factory.valid_data,
                self.data_factory.seq_valid,
                self.data_factory.f_valid,
                self.data_factory.rf_valid,
                vidx)
            vlosses = self.model.valid_loss(vreal, vreal_d, vseq, vseq_feat, vreal_feat, br)
            self._update_batch_id_valid_and_shuffle()

            # ---- Log to WandB ----
            if self.use_wandb and self.batch_id == 0:
                self.wandb.log({
                    'epoch': self.epoch_id,
                    **{f'G/{k}': v.numpy().item() if hasattr(v, 'numpy') else v
                       for k, v in g_losses.items()},
                    **{f'V/{k}': v.numpy().item() if hasattr(v, 'numpy') else v
                       for k, v in vlosses.items()},
                }, commit=False)

    def _update_batch_id_valid_and_shuffle(self):
        self.batch_id_valid += 1
        if self.batch_id_valid >= self.num_batch_valid:
            self.batch_id_valid = 0
            self.data_factory.shuffle_valid()

    def _update_batch_id_and_shuffle(self, ckpt_step, vis_freq):
        self.batch_id += 1
        if self.batch_id >= self.num_batch:
            self.epoch_id += 1
            self.batch_id = 0
            self.data_factory.shuffle_train()

            if self.epoch_id % ckpt_step == 0:
                ckpt = os.path.join(self.checkpoint_path, 'model.ckpt')
                self.model.save_model(ckpt)
                print(f'Saved checkpoint epoch {self.epoch_id}: {ckpt}')

            if self.epoch_id % vis_freq == 0:
                print(f'--- Epoch {self.epoch_id} ---')
                self._visualize()

    def _visualize(self):
        data_idx = self.batch_id * self.bs
        seq = self.data_factory.seq_train[data_idx:data_idx + self.bs]
        feat = self.data_factory.f_train[data_idx:data_idx + self.bs]

        z = z_samples(self.bs, self.cfg.model.latent_dims)
        fake = self.model.reconstruct(
            tf.constant(seq, dtype=tf.float32),
            tf.constant(feat[:, :, :], dtype=tf.float32),
            tf.constant(z, dtype=tf.float32))
        fake_np = fake.numpy()

        sample = fake_np[:, :, :22]
        samples = self.data_factory.recover_BALL_and_A(sample)
        samples = self.data_factory.recover_B(samples)
        fname = os.path.join(self.sample_path, f'reconstruct{self.epoch_id}.mp4')
        game_visualizer.plot_data(samples[0], self.cfg.training.seq_length,
                                  file_path=fname, if_save=True)

        if self.use_wandb:
            self.wandb.log({'sample': self.wandb.Video(fname)}, commit=False)


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Resolve paths relative to project root
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get() if hydra else None
    folder_path = os.path.abspath(cfg.paths.folder_path)
    data_path = os.path.abspath(cfg.paths.data_path)

    # --- Output directory ---
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

    # --- Load data ---
    real_data = np.load(os.path.join(data_path, '50Real.npy'))[:, :cfg.training.seq_length, :, :]
    seq_data = np.load(os.path.join(data_path, '50Seq.npy'))
    features_ = np.load(os.path.join(data_path, 'SeqCond.npy'))
    real_feat = np.load(os.path.join(data_path, 'RealCond.npy'))

    data_factory = DataFactory(real_data=real_data, seq_data=seq_data,
                               features_=features_, real_feat=real_feat)

    # --- Mixed precision (FP16 on T4, BF16 on A100) ---
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f'Compute dtype: {tf.keras.mixed_precision.global_policy().compute_dtype}')
    print(f'Variable dtype: {tf.keras.mixed_precision.global_policy().variable_dtype}')

    # --- Train ---
    trainer = Trainer(data_factory, cfg, checkpoint_path, sample_path)
    trainer()

    if cfg.wandb.enabled:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
