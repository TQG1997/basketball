"""BasketballGAN training entry point (TF 2.16+ / Keras 3)."""

import argparse
import os
import shutil
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils import DataFactory
from ThreeDiscrim import VAEGAN_Model
import game_visualizer


parser = argparse.ArgumentParser(description='Basketball VAE-GAN Training')

parser.add_argument('--folder_path', type=str, default=None, help='summary directory')
parser.add_argument('--data_path', type=str, default=None, help='data directory')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--latent_dims', type=int, default=150, help='latent variable dimension')
parser.add_argument('--seq_length', type=int, default=50, help='sequence length')
parser.add_argument('--features_', type=int, default=12, help='number of offence features')
parser.add_argument('--features_d', type=int, default=10, help='number of defence features')
parser.add_argument('--n_resblock', type=int, default=8, help='number of residual blocks')
parser.add_argument('--pretrain_D', type=int, default=25, help='epochs to pretrain D')
parser.add_argument('--train_D', type=int, default=5, help='D updates per G update')
parser.add_argument('--lr_', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lambda_', type=float, default=1.0, help='decaying lambda (unused)')
parser.add_argument('--n_filters', type=int, default=256, help='conv filters')
parser.add_argument('--keep_prob', type=float, default=1.0, help='dropout keep prob (unused)')
parser.add_argument('--beta', type=float, default=0.001, help='KL divergence weight')
parser.add_argument('--recon_weight', type=float, default=1.0, help='reconstruction loss weight (L1)')
parser.add_argument('--vis_freq', type=int, default=5, help='epochs between visualizations')
parser.add_argument('--max_epochs', type=int, default=None, help='max training epochs (None = forever)')
parser.add_argument('--checkpoint_step', type=int, default=100, help='epochs between checkpoint saves')


class TrainingConfig:
    def __init__(self, args):
        self.folder_path = args.folder_path
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.latent_dims = args.latent_dims
        self.seq_length = args.seq_length
        self.features_ = args.features_
        self.features_d = args.features_d
        self.n_filters = args.n_filters
        self.lr_ = args.lr_
        self.keep_prob = args.keep_prob
        self.beta = args.beta
        self.recon_weight = args.recon_weight
        self.n_resblock = args.n_resblock

    def show(self):
        for k, v in vars(self).items():
            print(f'  {k}: {v}')


def z_samples(batch_size, latent_dims):
    return np.random.normal(0., 1., size=[batch_size, latent_dims]).astype(np.float32)


class Trainer:
    def __init__(self, data_factory, config, checkpoint_path, sample_path):
        self.data_factory = data_factory
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.sample_path = sample_path
        self.model = VAEGAN_Model(config)
        self.num_data = data_factory.train_data['A'].shape[0]
        self.num_batch = self.num_data // args.batch_size
        self.num_batch_valid = data_factory.valid_data['A'].shape[0] // args.batch_size
        self.epoch_id = 0
        self.batch_id = 0
        self.batch_id_valid = 0
        print(f'num_batch: {self.num_batch}')
        print(f'num_batch_valid: {self.num_batch_valid}')

    def _get_batch(self, data_dict, seq_data, feat_data, real_feat_data, idx):
        """Extract a batch and apply the positional indexing hack."""
        batch = data_dict['A'][idx:idx + args.batch_size]
        batch_d = data_dict['B'][idx:idx + args.batch_size]
        seq = seq_data[idx:idx + args.batch_size]
        feat = feat_data[idx:idx + args.batch_size]
        real_feat = real_feat_data[idx:idx + args.batch_size]

        # Drop ball z (index 2): [B,T,13] → [B,T,12]
        batch = batch[:, :, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        return (tf.constant(batch, dtype=tf.float32),
                tf.constant(batch_d, dtype=tf.float32),
                tf.constant(seq, dtype=tf.float32),
                tf.constant(feat[:, :, :], dtype=tf.float32),
                tf.constant(real_feat[:, :, :], dtype=tf.float32))

    def _prep_basket_right(self):
        """Get normalized basket right position from DataFactory."""
        br = tf.constant(
            [self.data_factory.BASKET_RIGHT[0],
             self.data_factory.BASKET_RIGHT[1]],
            dtype=tf.float32)
        return br

    def __call__(self):
        br = self._prep_basket_right()
        while args.max_epochs is None or self.epoch_id < args.max_epochs:
            # Determine D steps (warmup: 10, normal: train_D)
            if self.epoch_id < args.pretrain_D:
                num_d = 10
            else:
                num_d = args.train_D

            # ---- Train D ----
            for _ in range(num_d):
                real, real_d, seq, seq_feat, real_feat = self._get_batch(
                    self.data_factory.train_data,
                    self.data_factory.seq_train,
                    self.data_factory.f_train,
                    self.data_factory.rf_train,
                    self.batch_id * args.batch_size)
                self.model.train_D(real, real_d, seq, seq_feat, real_feat, br)

            # ---- Train G ----
            real, real_d, seq, seq_feat, real_feat = self._get_batch(
                self.data_factory.train_data,
                self.data_factory.seq_train,
                self.data_factory.f_train,
                self.data_factory.rf_train,
                self.batch_id * args.batch_size)
            g_losses = self.model.train_G(real, real_d, seq, seq_feat, real_feat, br)

            self._update_batch_id_and_shuffle()

            # ---- Validation ----
            vidx = self.batch_id_valid * args.batch_size
            vreal, vreal_d, vseq, vseq_feat, vreal_feat = self._get_batch(
                self.data_factory.valid_data,
                self.data_factory.seq_valid,
                self.data_factory.f_valid,
                self.data_factory.rf_valid,
                vidx)
            self.model.valid_loss(vreal, vreal_d, vseq, vseq_feat, vreal_feat, br)
            self._update_batch_id_valid_and_shuffle()

    def _update_batch_id_valid_and_shuffle(self):
        self.batch_id_valid += 1
        if self.batch_id_valid >= self.num_batch_valid:
            self.batch_id_valid = 0
            self.data_factory.shuffle_valid()

    def _update_batch_id_and_shuffle(self):
        self.batch_id += 1
        if self.batch_id >= self.num_batch:
            self.epoch_id += 1
            self.batch_id = 0
            self.data_factory.shuffle_train()

            # Save checkpoint
            if self.epoch_id % args.checkpoint_step == 0:
                ckpt = os.path.join(self.checkpoint_path, 'model.ckpt')
                self.model.save_model(ckpt)
                print(f'Saved checkpoint epoch {self.epoch_id}: {ckpt}')

            # Generate visualization sample
            if self.epoch_id % args.vis_freq == 0:
                print(f'--- Epoch {self.epoch_id} ---')
                self._visualize()

    def _visualize(self):
        """Generate and save a sample play animation."""
        data_idx = self.batch_id * args.batch_size
        seq = self.data_factory.seq_train[data_idx:data_idx + args.batch_size]
        feat = self.data_factory.f_train[data_idx:data_idx + args.batch_size]

        z = z_samples(args.batch_size, args.latent_dims)
        fake = self.model.reconstruct(
            tf.constant(seq, dtype=tf.float32),
            tf.constant(feat[:, :, :], dtype=tf.float32),
            tf.constant(z, dtype=tf.float32))
        fake_np = fake.numpy()

        sample = fake_np[:, :, :22]
        samples = self.data_factory.recover_BALL_and_A(sample)
        samples = self.data_factory.recover_B(samples)
        game_visualizer.plot_data(
            samples[0], args.seq_length,
            file_path=os.path.join(
                self.sample_path, f'reconstruct{self.epoch_id}.mp4'),
            if_save=True)


def main(args):
    real_data = np.load(os.path.join(args.data_path, '50Real.npy'))[:, :args.seq_length, :, :]
    seq_data = np.load(os.path.join(args.data_path, '50Seq.npy'))
    features_ = np.load(os.path.join(args.data_path, 'SeqCond.npy'))
    real_feat = np.load(os.path.join(args.data_path, 'RealCond.npy'))

    print(f'Real Data:  {real_data.shape}')
    print(f'Seq Data:   {seq_data.shape}')
    print(f'Real Feat:  {real_feat.shape}')
    print(f'Seq Feat:   {features_.shape}')

    data_factory = DataFactory(
        real_data=real_data, seq_data=seq_data,
        features_=features_, real_feat=real_feat)

    config = TrainingConfig(args)
    config.show()

    trainer = Trainer(data_factory, config, CHECKPOINT_PATH, SAMPLE_PATH)
    trainer()


if __name__ == '__main__':
    args = parser.parse_args()
    CHECKPOINT_PATH = os.path.join(args.folder_path, 'Checkpoints')
    SAMPLE_PATH = os.path.join(args.folder_path, 'Samples')

    if os.path.exists(args.folder_path):
        ans = input(f'"{args.folder_path}" will be removed!! are you sure (y/N)? ')
        if ans.lower() == 'y':
            shutil.rmtree(args.folder_path)
            print(f'rm -rf "{args.folder_path}" complete!')
        else:
            exit()

    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(SAMPLE_PATH, exist_ok=True)
    main(args)
