import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
import numpy as np
import os
import shutil
import time

from utils import DataFactory
from ThreeDiscrim import VAEGAN_Model
import game_visualizer
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

os.environ[
    'TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

parser = argparse.ArgumentParser(description='Basketball VAE-GAN Training')

parser.add_argument('--folder_path', type=str, default=None, help='summeray directory')
parser.add_argument('--data_path', type=str, default=None, help='summary directory')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of input')
parser.add_argument('--latent_dims', type=int, default=150, help='dimension of latent variable')
parser.add_argument('--seq_length', type=int, default=50, help='sequence length')
parser.add_argument('--features_', type=int, default=12, help='number of offence features')
parser.add_argument('--features_d', type=int, default=10, help='number of defence features')
parser.add_argument('--n_resblock', type=int, default=8, help='number of residual blocks')
parser.add_argument('--pretrain_D', type=int, default=25, help='Epoch to pretrain D')
parser.add_argument('--train_D', type=int, default=5, help='Number of times to train D')
parser.add_argument('--lr_', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lambda_', type=float, default=1.0, help='Decaying lambda value')
parser.add_argument('--n_filters', type=int, default=256, help='number of filters in conv')
parser.add_argument('--keep_prob', type=float, default=1.0, help='keep prob of dropout')
parser.add_argument('--beta', type=float, default=0.001, help='KL divergence weight (beta-VAE)')
parser.add_argument('--recon_weight', type=float, default=1.0, help='Reconstruction loss weight (L1)')
parser.add_argument('--vis_freq', type=int, default=5, help='number of epoches to visulize samples')
parser.add_argument('--checkpoint_step', type=int, default=100, help='number of steps before saving checkpoint')

class Training_config(object):
    #Training configurations
    def __init__(self):
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
        print(vars(self))


#Generate sample
def encode(data, model):
    """Encode real play to latent code z using the VAE encoder."""
    x, x_d, seq, feat, real_feat = data
    return model.encode_latent(x, x_d, seq, feat, real_feat)


def reconstruct_(model, x, z, x2):
    return model.reconstruct_(x, z, x2)


def z_samples():
    return np.random.normal(0., 1., size=[args.batch_size, args.latent_dims])


class Trainer(object):
    def __init__(self, data_factory, config):
        self.data_factory = data_factory
        self.config = config
        self.model = VAEGAN_Model(config)
        self.num_data = self.data_factory.train_data['A'].shape[0]
        self.num_batch = self.num_data // args.batch_size
        self.num_batch_valid = self.data_factory.valid_data['A'].shape[
            0] // args.batch_size
        self.epoch_id = 0
        self.batch_id = 0
        self.batch_id_valid = 0
        print('self.num_batch:', self.num_batch)
        print('self.num_batch_valid:', self.num_batch_valid)

    def __call__(self):
        while True:
            if self.epoch_id < args.pretrain_D == 0:  # warming
                num_d = 10
            else:
                num_d = args.train_D
            start_time = time.time()
            for _ in range(num_d):
                self.train_D()

            start_time = time.time()
            self.train_G()
            # validation
            valid_idx = self.batch_id_valid * args.batch_size
            valid_ = self.data_factory.valid_data['A'][valid_idx:valid_idx +
                                                       args.batch_size]
            valid_D = self.data_factory.valid_data['B'][valid_idx:valid_idx +
                                                        args.batch_size]

            valid_ = valid_[:, :, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
            seq_v = self.data_factory.seq_valid[valid_idx:valid_idx +
                                                args.batch_size]
            rv_feat = self.data_factory.f_valid[valid_idx:valid_idx +
                                                args.batch_size]
            rfv_feat = self.data_factory.rf_valid[valid_idx:valid_idx +
                                                  args.batch_size]

            self.model.valid_loss(x=valid_,
                                  x2=valid_D,
                                  y=seq_v,
                                  z=z_samples(),
                                  feat_=rv_feat,
                                  feat2_=rfv_feat)
            self.update_batch_id_valid_and_shuffle()

    def train_G(self):
        data_idx = self.batch_id * args.batch_size
        training_data = self.data_factory.train_data
        f_train = self.data_factory.f_train
        seq_train = self.data_factory.seq_train
        rf_train = self.data_factory.rf_train

        real_ = training_data['A'][data_idx:data_idx + args.batch_size]
        real_D = training_data['B'][data_idx:data_idx + args.batch_size]

        seq_feat = f_train[data_idx:data_idx + args.batch_size]
        real_ = real_[:, :, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        seq_ = seq_train[data_idx:data_idx + args.batch_size]

        real_feat = rf_train[data_idx:data_idx + args.batch_size]

        self.model.update_gen(real=real_,
                              real_d=real_D,
                              x=seq_,
                              x2=seq_feat,
                              x3=real_feat,
                              z=z_samples())

        self.update_batch_id_and_shuffle()

    def train_D(self):
        data_idx = self.batch_id * args.batch_size
        training_data = self.data_factory.train_data
        f_train = self.data_factory.f_train
        seq_train = self.data_factory.seq_train
        rf_train = self.data_factory.rf_train

        real_ = training_data['A'][data_idx:data_idx + args.batch_size]
        real_D = training_data['B'][data_idx:data_idx + args.batch_size]

        seq_feat = f_train[data_idx:data_idx + args.batch_size]
        real_ = real_[:, :, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        seq_ = seq_train[data_idx:data_idx + args.batch_size]

        real_feat = rf_train[data_idx:data_idx + args.batch_size]

        self.model.update_discrim(x=real_,
                                  x2=real_D,
                                  y=seq_,
                                  z=z_samples(),
                                  feat_=seq_feat,
                                  feat2_=real_feat)

        self.update_batch_id_and_shuffle()

    def update_batch_id_valid_and_shuffle(self):
        self.batch_id_valid = self.batch_id_valid + 1
        if self.batch_id_valid >= self.num_batch_valid:
            self.batch_id_valid = 0
            self.data_factory.shuffle_valid()

    def update_batch_id_and_shuffle(self):
        self.batch_id = self.batch_id + 1
        if self.batch_id >= self.num_batch:
            self.epoch_id = self.epoch_id + 1
            self.batch_id = 0
            self.data_factory.shuffle_train()
            # save model
            if self.epoch_id % args.checkpoint_step == 0:
                checkpoint_ = os.path.join(CHECKPOINT_PATH, 'model.ckpt')
                self.model.save_model(checkpoint_)
                print("Saved model:", checkpoint_)
            # save generated sample
            if self.epoch_id % args.vis_freq == 0:
                print('epoch_id:', self.epoch_id)
                data_idx = self.batch_id * args.batch_size
                f_train = self.data_factory.f_train
                seq_train = self.data_factory.seq_train
                seq_feat = f_train[data_idx:data_idx + args.batch_size]
                seq_ = seq_train[data_idx:data_idx + args.batch_size]

                recon = reconstruct_(self.model, seq_, z_samples(), seq_feat)
                sample = recon[:, :, :22]
                samples = self.data_factory.recover_BALL_and_A(sample)
                samples = self.data_factory.recover_B(samples)
                game_visualizer.plot_data(
                    samples[0],
                    args.seq_length,
                    file_path=SAMPLE_PATH +
                    'reconstruct{}.mp4'.format(self.epoch_id),
                    if_save=True)


def main(args):
    with tf.get_default_graph().as_default() as graph:
        real_data = np.load(os.path.join(
            args.data_path, '50Real.npy'))[:, :args.seq_length, :, :]
        seq_data = np.load(os.path.join(args.data_path, '50Seq.npy'))
        features_ = np.load(os.path.join(args.data_path, 'SeqCond.npy'))
        real_feat = np.load(os.path.join(args.data_path, 'RealCond.npy'))

        print("Real Data: ", real_data.shape)
        print("Seq Data: ", seq_data.shape)
        print("Real Feat: ", real_feat.shape)
        print("Seq Feat: ", features_.shape)

        data_factory = DataFactory(real_data=real_data,
                                   seq_data=seq_data,
                                   features_=features_,
                                   real_feat=real_feat)

        config = Training_config()
        config.show()
        trainer = Trainer(data_factory, config)
        trainer()


if __name__ == '__main__':
    args = parser.parse_args()
    CHECKPOINT_PATH = os.path.join(args.folder_path, 'Checkpoints/')
    SAMPLE_PATH = os.path.join(args.folder_path, 'Samples/')
    if os.path.exists(args.folder_path):
        ans = input('"%s" will be removed!! are you sure (y/N)? ' %
                    args.folder_path)
        if ans == 'Y' or ans == 'y':
            shutil.rmtree(args.folder_path)
            print('rm -rf "%s" complete!' % args.folder_path)
        else:
            exit()
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    if not os.path.exists(SAMPLE_PATH):
        os.makedirs(SAMPLE_PATH)
    main(args)
