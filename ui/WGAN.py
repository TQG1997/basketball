"""WGAN model loading and inference (TF 2.16+ / Keras 3).

Usage:
    # Called programmatically from Main.py:
    model, data_factory = Load_Model()
    output = run_Model(model, data_factory)
"""

import os
import argparse
import numpy as np

# TF must be imported before ThreeDiscrim to avoid compat issues
import tensorflow as tf  # noqa: E402

from src.ThreeDiscrim import VAEGAN_Model
from src.Train_Triple import TrainingConfig, z_samples
from utils import DataFactory
import draw_feat


# ---- Default paths ----
DATA_PATH = os.path.join(os.path.dirname(__file__), 'Data')
MODEL_PATH_DEFAULT = os.path.join(DATA_PATH, 'checkpoints', 'model.ckpt-88200')
SAVE_PATH = os.path.join(DATA_PATH, 'output')

# ---- Inference constants ----
N_LATENT = 100
LATENT_DIM = 150
SEQ_LEN = 50


def _ensure_dirs():
    for path in [SAVE_PATH]:
        os.makedirs(path, exist_ok=True)


def _check_file(filepath, description):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f'{description} not found: {filepath}\n'
            f'See README.md for download links.')


def parse_args():
    parser = argparse.ArgumentParser(description='BasketballGAN: Inference from sketch')
    parser.add_argument('--checkpoint', type=str, default=MODEL_PATH_DEFAULT,
                        help='Path to checkpoint prefix (e.g. model.ckpt-88200)')
    parser.add_argument('--points', type=str, default='Points/points2.npy',
                        help='Path to input sketch .npy')
    parser.add_argument('--output', type=str, default=os.path.join(SAVE_PATH, 'output.npy'),
                        help='Output .npy path')
    return parser.parse_args()


def Load_Model(checkpoint_path=None):
    """Load the pre-trained model.

    Returns (model, data_factory) where model is a VAEGAN_Model instance.
    """
    if checkpoint_path is None:
        checkpoint_path = MODEL_PATH_DEFAULT

    data_dir = os.path.join(DATA_PATH, 'Model_data')

    # Check required files
    _check_file(os.path.join(data_dir, '50Seq.npy'), 'Sequence data')
    _check_file(os.path.join(data_dir, 'SeqCond.npy'), 'Sequence features')
    _check_file(os.path.join(data_dir, '50Real.npy'), 'Real data')
    _check_file(os.path.join(data_dir, 'RealCond.npy'), 'Real features')

    # Initialize DataFactory
    image_data = np.load(os.path.join(data_dir, '50Seq.npy'))
    features_ = np.load(os.path.join(data_dir, 'SeqCond.npy'))
    real_data = np.load(os.path.join(data_dir, '50Real.npy'))[:, :50, :, :]
    real_feat = np.load(os.path.join(data_dir, 'RealCond.npy'))
    print('real_data.shape', real_data.shape)
    data_factory = DataFactory(real_data, image_data, features_, real_feat)

    # Build model with dummy config
    class DummyConfig:
        pass
    config = DummyConfig()
    config.batch_size = 8
    config.seq_length = SEQ_LEN
    config.latent_dims = LATENT_DIM
    config.n_filters = 256
    config.n_resblock = 8
    config.lr_ = 1e-4
    config.beta = 0.001
    config.recon_weight = 1.0
    config.features_ = 12
    config.features_d = 10
    config.folder_path = DATA_PATH

    model = VAEGAN_Model(config)

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path + '.index'):
        model.load_model(checkpoint_path)
        print(f'Model loaded from {checkpoint_path}')
    else:
        # Check for new-format checkpoint
        alt_path = checkpoint_path.replace('.ckpt-', '.ckpt')
        if os.path.exists(os.path.dirname(checkpoint_path)):
            ckpt_files = [f for f in os.listdir(os.path.dirname(checkpoint_path))
                          if f.startswith('model.ckpt') and f.endswith('.index')]
            if ckpt_files:
                alt_path = os.path.join(os.path.dirname(checkpoint_path),
                                        ckpt_files[0].replace('.index', ''))
                model.load_model(alt_path)
                print(f'Model loaded from {alt_path}')
            else:
                print('WARNING: No checkpoint found — using random weights')
        else:
            print('WARNING: No checkpoint found — using random weights')

    return model, data_factory


def run_Model(model, data_factory, points_path=None, output_path=None):
    """Generate defensive plays from a sketched offensive play."""
    if points_path is None:
        points_path = 'Points/points2.npy'
    if output_path is None:
        output_path = os.path.join(SAVE_PATH, 'output.npy')

    _ensure_dirs()

    points = np.load(points_path)
    front = np.tile(points[0], (2, 1))
    points = np.concatenate([front, points])
    print('Points:', points.shape)
    extra = np.tile(points[-1], (4, 1))
    points = np.concatenate([points, extra])

    feature = draw_feat.get_feature(points)

    target_length = len(points)
    dims = 10
    points_batch = np.reshape(np.tile(points, (dims, 1)), [dims, target_length, 12])
    feature_batch = np.repeat(feature, dims, axis=0)

    # Normalize
    team_AB = np.concatenate([
        points_batch[:, :, :2].reshape([dims, target_length, 2]),
        points_batch[:, :, 2:12].reshape([dims, target_length, 10]),
        feature_batch.reshape([dims, target_length, 6]),
    ], axis=-1)
    team_AB = data_factory.normalize(team_AB)
    team_A = team_AB[:, :, :12]
    team_Feat = team_AB[:, :, 12:]

    results = []
    for idx in range(dims):
        real_conds = np.repeat(team_A[idx:idx + 1, :], N_LATENT, axis=0)
        real_feat = np.repeat(team_Feat[idx:idx + 1, :], N_LATENT, axis=0)

        z = z_samples(N_LATENT, LATENT_DIM)
        result = model.reconstruct(
            tf.constant(real_conds, dtype=tf.float32),
            tf.constant(real_feat, dtype=tf.float32),
            tf.constant(z, dtype=tf.float32))
        result_np = result.numpy()

        recovered = data_factory.recover_data(result_np[:, :, :22])
        recovered = np.concatenate([recovered, result_np[:, :, 22:]], axis=-1)
        results.append(recovered)

    results = np.stack(results, axis=1)
    print('Output shape:', results.shape)

    np.save(output_path, results.astype(np.float32).reshape(
        [N_LATENT, dims, target_length, 28]))
    print(f'!!Saved!! → {output_path}')


if __name__ == '__main__':
    args = parse_args()
    model, data_factory = Load_Model(checkpoint_path=args.checkpoint)
    run_Model(model, data_factory, points_path=args.points, output_path=args.output)
