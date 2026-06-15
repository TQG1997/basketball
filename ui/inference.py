"""Model loading and inference — PyTorch Diffusion.

Usage:
    model, data_factory = Load_Model()
    output = run_Model(model, data_factory)
"""

import os
import sys
import argparse
import numpy as np
import torch

# Make src/ importable
_ui_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(os.path.dirname(_ui_dir), 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from diffusion import GaussianDiffusion, DenoiserNet
from utils import DataFactory
import draw_feat


# ---- Default paths ----
DATA_PATH = os.path.join(os.path.dirname(__file__), 'Data')
MODEL_PATH_DEFAULT = os.path.join(DATA_PATH, 'checkpoints', 'model.pt')
SAVE_PATH = os.path.join(DATA_PATH, 'output')

# ---- Inference constants ----
N_LATENT = 100
SEQ_LEN = 50
DDIM_STEPS = 50


def _ensure_dirs():
    os.makedirs(SAVE_PATH, exist_ok=True)


def _check_file(filepath, description):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f'{description} not found: {filepath}\n'
            f'See README.md for download links.')


def parse_args():
    parser = argparse.ArgumentParser(description='Basketball Play Generation: Inference')
    parser.add_argument('--checkpoint', type=str, default=MODEL_PATH_DEFAULT,
                        help='Path to PyTorch checkpoint (.pt)')
    parser.add_argument('--points', type=str, default='Points/points2.npy')
    parser.add_argument('--output', type=str, default=os.path.join(SAVE_PATH, 'output.npy'))
    return parser.parse_args()


def Load_Model(checkpoint_path=None):
    """Load the pre-trained PyTorch diffusion model."""
    if checkpoint_path is None:
        checkpoint_path = MODEL_PATH_DEFAULT

    data_dir = os.path.join(DATA_PATH, 'Model_data')

    _check_file(os.path.join(data_dir, '50Seq.npy'), 'Sequence data')
    _check_file(os.path.join(data_dir, '50Real.npy'), 'Real data')

    image_data = np.load(os.path.join(data_dir, '50Seq.npy'))
    features_ = np.load(os.path.join(data_dir, 'SeqCond.npy'))
    real_data = np.load(os.path.join(data_dir, '50Real.npy'))[:, :50, :, :]
    real_feat = np.load(os.path.join(data_dir, 'RealCond.npy'))
    data_factory = DataFactory(real_data, image_data, features_, real_feat)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    diffusion = GaussianDiffusion(T=1000).to(device)
    denoiser = DenoiserNet(in_dim=16, cond_dim=18, n_filters=256,
                           n_resblock=4, num_heads=4, T=1000).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        denoiser.load_state_dict(ckpt.get('model_state_dict', ckpt))
        print(f'Model loaded from {checkpoint_path}')
    else:
        print('WARNING: No checkpoint — using random weights')

    denoiser.eval()
    return (diffusion, denoiser, device), data_factory


def run_Model(model_tuple, data_factory, points_path=None, output_path=None):
    """Generate defensive plays from a sketched offensive play."""
    diffusion, denoiser, device = model_tuple

    if points_path is None:
        points_path = 'Points/points2.npy'
    if output_path is None:
        output_path = os.path.join(SAVE_PATH, 'output.npy')
    _ensure_dirs()

    points = np.load(points_path)
    front = np.tile(points[0], (2, 1))
    points = np.concatenate([front, points])
    extra = np.tile(points[-1], (4, 1))
    points = np.concatenate([points, extra])

    feature = draw_feat.get_feature(points)
    target_length = len(points)
    dims = 10

    points_batch = np.reshape(np.tile(points, (dims, 1)), [dims, target_length, 12])
    feature_batch = np.repeat(feature, dims, axis=0)

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
        conds_np = np.repeat(team_A[idx:idx + 1, :], N_LATENT, axis=0)
        feat_np = np.repeat(team_Feat[idx:idx + 1, :], N_LATENT, axis=0)
        conds_full = np.concatenate([conds_np, feat_np], axis=-1)

        conds_t = torch.from_numpy(conds_full).float().to(device)
        with torch.no_grad():
            generated = diffusion.sample(denoiser, conds_t,
                                         [N_LATENT, target_length, 16],
                                         steps=DDIM_STEPS)
        result_np = generated.cpu().numpy()

        recovered = data_factory.recover_data(
            np.concatenate([conds_np, result_np[:, :, :10]], axis=-1))
        recovered = np.concatenate([recovered, result_np[:, :, 10:]], axis=-1)
        results.append(recovered)

    results = np.stack(results, axis=1)
    np.save(output_path, results.astype(np.float32).reshape(
        [N_LATENT, dims, target_length, 28]))
    print(f'!!Saved!! → {output_path}')


if __name__ == '__main__':
    args = parse_args()
    model, data_factory = Load_Model(checkpoint_path=args.checkpoint)
    run_Model(model, data_factory, points_path=args.points, output_path=args.output)
