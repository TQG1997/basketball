"""WGAN model loading and inference.

Usage:
    # Called programmatically from Main.py:
    graph, saver, config, data_factory = Load_Model()
    run_Model(graph, saver, config, data_factory)
"""

import os
import sys
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils import DataFactory
import draw_feat


# ---- Default paths ----
DATA_PATH = os.path.join(os.path.dirname(__file__), 'Data')
MODEL_PATH_DEFAULT = os.path.join(DATA_PATH, 'checkpoints', 'model.ckpt-88200')
SAVE_PATH = os.path.join(DATA_PATH, 'output')

# ---- Inference constants ----
BATCH_SIZE = 8
N_LATENT = 100
LATENT_DIM = 150
SEQ_LEN = 50


def _ensure_dirs():
    """Create required directories if they don't exist."""
    for path in [SAVE_PATH, os.path.dirname(MODEL_PATH_DEFAULT)]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)


def z_samples(n_latent=N_LATENT, latent_dim=LATENT_DIM):
    return np.random.normal(0., 1., size=[n_latent, latent_dim])


def _check_file(filepath, description):
    """Raise a user-friendly error if a required file is missing."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"{description} not found: {filepath}\n"
            f"Please download the required files. See README.md for download links."
        )


def parse_args():
    """Parse command-line arguments for standalone inference."""
    parser = argparse.ArgumentParser(
        description='BasketballGAN: Run inference from a sketch')
    parser.add_argument('--checkpoint', type=str, default=MODEL_PATH_DEFAULT,
                        help='Path to model checkpoint (without .meta extension)')
    parser.add_argument('--data_dir', type=str, default=os.path.join(DATA_PATH, 'Model_data'),
                        help='Directory containing 50Real.npy, 50Seq.npy, SeqCond.npy, RealCond.npy')
    parser.add_argument('--points', type=str, default='Points/points2.npy',
                        help='Path to input sketch .npy file')
    parser.add_argument('--output', type=str, default=os.path.join(SAVE_PATH, 'output.npy'),
                        help='Path for generated output .npy file')
    return parser.parse_args()


def Load_Model(model_path=None):
    """Load the pre-trained model and return (graph, saver, config, data_factory).

    Parameters
    ----------
    model_path : str or None
        Path to checkpoint (without .meta extension).
        Defaults to MODEL_PATH_DEFAULT.

    Returns
    -------
    graph : tf.Graph
    saver : tf.train.Saver
    config : tf.ConfigProto
    data_factory : DataFactory

    Raises
    ------
    FileNotFoundError
        If checkpoint or data files are missing, with download instructions.
    """
    if model_path is None:
        model_path = MODEL_PATH_DEFAULT

    meta_path = model_path + '.meta'
    data_dir = os.path.join(DATA_PATH, 'Model_data')

    # Check required files before starting TF
    _check_file(meta_path, "Model checkpoint (.meta file)")
    _check_file(os.path.join(data_dir, '50Seq.npy'), "Sequence data (50Seq.npy)")
    _check_file(os.path.join(data_dir, 'SeqCond.npy'), "Sequence features (SeqCond.npy)")
    _check_file(os.path.join(data_dir, '50Real.npy'), "Real data (50Real.npy)")
    _check_file(os.path.join(data_dir, 'RealCond.npy'), "Real features (RealCond.npy)")

    with tf.get_default_graph().as_default() as graph:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        try:
            saver = tf.train.import_meta_graph(meta_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model checkpoint: {meta_path}\n"
                f"Make sure the .meta, .index, and .data-* files are all present.\n"
                f"Original error: {e}"
            )

        print("Model found at:", model_path)

        image_data = np.load(os.path.join(data_dir, '50Seq.npy'))
        features_ = np.load(os.path.join(data_dir, 'SeqCond.npy'))
        real_data = np.load(os.path.join(data_dir, '50Real.npy'))[:, :50, :, :]
        real_feat = np.load(os.path.join(data_dir, 'RealCond.npy'))
        print('real_data.shape', real_data.shape)

        data_factory = DataFactory(real_data, image_data, features_, real_feat)

    return graph, saver, config, data_factory


def run_Model(graph, saver, config, data_factory, points_path=None, output_path=None):
    """Run inference: generate defensive plays from a sketched offensive play.

    Parameters
    ----------
    graph : tf.Graph
    saver : tf.train.Saver
    config : tf.ConfigProto
    data_factory : DataFactory
    points_path : str or None
        Path to sketch .npy file.
    output_path : str or None
        Where to save the generated output.
    """
    if points_path is None:
        points_path = 'Points/points2.npy'
    if output_path is None:
        output_path = os.path.join(SAVE_PATH, 'output.npy')

    _ensure_dirs()

    points = np.load(points_path)
    front = np.tile(points[0], (2, 1))
    points = np.concatenate([front, points])
    print("Points:", points.shape)
    print(points[-1])
    extra = np.tile(points[-1], (4, 1))
    points = np.concatenate([points, extra])

    feature = draw_feat.get_feature(points)

    with tf.Session(config=config) as sess:
        saver.restore(sess, MODEL_PATH_DEFAULT)

        result_t = graph.get_tensor_by_name('G_/concat_1:0')
        use_encoder_t = graph.get_tensor_by_name('use_encoder:0')
        latent_input_t = graph.get_tensor_by_name('Latent:0')
        feature_input = graph.get_tensor_by_name('Seq_feat:0')
        condition_input_t = graph.get_tensor_by_name('Cond_input:0')

        print("Loaded")

        target_length = len(points)
        dims = 10
        points = [points] * dims
        points = np.reshape(points, newshape=[dims, target_length, 12])

        feature = np.repeat(feature, dims, axis=0)

        print(points.shape)

        # Target data
        target_data = points
        target_feat = feature

        print('target_data.shape', target_data.shape)

        team_AB = np.concatenate(
            [
                # ball xy
                target_data[:, :, :2].reshape(
                    [target_data.shape[0], target_data.shape[1], 1 * 2]),
                # team A players xy
                target_data[:, :, 2:12].reshape(
                    [target_data.shape[0], target_data.shape[1], 5 * 2]),
                # feature
                target_feat[:, :, :].reshape(
                    [target_feat.shape[0], target_feat.shape[1], 6 * 1])
            ],
            axis=-1)
        team_AB = data_factory.normalize(team_AB)
        team_A = team_AB[:, :, :12]
        team_Feat = team_AB[:, :, 12:]

        # Result collector
        results_A_fake_B = []
        results_A_real_B = []

        print(team_AB.shape)
        print(team_AB.shape[0])

        for idx in range(team_AB.shape[0]):
            # Generate N_LATENT results for the same condition
            real_conds = team_A[idx:idx + 1, :target_length]
            real_conds = np.concatenate(
                [real_conds for _ in range(N_LATENT)], axis=0)

            real_feat = team_Feat[idx:idx + 1, :target_length]
            real_feat = np.concatenate(
                [real_feat for _ in range(N_LATENT)], axis=0)

            latents = z_samples()
            feed_dict = {
                use_encoder_t: False,
                latent_input_t: latents,
                condition_input_t: real_conds,
                feature_input: real_feat
            }

            result = sess.run(result_t, feed_dict=feed_dict)

            recovered_A_fake_B = data_factory.recover_data(result[:, :, :22])
            recovered_A_fake_B = np.concatenate(
                [recovered_A_fake_B, result[:, :, 22:]], axis=-1)

            temp_A_fake_B_concat = recovered_A_fake_B
            results_A_fake_B.append(temp_A_fake_B_concat)

        # Concat along conditions dimension (axis=1)
        results_A_fake_B = np.stack(results_A_fake_B, axis=1)

        # Save as numpy
        print(np.array(results_A_fake_B).shape)
        print(np.array(results_A_real_B).shape)

        np.save(output_path,
                np.array(results_A_fake_B).astype(np.float32).reshape(
                    [N_LATENT, team_AB.shape[0], team_AB.shape[1], 28]))

        print('!!Completely Saved!! →', output_path)


# ---- Standalone execution ----
if __name__ == '__main__':
    args = parse_args()
    graph, saver, config, data_factory = Load_Model(model_path=args.checkpoint)
    run_Model(graph, saver, config, data_factory,
              points_path=args.points,
              output_path=args.output)
