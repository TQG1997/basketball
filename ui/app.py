"""Basketball play generation — Gradio web interface.

Usage:
    python ui/app.py                     # local: http://127.0.0.1:7860
    python ui/app.py --share             # public link
    python ui/app.py --checkpoint path   # custom model checkpoint
"""

import os
import sys
import argparse
import tempfile
import numpy as np
import torch

# Make project root + src importable
_app_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_app_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if os.path.join(_project_root, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(_project_root, 'src'))

try:
    import gradio as gr
except ImportError:
    print('Install gradio: pip install gradio')
    sys.exit(1)

from shared import DataFactory
from diffusion import GaussianDiffusion, DenoiserNet
import game_visualizer
import draw_feat


# ---------------------------------------------------------------------------
#   Model loading
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(_app_dir, 'Data', 'checkpoints', 'model_epoch500.pt')
DATA_DIR = os.path.join(_app_dir, 'Data', 'Model_data')

# Court image for UI
COURT_IMAGE = os.path.join(_app_dir, 'images', 'court.png')

# Inference constants
DDIM_STEPS = 50
DIFFUSION_T = 1000
N_LATENT = 100
SEQ_LEN = 50


class ModelManager:
    """Lazy-loads model on first use."""

    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path or MODEL_PATH
        self._loaded = False
        self.diffusion = None
        self.denoiser = None
        self.device = None
        self.data_factory = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {self.device}')

        # DataFactory for normalization stats
        try:
            real_data = np.load(os.path.join(DATA_DIR, '50Real.npy'))[:, :50, :, :]
            seq_data = np.load(os.path.join(DATA_DIR, '50Seq.npy'))
            features_ = np.load(os.path.join(DATA_DIR, 'SeqCond.npy'))
            real_feat = np.load(os.path.join(DATA_DIR, 'RealCond.npy'))
            self.data_factory = DataFactory(real_data, seq_data, features_, real_feat)
        except FileNotFoundError:
            print('Data files not found — normalization will be approximate')
            self.data_factory = None

        # Build model
        self.diffusion = GaussianDiffusion(T=DIFFUSION_T).to(self.device)
        self.denoiser = DenoiserNet(
            in_dim=16, cond_dim=18, n_filters=256,
            n_resblock=4, num_heads=4, T=DIFFUSION_T).to(self.device)

        if os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            state = ckpt.get('ema_state_dict') or ckpt.get('model_state_dict')
            if state:
                self.denoiser.load_state_dict(state, strict=False)
            print(f'Loaded checkpoint: {self.checkpoint_path}')
        else:
            print('WARNING: No checkpoint — using untrained model. Generate will be random.')

        self.denoiser.eval()
        self._loaded = True

    def generate(self, conds):
        """Run DDIM sampling."""
        self._ensure_loaded()
        conds_t = torch.from_numpy(conds).float().to(self.device)
        B, T_len, _ = conds_t.shape
        with torch.no_grad():
            generated = self.diffusion.sample(
                self.denoiser, conds_t, [B, T_len, 16], steps=DDIM_STEPS)
        return generated.cpu().numpy()


# Global model manager
_model = ModelManager()


# ---------------------------------------------------------------------------
#   Core pipeline: points → conditioning → diffusion → video
# ---------------------------------------------------------------------------

def points_to_video(points_npy_path, seed=0):
    """Full pipeline: loaded points → normalize → generate → render video."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    points = np.load(points_npy_path)
    if points.ndim == 1:
        raise ValueError('Points file must be 2D: [N, 12]')

    # Pad edges (matching original inference logic)
    front = np.tile(points[0], (2, 1))
    points = np.concatenate([front, points])
    extra = np.tile(points[-1], (4, 1))
    points = np.concatenate([points, extra])

    feature = draw_feat.get_feature(points)
    target_length = len(points)
    dims = 10

    points_batch = np.reshape(np.tile(points, (dims, 1)), [dims, target_length, 12])
    feature_batch = np.repeat(feature, dims, axis=0)

    # Build conditioning [dims, T, 18]
    team_AB = np.concatenate([
        points_batch[:, :, :2].reshape([dims, target_length, 2]),
        points_batch[:, :, 2:12].reshape([dims, target_length, 10]),
        feature_batch.reshape([dims, target_length, 6]),
    ], axis=-1)

    if _model.data_factory:
        team_AB = _model.data_factory.normalize(team_AB)

    team_A = team_AB[:, :, :12]
    team_Feat = team_AB[:, :, 12:]
    conds_full = np.concatenate([team_A, team_Feat], axis=-1)

    # Run generation for each condition variant
    results = []
    for idx in range(dims):
        conds = np.repeat(conds_full[idx:idx + 1], N_LATENT, axis=0)
        gen = _model.generate(conds)
        results.append(gen)

    results = np.stack(results, axis=1)  # [N_LATENT, dims, T, 16]

    # Select best sample (median score based on movement smoothness)
    scores = []
    for i in range(N_LATENT):
        sample_i = results[i, 0]  # first condition variant
        # Score: prefer plays with moderate defense movement (not static, not wild)
        diff = np.diff(sample_i[:, :10], axis=0)
        score = np.mean(np.abs(diff))
        scores.append(score)
    median_idx = int(np.argsort(scores)[len(scores) // 2])

    best_gen = results[median_idx, 0]  # [T, 16]

    # Denormalize
    if _model.data_factory:
        off_np = team_A[0]  # [T, 12] — normalized
        full_sample = np.concatenate([off_np, best_gen[:, :10]], axis=-1)  # [T, 22]
        full_sample = full_sample[None, :, :]  # [1, T, 22]
        full_sample = _model.data_factory.recover_BALL_and_A(full_sample)
        full_sample = _model.data_factory.recover_B(full_sample)
        full_sample = full_sample[0]  # [T, 22]
    else:
        full_sample = np.concatenate([team_A[0], best_gen[:, :10]], axis=-1)

    # Render video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = f.name

    game_visualizer.plot_data(
        full_sample[None, :, :], SEQ_LEN,
        file_path=video_path, if_save=True)

    return video_path, full_sample


# ---------------------------------------------------------------------------
#   Interactive coordinate input
# ---------------------------------------------------------------------------

class ClickCollector:
    """Collects court clicks and converts to points for the model pipeline."""

    def __init__(self):
        self.clicks = []  # list of (x_pixel, y_pixel)
        self.court_w = 94   # feet
        self.court_h = 50   # feet
        self.img_w = 600
        self.img_h = 315     # 600 * 50/94 ≈ 319

    def add_click(self, evt: gr.SelectData):
        x, y = evt.index[0], evt.index[1]
        # Convert pixel → court coordinates
        court_x = x / self.img_w * self.court_w
        court_y = y / self.img_h * self.court_h
        self.clicks.append((court_x, court_y))
        return self._format_clicks()

    def clear(self):
        self.clicks = []
        return ''

    def save_and_get_path(self, seed=0):
        """Convert clicks to points.npy format and save."""
        if len(self.clicks) < 3:
            return None, 'Need at least 3 points for a play'

        clicks = np.array(self.clicks)
        # Build 12-dim points: ball_xy + 5 dummy offence players
        # Use clicks as ball trajectory; place dummy offence players nearby
        N = len(clicks)
        points = np.zeros((N, 12), dtype=np.float32)
        points[:, 0:2] = clicks  # ball x, y

        # Place dummy players at initial positions with offsets
        for j in range(5):
            ox = clicks[0, 0] + (j - 2) * 5  # spread players horizontally
            oy = clicks[0, 1] + 3             # slightly above ball
            points[0, 2 + j * 2] = ox
            points[0, 2 + j * 2 + 1] = oy
        # Interpolate player positions to follow ball approximately
        for j in range(5):
            for t in range(1, N):
                frac = t / (N - 1)
                points[t, 2 + j * 2] = points[0, 2 + j * 2] * (1 - frac) + clicks[t, 0] * frac
                points[t, 2 + j * 2 + 1] = points[0, 2 + j * 2 + 1] * (1 - frac) + clicks[t, 1] * frac

        # Scale from court feet to pixel-like (original code divides by 10)
        points = points * 10

        # Save
        save_dir = os.path.join(_app_dir, 'Points')
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'points_clicks_{seed}.npy')
        np.save(path, points)
        return path, f'Saved {N} points'


# ---------------------------------------------------------------------------
#   Gradio UI
# ---------------------------------------------------------------------------

collector = ClickCollector()

def on_court_click(evt: gr.SelectData):
    return collector.add_click(evt)

def on_clear():
    return collector.clear()

def on_generate_clicks(seed):
    path, msg = collector.save_and_get_path(seed)
    if path is None:
        return None, msg
    video_path, _ = points_to_video(path, seed)
    return video_path, msg

def on_generate_file(points_file, seed):
    if points_file is None:
        return None, 'Please upload a points .npy file'
    path = points_file.name if hasattr(points_file, 'name') else points_file
    video_path, _ = points_to_video(path, seed)
    return video_path, 'Done!'


def create_ui():
    with gr.Blocks(title='Basketball Play Generator', theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏀 Basketball Play Generator
        ### AI-powered defensive play simulation from offensive sketches
        """)

        with gr.Tabs():
            # ---- Tab 1: Upload .npy points file ----
            with gr.TabItem('📁 Upload Points File'):
                with gr.Row():
                    with gr.Column(scale=1):
                        points_input = gr.File(
                            label='Upload points.npy',
                            file_types=['.npy'])
                        seed1 = gr.Slider(0, 100, value=0, step=1, label='Random Seed')
                        gen_btn1 = gr.Button('Generate Defense', variant='primary', size='lg')
                        status1 = gr.Textbox(label='Status', interactive=False)

                    with gr.Column(scale=1):
                        video_output1 = gr.Video(label='Generated Play', height=420)

                gen_btn1.click(
                    on_generate_file,
                    inputs=[points_input, seed1],
                    outputs=[video_output1, status1])

                gr.Markdown("""
                **How to use:**
                1. Create a play sketch using a `.npy` file (shape `[N, 12]`: ball_xy + 5 offense player xy)
                2. Upload here and click Generate
                3. Watch the AI generate realistic defensive responses
                """)

            # ---- Tab 2: Interactive Point Placement ----
            with gr.TabItem('🖊️ Interactive Sketch'):
                with gr.Row():
                    with gr.Column(scale=1):
                        court_display = gr.Image(
                            value=COURT_IMAGE if os.path.exists(COURT_IMAGE) else None,
                            label='Click on court to place ball trajectory points',
                            type='filepath', height=350)
                        with gr.Row():
                            clear_btn = gr.Button('Clear Points')
                            seed2 = gr.Slider(0, 100, value=0, step=1, label='Seed')
                        points_display = gr.Textbox(
                            label='Placed Points (court coordinates)',
                            lines=6, interactive=False)
                        gen_btn2 = gr.Button('Generate Defense', variant='primary', size='lg')

                    with gr.Column(scale=1):
                        video_output2 = gr.Video(label='Generated Play', height=420)

                court_display.select(on_court_click, outputs=[points_display])
                clear_btn.click(on_clear, outputs=[points_display])
                gen_btn2.click(
                    on_generate_clicks,
                    inputs=[seed2],
                    outputs=[video_output2, points_display])

                gr.Markdown("""
                **How to use interactive mode:**
                1. Click on the court to place ball trajectory waypoints
                2. At least 3 points needed (more = smoother trajectory)
                3. Click **Generate Defense** to see the AI's defensive response
                4. Players are placed automatically around the ball starting position
                """)

            # ---- Tab 3: Info ----
            with gr.TabItem('ℹ️ About'):
                gr.Markdown("""
                ### How it works

                This uses a **Diffusion Model** (DDPM + DDIM) to generate basketball defensive plays.

                1. **Input**: Offensive play sketch (ball trajectory + player positions)
                2. **Model**: Denoising Diffusion Probabilistic Model conditions on the offense
                3. **Output**: Realistic defensive player movements

                ### Model Details
                - **Architecture**: DenoiserNet with ResBlocks + Self-Attention + Cross-Attention
                - **Diffusion Steps**: 1000 training, 50 DDIM sampling
                - **Conditioning**: 18 dims (offence positions + ball-possession features)
                - **Generated**: 16 dims (defence positions + ball features)

                ### Checkpoint
                Place your trained checkpoint at `ui/Data/checkpoints/model_epoch500.pt`
                or use `python ui/app.py --checkpoint <path>`.
                """)

    return demo


# ---------------------------------------------------------------------------
#   Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Basketball Play Generator Web UI')
    parser.add_argument('--checkpoint', type=str, default=MODEL_PATH,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--share', action='store_true',
                        help='Create a public Gradio link')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()

    # Update model path
    global _model
    _model = ModelManager(checkpoint_path=args.checkpoint)

    demo = create_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
