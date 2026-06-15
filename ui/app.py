"""Basketball play generation — Gradio web interface.

Usage:
    python ui/app.py                     # http://127.0.0.1:7860
    python ui/app.py --share             # public link
    python ui/app.py --checkpoint path   # custom model checkpoint
"""

import os
import sys
import argparse
import tempfile
import numpy as np
import torch

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


MODEL_PATH = os.path.join(_app_dir, 'Data', 'checkpoints', 'model_epoch500.pt')
DATA_DIR = os.path.join(_app_dir, 'Data', 'Model_data')
COURT_IMAGE = os.path.join(_app_dir, 'images', 'court.png')

DDIM_STEPS = 50
DIFFUSION_T = 1000
N_LATENT = 100
SEQ_LEN = 50

# Court dimensions (feet — half-court right side)
COURT_X_MIN, COURT_X_MAX = 47, 94
COURT_Y_MIN, COURT_Y_MAX = 0, 50

# Default offensive positions (spread formation at half-court right side)
DEFAULT_PLAYERS = {
    'PG': (60, 30),   # Point Guard — top of key area
    'SG': (70, 35),   # Shooting Guard — right wing
    'SF': (65, 20),   # Small Forward — left wing
    'PF': (55, 15),   # Power Forward — left post
    'C':  (50, 25),   # Center — paint area
}


# ---------------------------------------------------------------------------
#   Model Manager
# ---------------------------------------------------------------------------

class ModelManager:
    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path or MODEL_PATH
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {self.device}')
        try:
            real_data = np.load(os.path.join(DATA_DIR, '50Real.npy'))[:, :50, :, :]
            seq_data = np.load(os.path.join(DATA_DIR, '50Seq.npy'))
            features_ = np.load(os.path.join(DATA_DIR, 'SeqCond.npy'))
            real_feat = np.load(os.path.join(DATA_DIR, 'RealCond.npy'))
            self.data_factory = DataFactory(real_data, seq_data, features_, real_feat)
        except FileNotFoundError:
            print('Data files not found — normalization skipped')
            self.data_factory = None
        self.diffusion = GaussianDiffusion(T=DIFFUSION_T).to(self.device)
        self.denoiser = DenoiserNet(
            in_dim=16, cond_dim=18, n_filters=256,
            n_resblock=4, num_heads=4, T=DIFFUSION_T).to(self.device)
        if os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            state = ckpt.get('ema_state_dict') or ckpt.get('model_state_dict')
            if state:
                self.denoiser.load_state_dict(state, strict=False)
            print(f'Loaded: {self.checkpoint_path}')
        else:
            print('WARNING: no checkpoint — random weights')
        self.denoiser.eval()
        self._loaded = True

    def generate(self, conds):
        self._ensure_loaded()
        conds_t = torch.from_numpy(conds).float().to(self.device)
        B, T, _ = conds_t.shape
        with torch.no_grad():
            gen = self.diffusion.sample(self.denoiser, conds_t, [B, T, 16], steps=DDIM_STEPS)
        return gen.cpu().numpy()


_model = ModelManager()


# ---------------------------------------------------------------------------
#   Pipeline: player positions + ball clicks → video
# ---------------------------------------------------------------------------

class PlayDesigner:
    """Manages offensive play state: 5 players + ball trajectory."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.player_positions = dict(DEFAULT_PLAYERS)
        self.ball_clicks = []  # list of (x_court, y_court)

    def set_player(self, name, x, y=None):
        if y is None:
            y = self.player_positions[name][1]
        self.player_positions[name] = (float(x), float(y))

    def add_ball_click(self, court_x, court_y):
        self.ball_clicks.append((court_x, court_y))

    def clear_ball(self):
        self.ball_clicks = []

    def build_points(self):
        """Convert player positions + ball clicks → points.npy format.

        Returns (points_array, status_msg). points_array is [N, 12] where
        12 = ball_xy + PG_xy + SG_xy + SF_xy + PF_xy + C_xy.
        """
        if len(self.ball_clicks) < 3:
            return None, f'Need ≥3 ball trajectory points (have {len(self.ball_clicks)})'

        clicks = np.array(self.ball_clicks)
        N = len(clicks)
        points = np.zeros((N, 12), dtype=np.float32)

        # Ball trajectory
        points[:, 0:2] = clicks

        # Player paths: interpolate from initial positions toward ball
        player_names = ['PG', 'SG', 'SF', 'PF', 'C']
        for j, name in enumerate(player_names):
            px, py = self.player_positions[name]
            for t in range(N):
                frac = t / max(N - 1, 1)
                # Players move partially toward ball's current position
                target_x = clicks[t, 0] * 0.3 + px * 0.7
                target_y = clicks[t, 1] * 0.3 + py * 0.7
                points[t, 2 + j * 2] = target_x
                points[t, 2 + j * 2 + 1] = target_y

        points = points * 10  # scale to original coordinate system
        save_dir = os.path.join(_app_dir, 'Points')
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'current_play.npy')
        np.save(path, points)
        return path, f'OK: {N} frames, 5 players + ball'


_designer = PlayDesigner()
_last_generated_video = None
_last_sketch_video = None


def render_sketch_video():
    """Render current offensive sketch as animation."""
    path, _ = _designer.build_points()
    if path is None:
        return None
    points = np.load(path) / 10  # undo ×10 scale for display
    # Build full 22-dim: ball_xy + 5 players xy
    full = points[:, :12]  # [N, 12]
    # Pad to 22 dims (defense = zeros)
    padded = np.zeros((len(full), 22), dtype=np.float32)
    padded[:, :12] = full
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        vpath = f.name
    game_visualizer.plot_data(padded[None, :, :], SEQ_LEN, file_path=vpath, if_save=True)
    return vpath


def run_generate(seed):
    """Full generate pipeline. Returns (sim_video, sketch_video, status)."""
    global _last_generated_video, _last_sketch_video
    path, msg = _designer.build_points()
    if path is None:
        return None, None, msg

    # Render sketch first
    _last_sketch_video = render_sketch_video()

    torch.manual_seed(seed)
    np.random.seed(seed)

    points = np.load(path)
    front = np.tile(points[0], (2, 1))
    points = np.concatenate([front, points])
    extra = np.tile(points[-1], (4, 1))
    points = np.concatenate([points, extra])

    feature = draw_feat.get_feature(points)
    T_len = len(points)
    dims = 10
    points_batch = np.reshape(np.tile(points, (dims, 1)), [dims, T_len, 12])
    feature_batch = np.repeat(feature, dims, axis=0)

    team_AB = np.concatenate([
        points_batch[:, :, :2].reshape([dims, T_len, 2]),
        points_batch[:, :, 2:12].reshape([dims, T_len, 10]),
        feature_batch.reshape([dims, T_len, 6]),
    ], axis=-1)
    if _model.data_factory:
        team_AB = _model.data_factory.normalize(team_AB)
    team_A = team_AB[:, :, :12]
    team_Feat = team_AB[:, :, 12:]
    conds_full = np.concatenate([team_A, team_Feat], axis=-1)

    results = []
    for idx in range(dims):
        conds = np.repeat(conds_full[idx:idx + 1], N_LATENT, axis=0)
        gen = _model.generate(conds)
        results.append(gen)
    results = np.stack(results, axis=1)

    # Median score pick
    scores = [np.mean(np.abs(np.diff(results[i, 0, :, :10], axis=0)))
              for i in range(N_LATENT)]
    best_gen = results[int(np.argsort(scores)[len(scores) // 2]), 0]

    if _model.data_factory:
        off_np = team_A[0]
        full = np.concatenate([off_np, best_gen[:, :10]], axis=-1)[None, :, :]
        full = _model.data_factory.recover_BALL_and_A(full)
        full = _model.data_factory.recover_B(full)
        full = full[0]
    else:
        full = np.concatenate([team_A[0], best_gen[:, :10]], axis=-1)

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        vpath = f.name
    game_visualizer.plot_data(full[None, :, :], SEQ_LEN, file_path=vpath, if_save=True)
    _last_generated_video = vpath
    return _last_generated_video, _last_sketch_video, '✅ Generation complete'


# ---------------------------------------------------------------------------
#   Gradio event handlers
# ---------------------------------------------------------------------------

def on_court_click(evt: gr.SelectData):
    """Convert pixel click → court coordinates."""
    img_w, img_h = 600, 315
    cx = evt.index[0] / img_w * (COURT_X_MAX - COURT_X_MIN) + COURT_X_MIN
    cy = evt.index[1] / img_h * (COURT_Y_MAX - COURT_Y_MIN) + COURT_Y_MIN
    _designer.add_ball_click(cx, cy)
    return _format_state()

def on_clear():
    _designer.reset()
    return _format_state()

def on_generate(seed):
    sim_video, sketch_video, status = run_generate(seed)
    if sim_video is None:
        return None, None, status
    return sim_video, sketch_video, status

def on_view_mode_change(mode):
    """Switch between generated simulation and sketch animation."""
    if 'Simulation' in mode:
        return _last_generated_video, 'Showing generated defensive play'
    else:
        return _last_sketch_video, 'Showing offensive sketch'

def _format_state():
    lines = []
    lines.append(f'**🏀 Ball path:** {len(_designer.ball_clicks)} points')
    if _designer.ball_clicks:
        pts = ', '.join(f'({x:.0f},{y:.0f})' for x, y in _designer.ball_clicks[-5:])
        lines.append(f'Last 5: {pts}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
#   UI Layout
# ---------------------------------------------------------------------------

def create_ui():
    css = """
    .status-box { font-size: 0.9em; padding: 8px; background: #f0f0f0; border-radius: 6px; }
    footer { display: none !important; }
    #court-container { text-align: center; }
    """
    with gr.Blocks(title='Basketball Play Generator', theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown("""
        # 🏀 Basketball Play Generator
        ### Set offensive formation → Draw ball path → Generate defensive response
        """)

        # ---- Row 1: Player positions (compact) ----
        with gr.Row():
            with gr.Column(scale=2):
                pg_x = gr.Slider(47, 94, value=DEFAULT_PLAYERS['PG'][0], step=1, label='PG (Point Guard)')
            with gr.Column(scale=1):
                pg_y = gr.Slider(0, 50, value=DEFAULT_PLAYERS['PG'][1], step=1, label='')
        with gr.Row():
            with gr.Column(scale=2):
                sg_x = gr.Slider(47, 94, value=DEFAULT_PLAYERS['SG'][0], step=1, label='SG (Shooting Guard)')
            with gr.Column(scale=1):
                sg_y = gr.Slider(0, 50, value=DEFAULT_PLAYERS['SG'][1], step=1, label='')
        with gr.Row():
            with gr.Column(scale=2):
                sf_x = gr.Slider(47, 94, value=DEFAULT_PLAYERS['SF'][0], step=1, label='SF (Small Forward)')
            with gr.Column(scale=1):
                sf_y = gr.Slider(0, 50, value=DEFAULT_PLAYERS['SF'][1], step=1, label='')
        with gr.Row():
            with gr.Column(scale=2):
                pf_x = gr.Slider(47, 94, value=DEFAULT_PLAYERS['PF'][0], step=1, label='PF (Power Forward)')
            with gr.Column(scale=1):
                pf_y = gr.Slider(0, 50, value=DEFAULT_PLAYERS['PF'][1], step=1, label='')
        with gr.Row():
            with gr.Column(scale=2):
                c_x = gr.Slider(47, 94, value=DEFAULT_PLAYERS['C'][0], step=1, label='C (Center)')
            with gr.Column(scale=1):
                c_y = gr.Slider(0, 50, value=DEFAULT_PLAYERS['C'][1], step=1, label='')

        # ---- Row 2: Large centered court ----
        gr.Markdown("### 🏀 Ball Trajectory — click on court to draw")
        court_img = gr.Image(
            value=COURT_IMAGE if os.path.exists(COURT_IMAGE) else None,
            label='', type='filepath', height=420, show_label=False,
            elem_id='court-container')

        with gr.Row():
            state_display = gr.Markdown(_format_state(), elem_classes=['status-box'])

        # ---- Row 3: Controls ----
        with gr.Row():
            clear_btn = gr.Button('🔄 Reset All', variant='secondary', size='lg')
            seed_slider = gr.Slider(0, 200, value=0, step=1, label='🎲 Seed')
            gen_btn = gr.Button('⚡ Generate Defense', variant='primary', size='lg')

        # ---- Row 4: Result (2 modes like old PyQt5) ----
        gr.Markdown("---")
        with gr.Row():
            view_mode = gr.Radio(
                choices=['🎬 Generated Simulation', '✏️ Sketch Animation'],
                value='🎬 Generated Simulation', label='View Mode',
                interactive=True)

        video_out = gr.Video(label='', height=460)
        status_out = gr.Textbox(
            label='Status',
            value='Ready — adjust players, click court to draw ball path (≥3 points), then Generate',
            interactive=False)

        # ---- Event bindings ----
        court_img.select(on_court_click, outputs=[state_display])

        all_sliders = [pg_x, pg_y, sg_x, sg_y, sf_x, sf_y, pf_x, pf_y, c_x, c_y]
        player_names = ['PG', 'PG', 'SG', 'SG', 'SF', 'SF', 'PF', 'PF', 'C', 'C']
        for i, slider in enumerate(all_sliders):
            name = player_names[i]
            is_x = (i % 2 == 0)
            def make_update(n=name, ix=is_x):
                def update(v):
                    if ix:
                        _designer.set_player(n, v)
                    else:
                        _designer.set_player(n, _designer.player_positions[n][0], v)
                return update
            slider.change(make_update(), inputs=[slider])

        sketch_video = gr.Video(label='', height=460, visible=False)

        def on_gen_wrapper(seed):
            sim_video, sketch_v, status = on_generate(seed)
            return sim_video, sketch_v, status

        clear_btn.click(on_clear, outputs=[state_display])
        gen_btn.click(
            on_gen_wrapper,
            inputs=[seed_slider],
            outputs=[video_out, sketch_video, status_out])
        view_mode.change(
            on_view_mode_change,
            inputs=[view_mode],
            outputs=[video_out, status_out])

    return demo


def main():
    parser = argparse.ArgumentParser(description='Basketball Play Generator Web UI')
    parser.add_argument('--checkpoint', type=str, default=MODEL_PATH)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    global _model
    _model = ModelManager(checkpoint_path=args.checkpoint)
    create_ui().launch(server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
