"""Basketball play generation — Gradio web interface.

Mimics the original PyQt5 UX: visual court with players + ball trajectory,
click-based interaction, side-by-side sketch & simulation.

Usage:
    python ui/app.py                     # http://127.0.0.1:7860
    python ui/app.py --share             # public link
"""

import os, sys, argparse, tempfile, io
import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

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


# ---- Paths & constants ----
MODEL_PATH = os.path.join(_app_dir, 'Data', 'checkpoints', 'model_epoch500.pt')
DATA_DIR = os.path.join(_app_dir, 'Data', 'Model_data')
BACKGROUND = os.path.join(_app_dir, 'images', 'court.png')
DDIM_STEPS, DIFFUSION_T, N_LATENT, SEQ_LEN = 50, 1000, 100, 50

# Court bounds (half-court right side, in feet)
CX_MIN, CX_MAX = 47, 94
CY_MIN, CY_MAX = 0, 50
COURT_W, COURT_H = CX_MAX - CX_MIN, CY_MAX - CY_MIN

# Player names and default positions (spread offense)
PLAYER_DEFAULTS = {
    'PG': (60, 30), 'SG': (70, 35), 'SF': (65, 20),
    'PF': (55, 15), 'C':  (50, 25),
}
PLAYER_ORDER = ['PG', 'SG', 'SF', 'PF', 'C']
PLAYER_COLORS = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']
BASKET_POS = (89, 25)   # basket position in court coords
BALL_COLOR = '#f1c40f'


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
        try:
            real_data = np.load(os.path.join(DATA_DIR, '50Real.npy'))[:, :50, :, :]
            seq_data = np.load(os.path.join(DATA_DIR, '50Seq.npy'))
            features_ = np.load(os.path.join(DATA_DIR, 'SeqCond.npy'))
            real_feat = np.load(os.path.join(DATA_DIR, 'RealCond.npy'))
            self.data_factory = DataFactory(real_data, seq_data, features_, real_feat)
        except FileNotFoundError:
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
        self.denoiser.eval()
        self._loaded = True

    def generate(self, conds):
        self._ensure_loaded()
        conds_t = torch.from_numpy(conds).float().to(self.device)
        B, T, _ = conds_t.shape
        with torch.no_grad():
            return self.diffusion.sample(self.denoiser, conds_t, [B, T, 16], steps=DDIM_STEPS).cpu().numpy()


_model = ModelManager()


# ---------------------------------------------------------------------------
#   Play State
# ---------------------------------------------------------------------------

class PlayState:
    """Holds current offensive formation + ball path."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.players = dict(PLAYER_DEFAULTS)  # {name: (x, y)}
        self.ball_path = []                    # [(x, y), ...]

    @property
    def ball_path_count(self):
        return len(self.ball_path)


_state = PlayState()


# ---------------------------------------------------------------------------
#   Court Rendering (matplotlib → PNG)
# ---------------------------------------------------------------------------

def render_court():
    """Render court with players and ball trajectory as a PNG buffer."""
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=100)
    ax.set_xlim(CX_MIN, CX_MAX)
    ax.set_ylim(CY_MIN, CY_MAX)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Court background
    if os.path.exists(BACKGROUND):
        bg = plt.imread(BACKGROUND)
        ax.imshow(bg, extent=[0, 94, 50, 0], aspect='auto', zorder=0)
    ax.set_xlim(CX_MIN, CX_MAX)

    # Basket
    ax.add_patch(Circle(BASKET_POS, 1.0, color='orange', ec='darkorange', lw=2, zorder=10))
    ax.annotate('🏀', BASKET_POS, ha='center', va='center', fontsize=8, zorder=11)

    # Players
    for i, name in enumerate(PLAYER_ORDER):
        x, y = _state.players[name]
        ax.add_patch(Circle((x, y), 1.8, color=PLAYER_COLORS[i], ec='white', lw=1.5, zorder=10))
        ax.annotate(name, (x, y), ha='center', va='center', fontsize=6,
                    fontweight='bold', color='white', zorder=11)

    # Ball trajectory
    if len(_state.ball_path) >= 2:
        pts = np.array(_state.ball_path)
        ax.plot(pts[:, 0], pts[:, 1], '-', color=BALL_COLOR, lw=2, zorder=8, alpha=0.8)
    if _state.ball_path:
        bx, by = _state.ball_path[-1]
        ax.add_patch(Circle((bx, by), 0.9, color=BALL_COLOR, ec='#e67e22', lw=1, zorder=12))
        ax.add_patch(Circle((bx, by), 0.5, color='#f39c12', ec='none', zorder=13))

    # Legend
    ax.text(48, 2, f'🏀 {len(_state.ball_path)} points', fontsize=7, color='#555',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#ddd', alpha=0.9))

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
#   Interaction Logic
# ---------------------------------------------------------------------------

def handle_court_click(evt: gr.SelectData):
    """Click: if near a player → move them; else → add ball point."""
    # Convert pixel → court coords
    img_w, img_h = 600, 320
    cx = evt.index[0] / img_w * COURT_W + CX_MIN
    cy = evt.index[1] / img_h * COURT_H + CY_MIN

    # Check if near a player (within 5 feet)
    for name in PLAYER_ORDER:
        px, py = _state.players[name]
        if np.sqrt((cx - px)**2 + (cy - py)**2) < 4:
            _state.players[name] = (cx, cy)
            return render_court(), _status_text()

    # Otherwise add ball point
    _state.ball_path.append((cx, cy))
    return render_court(), _status_text()


def reset_all():
    _state.reset()
    return render_court(), _status_text(), None, None, 'Ready — click court to place players & ball path'


def _status_text():
    lines = []
    for name in PLAYER_ORDER:
        x, y = _state.players[name]
        lines.append(f'**{name}**: ({x:.0f}, {y:.0f})')
    lines.append(f'**🏀 Ball**: {_state.ball_path_count} points')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
#   Generation Pipeline
# ---------------------------------------------------------------------------

_last_sim_video = None
_last_sketch_video = None


def render_sketch_video():
    """Render current offensive sketch as animation."""
    if _state.ball_path_count < 3:
        return None
    pts = np.zeros((_state.ball_path_count, 22), dtype=np.float32)
    for t, (bx, by) in enumerate(_state.ball_path):
        pts[t, 0] = bx
        pts[t, 1] = by
        for j, name in enumerate(PLAYER_ORDER):
            px, py = _state.players[name]
            frac = t / max(_state.ball_path_count - 1, 1)
            pts[t, 2 + j * 2] = px * (1 - frac * 0.3)
            pts[t, 2 + j * 2 + 1] = py * (1 - frac * 0.3)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        vpath = f.name
    game_visualizer.plot_data(pts[None, :, :], SEQ_LEN, file_path=vpath, if_save=True)
    return vpath


def generate(seed):
    global _last_sim_video, _last_sketch_video
    if _state.ball_path_count < 3:
        return None, None, 'Need ≥3 ball path points'

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build points array [N, 12]
    N = _state.ball_path_count
    points = np.zeros((N, 12), dtype=np.float32)
    clicks = np.array(_state.ball_path)
    points[:, 0:2] = clicks
    for j, name in enumerate(PLAYER_ORDER):
        px, py = _state.players[name]
        for t in range(N):
            frac = t / max(N - 1, 1)
            points[t, 2 + j * 2] = px * (1 - frac * 0.3) + clicks[t, 0] * frac * 0.3
            points[t, 2 + j * 2 + 1] = py * (1 - frac * 0.3) + clicks[t, 1] * frac * 0.3
    points *= 10

    # Save + render sketch
    os.makedirs(os.path.join(_app_dir, 'Points'), exist_ok=True)
    np.save(os.path.join(_app_dir, 'Points', 'current_play.npy'), points)
    _last_sketch_video = render_sketch_video()

    # Pad like original
    front = np.tile(points[0], (2, 1))
    points = np.concatenate([front, points])
    points = np.concatenate([points, np.tile(points[-1], (4, 1))])

    feature = draw_feat.get_feature(points)
    T_len, dims = len(points), 10
    points_batch = np.reshape(np.tile(points, (dims, 1)), [dims, T_len, 12])
    feature_batch = np.repeat(feature, dims, axis=0)

    team_AB = np.concatenate([
        points_batch[:, :, :2].reshape([dims, T_len, 2]),
        points_batch[:, :, 2:12].reshape([dims, T_len, 10]),
        feature_batch.reshape([dims, T_len, 6]),
    ], axis=-1)
    if _model.data_factory:
        team_AB = _model.data_factory.normalize(team_AB)

    conds_full = np.concatenate([team_AB[:, :, :12], team_AB[:, :, 12:]], axis=-1)

    results = []
    for idx in range(dims):
        conds = np.repeat(conds_full[idx:idx + 1], N_LATENT, axis=0)
        results.append(_model.generate(conds))
    results = np.stack(results, axis=1)

    scores = [np.mean(np.abs(np.diff(results[i, 0, :, :10], axis=0))) for i in range(N_LATENT)]
    best = results[int(np.argsort(scores)[len(scores) // 2]), 0]

    if _model.data_factory:
        off = team_AB[0, :, :12]
        full = np.concatenate([off, best[:, :10]], axis=-1)[None, :, :]
        full = _model.data_factory.recover_BALL_and_A(full)
        full = _model.data_factory.recover_B(full)[0]
    else:
        full = np.concatenate([team_AB[0, :, :12], best[:, :10]], axis=-1)

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        vpath = f.name
    game_visualizer.plot_data(full[None, :, :], SEQ_LEN, file_path=vpath, if_save=True)
    _last_sim_video = vpath
    return _last_sim_video, _last_sketch_video, '✅ Done — generated defense play'


def switch_view(mode):
    if 'Simulation' in mode:
        return _last_sim_video, f'Showing: {mode}'
    else:
        return _last_sketch_video, f'Showing: {mode}'


# ---------------------------------------------------------------------------
#   UI
# ---------------------------------------------------------------------------

def create_ui():
    css = """
    footer { display: none !important; }
    .gradio-container { max-width: 1200px !important; }
    """
    with gr.Blocks(title='Basketball Play Generator', theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown("""
        # 🏀 Basketball Play Generator
        ### Click court to move players or draw ball path → Generate AI defense
        """)

        with gr.Row(equal_height=True):
            # ===== LEFT: Court =====
            with gr.Column(scale=5):
                court_display = gr.Image(
                    value=render_court(),
                    label='🖱️ Click near a player to move · Click empty space to draw ball path',
                    type='filepath', height=360, show_label=True)

                with gr.Row():
                    gr.Markdown(_status_text(), every=0.1, elem_id='status-md')
                    # Actually need a component for status
                    status_md = gr.Markdown(_status_text())

            # ===== RIGHT: Result =====
            with gr.Column(scale=5):
                video_out = gr.Video(label='Generated Play', height=380)
                status_out = gr.Textbox(label='Status', value='Ready — click court to begin', interactive=False)

        with gr.Row():
            clear_btn = gr.Button('🔄 Clear All', variant='secondary', size='lg')
            seed_slider = gr.Slider(0, 200, value=0, step=1, label='🎲 Seed')
            gen_btn = gr.Button('⚡ Generate Defense', variant='primary', size='lg')

        with gr.Row():
            view_mode = gr.Radio(
                ['🎬 Generated Simulation', '✏️ Sketch Animation'],
                value='🎬 Generated Simulation', label='View Mode')

        # ---- Events ----
        court_display.select(
            handle_court_click,
            outputs=[court_display, status_md])

        clear_btn.click(
            reset_all,
            outputs=[court_display, status_md, video_out, video_out, status_out])

        gen_btn.click(
            generate,
            inputs=[seed_slider],
            outputs=[video_out, video_out, status_out])

        view_mode.change(
            switch_view,
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
