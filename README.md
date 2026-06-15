# BasketballGAN

### Generate defensive basketball play simulations from offensive sketches.

A VAE-GAN model that generates realistic defensive player movements conditioned on an offensive play sequence, using three WGAN-GP discriminators.

## Prerequisites

- Linux / macOS
- NVIDIA GPU (for training)
- Python 3.8+

## Quick Start

```bash
# 1. Clone
git clone https://github.com/TQG1997/basketball.git
cd basketball

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Google Drive and place .npy files under data/
#    https://drive.google.com/drive/folders/1uNPw7LOA3xENclQRtSlUftiR7tlVNOts

# 4. Train
python src/Train_Triple.py --folder_path='output' --data_path='data'
```

Checkpoints and sample animations are saved under the `--folder_path` directory.

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TQG1997/basketball/blob/main/notebooks/train.ipynb)

Use **Runtime → Change runtime type → T4 GPU**. The notebook handles cloning, dependency installation, dataset download, Drive mounting, and training in one go.

> Set `--max_epochs` to avoid exhausting the Colab session limit (~12h for free tier).

## UI Application

Interactive PyQt5 desktop app for sketching offensive plays and visualizing generated defensive responses:

```bash
cd ui
python Main.py
```

Requires a pre-trained model checkpoint placed at `ui/Data/checkpoints/`.

## Project Structure

```
Basketball/
├── src/                    # VAE-GAN model + training pipeline
│   ├── Train_Triple.py     # Training entry point
│   ├── ThreeDiscrim.py     # VAE-GAN model (encoder, generator, 3 discriminators)
│   ├── ops.py              # Spectral norm, conv1d, residual blocks
│   ├── utils.py            # DataFactory re-export
│   └── game_visualizer.py  # Play animation (matplotlib)
├── ui/                     # PyQt5 interactive sketch-to-play app
│   ├── Main.py             # Main window
│   ├── Drawingboard.py     # Sketch input (ball + 5 offence players)
│   ├── Court.py            # Court playback (matplotlib canvas)
│   ├── WGAN.py             # Model loading & inference
│   ├── SavePos.py          # Bezier curve trajectory smoothing
│   ├── draw_feat.py        # Ball-possession feature extraction
│   └── ...
├── shared/                 # Shared code (DataFactory singleton)
├── DataTranslater/         # Data conversion utilities
├── notebooks/              # Colab training notebook
├── data/                   # Dataset (.npy files, excluded from git)
└── requirements.txt
```

## Dataset

### Files

| File | Shape | Type | Size | Description |
|---|---|---|---|---|
| `50Real.npy` | (14032, 50, 11, 4) | float64 | 236MB | Ground truth plays: ball + player positions (50 timesteps) |
| `50Seq.npy` | (14032, 50, 12) | float64 | 64MB | Offence conditioning (ball + 5 offence players x,y) |
| `FEATURES-4.npy` | (11863, 100, 11, 4) | float64 | 398MB | Full-length ground truth (100 timesteps) |
| `RealCond.npy` | (14032, 50, 6) | int32 | 16MB | Ball status features for ground truth plays |
| `SeqCond.npy` | (14032, 50, 6) | int32 | 16MB | Ball status features for conditioning sequences |

### Feature Layout

**50Real.npy** — 4D tensor `[sample, timestep, entity, feature]`:

```
entity 0:     ball           (x, y, z, flag)
entity 1-5:   offence A1-A5  (x, y, z, flag)
entity 6-10:  defence B1-B5  (x, y, z, flag)
```

- `x, y`: court coordinates (normalised during training)
- `z`: ball height (ball only; player z is 0)
- `flag`: `1` for ball, `0` for players

**50Seq.npy** — 3D tensor `[sample, timestep, feature]`, each timestep: `[ball.x, ball.y, A1.x, A1.y, A2.x, A2.y, A3.x, A3.y, A4.x, A4.y, A5.x, A5.y]`

**RealCond.npy / SeqCond.npy** — 3D tensor `[sample, timestep, feature]`, one-hot ball-possession: `[dribble_by_A1, ..., dribble_by_A5, pass]`

### Source

Download from [Google Drive](https://drive.google.com/drive/folders/1uNPw7LOA3xENclQRtSlUftiR7tlVNOts?usp=share_link). Data consists of NBA play-by-play tracking processed into fixed-length game segments, split 9:1 into train/validation by `DataFactory`.

## Training Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--folder_path` | — | Output directory (checkpoints, samples, logs) |
| `--data_path` | — | Directory containing .npy dataset files |
| `--batch_size` | 128 | Batch size (reduce if OOM) |
| `--max_epochs` | None | Stop after N epochs (None = run forever) |
| `--latent_dims` | 150 | Latent z dimension |
| `--lr_` | 1e-4 | Adam learning rate |
| `--beta` | 0.001 | KL divergence weight (β-VAE) |
| `--recon_weight` | 1.0 | L1 reconstruction loss weight |
| `--n_filters` | 256 | Conv filters per layer |
| `--n_resblock` | 8 | Residual blocks per network |
| `--checkpoint_step` | 100 | Epochs between checkpoint saves |
| `--vis_freq` | 5 | Epochs between sample visualizations |
