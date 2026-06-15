# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

BasketballGAN is a research project (ACMMM 2019 paper) that generates defensive basketball play simulations given an offensive play sketch. It uses a **VAE-GAN** with three WGAN-GP discriminators. There are two entry points: a training pipeline (`src/`) and an interactive PyQt5 sketching UI (`ui/`).

## Commands

```bash
# Training (requires TF1 compat, GPU, and the dataset .npy files in data/)
cd src && python Train_Triple.py --folder_path='tmp' --data_path='../data'

# Interactive UI (requires pre-trained checkpoint at ui/Data/checkpoints/)
cd ui && python Main.py

# Training via Docker (original setup)
docker run --runtime=nvidia -it --rm -v $PWD:$PWD --net host nvcr.io/nvidia/tensorflow:19.06-py2 bash
```

There are no tests, no linting, and no CI in this repository.

## Architecture

### TensorFlow constraint

The entire codebase uses **TensorFlow 1.x compat mode**. Every `.py` file that imports TF begins with:
```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```
All graph construction uses `tf.placeholder`, `tf.get_variable`, `tf.Session`, etc. Do not use TF2 eager APIs or Keras layers when modifying model code.

### VAE-GAN model (`src/ThreeDiscrim.py`)

The core model (`VAEGAN_Model`) has four sub-networks:

- **Encoder (E_)**: Takes `(offence conditioning + ground-truth play)` → `z_mean, z_log_var`. Architecture: concat → conv1d → N residual blocks → global temporal pooling → dense layers.
- **Generator/Decoder (G_)**: Takes `(offence conditioning + latent z)` → fake play `[batch, seq_len, 28]`. The 28 output channels are: 22 player positions (ball xy + 10 players xy) + 6 ball-status features (sigmoid).
- **Three Discriminators** (shared `discriminator` method, each in its own variable scope):
  - `O_disc`: Offence discriminator — conditioned on ground-truth defence
  - `D_disc`: Defence discriminator — conditioned on ground-truth offence
  - `P_disc`: Full-play discriminator — conditioned on offence sequence + features

All use **spectral normalization** (via `ops.spectral_norm`) and **WGAN-GP gradient penalty** (`loss_d`, λ=10).

**VAE loss**: L1 reconstruction + β-KL divergence (weighted by `--beta`, default 0.001).

**Domain-specific penalties** (added to generator loss, scaled by |g_mean_cost|):
- `dribbler_penalty` — distance between ball and closest offensive player near the basket
- `_open_shot_penalty` — measures how "open" an offensive player is for a shot
- `_pass_ball_penalty` — ball trajectory smoothness during passes
- `_acc_penalty` — player acceleration consistency with real data

Training loop: pretrain D for N epochs → alternating updates (train D `train_D` times, then train G once). The generator optimizer updates **both encoder and decoder variables** jointly.

### Data pipeline (`src/utils.py`)

`DataFactory` is a **singleton**. On first instantiation with raw `.npy` arrays, it:
1. Normalizes x/y/z positions (z-normalization, stores mean/std for recovery)
2. Splits data 9:1 into train/valid (no shuffle before split — assumes random ordering)
3. Extracts team A (ball + offence) and team B (defence) subsets
4. Provides `shuffle_train()` / `shuffle_valid()` for per-epoch shuffling

Data layout:
- `50Real.npy` shape `[N, 50, 11, 4]`: entity 0=ball, 1-5=offence A, 6-10=defence B; features: x, y, z, flag
- `50Seq.npy` shape `[N, 50, 12]`: [ball.x, ball.y, A1.x, A1.y, …, A5.x, A5.y]
- `SeqCond.npy` / `RealCond.npy` shape `[N, 50, 6]`: one-hot ball-possession indicators (dribble_by_A1…A5, pass)

The positional indexing hack in training: `real_[:, :, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]` drops index 2 (ball z) — this maps the 13-element `[ball.xyz + 5 players × xy]` down to 12 features.

### Ops (`src/ops.py`)

- `spectral_norm(w, iteration=1)`: Single power iteration spectral normalization for conv kernels
- `conv1d_sn`: 1D convolution with spectral norm, no activation, kernel size 5
- `res_block`: Residual block with manual "same" padding (concatenating edge frames), leaky ReLU (α=0.2), skip connection scaled by `residual_alpha`

### UI application (`ui/`)

The UI is a **PyQt5** desktop app with two panels:
- **Left panel**: `Drawingboard` — a `QGraphicsView` where users place 5 offensive players (double-click) and sketch ball trajectories (click-release). Uses `MovableDisk` (players) and `BallDisk` (ball with shot/pass logic).
- **Right panel**: `Court` — matplotlib `FigureCanvas` embedded in Qt, plays back the sketch animation or the generated simulation.

**Workflow**: User sketches → "Generate" button clicked → `Main.run_model()`:
1. `Scene_.savePos()` saves the sketched trajectory to `Points/points2.npy`
2. `WGAN.run_Model()`: loads the pre-trained checkpoint, runs inference with `use_encoder=False`, generates `n_Latent=100` defense variants for 10 condition duplicates → saves `Data/output/output.npy`
3. User toggles "Sketch Animation" (shows input) or "Play Simulation" (shows generated defense)

**Sketch data flow**: `Drawingboard` mouse events → `BallDisk.segData` (raw segments) → `SavePos.save_pos()` applies Bezier curve smoothing → `draw_feat.get_feature()` computes ball-possession features → `WGAN.run_Model()` normalizes via `DataFactory.normalize()` and runs the generator.

**Pre-trained checkpoint**: The UI expects `ui/Data/checkpoints/model.ckpt-88200.meta` (hardcoded in `WGAN.py:9-10`). Model data files (`50Seq.npy`, `SeqCond.npy`, `50Real.npy`, `RealCond.npy`) must also be present in `ui/Data/Model_data/`.

### Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--latent_dims` | 150 | Latent z dimension |
| `--seq_length` | 50 | Timesteps per play |
| `--n_filters` | 256 | Conv filters |
| `--n_resblock` | 8 | Residual blocks per network |
| `--lr_` | 1e-4 | Adam learning rate |
| `--beta` | 0.001 | KL divergence weight (β-VAE) |
| `--recon_weight` | 1.0 | L1 reconstruction loss weight |
| `--pretrain_D` | 25 | Epochs to pretrain discriminator |
| `--train_D` | 5 | D updates per G update |
| `--batch_size` | 128 | Batch size |
| `--lambda_` | 1.0 | Decaying lambda (unused currently) |
