# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Basketball play generation via **Diffusion Model** (DDPM + DDIM). Given an offensive play sketch (ball trajectory + 5 players), the model generates realistic defensive player movements. Uses PyTorch with ResBlock1D + Self-Attention + Cross-Attention architecture.

Two entry points: training (`src/train.py`) and Gradio web UI (`ui/app.py`).

## Commands

```bash
# Training
python src/train.py --data_path=data --output=output --max_epochs=500

# Gradio Web UI
python ui/app.py              # http://127.0.0.1:7860
python ui/app.py --share      # public link

# Standalone visualization
python src/game_visualizer.py --data_path=path/to/data.npy
```

There are no tests, linting, or CI.

## Architecture

### Diffusion model (`src/diffusion.py`)

**GaussianDiffusion** (inherits `nn.Module`):
- Forward process (q_sample): x0 → xt = √ᾱ·x0 + √(1-ᾱ)·ε
- Reverse process: DDIM sampling (50 steps, deterministic η=0)
- Noise schedule: linear β from 1e-4 to 0.02 over T=1000 steps
- Uses `register_buffer` for device-safe schedule tensors

**DenoiserNet** (inherits `nn.Module`):
- Time embedding: sinusoidal → MLP(SiLU) → [B, 1, n_filters]
- Input projection: Conv1d(16→n_filters), Cond projection: Conv1d(18→n_filters)
- N ResBlock1D (LayerNorm+SiLU+Conv1d, no spectral norm)
- Self-attention every 2 blocks (MultiheadAttention, batch_first=True)
- Cross-attention to conditioning at output
- All tensors in [B, T, C] format (permute for Conv1d)

Data flow:
- Target (diffused): concat(defence_xy[10], ball_features[6]) = [B,T,16]
- Conditioning: concat(offence_xy[12], seq_feat[6]) = [B,T,18]

### Custom layers (`src/ops.py`)

- `Conv1D_SN`: Conv1d with `torch.nn.utils.spectral_norm`. [B,T,C] → permute → conv → permute back
- `ResBlock1D`: LayerNorm → SiLU → Conv1d (×2) + skip connection
- `SelfAttentionBlock`: MultiheadAttention(batch_first=True) + LayerNorm + residual
- `CrossAttentionBlock`: MultiheadAttention(query≠key/value) + LayerNorm + residual

### Training (`src/train.py`)

- `BasketballDataset`: Pre-concatenates target/conds arrays in `__init__` for speed
- Training loop: AMP (GradScaler + autocast), EMA (AveragedModel, decay=0.9999)
- Optimizer: AdamW (β₁=0.9, β₂=0.999 — standard diffusion, no GAN β₁=0.5 hack)
- Loss: MSE(ε_pred, ε_true)
- Validation: configurable `valid_freq`, computes MSE on held-out set
- Checkpoint: PyTorch `.pt` with model+EMA+optimizer state_dicts
- Visualization: DDIM sample → recover_BALL_and_A → recover_B → MP4 via matplotlib

### Data pipeline (`shared/__init__.py`)

`DataFactory` singleton — pure numpy, framework-agnostic:
1. Z-normalizes x/y/z positions on `__real_data` (stores mean/std)
2. Splits 9:1 into train/valid
3. Extracts team_A (ball xyz + 5 offence xy) = [N,T,13], team_B (5 defence xy) = [N,T,10]
4. `normalize()` method for UI inference sketches
5. `recover_*()` methods for denormalizing generated plays

Data indexing hack: offence [N,T,13] → drop ball z (index 2) → [N,T,12]

### UI (`ui/app.py`, `ui/inference.py`)

- Gradio web app with 3 tabs: file upload, interactive click-to-place, model info
- `ModelManager`: lazy-loads diffusion model + DataFactory
- `ClickCollector`: pixel→court coordinate conversion, auto-places dummy offence players
- `inference.py`: PyTorch pipeline (load model → normalize → DDIM sample → recover → save)

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_filters` | 256 | Conv filters |
| `--n_resblock` | 4 | Residual blocks per network |
| `--num_heads` | 4 | Attention heads |
| `--T` | 1000 | Diffusion timesteps |
| `--ddim_steps` | 50 | DDIM sampling steps |
| `--batch_size` | 64 | Batch size |
| `--lr` | 1e-4 | AdamW learning rate |
| `--max_epochs` | null | Stop after N epochs |
