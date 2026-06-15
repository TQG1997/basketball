# Basketball Play Generator

### Generate defensive basketball play simulations from offensive sketches.

A **Diffusion Model** (DDPM + DDIM) that generates realistic defensive player movements conditioned on an offensive play sequence, using ResBlock1D + Self-Attention + Cross-Attention.

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
python src/train.py --data_path=data --output=output --max_epochs=500
```

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TQG1997/basketball/blob/main/notebooks/train.ipynb)

Use **Runtime → Change runtime type → T4 GPU**. The notebook handles everything end-to-end.

> Set `--max_epochs` to avoid exhausting the Colab session limit (~12h for free tier).

## Web UI

Modern Gradio web interface — sketch plays and visualize generated defense in your browser:

```bash
pip install gradio
python ui/app.py              # http://127.0.0.1:7860
python ui/app.py --share      # public link
```

Three tabs: upload `.npy` sketch file, interactive court click-to-place, or read about the model.

Requires a trained checkpoint at `ui/Data/checkpoints/model_epoch500.pt`.

## Project Structure

```
Basketball/
├── src/                        # PyTorch diffusion model + training
│   ├── train.py                # Training entry (AMP, EMA, validation)
│   ├── diffusion.py            # DDPM/DDIM + DenoiserNet
│   ├── ops.py                  # Conv1D_SN, ResBlock1D, Self/CrossAttention
│   ├── game_visualizer.py      # Play animation (matplotlib)
│   └── utils.py                # DataFactory re-export
├── ui/                         # Web UI + inference
│   ├── app.py                  # Gradio interface
│   ├── inference.py            # PyTorch inference pipeline
│   └── draw_feat.py            # Ball-possession feature extraction
├── shared/                     # DataFactory (numpy, framework-agnostic)
├── config/                     # Training YAML configuration
├── notebooks/                  # Colab training notebook
├── data/                       # Dataset (.npy, excluded from git)
└── requirements.txt
```

## Training Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_path` | `data` | Dataset directory |
| `--output` | `output` | Output dir (checkpoints, samples) |
| `--max_epochs` | `null` | Stop after N epochs (null = forever) |
| `--batch_size` | `64` | Batch size |
| `--lr` | `1e-4` | AdamW learning rate |
| `--n_filters` | `256` | Conv filters |
| `--n_resblock` | `4` | Residual blocks per network |
| `--num_heads` | `4` | Attention heads |
| `--T` | `1000` | Diffusion timesteps |
| `--ddim_steps` | `50` | DDIM sampling steps |
| `--checkpoint_step` | `100` | Epochs between checkpoints |
| `--vis_freq` | `10` | Epochs between visualizations |

See `config/config.yaml` for all settings.

## Architecture

```
Offensive Play (conditioning)          Random Noise
        │                                   │
        ▼                                   ▼
  ┌──────────┐                      ┌──────────────┐
  │ Cond Proj │                      │  Input Proj  │
  └──────────┘                      └──────────────┘
        │                                   │
        └─────────────┬─────────────────────┘
                      ▼
          ┌───────────────────┐
          │  ResBlock1D × N   │   (LayerNorm + SiLU + Conv1D)
          │  Self-Attention   │   (temporal interactions)
          │  Cross-Attention  │   (attend to conditioning)
          └───────────────────┘
                      │
                      ▼
               ┌──────────┐
               │ Output   │  →  Defence (10) + Ball Features (6)
               └──────────┘

  Training:  DDPM forward process → predict noise → MSE loss
  Inference: DDIM reverse process — 50 steps from noise to trajectory
```

## Dataset

| File | Shape | Size | Description |
|---|---|---|---|
| `50Real.npy` | (14032, 50, 11, 4) | 236MB | Ground truth plays (ball + 10 players) |
| `50Seq.npy` | (14032, 50, 12) | 64MB | Offence conditioning |
| `SeqCond.npy` | (14032, 50, 6) | 16MB | Ball-possession features (conditioning) |
| `RealCond.npy` | (14032, 50, 6) | 16MB | Ball-possession features (ground truth) |

**Feature layout**: entity 0 = ball (x, y, z, flag), entities 1-5 = offence A1-A5, entities 6-10 = defence B1-B5.

Download from [Google Drive](https://drive.google.com/drive/folders/1uNPw7LOA3xENclQRtSlUftiR7tlVNOts?usp=share_link). Data split 9:1 train/valid by `DataFactory`.
