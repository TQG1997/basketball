# Basketball Play Generator

### Generate defensive basketball play simulations from offensive sketches.

A **Diffusion Model** (DDPM + DDIM) that generates realistic defensive player movements conditioned on an offensive play sequence, using ResBlock1D + Self-Attention + Cross-Attention.

## Quick Start

```bash
git clone https://github.com/TQG1997/basketball.git
cd basketball
pip install -r requirements.txt

# Download dataset (~730MB)
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1uNPw7LOA3xENclQRtSlUftiR7tlVNOts -O data/

# Train
python src/train.py --data_path=data --output=output --max_epochs=500
```

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TQG1997/basketball/blob/main/notebooks/train.ipynb)

One-click training on free GPU. Use **Runtime → Change runtime type → T4 GPU**.

## Web UI

```bash
pip install gradio
python ui/app.py              # http://127.0.0.1:7860
python ui/app.py --share      # public link
```

Interactive browser interface: upload sketches, place points on court, view generated plays. Requires a trained checkpoint at `ui/Data/checkpoints/model_epoch500.pt`.

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

## Problem Formulation

```
Input (Conditioning):            Output (Generated):
┌──────────────────────┐        ┌──────────────────────┐
│ Offence [T×12]        │        │ Defence [T×10]        │
│  ball x,y             │        │  B1..B5 x,y           │
│  A1..A5 x,y           │   →    │                       │
│                       │        │ Ball Features [T×6]   │
│ Seq Feat [T×6]        │        │  dribble/pass status  │
│  ball possession      │        │                       │
└──────────────────────┘        └──────────────────────┘

  T = 50 frames (~8 seconds of gameplay)
  Conditional time-series generation: given offense, predict defense
```

## Future Directions

1. **Flow Matching** — Replace DDPM with flow matching for faster training and sampling (10-20 steps instead of 50-1000). [Reference](https://arxiv.org/abs/2210.02747)

2. **Graph Neural Network for Player Interactions** — Model 10 players + ball as graph nodes with edges encoding distance and defensive matchups, replacing the current flat 16-dim vector representation. More physically grounded player movement modeling.

3. **Variable-Length Sequences via Perceiver IO** — Remove the fixed 50-frame limit. Handle plays of arbitrary length with attention-based pooling over time, enabling the model to generalize across different play durations.

## Dataset

| File | Shape | Size | Description |
|---|---|---|---|
| `50Real.npy` | (14032, 50, 11, 4) | 236MB | Ground truth plays (ball + 10 players) |
| `50Seq.npy` | (14032, 50, 12) | 64MB | Offence conditioning |
| `SeqCond.npy` | (14032, 50, 6) | 16MB | Ball-possession features (conditioning) |
| `RealCond.npy` | (14032, 50, 6) | 16MB | Ball-possession features (ground truth) |

**Feature layout**: entity 0 = ball (x, y, z, flag), entities 1-5 = offence A1-A5, entities 6-10 = defence B1-B5.

Download from [Google Drive](https://drive.google.com/drive/folders/1uNPw7LOA3xENclQRtSlUftiR7tlVNOts?usp=share_link). Data split 9:1 train/valid by `DataFactory`.
