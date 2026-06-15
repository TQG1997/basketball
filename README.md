# BasketballGAN

### Generate the ghosting defensive strategies given offensive sketch.

[Paper](https://arxiv.org/abs/1909.07088) | [CGVLab](https://people.cs.nctu.edu.tw/~yushuen/) | [Video](https://youtu.be/NTir0-znPyw)

**BasketballGAN: Generating Basketball Play Simulation through Sketching**

Hsin-Ying Hsieh<sup>1</sup>, Chieh-Yu Chen<sup>2</sup>, Yu-Shuen Wang<sup>1</sup>, Jung-Hong Chuang<sup>1</sup>

<sup>1</sup>National Chiao Tung University, <sup>2</sup>NVIDIA Corporation

Accepted paper in ACMMM 2019.

## Project Structure

```
Basketball/
├── src/               # GAN model source code
│   ├── Train_Triple.py      # Training entry
│   ├── ThreeDiscrim.py      # Discriminator model
│   ├── game_visualizer.py   # Game visualization utilities
│   ├── ops.py               # TensorFlow ops
│   ├── utils.py             # Utility functions
│   └── court.png            # Court background (high-res for training)
├── ui/                # Interactive UI application (PyQt5)
│   ├── Main.py              # Main application entry
│   ├── Drawingboard.py      # Drawing board for sketching plays
│   ├── Court.py             # Court rendering
│   ├── Players.py           # Player rendering
│   ├── Ball.py / Bezier.py  # Ball trajectory
│   ├── WGAN.py              # WGAN model integration
│   ├── draw_feat.py         # Feature drawing
│   ├── SavePos.py           # Position saving/loading
│   ├── CreateTraj.py        # Trajectory creation
│   ├── utils.py             # UI utilities
│   ├── images/              # UI assets (icons, court)
│   ├── Points/              # Saved position data
│   ├── Data/
│   │   ├── checkpoints/     # Pre-trained model checkpoints
│   │   └── output/          # Generated play output
│   └── run_ui_container.sh  # Docker run script
├── data/              # Dataset (.npy files)
│   ├── 50Real.npy
│   ├── 50Seq.npy
│   ├── FEATURES-4.npy
│   ├── RealCond.npy
│   └── SeqCond.npy
├── DataTranslater/    # Data conversion tools
│   └── ToCsv.py
└── requirements.txt   # Python dependencies
```

## Prerequisites

- Linux / macOS
- NVIDIA GPU (for training)
- Docker (recommended for training)
- Python 3.6+ (for UI)

## Getting Started

### Training

```bash
cd src
python Train_Triple.py --folder_path='tmp' --data_path='../data'
```

### UI Application

```bash
cd ui
python Main.py
```

### Docker (Recommended for Training)

```bash
docker run --runtime=nvidia -it --rm -v $PWD:$PWD --net host nvcr.io/nvidia/tensorflow:19.06-py2 bash
```

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TQG1997/basketball/blob/main/notebooks/train.ipynb)

Or set up manually in a Colab notebook:

```python
# 1. Clone the repository
!git clone https://github.com/TQG1997/basketball.git
%cd basketball

# 2. Install dependencies (Colab has TF2 pre-installed)
!pip install -r requirements.txt

# 3. Download dataset from Google Drive (see links below)
#    Place the .npy files under data/

# 4. Mount Google Drive for checkpoint persistence
from google.colab import drive
drive.mount('/content/drive')

# 5. Train (checkpoints + logs saved to Drive)
!python src/Train_Triple.py \
    --folder_path='/content/drive/MyDrive/basketballgan/tmp' \
    --data_path='data' \
    --max_epochs=500 \
    --batch_size=64
```

> **Note:** Use **Runtime → Change runtime type → T4 GPU** for free GPU acceleration.
> Set `--max_epochs` to avoid exhausting the Colab session limit (~12h for free tier).

## Dataset

### Files

| File | Shape | Type | Size | Description |
|---|---|---|---|---|
| `50Real.npy` | (14032, 50, 11, 4) | float64 | 236MB | Ground truth plays: ball trajectory + player positions (50 timesteps) |
| `50Seq.npy` | (14032, 50, 12) | float64 | 64MB | Offence conditioning sequences (ball + 5 offence players x,y) |
| `FEATURES-4.npy` | (11863, 100, 11, 4) | float64 | 398MB | Full-length ground truth (100 timesteps), no truncation |
| `RealCond.npy` | (14032, 50, 6) | int32 | 16MB | Ball status features for ground truth plays (binary indicators) |
| `SeqCond.npy` | (14032, 50, 6) | int32 | 16MB | Ball status features for conditioning sequences (binary indicators) |

### Feature Layout

**50Real.npy** — 4D tensor organized as `[sample, timestep, entity, feature]`:

```
entity 0:     ball           (x, y, z, flag)
entity 1-5:   offence A1-A5  (x, y, z, flag)
entity 6-10:  defence B1-B5  (x, y, z, flag)
```

- `x, y`: court coordinates (normalised during training)
- `z`: ball height (ball only; player z is 0)
- `flag`: `1` for ball, `0` for players

**50Seq.npy** — 3D tensor `[sample, timestep, feature]`, each timestep contains 12 values:

```
[ball.x, ball.y, A1.x, A1.y, A2.x, A2.y, A3.x, A3.y, A4.x, A4.y, A5.x, A5.y]
```

**RealCond.npy / SeqCond.npy** — 3D tensor `[sample, timestep, feature]`, each timestep has 6 binary ball-status indicators:

```
[dribble_by_A1, dribble_by_A2, dribble_by_A3, dribble_by_A4, dribble_by_A5, pass]
```

Exactly one is `1` per timestep (one-hot state of which player has possession).

### Data Split

The dataset is split 9:1 into training and validation sets (by the `DataFactory` class in `utils.py`). No shuffle is applied before the split; it assumes the data is already randomly ordered.

### Source

Download the original dataset from the [BasketballGAN Google Drive folder](https://drive.google.com/drive/folders/1uNPw7LOA3xENclQRtSlUftiR7tlVNOts?usp=share_link). The data consists of NBA play-by-play tracking data, processed into fixed-length game segments.


## Citation

```
@article{hsieh2019basketballgan,
  title={BasketballGAN: Generating Basketball Play Simulation Through Sketching},
  author={Hsieh, Hsin-Ying and Chen, Chieh-Yu and Wang, Yu-Shuen and Chuang, Jung-Hong},
  journal={arXiv preprint arXiv:1909.07088},
  year={2019}
}
```
