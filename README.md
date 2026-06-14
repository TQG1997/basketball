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

## Dataset

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1uNPw7LOA3xENclQRtSlUftiR7tlVNOts?usp=share_link) and place `.npy` files under `data/`.

## Citation

```
@article{hsieh2019basketballgan,
  title={BasketballGAN: Generating Basketball Play Simulation Through Sketching},
  author={Hsieh, Hsin-Ying and Chen, Chieh-Yu and Wang, Yu-Shuen and Chuang, Jung-Hong},
  journal={arXiv preprint arXiv:1909.07088},
  year={2019}
}
```
