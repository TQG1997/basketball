"""BasketballGAN — PyTorch diffusion training.

Usage:
    python src/train.py --data_path=data --output=output --max_epochs=500
"""

import os
import sys
import shutil
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Make shared importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import DataFactory

from diffusion import GaussianDiffusion, DenoiserNet
import game_visualizer


# ---------------------------------------------------------------------------
#   Dataset (pre-computed concatenation for speed)
# ---------------------------------------------------------------------------

class BasketballDataset(Dataset):
    """Yields (target, conditioning) pairs.

    target:       defence(10) + ball_features(6) = 16 dims
    conditioning: offence(12) + seq_feat(6) = 18 dims
    """

    def __init__(self, data, seq, feat, real_feat):
        # Drop ball z (index 2): [B,T,13] → [B,T,12]
        offence = data['A'][:, :, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        # Pre-concatenate for speed (avoid np.concatenate per __getitem__)
        self.target = np.concatenate([data['B'], real_feat[:, :, :]], axis=-1).astype(np.float32)
        self.conds = np.concatenate([offence, feat[:, :, :]], axis=-1).astype(np.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.target[idx]),
                torch.from_numpy(self.conds[idx]))


# ---------------------------------------------------------------------------
#   Training entry
# ---------------------------------------------------------------------------

def train(config, data_path, output_path):
    """Main training loop with EMA, AMP, and DDIM visualization."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    m = config['model']
    d = config['diffusion']
    t = config['training']

    # --- Load data ---
    real_data = np.load(os.path.join(data_path, '50Real.npy'))[:, :t['seq_length'], :, :]
    seq_data = np.load(os.path.join(data_path, '50Seq.npy'))
    features_ = np.load(os.path.join(data_path, 'SeqCond.npy'))
    real_feat = np.load(os.path.join(data_path, 'RealCond.npy'))

    df = DataFactory(real_data=real_data, seq_data=seq_data,
                     features_=features_, real_feat=real_feat)

    train_dataset = BasketballDataset(
        df.train_data, df.seq_train, df.f_train, df.rf_train)
    train_loader = DataLoader(
        train_dataset, batch_size=t['batch_size'], shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)

    # --- Build model ---
    diffusion = GaussianDiffusion(
        T=d['T'], beta_start=d['beta_start'], beta_end=d['beta_end']).to(device)
    denoiser = DenoiserNet(
        in_dim=16, cond_dim=18,
        n_filters=m['n_filters'],
        n_resblock=m['n_resblock'],
        num_heads=m['num_heads'],
        T=d['T']).to(device)

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=t['lr_'])

    # EMA for sampling quality (stochastic weight averaging)
    ema = torch.optim.swa_utils.AveragedModel(
        denoiser, avg_fn=lambda avg, model, _: avg * 0.9999 + model * 0.0001)

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    use_amp = scaler is not None

    # Checkpoint paths
    ckpt_dir = os.path.join(output_path, 'Checkpoints')
    sample_dir = os.path.join(output_path, 'Samples')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    num_batches = len(train_loader)
    epoch = 0

    print(f'Batches/epoch: {num_batches}  Samples: {len(train_dataset)}')
    print(f'Diffusion T={d["T"]}  DDIM steps={d["ddim_steps"]}')
    if use_amp:
        print('AMP: enabled (float16)')

    while t['max_epochs'] is None or epoch < t['max_epochs']:
        epoch_loss = 0.0

        for target, conds in train_loader:
            target = target.to(device)
            conds = conds.to(device)
            B = target.shape[0]

            optimizer.zero_grad()

            timesteps = torch.randint(0, d['T'], (B,), device=device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    xt, noise = diffusion.q_sample(target, timesteps)
                    pred_noise = denoiser(xt, timesteps, conds)
                    loss = F.mse_loss(pred_noise, noise)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                xt, noise = diffusion.q_sample(target, timesteps)
                pred_noise = denoiser(xt, timesteps, conds)
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                optimizer.step()

            # Update EMA weights after optimizer step
            ema.update_parameters(denoiser)

            epoch_loss += loss.item()

        epoch += 1
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch:4d}  loss={avg_loss:.6f}')

        # Checkpoint
        if epoch % t['checkpoint_step'] == 0:
            ckpt_path = os.path.join(ckpt_dir, f'model_epoch{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': denoiser.state_dict(),
                'ema_state_dict': ema.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f'  Saved: {ckpt_path}')

        # Visualization (use EMA weights for best quality)
        if epoch % t['vis_freq'] == 0:
            denoiser.eval()
            ema_denoiser = ema.module  # EMA-averaged weights
            with torch.no_grad():
                _, conds = next(iter(train_loader))
                conds = conds[:1].to(device)
                generated = diffusion.sample(
                    ema_denoiser, conds, [1, t['seq_length'], 16],
                    steps=d['ddim_steps'])
                gen_np = generated[0].cpu().numpy()
                off_np = conds[0, :, :12].cpu().numpy()
                sample = np.concatenate([off_np, gen_np[:, :10]], axis=-1)
                sample = sample[None, :, :]

                samples = df.recover_BALL_and_A(sample)
                samples = df.recover_B(samples)
                fname = os.path.join(sample_dir, f'reconstruct{epoch}.mp4')
                game_visualizer.plot_data(
                    samples[0], t['seq_length'], file_path=fname, if_save=True)
            denoiser.train()

    print('Training complete!')


# ---------------------------------------------------------------------------
#   CLI
# ---------------------------------------------------------------------------

def parse_config():
    parser = argparse.ArgumentParser(description='BasketballGAN PyTorch Training')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_filters', type=int, default=256)
    parser.add_argument('--n_resblock', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--T', type=int, default=1000, help='Diffusion steps')
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--checkpoint_step', type=int, default=100)
    parser.add_argument('--vis_freq', type=int, default=10)
    args = parser.parse_args()

    return {
        'model': {
            'n_filters': args.n_filters,
            'n_resblock': args.n_resblock,
            'num_heads': args.num_heads,
        },
        'diffusion': {
            'T': args.T,
            'beta_start': 1e-4,
            'beta_end': 0.02,
            'ddim_steps': args.ddim_steps,
        },
        'training': {
            'batch_size': args.batch_size,
            'seq_length': 50,
            'lr_': args.lr,
            'max_epochs': args.max_epochs,
            'checkpoint_step': args.checkpoint_step,
            'vis_freq': args.vis_freq,
        },
    }, args.data_path, args.output


if __name__ == '__main__':
    config, data_path, output_path = parse_config()

    if os.path.exists(output_path):
        ans = input(f'"{output_path}" will be removed!! are you sure (y/N)? ')
        if ans.lower() != 'y':
            exit(0)
        shutil.rmtree(output_path)

    train(config, data_path, output_path)
