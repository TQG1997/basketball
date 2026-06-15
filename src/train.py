"""BasketballGAN — PyTorch Lightning diffusion training.

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

# Optional: PyTorch Lightning
try:
    import lightning as pl
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False


# ---------------------------------------------------------------------------
#   Dataset
# ---------------------------------------------------------------------------

class BasketballDataset(Dataset):
    """Yields (target, conditioning) pairs from numpy data.

    target:       defence(10) + ball_features(6) = 16 dims
    conditioning: offence(12) + seq_feat(6) = 18 dims
    """

    def __init__(self, data, seq, feat, real_feat):
        self.offence = data['A'][:, :, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        self.defence = data['B']
        self.seq = seq
        self.feat = feat[:, :, :]
        self.real_feat = real_feat[:, :, :]

    def __len__(self):
        return len(self.offence)

    def __getitem__(self, idx):
        target = np.concatenate([self.defence[idx], self.real_feat[idx]], axis=-1)
        conds = np.concatenate([self.offence[idx], self.feat[idx]], axis=-1)
        return (torch.from_numpy(target).float(),
                torch.from_numpy(conds).float())


# ---------------------------------------------------------------------------
#   Lightning Module
# ---------------------------------------------------------------------------

class BasketballDiffusion(pl.LightningModule if HAS_LIGHTNING else object):
    """PyTorch Lightning module for basketball trajectory diffusion."""

    def __init__(self, config):
        if HAS_LIGHTNING:
            super().__init__()
            self.save_hyperparameters()
        else:
            super().__init__()

        self.cfg = config
        m = config['model']
        d = config['diffusion']
        t = config['training']

        self.diffusion = GaussianDiffusion(
            T=d['T'], beta_start=d['beta_start'], beta_end=d['beta_end'])

        self.denoiser = DenoiserNet(
            in_dim=16, cond_dim=18,
            n_filters=m['n_filters'],
            n_resblock=m['n_resblock'],
            num_heads=m['num_heads'],
            T=d['T'])

        self.lr = t['lr_']
        self.ddim_steps = d['ddim_steps']
        self.seq_len = t['seq_length']

        # EMA
        self.ema_decay = config.get('ema_decay', 0.9999)
        self.ema_model = None

    def forward(self, conds, steps=None):
        """Generate via DDIM sampling."""
        if steps is None:
            steps = self.ddim_steps
        B = conds.shape[0]
        T_len = conds.shape[1]
        return self.diffusion.sample(
            self.denoiser, conds, [B, T_len, 16], steps=steps)

    def training_step(self, batch, batch_idx):
        target, conds = batch
        B = target.shape[0]
        device = target.device

        # Forward diffusion
        t = torch.randint(0, self.cfg['diffusion']['T'], (B,), device=device)
        xt, noise = self.diffusion.q_sample(target, t)

        # Predict noise
        pred_noise = self.denoiser(xt, t, conds)

        loss = F.mse_loss(pred_noise, noise)

        if HAS_LIGHTNING:
            self.log('train_loss', loss, prog_bar=True, on_step=True)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return opt

    @torch.no_grad()
    def generate_sample(self, conds, data_factory):
        """Generate a sample play for visualization."""
        device = next(self.parameters()).device
        conds = conds.to(device)

        generated = self(conds)                            # [B, T, 16]
        gen_np = generated.cpu().numpy()

        defence_gen = gen_np[0, :, :10]                    # defence
        offence_np = conds[0, :, :12].cpu().numpy()        # offence from conditioning

        sample = np.concatenate([offence_np, defence_gen], axis=-1)  # [T, 22]
        sample = sample[None, :, :]                                  # [1, T, 22]

        samples = data_factory.recover_BALL_and_A(sample)
        samples = data_factory.recover_B(samples)
        return samples


# ---------------------------------------------------------------------------
#   Training entry (plain PyTorch — no Lightning dependency required)
# ---------------------------------------------------------------------------

def train(config, data_path, output_path):
    """Main training loop in plain PyTorch (PyTorch Lightning optional)."""
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

    # Split into train/valid
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
        print('AMP: enabled (bfloat16/float16)')

    while t['max_epochs'] is None or epoch < t['max_epochs']:
        df.shuffle_train()
        epoch_loss = 0.0

        for batch_idx, (target, conds) in enumerate(train_loader):
            target = target.to(device)
            conds = conds.to(device)
            B = target.shape[0]

            optimizer.zero_grad()

            # Forward diffusion + noise prediction
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
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f'  Saved: {ckpt_path}')

        # Visualization
        if epoch % t['vis_freq'] == 0:
            denoiser.eval()
            with torch.no_grad():
                # Use fixed conditioning for consistent comparison
                target, conds = next(iter(train_loader))
                conds = conds[:1].to(device)  # single sample
                generated = diffusion.sample(
                    denoiser, conds, [1, t['seq_length'], 16],
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
    """Minimal config parser — can also use Hydra or YAML."""
    parser = argparse.ArgumentParser(description='BasketballGAN PyTorch Training')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file (optional, uses defaults if omitted)')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output', type=str, default='output_torch')
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

    # Build config dict (supports YAML override if provided)
    config = {
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
    }

    # If YAML config provided, merge (shallow — can enhance with OmegaConf)
    if args.config:
        try:
            import yaml
            with open(args.config) as f:
                yaml_cfg = yaml.safe_load(f)
            # Deep merge (simple version)
            for section in yaml_cfg:
                if section in config:
                    config[section].update(yaml_cfg[section])
                else:
                    config[section] = yaml_cfg[section]
        except ImportError:
            print('PyYAML not installed — ignoring --config')

    return config, args.data_path, args.output


if __name__ == '__main__':
    config, data_path, output_path = parse_config()

    if os.path.exists(output_path):
        ans = input(f'"{output_path}" will be removed!! are you sure (y/N)? ')
        if ans.lower() != 'y':
            exit(0)
        shutil.rmtree(output_path)

    train(config, data_path, output_path)
