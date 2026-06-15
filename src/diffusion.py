"""DDPM/DDIM diffusion for basketball trajectory generation — PyTorch.

Diffuses: defence(10) + ball features(6) = 16 dims
Conditioning: offence(12) + seq_feat(6) = 18 dims
"""

import math
import torch
import torch.nn as nn
from ops import ResBlock1D, SelfAttentionBlock, CrossAttentionBlock


# ---------------------------------------------------------------------------
#   Timestep embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t, dim, max_period=10000):
    """Transformer-style sinusoidal timestep encoding."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    freqs = freqs.to(t.device)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
#   Diffusion process
# ---------------------------------------------------------------------------

class GaussianDiffusion(nn.Module):
    """DDPM forward + DDIM reverse sampling.

    Inherits nn.Module so buffers move automatically with model.to(device).
    """

    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

    def q_sample(self, x0, t, noise=None):
        """Forward: x0 → xt. Returns (xt, noise)."""
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bars[t][:, None, None]
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        return xt, noise

    @torch.no_grad()
    def ddim_step(self, model, xt, t, conds, mask=None):
        """Single DDIM reverse step (deterministic, eta=0)."""
        B = xt.shape[0]
        pred_noise = model(xt, t, conds, mask=mask)

        alpha_bar_t = self.alpha_bars[t][:, None, None]
        pred_x0 = (xt - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

        if t[0] > 0:
            alpha_bar_prev = self.alpha_bars[t - 1][:, None, None]
            direction = torch.sqrt(1 - alpha_bar_prev) * pred_noise
            xt_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + direction
        else:
            xt_prev = pred_x0
        return xt_prev

    @torch.no_grad()
    def sample(self, model, conds, shape, steps=50, mask=None):
        """Full DDIM sampling loop."""
        B, T, D = shape
        xt = torch.randn(B, T, D, device=conds.device)

        stride = max(1, self.T // steps)
        timesteps = list(range(self.T - 1, -1, -stride))

        for t_val in timesteps:
            t_batch = torch.full((B,), t_val, dtype=torch.long, device=conds.device)
            xt = self.ddim_step(model, xt, t_batch, conds, mask=mask)

        return xt


# ---------------------------------------------------------------------------
#   Denoiser Network (PyTorch)
# ---------------------------------------------------------------------------

class DenoiserNet(nn.Module):
    """Predicts noise given (noisy_traj, t, conditioning).

    Input:  xt [B,T,16],  t [B],  conds [B,T,18]
    Output: predicted noise [B,T,16]
    """

    def __init__(self, in_dim=16, cond_dim=18, n_filters=256, n_resblock=4,
                 num_heads=4, T=1000):
        super().__init__()
        self.n_filters = n_filters
        self.T = T

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(n_filters, n_filters * 2),
            nn.SiLU(),
            nn.Linear(n_filters * 2, n_filters),
        )

        # Input projections
        self.input_proj = nn.Conv1d(in_dim, n_filters, 1)
        self.cond_proj = nn.Conv1d(cond_dim, n_filters, 1)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock1D(n_filters) for _ in range(n_resblock)
        ])

        # Attention (interleaved every 2 resblocks)
        self.attn_blocks = nn.ModuleList([
            SelfAttentionBlock(n_filters, num_heads=num_heads)
            for _ in range(n_resblock // 2)
        ])
        self.cross_attn = CrossAttentionBlock(n_filters, num_heads=num_heads)

        # Output
        self.output_proj = nn.Conv1d(n_filters, in_dim, 1)

    def forward(self, xt, t, conds, mask=None):
        # Time embedding
        t_emb = sinusoidal_embedding(t, self.n_filters)
        t_emb = self.time_mlp(t_emb)                          # [B, n_filters]
        t_emb = t_emb.unsqueeze(1)                             # [B, 1, n_filters]

        # Project inputs (permute for Conv1d: [B,T,C] → [B,C,T] → [B,C,T] → [B,T,C])
        x = self.input_proj(xt.permute(0, 2, 1)).permute(0, 2, 1)
        c = self.cond_proj(conds.permute(0, 2, 1)).permute(0, 2, 1)

        # Combine
        x = x + c + t_emb

        # ResBlocks + attention
        attn_idx = 0
        for i, block in enumerate(self.res_blocks):
            x = block(x)
            if i % 2 == 0 and attn_idx < len(self.attn_blocks):
                x = self.attn_blocks[attn_idx](x)
                attn_idx += 1

        # Cross-attention to conditioning
        x = self.cross_attn(x, c)

        # Output (permute for Conv1d)
        out = self.output_proj(x.permute(0, 2, 1)).permute(0, 2, 1)
        return out
