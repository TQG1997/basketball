"""Custom PyTorch layers matching the TF ops API.

All layers use [B, T, C] format (batch_first) for consistency with
MultiheadAttention and the original TF code. Conv1d uses internal permute.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ---------------------------------------------------------------------------
#   Spectral-norm Conv1D
# ---------------------------------------------------------------------------

class Conv1D_SN(nn.Module):
    """Conv1D with spectral normalization, [B, T, C] → [B, T, filters]."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding='valid'):
        super().__init__()
        self.stride = stride
        p = kernel_size // 2 if padding == 'same' else 0
        self.pad = p
        self.conv = spectral_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=p))

    def forward(self, x):
        # x: [B, T, C] → [B, C, T] → conv → [B, T, C_out]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x


# ---------------------------------------------------------------------------
#   Residual block (no spectral norm — for diffusion)
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    """Conv1D residual block with LayerNorm + SiLU. [B, T, C] ↔ [B, T, C]."""

    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        shortcut = x
        # LayerNorm + SiLU + Conv (permute for Conv1d)
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h.permute(0, 2, 1)).permute(0, 2, 1)
        return h + shortcut


# ---------------------------------------------------------------------------
#   Attention blocks
# ---------------------------------------------------------------------------

class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention with residual + LayerNorm. [B, T, C]."""

    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        return self.norm(x + attn_out)


class CrossAttentionBlock(nn.Module):
    """Cross-attention: query attends to key/value, residual + LayerNorm. [B, T, C]."""

    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key_value, key_padding_mask=None):
        attn_out, _ = self.attn(query, key_value, key_value,
                                 key_padding_mask=key_padding_mask)
        return self.norm(query + attn_out)
