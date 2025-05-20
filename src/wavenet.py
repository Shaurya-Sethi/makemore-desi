"""
Wavenet-style character language model
=====================================

•  Embedding lookup  → stack of dilated causal conv residual blocks
•  Gated activations (tanh ⊙ sigmoid) with residual & skip connections
•  Final 1×1 conv projects to vocabulary size logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1-D causal (left-padded) convolution."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            dilation=dilation,
            padding=self.pad,
        )

    def forward(self, x):
        out = self.conv(x)
        # remove padding on the right to keep causality
        return out[:, :, :-self.pad] if self.pad > 0 else out


class ResidualBlock(nn.Module):
    """
    Gated residual block used in original Wavenet.

    x ─► CausalConv (filter) ─► tanh ─┐
      │                               × (gated)
      └► CausalConv (gate) ─► sigmoid ┘
            │
            └► 1×1 conv → residual, skip
    """
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv_filter = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv_gate   = CausalConv1d(channels, channels, kernel_size, dilation)

        # 1×1 convs for residual + skip
        self.conv_res = nn.Conv1d(channels, channels, 1)
        self.conv_skip = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        f = torch.tanh(self.conv_filter(x))
        g = torch.sigmoid(self.conv_gate(x))
        z = f * g  # gated

        residual = self.conv_res(z)
        skip     = self.conv_skip(z)
        return (x + residual), skip


class WaveNetLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        channels: int = 128,
        kernel_size: int = 3,
        dilations=(1, 2, 4, 8, 16, 32),
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # first 1×1 conv to expand embedding dim → residual channels
        self.conv_in = nn.Conv1d(emb_dim, channels, 1)

        # stack residual blocks
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(channels, kernel_size, d)
                for d in dilations
            ]
        )

        # post-processing: ReLU → 1×1 conv → ReLU → 1×1 conv logits
        self.post = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv1d(channels, vocab_size, 1),
        )

    def forward(self, idx: torch.Tensor):
        """
        idx: (B, T) int64
        returns logits: (B, T, vocab_size)
        """
        x = self.embedding(idx)          # (B, T, emb)
        x = x.permute(0, 2, 1)           # (B, emb, T)
        x = self.conv_in(x)              # (B, C, T)

        skip_connections = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_connections = skip_connections + skip

        out = self.post(skip_connections)
        out = out.permute(0, 2, 1)       # (B, T, vocab)
        return out
