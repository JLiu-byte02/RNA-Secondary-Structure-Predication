"""src/pair_head.py -- 轻量配对预测头（计划 §14 / README §5.2）

冻结 RiNALMo embedding (L,1280)
  -> LayerNorm -> Linear(1280,64) -> GELU        => z (L,64)
  -> 逐对拼接 [z_i ; z_j]                         => (L,L,128)
  -> Conv1x1 128->64
  -> 残差块 (1x1)
  -> 残差块 (3x3, padding=1)
  -> Conv1x1 64->1                                => (L,L) logits
  -> (s_ij + s_ji)/2                              => 对称 logits

残差块: Conv -> InstanceNorm(affine) -> ReLU -> Conv -> InstanceNorm -> 加输入 -> ReLU
"""
from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.relu(x + h)


class PairHead(nn.Module):
    def __init__(self, in_dim: int = 1280, projection_dim: int = 64,
                 pair_channels: int = 64, resnet_kernel_sizes=(1, 3),
                 symmetrize: bool = True):
        super().__init__()
        self.symmetrize = symmetrize
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, projection_dim)
        self.act = nn.GELU()
        self.compress = nn.Conv2d(2 * projection_dim, pair_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(pair_channels, k) for k in resnet_kernel_sizes]
        )
        self.out = nn.Conv2d(pair_channels, 1, kernel_size=1)

    def _forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: (L,1280) 或 (B,L,1280) -> logits (L,L) 或 (B,L,L)。"""
        squeeze = emb.dim() == 2
        if squeeze:
            emb = emb.unsqueeze(0)
        B, L, _ = emb.shape

        z = self.act(self.proj(self.norm(emb)))                     # B,L,P
        zi = z.unsqueeze(2).expand(B, L, L, z.shape[-1])            # B,L,L,P
        zj = z.unsqueeze(1).expand(B, L, L, z.shape[-1])
        x = torch.cat([zi, zj], dim=-1)                             # B,L,L,2P
        x = x.permute(0, 3, 1, 2).contiguous()                      # B,2P,L,L
        x = self.compress(x)
        for blk in self.blocks:
            x = blk(x)
        s = self.out(x)[:, 0]                                       # B,L,L
        if self.symmetrize:
            s = (s + s.transpose(1, 2)) / 2.0
        return s[0] if squeeze else s

    def forward(self, emb):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=emb.is_cuda):
            return self._forward(emb)
