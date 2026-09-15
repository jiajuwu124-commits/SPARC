"""Compact scenario-conditioned residual restoration for SPARC."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConditionalBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int, dilation: int = 1):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv1 = nn.Conv2d(
            channels, channels, 3, padding=dilation, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Linear(condition_dim, 2 * channels)

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, bias = self.film(condition).chunk(2, dim=1)
        hidden = self.norm1(inputs)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]
        hidden = self.conv1(F.gelu(hidden))
        hidden = self.conv2(F.gelu(self.norm2(hidden)))
        return inputs + hidden


class ConditionalResidualRestorer(nn.Module):
    """A sub-million-parameter U-shaped restorer with frozen scenario conditioning."""

    def __init__(self, scenario_count: int, width: int = 32, condition_dim: int = 32):
        super().__init__()
        self.scenario_count = int(scenario_count)
        self.width = int(width)
        self.condition_dim = int(condition_dim)
        self.embedding = nn.Embedding(self.scenario_count, condition_dim)
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.enc1 = ConditionalBlock(width, condition_dim)
        self.down1 = nn.Conv2d(width, 2 * width, 4, stride=2, padding=1)
        self.enc2 = ConditionalBlock(2 * width, condition_dim)
        self.down2 = nn.Conv2d(2 * width, 3 * width, 4, stride=2, padding=1)
        self.middle = nn.ModuleList([
            ConditionalBlock(3 * width, condition_dim, dilation=value)
            for value in (1, 2, 4, 2)
        ])
        self.up2 = nn.Conv2d(3 * width, 2 * width, 3, padding=1)
        self.dec2 = ConditionalBlock(2 * width, condition_dim)
        self.up1 = nn.Conv2d(2 * width, width, 3, padding=1)
        self.dec1 = ConditionalBlock(width, condition_dim)
        self.head = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, images: torch.Tensor, scenario: torch.Tensor) -> torch.Tensor:
        condition = self.embedding(scenario)
        skip1 = self.enc1(self.stem(images), condition)
        skip2 = self.enc2(self.down1(skip1), condition)
        hidden = self.down2(skip2)
        for block in self.middle:
            hidden = block(hidden, condition)
        hidden = F.interpolate(hidden, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.dec2(self.up2(hidden) + skip2, condition)
        hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.dec1(self.up1(hidden) + skip1, condition)
        return (images + torch.tanh(self.head(hidden))).clamp(0.0, 1.0)

