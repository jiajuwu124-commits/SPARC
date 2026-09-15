"""Occlusion-aware residual-gated restoration for severe synthetic snow.

The module separates the dense, low-frequency tone shift introduced by the
ImageNet-C snow process from sparse/high-frequency snow occlusions.  A bounded
global affine branch corrects the former.  A learned soft mask gates the local
residual branch so that local corrections cannot freely rewrite the complete
image.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SnowSpecialistOutput:
    restored: torch.Tensor
    gate: torch.Tensor
    global_delta: torch.Tensor
    local_delta: torch.Tensor


def snow_evidence(images: torch.Tensor) -> torch.Tensor:
    """Return differentiable snow cues without labels or clean references.

    Evidence combines RGB with brightness, low saturation, bright local excess,
    high-frequency energy, and gradient energy.  The cues are observations of
    the deployed image only; paired clean images are used solely by the
    development-time training loss.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("expected RGB images with shape [batch, 3, height, width]")
    if not images.is_floating_point():
        raise TypeError("expected floating-point RGB values in [0, 1]")
    maximum = images.amax(dim=1, keepdim=True)
    minimum = images.amin(dim=1, keepdim=True)
    luminance = (
        0.2126 * images[:, 0:1]
        + 0.7152 * images[:, 1:2]
        + 0.0722 * images[:, 2:3]
    )
    saturation = (maximum - minimum) / maximum.clamp_min(1e-3)
    local_mean = F.avg_pool2d(luminance, 7, stride=1, padding=3)
    bright_excess = F.relu(luminance - local_mean)
    high_frequency = (luminance - local_mean).abs()
    dx = F.pad((luminance[..., 1:] - luminance[..., :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((luminance[..., 1:, :] - luminance[..., :-1, :]).abs(), (0, 0, 0, 1))
    gradient = 0.5 * (dx + dy)
    whiteness = luminance * (1.0 - saturation)
    return torch.cat(
        (images, luminance, saturation, whiteness, bright_excess, high_frequency, gradient),
        dim=1,
    )


def analytic_snow_tone_inverse(images: torch.Tensor) -> torch.Tensor:
    """Approximate the invertible dense part of ImageNet-C severity-5 snow.

    Before adding flakes, that corruption mixes RGB with
    ``1.5 * gray + 0.5`` using weights 0.55/0.45.  Where the max branch is the
    grayscale expression and no clipping occurs, the expression below is the
    exact inverse.  The network learns a bounded strength for this prior and
    therefore does not have to rediscover the global transform from 800 images.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("expected RGB images with shape [batch, 3, height, width]")
    mix_rgb = 0.55
    mix_gray = 0.45 * 1.5
    offset = 0.45 * 0.5
    shifted = images - offset
    gray_shifted = (
        0.2989 * shifted[:, 0:1]
        + 0.5870 * shifted[:, 1:2]
        + 0.1140 * shifted[:, 2:3]
    )
    estimated_gray = gray_shifted / (mix_rgb + mix_gray)
    restored = (shifted - mix_gray * estimated_gray) / mix_rgb
    return restored.clamp(0.0, 1.0)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv1 = nn.Conv2d(
            channels, channels, 3, padding=dilation, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.gelu(self.norm1(inputs)))
        hidden = self.conv2(F.gelu(self.norm2(hidden)))
        return inputs + hidden


class SnowSpecialist(nn.Module):
    """Small snow-specific restorer with an explicit preservation gate."""

    def __init__(
        self,
        width: int = 32,
        max_gain: float = 0.45,
        max_bias: float = 0.45,
        max_local_residual: float = 0.60,
    ):
        super().__init__()
        if width % 4:
            raise ValueError("width must be divisible by four")
        self.width = int(width)
        self.max_gain = float(max_gain)
        self.max_bias = float(max_bias)
        self.max_local_residual = float(max_local_residual)
        self.stem = nn.Conv2d(9, width, 3, padding=1)
        self.blocks = nn.Sequential(
            ResidualBlock(width, 1),
            ResidualBlock(width, 2),
            ResidualBlock(width, 4),
            ResidualBlock(width, 2),
            ResidualBlock(width, 1),
        )
        self.gate_head = nn.Conv2d(width, 1, 3, padding=1)
        self.local_head = nn.Conv2d(width, 3, 3, padding=1)
        self.global_head = nn.Sequential(
            nn.Linear(width + 3, width),
            nn.GELU(),
            nn.Linear(width, 7),
        )

        # Exact identity at initialization.  This makes accidental application
        # of an untrained checkpoint benign and gives a deterministic baseline.
        nn.init.zeros_(self.local_head.weight)
        nn.init.zeros_(self.local_head.bias)
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)

    def forward(
        self, images: torch.Tensor, *, return_aux: bool = False
    ) -> torch.Tensor | SnowSpecialistOutput:
        evidence = snow_evidence(images)
        hidden = self.blocks(self.stem(evidence))

        pooled = torch.cat(
            (hidden.mean(dim=(-2, -1)), images.mean(dim=(-2, -1))), dim=1
        )
        global_parameters = self.global_head(pooled)
        gain, bias, tone_logit = global_parameters[:, :3], global_parameters[:, 3:6], global_parameters[:, 6:]
        gain = self.max_gain * torch.tanh(gain)[:, :, None, None]
        bias = self.max_bias * torch.tanh(bias)[:, :, None, None]
        tone_strength = torch.tanh(tone_logit)[:, :, None, None]
        tone_prior = analytic_snow_tone_inverse(images)
        global_delta = tone_strength * (tone_prior - images) + images * gain + bias

        gate = torch.sigmoid(self.gate_head(hidden))
        local_proposal = self.max_local_residual * torch.tanh(self.local_head(hidden))
        local_delta = gate * local_proposal
        restored = (images + global_delta + local_delta).clamp(0.0, 1.0)
        if return_aux:
            return SnowSpecialistOutput(restored, gate, global_delta, local_delta)
        return restored


def decompose_snow_target(
    source: torch.Tensor, target: torch.Tensor, kernel_size: int = 31
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split paired restoration residual into dense tone and local occlusion.

    The soft target mask is derived only from the development pair and is never
    needed by ``SnowSpecialist.forward``.  Per-image normalization prevents a
    globally bright snow transform from labeling every pixel as an occlusion.
    """
    if source.shape != target.shape or source.ndim != 4:
        raise ValueError("source and target must have the same BCHW shape")
    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd and greater than one")
    correction = target - source
    padding = kernel_size // 2
    global_target = F.avg_pool2d(
        correction, kernel_size, stride=1, padding=padding, count_include_pad=False
    )
    local_target = correction - global_target
    energy = local_target.abs().mean(dim=1, keepdim=True)
    scale = torch.quantile(energy.flatten(1), 0.75, dim=1).clamp_min(0.02)
    mask = (energy / (1.5 * scale[:, None, None, None])).clamp(0.0, 1.0)
    return global_target, local_target, mask


def snow_specialist_loss(
    output: SnowSpecialistOutput,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    semantic_loss: torch.Tensor | None = None,
    semantic_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Paired, mask-aware objective with no ImageNet category labels."""
    global_target, local_target, mask_target = decompose_snow_target(source, target)
    eps = 1e-6
    reconstruction = torch.sqrt((output.restored - target).square() + eps).mean()
    global_loss = torch.sqrt(
        (output.global_delta - global_target).square() + eps
    ).mean()
    local_weight = 1.0 + 3.0 * mask_target
    local_loss = (
        torch.sqrt((output.local_delta - local_target).square() + eps) * local_weight
    ).mean() / local_weight.mean()
    # BCE on probabilities is intentionally computed in FP32 because PyTorch
    # rejects this operation under autocast (the rest of the objective remains
    # mixed precision).  The target is soft, so a logits-only BCE replacement
    # would otherwise require changing the public diagnostic output contract.
    with torch.autocast(device_type=output.gate.device.type, enabled=False):
        gate_loss = F.binary_cross_entropy(output.gate.float(), mask_target.float())
    leakage = (
        output.local_delta.abs() * (1.0 - mask_target)
    ).mean()
    dx = (output.restored[..., 1:] - output.restored[..., :-1]) - (
        target[..., 1:] - target[..., :-1]
    )
    dy = (output.restored[..., 1:, :] - output.restored[..., :-1, :]) - (
        target[..., 1:, :] - target[..., :-1, :]
    )
    gradient = dx.abs().mean() + dy.abs().mean()
    if semantic_loss is None:
        semantic_loss = reconstruction.new_zeros(())
    total = (
        reconstruction
        + 0.35 * global_loss
        + 0.75 * local_loss
        + 0.10 * gate_loss
        + 0.25 * leakage
        + 0.10 * gradient
        + float(semantic_weight) * semantic_loss
    )
    terms = {
        "total": total,
        "reconstruction": reconstruction,
        "global": global_loss,
        "local": local_loss,
        "gate": gate_loss,
        "leakage": leakage,
        "gradient": gradient,
        "semantic": semantic_loss,
    }
    return total, terms
