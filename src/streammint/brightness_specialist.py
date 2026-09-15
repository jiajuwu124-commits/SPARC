"""Clipping-aware specialist for ImageNet-C brightness at severity five.

The corruption adds 0.5 to HSV value and clips the result.  Consequently the
inverse is known exactly (up to uint8 quantization) wherever the observed value
is below one.  Only clipped pixels are ambiguous.  This module hard-codes that
separation: an analytic branch handles invertible pixels, while a small context
network predicts value only inside the clipped support.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BrightnessSpecialistOutput:
    restored: torch.Tensor
    clipped_mask: torch.Tensor
    analytic_rgb: torch.Tensor
    analytic_value: torch.Tensor
    context_value: torch.Tensor
    context_gate: torch.Tensor
    context_prior: torch.Tensor


def clipping_aware_analytic_inverse(
    images: torch.Tensor,
    *,
    offset: float = 0.5,
    saturation_threshold: float = 254.5 / 255.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invert the known value shift outside its clipped support.

    Scaling RGB by the ratio of old and new value preserves HSV hue and
    saturation.  Pixels whose observed maximum channel is 255 are marked as
    ambiguous; their provisional value is the clipping boundary (0.5).
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("expected RGB images with shape [batch, 3, height, width]")
    if not images.is_floating_point():
        raise TypeError("expected floating-point RGB values in [0, 1]")
    if not 0.0 < offset < 1.0:
        raise ValueError("offset must be between zero and one")
    observed_value = images.amax(dim=1, keepdim=True)
    clipped_mask = (observed_value >= saturation_threshold).to(images.dtype)
    analytic_value = (observed_value - offset).clamp(0.0, 1.0)
    analytic_rgb = images * (analytic_value / observed_value.clamp_min(1e-6))
    return analytic_rgb, analytic_value, clipped_mask


def _masked_local_mean(
    values: torch.Tensor, known: torch.Tensor, kernel_size: int, fallback: float
) -> tuple[torch.Tensor, torch.Tensor]:
    padding = kernel_size // 2
    numerator = F.avg_pool2d(
        values * known,
        kernel_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    )
    denominator = F.avg_pool2d(
        known,
        kernel_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    )
    mean = numerator / denominator.clamp_min(1e-6)
    mean = torch.where(denominator > 1e-4, mean, mean.new_full((), fallback))
    return mean, denominator


def multiscale_context_prior(
    analytic_value: torch.Tensor,
    clipped_mask: torch.Tensor,
    *,
    kernels: tuple[int, ...] = (15, 31, 63),
    fallback: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extrapolate clipped values from nearby invertible pixels.

    Known neighbors necessarily have clean value below 0.5.  Adding the
    clipping boundary to their weighted mean turns that local contrast cue into
    a prior over the lost interval [0.5, 1].  With no known context, the default
    contextual mean 0.25 yields the neutral midpoint prior 0.75.
    """
    if not kernels or any(kernel <= 1 or kernel % 2 == 0 for kernel in kernels):
        raise ValueError("context kernels must be non-empty odd integers > 1")
    known = 1.0 - clipped_mask
    means, coverages = zip(
        *(
            _masked_local_mean(analytic_value, known, kernel, fallback)
            for kernel in kernels
        )
    )
    # Prefer local evidence when it exists, then smoothly back off to wider
    # windows.  These weights depend only on observed clipping support.
    remaining = torch.ones_like(analytic_value)
    prior = torch.zeros_like(analytic_value)
    for mean, coverage in zip(means, coverages):
        weight = remaining * (coverage / 0.20).clamp(0.0, 1.0)
        prior = prior + weight * mean
        remaining = remaining - weight
    prior = 0.5 + prior + remaining * fallback
    coverage = torch.stack(coverages, dim=0).amax(dim=0)
    return prior.clamp(0.5, 1.0), coverage


class _ResidualBlock(nn.Module):
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


class BrightnessClippingSpecialist(nn.Module):
    """Analytic inverse plus a learned predictor restricted to clipped pixels."""

    def __init__(
        self,
        width: int = 32,
        offset: float = 0.5,
        saturation_threshold: float = 254.5 / 255.0,
        max_context_residual: float = 0.35,
    ):
        super().__init__()
        if width % 4:
            raise ValueError("width must be divisible by four")
        self.width = int(width)
        self.offset = float(offset)
        self.saturation_threshold = float(saturation_threshold)
        self.max_context_residual = float(max_context_residual)
        # RGB, provisional analytic RGB, observed V, saturation, clip mask,
        # three context priors, and widest-window known-pixel coverage.
        self.stem = nn.Conv2d(13, width, 3, padding=1)
        self.blocks = nn.Sequential(
            _ResidualBlock(width, 1),
            _ResidualBlock(width, 2),
            _ResidualBlock(width, 4),
            _ResidualBlock(width, 8),
            _ResidualBlock(width, 4),
            _ResidualBlock(width, 2),
            _ResidualBlock(width, 1),
        )
        self.value_head = nn.Conv2d(width, 1, 3, padding=1)
        self.gate_head = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def forward(
        self, images: torch.Tensor, *, return_aux: bool = False
    ) -> torch.Tensor | BrightnessSpecialistOutput:
        analytic_rgb, analytic_value, clipped_mask = clipping_aware_analytic_inverse(
            images,
            offset=self.offset,
            saturation_threshold=self.saturation_threshold,
        )
        observed_value = images.amax(dim=1, keepdim=True)
        observed_minimum = images.amin(dim=1, keepdim=True)
        saturation = (observed_value - observed_minimum) / observed_value.clamp_min(1e-6)
        priors = []
        coverages = []
        known = 1.0 - clipped_mask
        for kernel in (15, 31, 63):
            prior, coverage = _masked_local_mean(
                analytic_value, known, kernel, fallback=0.25
            )
            priors.append(prior.clamp(0.0, 0.5))
            coverages.append(coverage)
        context_prior, widest_coverage = multiscale_context_prior(
            analytic_value, clipped_mask
        )
        evidence = torch.cat(
            (
                images,
                analytic_rgb,
                observed_value,
                saturation,
                clipped_mask,
                *priors,
                widest_coverage,
            ),
            dim=1,
        )
        hidden = self.blocks(self.stem(evidence))
        proposal = (
            context_prior
            + self.max_context_residual * torch.tanh(self.value_head(hidden))
        ).clamp(0.5, 1.0)
        context_gate = torch.sigmoid(self.gate_head(hidden))
        # A low learned gate falls back to the deterministic multiscale prior.
        context_value = context_prior + context_gate * (proposal - context_prior)
        restored_value = (
            (1.0 - clipped_mask) * analytic_value + clipped_mask * context_value
        )
        restored = images * (restored_value / observed_value.clamp_min(1e-6))
        restored = restored.clamp(0.0, 1.0)
        if return_aux:
            return BrightnessSpecialistOutput(
                restored=restored,
                clipped_mask=clipped_mask,
                analytic_rgb=analytic_rgb,
                analytic_value=analytic_value,
                context_value=context_value,
                context_gate=context_gate,
                context_prior=context_prior,
            )
        return restored


def brightness_specialist_loss(
    output: BrightnessSpecialistOutput,
    target: torch.Tensor,
    *,
    semantic_loss: torch.Tensor | None = None,
    semantic_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Paired objective using RGB/structure and clean-feature supervision."""
    if output.restored.shape != target.shape:
        raise ValueError("restored and target tensors must have equal shapes")
    eps = 1e-6
    target_value = target.amax(dim=1, keepdim=True)
    reconstruction = torch.sqrt((output.restored - target).square() + eps).mean()
    saturated_weight = output.clipped_mask.sum().clamp_min(1.0)
    clipped_value = (
        torch.sqrt((output.context_value - target_value).square() + eps)
        * output.clipped_mask
    ).sum() / saturated_weight
    clipped_rgb = (
        torch.sqrt((output.restored - target).square() + eps)
        * output.clipped_mask
    ).sum() / (3.0 * saturated_weight)
    dx = (output.restored[..., 1:] - output.restored[..., :-1]) - (
        target[..., 1:] - target[..., :-1]
    )
    dy = (output.restored[..., 1:, :] - output.restored[..., :-1, :]) - (
        target[..., 1:, :] - target[..., :-1, :]
    )
    gradient = dx.abs().mean() + dy.abs().mean()
    # Discourage unnecessary high-confidence extrapolation where broad context
    # is weak; this operates only through the context branch.
    gate_regularizer = (
        output.context_gate * output.clipped_mask
    ).sum() / saturated_weight
    if semantic_loss is None:
        semantic_loss = reconstruction.new_zeros(())
    total = (
        reconstruction
        + 1.25 * clipped_value
        + 0.50 * clipped_rgb
        + 0.10 * gradient
        + 0.01 * gate_regularizer
        + float(semantic_weight) * semantic_loss
    )
    terms = {
        "total": total,
        "reconstruction": reconstruction,
        "clipped_value": clipped_value,
        "clipped_rgb": clipped_rgb,
        "gradient": gradient,
        "gate_regularizer": gate_regularizer,
        "semantic": semantic_loss,
        "clipped_fraction": output.clipped_mask.mean(),
    }
    return total, terms
