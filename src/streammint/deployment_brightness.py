"""Deployment-supervised brightness restoration with image-only inference.

The architecture retains the clipping-aware analytic support constraint. Its
training objective may use deployment-domain clean pairs and ImageNet class
labels, while ``forward`` accepts only an RGB tensor and never a label or
corruption identifier.
"""
from __future__ import annotations

import torch

from .brightness_specialist import (
    BrightnessClippingSpecialist,
    BrightnessSpecialistOutput,
    brightness_specialist_loss,
)


class DeploymentBrightnessRestorer(BrightnessClippingSpecialist):
    """Clipping-aware restorer trained for a declared deployment domain."""


def deployment_brightness_loss(
    output: BrightnessSpecialistOutput,
    target: torch.Tensor,
    semantic_cross_entropy: torch.Tensor,
    *,
    semantic_weight: float = 0.05,
    residual_weight: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reconstruction + task-semantic + conservative residual objective."""
    if semantic_cross_entropy.ndim != 0 or not torch.isfinite(semantic_cross_entropy):
        raise ValueError("semantic_cross_entropy must be a finite scalar")
    base, terms = brightness_specialist_loss(output, target, semantic_weight=0.0)
    clipped_count = output.clipped_mask.sum().clamp_min(1.0)
    learned_residual = (
        (output.context_value - output.context_prior).square() * output.clipped_mask
    ).sum() / clipped_count
    total = (
        base
        + float(semantic_weight) * semantic_cross_entropy
        + float(residual_weight) * learned_residual
    )
    return total, {
        **terms,
        "total": total,
        "semantic_cross_entropy": semantic_cross_entropy,
        "learned_residual_regularizer": learned_residual,
    }
