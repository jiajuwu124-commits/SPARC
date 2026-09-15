from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


TEMPLATES = (
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
)


@torch.no_grad()
def build_text_weights(
    model: nn.Module, class_names: Sequence[str], templates: Sequence[str] = TEMPLATES
) -> torch.Tensor:
    device = next(model.parameters()).device
    weights = []
    for class_name in class_names:
        prompts = [template.format(class_name.replace("_", " ")) for template in templates]
        encoded = model.encode_text(clip.tokenize(prompts).to(device))
        encoded = F.normalize(encoded.float(), dim=-1).mean(dim=0)
        weights.append(F.normalize(encoded, dim=0))
    return torch.stack(weights)


def configure_visual_norm(model: nn.Module) -> list[nn.Parameter]:
    model.eval()
    model.requires_grad_(False)
    for module in model.visual.modules():
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()
            if module.weight is not None:
                module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


class WeightedMeanBank:
    def __init__(self, classes: int, dim: int, device: torch.device) -> None:
        self.class_sum = torch.zeros(classes, dim, device=device)
        self.class_mass = torch.zeros(classes, device=device)
        self.total_sum = torch.zeros(dim, device=device)
        self.total_mass = torch.tensor(0.0, device=device)

    @torch.no_grad()
    def update(
        self, features: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None
    ) -> None:
        features = features.detach().float()
        labels = labels.detach()
        if weights is None:
            weights = torch.ones(len(features), device=features.device)
        weights = weights.detach().float()
        weighted = features * weights[:, None]
        self.class_sum.index_add_(0, labels, weighted)
        self.class_mass.index_add_(0, labels, weights)
        self.total_sum.add_(weighted.sum(dim=0))
        self.total_mass.add_(weights.sum())

    def class_means(self) -> torch.Tensor:
        means = self.class_sum / self.class_mass.clamp_min(1e-12)[:, None]
        means[self.class_mass == 0] = 0
        return means

    def total_mean(self) -> torch.Tensor:
        return self.total_sum / self.total_mass.clamp_min(1e-12)

    def clear(self) -> None:
        self.class_sum.zero_()
        self.class_mass.zero_()
        self.total_sum.zero_()
        self.total_mass.zero_()


@dataclass
class AdapterDiagnostics:
    batches: int = 0
    mean_reliability: float = 0.0
    mean_present_classes: float = 0.0
    mean_inter_variance: float = 0.0
    mean_gradient_cosine: float = 0.0
    min_gradient_cosine: float = 1.0
    mean_fast_gate: float = 0.0
    max_fast_gate: float = 0.0

    def update(
        self,
        reliability: float,
        classes: int,
        inter_variance: float,
        gradient_cosine: float,
        fast_gate: float,
    ) -> None:
        self.batches += 1
        rate = 1.0 / self.batches
        self.mean_reliability += rate * (reliability - self.mean_reliability)
        self.mean_present_classes += rate * (classes - self.mean_present_classes)
        self.mean_inter_variance += rate * (inter_variance - self.mean_inter_variance)
        self.mean_gradient_cosine += rate * (gradient_cosine - self.mean_gradient_cosine)
        self.min_gradient_cosine = min(self.min_gradient_cosine, gradient_cosine)
        self.mean_fast_gate += rate * (fast_gate - self.mean_fast_gate)
        self.max_fast_gate = max(self.max_fast_gate, fast_gate)


@dataclass
class TDADiagnostics:
    """Compact provenance for the official TDA cache dynamics."""

    samples: int = 0
    positive_items: int = 0
    negative_items: int = 0


class TDAAdapter(nn.Module):
    """Training-free Dynamic Adapter (TDA) with the published ImageNet settings.

    This is a batched-feature, sequential-state implementation of the official
    CVPR 2024 code.  Images are encoded together for throughput, while cache
    insertion and prediction remain sample ordered exactly as in the reference
    implementation.
    """

    def __init__(
        self,
        model: nn.Module,
        text_weights: torch.Tensor,
        positive_capacity: int = 3,
        positive_alpha: float = 2.0,
        positive_beta: float = 5.0,
        negative_capacity: int = 2,
        negative_alpha: float = 0.117,
        negative_beta: float = 1.0,
        entropy_bounds: tuple[float, float] = (0.2, 0.5),
        mask_bounds: tuple[float, float] = (0.03, 1.0),
    ) -> None:
        super().__init__()
        self.model = model
        self.text_weights = text_weights.float()
        self.positive_capacity = int(positive_capacity)
        self.positive_alpha = float(positive_alpha)
        self.positive_beta = float(positive_beta)
        self.negative_capacity = int(negative_capacity)
        self.negative_alpha = float(negative_alpha)
        self.negative_beta = float(negative_beta)
        self.entropy_bounds = tuple(float(value) for value in entropy_bounds)
        self.mask_bounds = tuple(float(value) for value in mask_bounds)
        classes, dimension = self.text_weights.shape
        device = self.text_weights.device
        self.positive_features = torch.zeros(
            classes, self.positive_capacity, dimension, device=device
        )
        self.positive_entropies = torch.full(
            (classes, self.positive_capacity), float("inf"), device=device
        )
        self.positive_valid = torch.zeros(
            classes, self.positive_capacity, dtype=torch.bool, device=device
        )
        self.negative_features = torch.zeros(
            classes, self.negative_capacity, dimension, device=device
        )
        self.negative_entropies = torch.full(
            (classes, self.negative_capacity), float("inf"), device=device
        )
        self.negative_valid = torch.zeros(
            classes, self.negative_capacity, dtype=torch.bool, device=device
        )
        self.negative_masks = torch.zeros(
            classes, self.negative_capacity, classes, device=device
        )
        self.last_grid_logits: dict[str, torch.Tensor] = {}
        self.batch_trace: list[dict[str, float | int]] = []
        self.diagnostics = TDADiagnostics()

    @staticmethod
    def _update_cache(
        features: torch.Tensor,
        entropies: torch.Tensor,
        valid: torch.Tensor,
        predicted_class: int,
        feature: torch.Tensor,
        entropy: float,
        value_masks: torch.Tensor | None = None,
        value_mask: torch.Tensor | None = None,
    ) -> None:
        empty = (~valid[predicted_class]).nonzero(as_tuple=False).flatten()
        if len(empty):
            slot = int(empty[0].item())
        else:
            slot = int(entropies[predicted_class].argmax().item())
            if entropy >= float(entropies[predicted_class, slot].item()):
                return
        features[predicted_class, slot].copy_(feature.detach().float())
        entropies[predicted_class, slot] = entropy
        valid[predicted_class, slot] = True
        if value_masks is not None:
            assert value_mask is not None
            value_masks[predicted_class, slot].copy_(value_mask.detach().float())

    def _cache_logits(
        self,
        feature: torch.Tensor,
        features: torch.Tensor,
        valid: torch.Tensor,
        alpha: float,
        beta: float,
        value_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat_features = features.flatten(0, 1)
        flat_valid = valid.flatten().float()
        affinity = feature @ flat_features.T
        weights = torch.exp(-beta + beta * affinity) * flat_valid
        if value_masks is None:
            return alpha * weights.view_as(valid).sum(dim=1)
        return alpha * (weights @ value_masks.flatten(0, 1))

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type="cuda", enabled=images.is_cuda):
            features = F.normalize(self.model.encode_image(images).float(), dim=-1)
        source_logits = 100.0 * features @ self.text_weights.T
        adapted: list[torch.Tensor] = []
        entropy_normalizer = math.log2(self.text_weights.shape[0])

        for feature, logits in zip(features, source_logits):
            probability = logits.softmax(dim=-1)
            entropy = float(
                (-(probability * probability.clamp_min(1e-12).log()).sum()).item()
            )
            predicted_class = int(logits.argmax().item())
            self._update_cache(
                self.positive_features,
                self.positive_entropies,
                self.positive_valid,
                predicted_class,
                feature,
                entropy,
            )
            proportional_entropy = entropy / entropy_normalizer
            if self.entropy_bounds[0] < proportional_entropy < self.entropy_bounds[1]:
                lower, upper = self.mask_bounds
                self._update_cache(
                    self.negative_features,
                    self.negative_entropies,
                    self.negative_valid,
                    predicted_class,
                    feature,
                    entropy,
                    self.negative_masks,
                    (probability > lower) & (probability < upper),
                )

            final_logits = logits.float().clone()
            final_logits += self._cache_logits(
                feature,
                self.positive_features,
                self.positive_valid,
                self.positive_alpha,
                self.positive_beta,
            )
            if self.negative_valid.any():
                final_logits -= self._cache_logits(
                    feature,
                    self.negative_features,
                    self.negative_valid,
                    self.negative_alpha,
                    self.negative_beta,
                    self.negative_masks,
                )
            adapted.append(final_logits)

        self.diagnostics.samples += len(images)
        self.diagnostics.positive_items = int(self.positive_valid.sum().item())
        self.diagnostics.negative_items = int(self.negative_valid.sum().item())
        return torch.stack(adapted)

    def prototype_diagnostics(self) -> dict[str, int]:
        return {
            "positive_classes": int(self.positive_valid.any(dim=1).sum().item()),
            "positive_items": self.diagnostics.positive_items,
            "negative_classes": int(self.negative_valid.any(dim=1).sum().item()),
            "negative_items": self.diagnostics.negative_items,
        }

    def reset(self) -> None:
        self.positive_features.zero_()
        self.positive_entropies.fill_(float("inf"))
        self.positive_valid.zero_()
        self.negative_features.zero_()
        self.negative_entropies.fill_(float("inf"))
        self.negative_valid.zero_()
        self.negative_masks.zero_()
        self.last_grid_logits = {}
        self.batch_trace = []
        self.diagnostics = TDADiagnostics()


class StreamingAdapter(nn.Module):
    """Independent implementation of Mint and support-calibrated variants.

    ``mint`` follows the published cumulative mean/gradient equations and global
    prototype prior. ``support`` changes only the inference prototype shrinkage.
    ``robust`` additionally weights pseudo-observations by temperature-scaled
    normalized entropy and uses a clipped gradient innovation.
    """

    def __init__(
        self,
        model: nn.Module,
        text_weights: torch.Tensor,
        method: str = "mint",
        lr: float = 0.015,
        prior: float = 10000.0,
        class_prior: float = 10.0,
        reliability_temperature: float = 10.0,
        reliability_floor: float = 0.05,
        gradient_clip_ratio: float = 2.5,
        agreement_threshold: float = 0.15,
        agreement_temperature: float = 0.05,
        agreement_grid: Sequence[tuple[float, float]] = (),
        gradient_ema: float = 0.9,
        gradient_cosine_threshold: float = 0.5,
        gradient_gate_temperature: float = 0.1,
        center_strength: float = 1.0,
        center_grid: Sequence[float] = (),
    ) -> None:
        super().__init__()
        if method not in {
            "mint", "encoder", "support", "robust", "agreement", "ema", "dual",
            "dn", "centered", "adaptive_centered"
        }:
            raise ValueError(f"unknown method: {method}")
        self.model = model
        self.text_weights = text_weights.float()
        self.method = method
        self.prior = float(prior)
        self.class_prior = float(class_prior)
        self.reliability_temperature = float(reliability_temperature)
        self.reliability_floor = float(reliability_floor)
        self.gradient_clip_ratio = float(gradient_clip_ratio)
        self.agreement_threshold = float(agreement_threshold)
        self.agreement_temperature = float(agreement_temperature)
        self.agreement_grid = tuple((float(a), float(b)) for a, b in agreement_grid)
        self.gradient_ema = float(gradient_ema)
        self.gradient_cosine_threshold = float(gradient_cosine_threshold)
        self.gradient_gate_temperature = float(gradient_gate_temperature)
        self.center_strength = float(center_strength)
        self.center_grid = tuple(float(value) for value in center_grid)
        self.last_grid_logits: dict[str, torch.Tensor] = {}

        self.parameters_to_adapt = configure_visual_norm(model)
        self.source_parameters = [p.detach().clone() for p in self.parameters_to_adapt]
        self.optimizer = torch.optim.Adam(self.parameters_to_adapt, lr=lr)
        classes, dim = self.text_weights.shape
        device = self.text_weights.device
        self.pre_bank = WeightedMeanBank(classes, dim, device)
        self.post_bank = WeightedMeanBank(classes, dim, device)
        self.gradient_mean: torch.Tensor | None = None
        self.gradient_fast: torch.Tensor | None = None
        self.gradient_batches = 0
        self.gradient_scale = torch.tensor(0.0, device=device)
        self.last_gradient_cosine = 1.0
        self.last_fast_gate = 0.0
        self.diagnostics = AdapterDiagnostics()
        self.batch_trace: list[dict[str, float | int]] = []

    @torch.no_grad()
    def _batch_shift_signals(
        self, features: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, float | int]:
        """Return label-free stream diagnostics before updating the state banks."""
        batch_mean = features.detach().float().mean(dim=0)
        batch_hist = torch.bincount(
            labels.detach(), minlength=self.text_weights.shape[0]
        ).float()
        batch_hist /= batch_hist.sum().clamp_min(1.0)
        probabilities = F.softmax(logits.detach().float() / 10.0, dim=-1)
        entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1) / math.log(probabilities.shape[-1])

        if self.pre_bank.total_mass.item() == 0:
            mean_cosine = 1.0
            mean_l2 = 0.0
            label_js = 0.0
        else:
            reference_mean = self.pre_bank.total_mean()
            mean_cosine = float(F.cosine_similarity(
                batch_mean[None], reference_mean[None], dim=-1
            ).item())
            mean_l2 = float((batch_mean - reference_mean).norm().item())
            reference_hist = self.pre_bank.class_mass / self.pre_bank.total_mass
            midpoint = 0.5 * (batch_hist + reference_hist)
            label_js = float((
                0.5 * (
                    batch_hist
                    * (batch_hist.clamp_min(1e-12) / midpoint.clamp_min(1e-12)).log()
                ).sum()
                + 0.5 * (
                    reference_hist
                    * (reference_hist.clamp_min(1e-12) / midpoint.clamp_min(1e-12)).log()
                ).sum()
            ).item())

        top2 = logits.detach().float().topk(2, dim=-1).values
        return {
            "batch": len(self.batch_trace),
            "feature_mean_norm": float(batch_mean.norm().item()),
            "feature_mean_cosine_reference": mean_cosine,
            "feature_mean_l2_reference": mean_l2,
            "pseudo_label_js_reference": label_js,
            "normalized_entropy": float(entropy.mean().item()),
            "logit_margin": float((top2[:, 0] - top2[:, 1]).mean().item()),
        }

    def _encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = F.normalize(self.model.encode_image(images).float(), dim=-1)
        logits = 100.0 * features @ self.text_weights.T
        return features, logits

    def _reliability(self, logits: torch.Tensor) -> torch.Tensor:
        if self.method != "robust":
            return torch.ones(logits.shape[0], device=logits.device)
        probabilities = F.softmax(logits / self.reliability_temperature, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        normalized = 1.0 - entropy / math.log(probabilities.shape[-1])
        return normalized.clamp(self.reliability_floor, 1.0)

    def _accumulate_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        gradient = gradient.detach().float()
        if self.gradient_mean is None:
            self.gradient_mean = gradient.clone()
            self.gradient_fast = gradient.clone()
            self.gradient_scale = gradient.norm().detach()
            self.last_gradient_cosine = 1.0
            self.last_fast_gate = 0.0
        else:
            cosine = F.cosine_similarity(
                gradient[None], self.gradient_mean[None], dim=-1
            ).item()
            self.last_gradient_cosine = cosine
            if self.method == "robust":
                innovation = gradient - self.gradient_mean
                threshold = self.gradient_clip_ratio * self.gradient_scale.clamp_min(1e-12)
                factor = (threshold / innovation.norm().clamp_min(1e-12)).clamp(max=1.0)
                gradient = self.gradient_mean + factor * innovation
            n = self.gradient_batches
            self.gradient_mean.mul_(n / (n + 1)).add_(gradient, alpha=1.0 / (n + 1))
            assert self.gradient_fast is not None
            self.gradient_fast.mul_(self.gradient_ema).add_(
                gradient, alpha=1.0 - self.gradient_ema
            )
            scale = gradient.norm().detach()
            self.gradient_scale.mul_(0.95).add_(scale, alpha=0.05)
        self.gradient_batches += 1
        if self.method == "ema":
            self.last_fast_gate = 1.0
            assert self.gradient_fast is not None
            return self.gradient_fast
        if self.method == "dual" and self.gradient_batches > 1:
            gate = torch.sigmoid(torch.tensor(
                (self.gradient_cosine_threshold - self.last_gradient_cosine)
                / self.gradient_gate_temperature,
                device=gradient.device,
            ))
            self.last_fast_gate = float(gate.item())
            assert self.gradient_fast is not None
            return (1.0 - gate) * self.gradient_mean + gate * self.gradient_fast
        return self.gradient_mean

    @torch.no_grad()
    def _restore_source(self) -> None:
        for parameter, source in zip(self.parameters_to_adapt, self.source_parameters):
            parameter.copy_(source)
        self.optimizer.state.clear()

    def _agreement_weights(self, threshold: float, temperature: float) -> torch.Tensor:
        means = self.post_bank.class_means()
        global_ratio = self.post_bank.total_mass / (self.prior + self.post_bank.total_mass)
        directions = F.normalize(means, dim=-1)
        alignment = (directions * self.text_weights).sum(dim=-1)
        concentration = means.norm(dim=-1)
        evidence = alignment.clamp_min(0.0) * concentration
        gate = torch.sigmoid((evidence - threshold) / temperature)[:, None]
        ratio = global_ratio * gate
        mixed = (1.0 - ratio) * self.text_weights + ratio * means
        return F.normalize(mixed, dim=-1)

    def _prototype_weights(self) -> torch.Tensor:
        means = self.post_bank.class_means()
        if self.method == "encoder":
            mixed = self.text_weights
        elif self.method in {
            "mint", "robust", "ema", "dual", "centered", "adaptive_centered"
        }:
            ratio = self.post_bank.total_mass / (self.prior + self.post_bank.total_mass)
            mixed = (1.0 - ratio) * self.text_weights + ratio * means
        elif self.method == "support":
            mass = self.post_bank.class_mass[:, None]
            ratio = mass / (self.class_prior + mass)
            mixed = (1.0 - ratio) * self.text_weights + ratio * means
        else:
            return self._agreement_weights(
                self.agreement_threshold, self.agreement_temperature
            )
        return F.normalize(mixed, dim=-1)

    def _centered_logits(
        self, features: torch.Tensor, weights: torch.Tensor, strength: float
    ) -> torch.Tensor:
        image_mean = self.post_bank.total_mean()
        text_mean = weights.mean(dim=0)
        centered_features = F.normalize(features - strength * image_mean, dim=-1)
        centered_weights = F.normalize(weights - strength * text_mean, dim=-1)
        return 100.0 * centered_features @ centered_weights.T

    @torch.no_grad()
    def prototype_diagnostics(self) -> dict[str, float | int | list[float]]:
        seen = self.post_bank.class_mass > 0
        if not seen.any():
            return {"seen_classes": 0}
        means = self.post_bank.class_means()[seen]
        text = self.text_weights[seen]
        concentration = means.norm(dim=-1)
        alignment = (F.normalize(means, dim=-1) * text).sum(dim=-1)
        evidence = concentration * alignment.clamp_min(0.0)
        mass = self.post_bank.class_mass[seen]

        def summary(values: torch.Tensor) -> list[float]:
            quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=values.device)
            return [float(v) for v in torch.quantile(values.float(), quantiles).cpu()]

        return {
            "seen_classes": int(seen.sum().item()),
            "mass_quantiles": summary(mass),
            "concentration_quantiles": summary(concentration),
            "alignment_quantiles": summary(alignment),
            "evidence_quantiles": summary(evidence),
            "image_mean_norm": float(self.post_bank.total_mean().norm().item()),
            "text_mean_norm": float(self.text_weights.mean(dim=0).norm().item()),
            "image_text_mean_cosine": float(
                F.cosine_similarity(
                    self.post_bank.total_mean()[None],
                    self.text_weights.mean(dim=0)[None],
                    dim=-1,
                ).item()
            ),
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.method == "dn":
            with torch.no_grad(), torch.amp.autocast(
                device_type="cuda", enabled=images.is_cuda
            ):
                features, logits = self._encode(images)
                labels = logits.argmax(dim=-1)
                self.post_bank.update(features, labels)
                adapted_logits = self._centered_logits(
                    features, self.text_weights, self.center_strength
                )
                self.last_grid_logits = {
                    f"center={value:g}": self._centered_logits(
                        features, self.text_weights, value
                    )
                    for value in self.center_grid
                }
            return adapted_logits

        amp = torch.amp.autocast(device_type="cuda", enabled=images.is_cuda)
        with amp:
            features, logits = self._encode(images)
            labels = logits.argmax(dim=-1)
            trace_row = self._batch_shift_signals(features, logits, labels)
            reliability = self._reliability(logits)
            self.pre_bank.update(features, labels, reliability)

            class_means = self.pre_bank.class_means()
            total_mean = self.pre_bank.total_mean()
            present = torch.unique(labels)
            current_mass = torch.zeros(
                self.text_weights.shape[0], device=features.device, dtype=torch.float32
            )
            current_mass.index_add_(0, labels, reliability)
            sample_weights = reliability / current_mass[labels].clamp_min(1e-12)
            sample_weights = sample_weights / len(present)
            intra = ((features - class_means[labels]) ** 2).sum(dim=-1)
            total = ((features - total_mean) ** 2).sum(dim=-1)
            inter = ((total - intra) * sample_weights).sum()
            loss = -inter

        loss.backward()
        flat = torch.cat(
            [parameter.grad.detach().float().reshape(-1) for parameter in self.parameters_to_adapt]
        )
        aggregate = self._accumulate_gradient(flat)
        offset = 0
        with torch.no_grad():
            for parameter in self.parameters_to_adapt:
                count = parameter.numel()
                parameter.grad.copy_(aggregate[offset : offset + count].reshape_as(parameter))
                offset += count
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=images.is_cuda):
            post_features, post_logits = self._encode(images)
            post_labels = post_logits.argmax(dim=-1)
            post_reliability = self._reliability(post_logits)
            self.post_bank.update(post_features, post_labels, post_reliability)
            prototype_weights = self._prototype_weights()
            if self.method in {"centered", "adaptive_centered"}:
                adapted_logits = self._centered_logits(
                    post_features, prototype_weights, self.center_strength
                )
            else:
                adapted_logits = 100.0 * post_features @ prototype_weights.T
            self.last_grid_logits = {}
            if self.method == "agreement":
                for threshold, temperature in self.agreement_grid:
                    key = f"threshold={threshold:g},temperature={temperature:g}"
                    weights = self._agreement_weights(threshold, temperature)
                    self.last_grid_logits[key] = 100.0 * post_features @ weights.T
            if self.method in {"centered", "adaptive_centered"}:
                candidate_logits = []
                for value in self.center_grid:
                    key = f"center={value:g}"
                    logits_for_value = self._centered_logits(
                        post_features, prototype_weights, value
                    )
                    self.last_grid_logits[key] = logits_for_value
                    candidate_logits.append(logits_for_value)
                if candidate_logits:
                    stacked = torch.stack(candidate_logits, dim=0)
                    top2 = stacked.topk(2, dim=-1).values
                    margins = top2[..., 0] - top2[..., 1]
                    batch_margin_idx = margins.mean(dim=1).argmax()
                    self.last_grid_logits["select=batch_margin"] = stacked[batch_margin_idx]

                    probabilities = F.softmax(stacked / 10.0, dim=-1)
                    entropies = -(
                        probabilities * probabilities.clamp_min(1e-12).log()
                    ).sum(dim=-1)
                    batch_entropy_idx = entropies.mean(dim=1).argmin()
                    self.last_grid_logits["select=batch_entropy"] = stacked[
                        batch_entropy_idx
                    ]

                    sample_index = torch.arange(stacked.shape[1], device=stacked.device)
                    sample_margin_idx = margins.argmax(dim=0)
                    self.last_grid_logits["select=sample_margin"] = stacked[
                        sample_margin_idx, sample_index
                    ]
                    sample_entropy_idx = entropies.argmin(dim=0)
                    self.last_grid_logits["select=sample_entropy"] = stacked[
                        sample_entropy_idx, sample_index
                    ]
                    if self.method == "adaptive_centered":
                        adapted_logits = stacked[batch_margin_idx]

        self.diagnostics.update(
            reliability.mean().item(),
            len(present),
            inter.detach().item(),
            self.last_gradient_cosine,
            self.last_fast_gate,
        )
        trace_row["inter_variance"] = float(inter.detach().item())
        trace_row["gradient_cosine_reference"] = float(self.last_gradient_cosine)
        self.batch_trace.append(trace_row)
        self._restore_source()
        return adapted_logits

    def reset(self) -> None:
        self._restore_source()
        self.pre_bank.clear()
        self.post_bank.clear()
        self.gradient_mean = None
        self.gradient_fast = None
        self.gradient_batches = 0
        self.gradient_scale.zero_()
        self.last_gradient_cosine = 1.0
        self.last_fast_gate = 0.0
        self.diagnostics = AdapterDiagnostics()
        self.last_grid_logits = {}
        self.batch_trace = []
