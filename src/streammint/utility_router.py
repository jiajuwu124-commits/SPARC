"""Direct expert-utility routing from masked-risk descriptors.

The router deliberately does not consume corruption names, image labels, or
backend predictions.  It estimates each expert's utility relative to identity
from the 18-D masked-risk descriptor and optional batch statistics that are
available at inference time.  Identity is selected whenever the best predicted
utility is too small or insufficiently separated from the alternatives.

This module is NumPy-only so that fitting, auditing, and inference can be run on
CPU without adding a new dependency to the release package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


RISK_FEATURE_ORDER = tuple(
    [f"log10_risk_v{view}_{loss}" for view in range(4) for loss in ("mse", "mae")]
    + [f"normalized_risk_v{view}_{loss}" for view in range(4) for loss in ("mse", "mae")]
    + ["identity_gain_mse", "identity_gain_mae"]
)


def risk_features(risks: np.ndarray) -> np.ndarray:
    """Convert [N, 4 probes, (MSE, MAE)] risks into the frozen 18-D descriptor."""
    risks = np.asarray(risks)
    if risks.ndim != 3 or risks.shape[1:] != (4, 2):
        raise ValueError(f"expected risks with shape [N,4,2], got {risks.shape}")
    if not np.isfinite(risks).all() or np.any(risks < 0):
        raise ValueError("risks must be finite and non-negative")
    lower = risks.min(axis=1, keepdims=True)
    span = np.maximum(risks.max(axis=1, keepdims=True) - lower, 1e-8)
    normalized = (risks - lower) / span
    gain = (risks[:, 0] - risks.min(axis=1)) / np.maximum(risks[:, 0], 1e-8)
    return np.concatenate(
        (
            np.log10(risks.reshape(len(risks), -1) + 1e-10),
            normalized.reshape(len(risks), -1),
            gain,
        ),
        axis=1,
    ).astype(np.float64)


def augment_batch_statistics(
    features: np.ndarray,
    batch_ids: Sequence[object] | np.ndarray,
    mode: str = "mean_std_count",
) -> np.ndarray:
    """Append inference-available batch statistics without using targets.

    Statistics are computed independently for every batch identifier.  The
    returned row contains its own descriptor, the batch mean and standard
    deviation, and log(1 + batch size).  ``mode='none'`` returns a copy of the
    descriptor and is useful for the no-context ablation.
    """
    x = np.asarray(features, dtype=np.float64)
    batches = np.asarray(batch_ids)
    if x.ndim != 2 or x.shape[1] != len(RISK_FEATURE_ORDER):
        raise ValueError(f"expected [N,{len(RISK_FEATURE_ORDER)}] features, got {x.shape}")
    if batches.shape != (len(x),):
        raise ValueError("batch_ids must have one value per feature row")
    if mode == "none":
        return x.copy()
    if mode != "mean_std_count":
        raise ValueError(f"unknown batch-statistic mode: {mode}")

    means = np.empty_like(x)
    stds = np.empty_like(x)
    counts = np.empty((len(x), 1), dtype=np.float64)
    for batch in np.unique(batches):
        selected = batches == batch
        values = x[selected]
        means[selected] = values.mean(axis=0)
        stds[selected] = values.std(axis=0, ddof=0)
        counts[selected, 0] = np.log1p(len(values))
    return np.concatenate((x, means, stds, counts), axis=1)


def seeded_group_folds(
    groups: Sequence[object] | np.ndarray,
    n_splits: int = 5,
    seed: int = 1109,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield deterministic train/validation indices with disjoint groups."""
    groups_a = np.asarray(groups)
    if groups_a.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    unique, inverse, counts = np.unique(groups_a, return_inverse=True, return_counts=True)
    if n_splits < 2 or n_splits > len(unique):
        raise ValueError("n_splits must be between 2 and the number of groups")

    rng = np.random.default_rng(seed)
    tie_break = rng.random(len(unique))
    order = np.lexsort((tie_break, -counts))
    fold_load = np.zeros(n_splits, dtype=np.int64)
    assignment = np.empty(len(unique), dtype=np.int64)
    for group_index in order:
        fold = int(np.argmin(fold_load))
        assignment[group_index] = fold
        fold_load[fold] += counts[group_index]
    row_folds = assignment[inverse]
    all_indices = np.arange(len(groups_a))
    for fold in range(n_splits):
        validation = all_indices[row_folds == fold]
        training = all_indices[row_folds != fold]
        yield training, validation


def connected_component_groups(
    primary_groups: Sequence[object] | np.ndarray,
    batch_ids: Sequence[object] | np.ndarray,
) -> np.ndarray:
    """Build deterministic components that keep identities and batches intact."""
    primary = np.asarray(primary_groups)
    batches = np.asarray(batch_ids)
    if primary.ndim != 1 or batches.ndim != 1 or primary.shape != batches.shape:
        raise ValueError("primary_groups and batch_ids must be aligned vectors")
    unique_primary, inverse = np.unique(primary, return_inverse=True)
    parent = np.arange(len(unique_primary), dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for batch in np.unique(batches):
        members = np.unique(inverse[batches == batch])
        anchor = int(members[0])
        for member in members[1:]:
            union(anchor, int(member))
    roots = np.asarray([find(int(value)) for value in inverse], dtype=np.int64)
    unique_roots = np.unique(roots)
    remap = {int(root): index for index, root in enumerate(unique_roots.tolist())}
    return np.asarray([remap[int(root)] for root in roots], dtype=np.int64)


@dataclass
class UtilityRouter:
    """Independent ridge regressors followed by a conservative identity gate."""

    expert_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    fitted: np.ndarray
    ridge_alpha: float
    feature_mode: str = "mean_std_count"
    utility_threshold: float = 0.0
    min_margin: float = 0.0

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        utilities: np.ndarray,
        observed: np.ndarray,
        expert_names: Sequence[str],
        *,
        ridge_alpha: float = 1.0,
        feature_mode: str = "mean_std_count",
    ) -> "UtilityRouter":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(utilities, dtype=np.float64)
        mask = np.asarray(observed, dtype=bool)
        names = tuple(expert_names)
        if x.ndim != 2:
            raise ValueError("features must be a matrix")
        if y.shape != mask.shape or y.shape != (len(x), len(names)):
            raise ValueError("utilities/observed shape must be [N, num_experts]")
        if ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")

        mean = x.mean(axis=0)
        scale = x.std(axis=0, ddof=0)
        scale[scale < 1e-12] = 1.0
        z = (x - mean) / scale
        coefficients = np.zeros((len(names), x.shape[1]), dtype=np.float64)
        intercepts = np.zeros(len(names), dtype=np.float64)
        fitted = np.zeros(len(names), dtype=bool)
        for expert in range(len(names)):
            selected = mask[:, expert] & np.isfinite(y[:, expert])
            if selected.sum() < 2:
                continue
            design = np.column_stack((np.ones(selected.sum()), z[selected]))
            penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_alpha
            penalty[0, 0] = 0.0
            try:
                weights = np.linalg.solve(design.T @ design + penalty, design.T @ y[selected, expert])
            except np.linalg.LinAlgError:
                weights = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y[selected, expert]
            intercepts[expert] = weights[0]
            coefficients[expert] = weights[1:]
            fitted[expert] = True
        if not fitted.any():
            raise ValueError("no expert has enough observed utility labels")
        return cls(
            expert_names=names,
            mean=mean,
            scale=scale,
            coefficients=coefficients,
            intercepts=intercepts,
            fitted=fitted,
            ridge_alpha=float(ridge_alpha),
            feature_mode=feature_mode,
        )

    def predict_utility(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.mean.shape[0]:
            raise ValueError(f"expected feature width {self.mean.shape[0]}, got {x.shape}")
        predicted = ((x - self.mean) / self.scale) @ self.coefficients.T + self.intercepts
        predicted[:, ~self.fitted] = -np.inf
        return predicted

    def select(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return expert indices (-1 means identity/abstain) and predictions."""
        predicted = self.predict_utility(features)
        best = np.argmax(predicted, axis=1)
        best_value = predicted[np.arange(len(predicted)), best]
        if predicted.shape[1] > 1:
            second = np.partition(predicted, -2, axis=1)[:, -2]
        else:
            second = np.zeros(len(predicted), dtype=np.float64)
        margin = best_value - np.maximum(second, 0.0)
        accepted = (best_value > self.utility_threshold) & (margin >= self.min_margin)
        return np.where(accepted, best, -1), predicted

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "expert_names": np.asarray(self.expert_names),
            "mean": self.mean,
            "scale": self.scale,
            "coefficients": self.coefficients,
            "intercepts": self.intercepts,
            "fitted": self.fitted,
            "ridge_alpha": np.asarray(self.ridge_alpha),
            "feature_mode": np.asarray(self.feature_mode),
            "utility_threshold": np.asarray(self.utility_threshold),
            "min_margin": np.asarray(self.min_margin),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, np.ndarray]) -> "UtilityRouter":
        return cls(
            expert_names=tuple(str(value) for value in state["expert_names"].tolist()),
            mean=np.asarray(state["mean"], dtype=np.float64),
            scale=np.asarray(state["scale"], dtype=np.float64),
            coefficients=np.asarray(state["coefficients"], dtype=np.float64),
            intercepts=np.asarray(state["intercepts"], dtype=np.float64),
            fitted=np.asarray(state["fitted"], dtype=bool),
            ridge_alpha=float(np.asarray(state["ridge_alpha"]).item()),
            feature_mode=str(np.asarray(state["feature_mode"]).item()),
            utility_threshold=float(np.asarray(state["utility_threshold"]).item()),
            min_margin=float(np.asarray(state["min_margin"]).item()),
        )

    def save(self, path: str) -> None:
        np.savez_compressed(path, **self.state_dict())

    @classmethod
    def load(cls, path: str) -> "UtilityRouter":
        with np.load(path, allow_pickle=False) as stored:
            return cls.from_state_dict({key: stored[key] for key in stored.files})
