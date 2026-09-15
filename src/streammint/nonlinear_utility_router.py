"""Nonlinear direct-utility router with one deterministic model per expert."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


SEED = 1109


@dataclass
class NonlinearUtilityRouter:
    """Frozen per-expert regressors followed by a conservative identity gate."""

    expert_names: tuple[str, ...]
    models: tuple[Any | None, ...]
    fitted: np.ndarray
    feature_width: int
    feature_mode: str = "mean_std_count"
    utility_threshold: float = 0.0
    min_margin: float = 0.0
    seed: int = SEED
    model_parameters: dict[str, Any] | None = None

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        utilities: np.ndarray,
        observed: np.ndarray,
        expert_names: Sequence[str],
        *,
        feature_mode: str = "mean_std_count",
        seed: int = SEED,
        model_parameters: dict[str, Any] | None = None,
    ) -> "NonlinearUtilityRouter":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(utilities, dtype=np.float64)
        mask = np.asarray(observed, dtype=bool)
        names = tuple(str(value) for value in expert_names)
        if seed != SEED:
            raise ValueError("nonlinear utility router requires seed 1109")
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("features must be a finite matrix")
        if y.shape != mask.shape or y.shape != (len(x), len(names)):
            raise ValueError("utilities/observed must have shape [N,num_experts]")
        parameters = {
            "loss": "squared_error",
            "learning_rate": 0.05,
            "max_iter": 150,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 40,
            "l2_regularization": 1.0,
            "early_stopping": False,
            "random_state": seed,
        }
        if model_parameters is not None:
            unsupported = set(model_parameters) - set(parameters)
            if unsupported:
                raise ValueError(f"unsupported model parameters: {sorted(unsupported)}")
            parameters.update(model_parameters)
        models: list[Any | None] = []
        fitted = np.zeros(len(names), dtype=bool)
        for expert in range(len(names)):
            selected = mask[:, expert] & np.isfinite(y[:, expert])
            if selected.sum() < max(2, int(parameters["min_samples_leaf"])):
                models.append(None)
                continue
            model = HistGradientBoostingRegressor(**parameters)
            model.fit(x[selected], y[selected, expert])
            models.append(model)
            fitted[expert] = True
        if not fitted.any():
            raise ValueError("no expert has enough observed utility labels")
        return cls(
            expert_names=names,
            models=tuple(models),
            fitted=fitted,
            feature_width=x.shape[1],
            feature_mode=feature_mode,
            seed=seed,
            model_parameters=parameters,
        )

    def predict_utility(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.feature_width:
            raise ValueError(f"expected feature width {self.feature_width}, got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("features contain non-finite values")
        predicted = np.full((len(x), len(self.expert_names)), -np.inf, dtype=np.float64)
        for index, model in enumerate(self.models):
            if model is not None and self.fitted[index]:
                predicted[:, index] = model.predict(x)
        return predicted

    def select(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predicted = self.predict_utility(features)
        best = np.argmax(predicted, axis=1)
        best_value = predicted[np.arange(len(predicted)), best]
        second = (
            np.partition(predicted, -2, axis=1)[:, -2]
            if predicted.shape[1] > 1
            else np.zeros(len(predicted), dtype=np.float64)
        )
        margin = best_value - np.maximum(second, 0.0)
        accepted = (best_value > self.utility_threshold) & (margin >= self.min_margin)
        return np.where(accepted, best, -1), predicted

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path, compress=3, protocol=5)

    @classmethod
    def load(cls, path: str | Path) -> "NonlinearUtilityRouter":
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError("joblib artifact is not a NonlinearUtilityRouter")
        if value.seed != SEED:
            raise RuntimeError("nonlinear router artifact seed mismatch")
        if len(value.models) != len(value.expert_names) or value.fitted.shape != (
            len(value.expert_names),
        ):
            raise RuntimeError("nonlinear router artifact schema mismatch")
        return value
