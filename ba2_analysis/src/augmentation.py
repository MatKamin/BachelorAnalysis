from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentationSpec:
    kind: str
    level: float

    @property
    def label(self) -> str:
        return f"{self.kind}={self.level:g}"


def add_gaussian_noise(X: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    if level <= 0:
        return X.copy()
    scale = np.std(X, axis=0, ddof=0) * level
    return X + rng.normal(0.0, scale, size=X.shape)


def random_dropout(X: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    if rate <= 0:
        return X.copy()
    out = X.copy()
    mask = rng.random(X.shape) < rate
    col_means = np.nanmean(X, axis=0)
    out[mask] = np.take(col_means, np.where(mask)[1])
    return out


def sensor_drift(X: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    if level <= 0:
        return X.copy()
    n, d = X.shape
    t = np.linspace(0, 1, n)[:, None]
    direction = rng.normal(0, 1, size=(1, d))
    drift = direction * t * level * np.std(X, axis=0)
    return X + drift


def apply_augmentation(X: np.ndarray, spec: AugmentationSpec,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    if spec.kind == "noise":
        return add_gaussian_noise(X, spec.level, rng)
    if spec.kind == "dropout":
        return random_dropout(X, spec.level, rng)
    if spec.kind == "drift":
        return sensor_drift(X, spec.level, rng)
    raise ValueError(f"unknown augmentation kind: {spec.kind}")
