from __future__ import annotations

import numpy as np


def knee_point(k_values: list[int], scores: list[float]) -> int:
    k = np.asarray(k_values, dtype=float)
    s = np.asarray(scores, dtype=float)
    if len(k) < 3:
        return int(k[0])
    s_norm = (s - s.min()) / max(s.max() - s.min(), 1e-12)
    k_norm = (k - k.min()) / max(k.max() - k.min(), 1e-12)
    diff = k_norm - s_norm
    return int(k[int(np.argmax(diff))])


def max_cluster_share(labels: np.ndarray) -> float:
    mask = labels != -1
    if not mask.any():
        return 1.0
    unique, counts = np.unique(labels[mask], return_counts=True)
    return float(counts.max() / mask.sum())


def is_degenerate(labels: np.ndarray, max_share: float = 0.9) -> bool:
    return max_cluster_share(labels) > max_share
