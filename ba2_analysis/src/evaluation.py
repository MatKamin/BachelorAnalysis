from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)


@dataclass
class InternalMetrics:
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    n_clusters: int
    n_noise: int


@dataclass
class ExternalMetrics:
    ari: float
    nmi: float
    v_measure: float
    fmi: float


def internal_metrics(X: np.ndarray, labels: np.ndarray) -> InternalMetrics:
    mask = labels != -1
    Xm, lm = X[mask], labels[mask]
    uniq = np.unique(lm)
    n_clusters = len(uniq)
    n_noise = int(np.sum(~mask))
    if n_clusters < 2 or len(lm) < 3:
        return InternalMetrics(np.nan, np.nan, np.nan, n_clusters, n_noise)
    try:
        sil = silhouette_score(Xm, lm)
    except Exception:
        sil = np.nan
    try:
        db = davies_bouldin_score(Xm, lm)
    except Exception:
        db = np.nan
    try:
        ch = calinski_harabasz_score(Xm, lm)
    except Exception:
        ch = np.nan
    return InternalMetrics(float(sil), float(db), float(ch), n_clusters, n_noise)


def external_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ExternalMetrics:
    return ExternalMetrics(
        ari=float(adjusted_rand_score(y_true, y_pred)),
        nmi=float(normalized_mutual_info_score(y_true, y_pred)),
        v_measure=float(v_measure_score(y_true, y_pred)),
        fmi=float(fowlkes_mallows_score(y_true, y_pred)),
    )


def bootstrap_stability(fit_fn, X: np.ndarray, n_bootstrap: int = 20,
                        fraction: float = 0.8, random_state: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    size = int(n * fraction)
    aris: list[float] = []
    base = fit_fn(X).labels
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=size, replace=False)
        try:
            sub_labels = fit_fn(X[idx]).labels
            aris.append(adjusted_rand_score(base[idx], sub_labels))
        except Exception:
            continue
    if not aris:
        return {"mean_ari": np.nan, "std_ari": np.nan, "median_ari": np.nan}
    aris = np.asarray(aris)
    return {
        "mean_ari": float(np.mean(aris)),
        "std_ari": float(np.std(aris)),
        "median_ari": float(np.median(aris)),
    }


def co_assignment_consensus(labels_list: list[np.ndarray]) -> np.ndarray:
    n = len(labels_list[0])
    consensus = np.zeros((n, n), dtype=np.float32)
    for labels in labels_list:
        same = (labels[:, None] == labels[None, :]).astype(np.float32)
        consensus += same
    return consensus / len(labels_list)
