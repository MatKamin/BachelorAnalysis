from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


@dataclass
class ClusterResult:
    name: str
    labels: np.ndarray
    n_clusters: int
    n_noise: int
    params: dict[str, Any] = field(default_factory=dict)
    model: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


def _summary(name: str, labels: np.ndarray, params: dict, model=None, extras=None) -> ClusterResult:
    uniq = set(int(x) for x in labels)
    noise = int(np.sum(labels == -1))
    n_clusters = len(uniq - {-1})
    return ClusterResult(
        name=name, labels=labels.astype(int), n_clusters=n_clusters,
        n_noise=noise, params=params, model=model, extras=extras or {},
    )


def fit_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> ClusterResult:
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    return _summary("kmeans", labels, {"k": k}, model,
                    extras={"inertia": float(model.inertia_)})


def fit_gmm(X: np.ndarray, k: int, covariance_type: str = "full",
            random_state: int = 42) -> ClusterResult:
    model = GaussianMixture(n_components=k, covariance_type=covariance_type,
                            random_state=random_state, max_iter=300, n_init=3)
    model.fit(X)
    labels = model.predict(X)
    return _summary("gmm", labels, {"k": k, "cov": covariance_type}, model,
                    extras={"bic": float(model.bic(X)), "aic": float(model.aic(X))})


def fit_ahc(X: np.ndarray, k: int, linkage: str = "ward") -> ClusterResult:
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X)
    return _summary("ahc", labels, {"k": k, "linkage": linkage}, model)


def fit_spectral(X: np.ndarray, k: int, random_state: int = 42) -> ClusterResult:
    n = X.shape[0]
    n_neighbors = int(min(max(5, np.sqrt(n)), 30))
    model = SpectralClustering(
        n_clusters=k, assign_labels="kmeans", affinity="nearest_neighbors",
        n_neighbors=n_neighbors, random_state=random_state,
    )
    labels = model.fit_predict(X)
    return _summary("spectral", labels, {"k": k, "n_neighbors": n_neighbors}, model)


def _suggest_eps(X: np.ndarray, min_samples: int) -> float:
    n = min(X.shape[0], 2000)
    idx = np.random.default_rng(0).choice(X.shape[0], n, replace=False)
    Xs = X[idx]
    nn = NearestNeighbors(n_neighbors=min_samples).fit(Xs)
    dists, _ = nn.kneighbors(Xs)
    kdist = np.sort(dists[:, -1])
    return float(np.percentile(kdist, 90))


def fit_dbscan(X: np.ndarray, eps: float | None = None,
               min_samples: int | None = None) -> ClusterResult:
    if min_samples is None:
        min_samples = max(2 * X.shape[1], 5)
    if eps is None:
        eps = _suggest_eps(X, min_samples)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return _summary("dbscan", labels, {"eps": eps, "min_samples": min_samples}, model)


def algorithm_registry(k: int, random_state: int = 42) -> dict[str, Callable[[np.ndarray], ClusterResult]]:
    return {
        "kmeans": lambda X: fit_kmeans(X, k=k, random_state=random_state),
        "gmm_full": lambda X: fit_gmm(X, k=k, covariance_type="full", random_state=random_state),
        "gmm_diag": lambda X: fit_gmm(X, k=k, covariance_type="diag", random_state=random_state),
        "ahc_ward": lambda X: fit_ahc(X, k=k, linkage="ward"),
        "ahc_avg": lambda X: fit_ahc(X, k=k, linkage="average"),
        "spectral": lambda X: fit_spectral(X, k=k, random_state=random_state),
        "dbscan": lambda X: fit_dbscan(X),
    }
