
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score


@dataclass
class ALStep:
    round_idx: int
    n_labeled: int
    ari: float
    mean_entropy_unlabeled: float


def _stratified_seed(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    classes = np.unique(y)
    picks: list[int] = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        if idx_c.size:
            picks.append(int(rng.choice(idx_c)))
    if len(picks) < n:
        remaining = np.setdiff1d(np.arange(len(y)), np.asarray(picks))
        extra = rng.choice(remaining, size=n - len(picks), replace=False)
        picks.extend(int(i) for i in extra)
    return np.asarray(picks[:n], dtype=int)


def _fit_predict(X: np.ndarray, labeled_idx: np.ndarray, y_lab: np.ndarray,
                 random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clf = LogisticRegression(
        max_iter=1000, solver="lbfgs", random_state=random_state,
    )
    clf.fit(X[labeled_idx], y_lab)
    proba = clf.predict_proba(X)
    proba_safe = np.clip(proba, 1e-12, 1.0)
    ent = -np.sum(proba_safe * np.log(proba_safe), axis=1)
    preds_internal = proba.argmax(axis=1)
    preds = clf.classes_[preds_internal]
    return preds, ent, clf.classes_


def run_active_learning(X: np.ndarray, y_true: np.ndarray, n_components: int,
                        initial_labeled: int, queries_per_round: int,
                        total_rounds: int, strategy: str = "uncertainty",
                        random_state: int = 42) -> list[ALStep]:
    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    labeled_idx = _stratified_seed(y_true, initial_labeled, rng)
    y_labeled = y_true[labeled_idx].copy()

    steps: list[ALStep] = []
    for r in range(total_rounds + 1):
        unlabeled_mask = np.ones(n, dtype=bool)
        unlabeled_mask[labeled_idx] = False

        if np.unique(y_labeled).size < 2:
            # Degenerate; classifier needs at least 2 classes.
            preds = np.full(n, y_labeled[0])
            ent = np.zeros(n)
        else:
            preds, ent, _ = _fit_predict(X, labeled_idx, y_labeled, random_state)

        mean_ent_unl = float(ent[unlabeled_mask].mean()) if unlabeled_mask.any() else 0.0
        ari = float(adjusted_rand_score(y_true, preds))
        steps.append(ALStep(r, len(labeled_idx), ari, mean_ent_unl))

        if r == total_rounds or not unlabeled_mask.any():
            break

        unl_idx = np.where(unlabeled_mask)[0]
        if strategy == "uncertainty":
            scores = ent[unl_idx]
            choose = unl_idx[np.argsort(-scores)[:queries_per_round]]
        elif strategy == "random":
            k = min(queries_per_round, unl_idx.size)
            choose = rng.choice(unl_idx, size=k, replace=False)
        else:
            raise ValueError(strategy)

        labeled_idx = np.concatenate([labeled_idx, choose])
        y_labeled = np.concatenate([y_labeled, y_true[choose]])

    return steps
