from __future__ import annotations

import _paths
from _common import load_windows, scale, split_features

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import config as cfg
from src import augmentation as aug
from src import clustering as cl
from src import evaluation as ev
from src import model_selection as ms
from src import visualization as viz


def run_one(X_clean: np.ndarray, base_labels: dict[str, np.ndarray],
            spec: aug.AugmentationSpec, k_best: int) -> list[dict]:
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    X_pert = aug.apply_augmentation(X_clean, spec, rng)
    rows = []
    for name, fit_fn in cl.algorithm_registry(k_best, cfg.RANDOM_STATE).items():
        try:
            res = fit_fn(X_pert)
            m = ev.internal_metrics(X_pert, res.labels)
            stability = float(adjusted_rand_score(base_labels[name], res.labels))
            share = ms.max_cluster_share(res.labels)
        except Exception as exc:
            print(f"[warn] {name} failed for {spec.label}: {exc}")
            m = ev.InternalMetrics(np.nan, np.nan, np.nan, 0, 0)
            stability = np.nan
            share = 1.0
        rows.append({
            "algorithm": name, "augmentation": spec.kind, "level": spec.level,
            "silhouette": m.silhouette, "davies_bouldin": m.davies_bouldin,
            "calinski_harabasz": m.calinski_harabasz,
            "stability_ari": stability,
            "max_cluster_share": share,
            "degenerate": share > 0.9,
        })
    return rows


def main() -> None:
    real_df = load_windows(cfg.FEATURES_DIR / "real_windows.parquet")
    best = pd.read_csv(cfg.TABLES_DIR / "real_best_choice.csv").iloc[0]
    k_best = int(best["k_used"])
    print(f"[robust] using k={k_best} from real-world benchmark")  

    X_df, _ = split_features(real_df)
    X, _ = scale(X_df)

    base_labels = {name: fit_fn(X).labels for name, fit_fn in
                   cl.algorithm_registry(k_best, cfg.RANDOM_STATE).items()}

    specs = (
        [aug.AugmentationSpec("noise", lvl) for lvl in cfg.NOISE_LEVELS]
        + [aug.AugmentationSpec("dropout", lvl) for lvl in cfg.DROPOUT_RATES]
        + [aug.AugmentationSpec("drift", lvl) for lvl in cfg.DRIFT_LEVELS]
    )

    rows: list[dict] = []
    for spec in specs:
        print(f"[robust] {spec.label}")
        rows.extend(run_one(X, base_labels, spec, k_best))

    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES_DIR / "robustness.csv", index=False)

    for kind in ["noise", "dropout", "drift"]:
        for metric in ["silhouette", "stability_ari"]:
            viz.plot_robustness_curves(
                df, kind, metric,
                cfg.FIGURES_DIR / f"robust_{kind}_{metric}.png",
            )
    print("[robust] done")


if __name__ == "__main__":
    main()
