from __future__ import annotations

import _paths
from _common import load_windows, scale, split_features

from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from src import clustering as cl
from src import evaluation as ev
from src import model_selection as ms
from src import visualization as viz


def _selection_kmeans(X: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in cfg.K_RANGE:
        res = cl.fit_kmeans(X, k, cfg.RANDOM_STATE)
        m = ev.internal_metrics(X, res.labels)
        rows.append({
            "k": k, "inertia": res.extras["inertia"],
            "silhouette": m.silhouette, "davies_bouldin": m.davies_bouldin,
            "calinski_harabasz": m.calinski_harabasz,
        })
    return pd.DataFrame(rows)


def _selection_gmm(X: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in cfg.K_RANGE:
        res = cl.fit_gmm(X, k, "full", cfg.RANDOM_STATE)
        m = ev.internal_metrics(X, res.labels)
        rows.append({
            "k": k, "bic": res.extras["bic"], "aic": res.extras["aic"],
            "silhouette": m.silhouette,
        })
    return pd.DataFrame(rows)


def _choose_k(km: pd.DataFrame, gmm: pd.DataFrame) -> dict[str, int]:
    """Use silhouette-max + BIC-knee (avoid monotone-BIC trap)."""
    k_sil = int(km.loc[km["silhouette"].idxmax(), "k"])
    k_bic_knee = ms.knee_point(gmm["k"].tolist(), gmm["bic"].tolist())
    k_used = max(k_sil, k_bic_knee)
    return {"k_silhouette": k_sil, "k_bic_knee": k_bic_knee, "k_used": k_used}


def _benchmark_internal(X: np.ndarray, k_best: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    label_map: dict[str, np.ndarray] = {}
    for name, fit_fn in cl.algorithm_registry(k_best, cfg.RANDOM_STATE).items():
        res = fit_fn(X)
        m = ev.internal_metrics(X, res.labels)
        stab = ev.bootstrap_stability(fit_fn, X, n_bootstrap=cfg.N_BOOTSTRAP,
                                      fraction=cfg.BOOTSTRAP_FRACTION,
                                      random_state=cfg.RANDOM_STATE)
        share = ms.max_cluster_share(res.labels)
        rows.append({
            "algorithm": name, "n_clusters": res.n_clusters, "n_noise": res.n_noise,
            "max_cluster_share": share,
            "degenerate": share > 0.9,
            "silhouette": m.silhouette, "davies_bouldin": m.davies_bouldin,
            "calinski_harabasz": m.calinski_harabasz,
            "stability_mean_ari": stab["mean_ari"], "stability_std_ari": stab["std_ari"],
        })
        label_map[name] = res.labels
    return pd.DataFrame(rows), label_map


def _benchmark_external(X: np.ndarray, y: np.ndarray, k_best: int) -> pd.DataFrame:
    rows = []
    for name, fit_fn in cl.algorithm_registry(k_best, cfg.RANDOM_STATE).items():
        res = fit_fn(X)
        ext = ev.external_metrics(y, res.labels)
        m = ev.internal_metrics(X, res.labels)
        share = ms.max_cluster_share(res.labels)
        rows.append({
            "algorithm": name, "n_clusters": res.n_clusters,
            "max_cluster_share": share, "degenerate": share > 0.9,
            "ari": ext.ari, "nmi": ext.nmi, "v_measure": ext.v_measure, "fmi": ext.fmi,
            "silhouette": m.silhouette,
        })
    return pd.DataFrame(rows)


def _post_hoc(df: pd.DataFrame, labels: np.ndarray, out_csv: Path) -> pd.DataFrame:
    feats, _ = split_features(df)
    sig_df = feats.copy()
    sig_df["cluster"] = labels
    sig = sig_df.groupby("cluster").mean(numeric_only=True)
    sig.to_csv(out_csv)
    return sig


def _best_non_degenerate(bench: pd.DataFrame, metric: str, ascending: bool = False) -> str:
    candidates = bench[~bench["degenerate"]].copy()
    if candidates.empty:
        candidates = bench
    return candidates.sort_values(metric, ascending=ascending).iloc[0]["algorithm"]


def run_real_track(real_df: pd.DataFrame, prefix: str) -> None:
    X_df, _ = split_features(real_df)
    X, _ = scale(X_df)

    km = _selection_kmeans(X)
    km.to_csv(cfg.TABLES_DIR / f"{prefix}_kmeans_selection.csv", index=False)
    viz.plot_elbow_silhouette(km["k"].tolist(), km["inertia"].tolist(),
                              km["silhouette"].tolist(),
                              cfg.FIGURES_DIR / f"{prefix}_01_kmeans_elbow.png")

    gmm = _selection_gmm(X)
    gmm.to_csv(cfg.TABLES_DIR / f"{prefix}_gmm_selection.csv", index=False)
    viz.plot_bic_aic(gmm["k"].tolist(), gmm["bic"].tolist(), gmm["aic"].tolist(),
                     cfg.FIGURES_DIR / f"{prefix}_02_gmm_bic_aic.png")

    chosen = _choose_k(km, gmm)
    pd.DataFrame([chosen]).to_csv(cfg.TABLES_DIR / f"{prefix}_k_selection.csv", index=False)
    k_best = chosen["k_used"]
    print(f"[{prefix}] k_silhouette={chosen['k_silhouette']}, "
          f"k_bic_knee={chosen['k_bic_knee']}, k_used={k_best}")

    bench, labels = _benchmark_internal(X, k_best)
    bench.to_csv(cfg.TABLES_DIR / f"{prefix}_benchmark.csv", index=False)

    for metric in ["silhouette", "davies_bouldin", "calinski_harabasz", "stability_mean_ari"]:
        viz.plot_metric_comparison(bench, metric,
                                   cfg.FIGURES_DIR / f"{prefix}_03_{metric}.png",
                                   title=f"{prefix}: {metric}")

    best_algo = _best_non_degenerate(bench, "stability_mean_ari")
    print(f"[{prefix}] best non-degenerate (stability ARI): {best_algo}")
    labels_best = labels[best_algo]

    viz.plot_2d_embedding(X, labels_best,
                          cfg.FIGURES_DIR / f"{prefix}_05_pca.png",
                          title=f"{prefix} PCA ({best_algo}, k={k_best})")

    sig = _post_hoc(real_df, labels_best,
                    cfg.TABLES_DIR / f"{prefix}_cluster_signatures.csv")
    top = sig.var(axis=0).sort_values(ascending=False).head(6).index.tolist()
    viz.plot_feature_signature(real_df, labels_best, top,
                               cfg.FIGURES_DIR / f"{prefix}_07_feature_signatures.png")

    np.save(cfg.TABLES_DIR / f"{prefix}_labels_{best_algo}.npy", labels_best)
    pd.DataFrame([{"prefix": prefix, "best_algorithm": best_algo, "k_used": k_best}]).to_csv(
        cfg.TABLES_DIR / f"{prefix}_best_choice.csv", index=False)


def run_synth_track(synth_df: pd.DataFrame) -> None:
    X_df, meta = split_features(synth_df)
    X, _ = scale(X_df)
    y = pd.Categorical(meta["persona"]).codes
    k_true = int(meta["persona"].nunique())
    print(f"[synth] k_true={k_true} personas, n={len(X)}")

    bench = _benchmark_external(X, y, k_true)
    bench.to_csv(cfg.TABLES_DIR / "synth_benchmark.csv", index=False)
    for metric in ["ari", "nmi", "v_measure", "fmi"]:
        viz.plot_metric_comparison(bench, metric,
                                   cfg.FIGURES_DIR / f"synth_04_{metric}.png",
                                   title=f"Synthetic GT: {metric}")
    viz.plot_2d_embedding(X, y, cfg.FIGURES_DIR / "synth_06_pca_truth.png",
                          title=f"Synthetic PCA - Ground Truth ({k_true} personas)")
    best_algo = _best_non_degenerate(bench, "ari")
    print(f"[synth] best non-degenerate (ARI): {best_algo}")
    pd.DataFrame([{"best_algorithm": best_algo, "k_true": k_true}]).to_csv(
        cfg.TABLES_DIR / "synth_best_choice.csv", index=False)


def main() -> None:
    real = load_windows(cfg.FEATURES_DIR / "real_windows.parquet")
    synth = load_windows(cfg.FEATURES_DIR / "synth_windows.parquet")
    print(f"[bench] real={len(real)}, synth={len(synth)}")

    run_real_track(real, "real")
    run_synth_track(synth)
    print("[bench] done")


if __name__ == "__main__":
    main()
