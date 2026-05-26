from __future__ import annotations

import _paths
from _common import load_windows, scale, split_features

import pandas as pd

import config as cfg
from src import active_learning as al
from src import visualization as viz


def main() -> None:
    df = load_windows(cfg.FEATURES_DIR / "synth_windows.parquet")
    X_df, meta = split_features(df)
    X, _ = scale(X_df)
    y = pd.Categorical(meta["persona"]).codes
    n_components = int(meta["persona"].nunique())
    print(f"[al] X shape={X.shape}, n_components={n_components}")

    rows = []
    for strategy in ["uncertainty", "random"]:
        steps = al.run_active_learning(
            X, y, n_components,
            initial_labeled=cfg.AL_INITIAL_LABELED,
            queries_per_round=cfg.AL_QUERIES_PER_ROUND,
            total_rounds=cfg.AL_TOTAL_ROUNDS,
            strategy=strategy,
            random_state=cfg.RANDOM_STATE,
        )
        for s in steps:
            rows.append({"strategy": strategy, "round": s.round_idx,
                         "n_labeled": s.n_labeled, "ari": s.ari,
                         "mean_entropy_unlabeled": s.mean_entropy_unlabeled})

    out = pd.DataFrame(rows)
    out.to_csv(cfg.TABLES_DIR / "active_learning.csv", index=False)
    viz.plot_active_learning(out, cfg.FIGURES_DIR / "active_learning.png")
    print("[al] done")


if __name__ == "__main__":
    main()
