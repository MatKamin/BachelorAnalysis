from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sns.set_theme(style="whitegrid", context="notebook")
CAPTION_PROPS = dict(fontsize=9, color="dimgray", wrap=True)


def _save(fig: plt.Figure, path: Path, caption: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.5, -0.02, caption, ha="center", va="top", **CAPTION_PROPS)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_elbow_silhouette(k_values: list[int], inertia: list[float],
                          silhouette: list[float], path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(k_values, inertia, "o-", color="tab:blue", label="Inertia (K-Means)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(k_values, silhouette, "s--", color="tab:orange", label="Silhouette")
    ax2.set_ylabel("Silhouette", color="tab:orange")
    ax1.set_title("Elbow- und Silhouette-Analyse fuer K-Means")
    caption = (
        "Inertia (blau) und Silhouette (orange) ueber k. Der 'Knick' der Inertia "
        "und das Silhouette-Maximum geben Hinweise auf eine plausible Clusterzahl."
    )
    _save(fig, path, caption)


def plot_bic_aic(k_values: list[int], bic: list[float], aic: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, bic, "o-", label="BIC")
    ax.plot(k_values, aic, "s--", label="AIC")
    ax.set_xlabel("Anzahl Komponenten k")
    ax.set_ylabel("Score (kleiner = besser)")
    ax.set_title("BIC/AIC fuer GMM-Modellauswahl")
    ax.legend()
    caption = (
        "BIC und AIC ueber die Komponentenzahl k der GMM. Beide Kurven helfen, "
        "die Modellkomplexitaet zu kalibrieren; das jeweilige Minimum signalisiert "
        "das bevorzugte Modell."
    )
    _save(fig, path, caption)


def plot_metric_comparison(df: pd.DataFrame, metric: str, path: Path,
                           title: str | None = None) -> None:
    pivot = df.pivot_table(index="algorithm", values=metric, aggfunc="mean").sort_values(metric)
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot[metric].plot(kind="barh", ax=ax, color=sns.color_palette("crest", len(pivot)))
    ax.set_xlabel(metric)
    ax.set_title(title or f"Algorithmenvergleich: {metric}")
    caption = (
        f"Mittelwert von {metric} pro Algorithmus auf dem Echtdatensatz. "
        "Hoehere Silhouette/Calinski-Harabasz und niedrigere Davies-Bouldin sind besser."
    )
    _save(fig, path, caption)


def plot_2d_embedding(X: np.ndarray, labels: np.ndarray, path: Path,
                      title: str, method: str = "pca",
                      max_points: int = 4000) -> None:
    if X.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(X.shape[0], max_points, replace=False)
        X, labels = X[idx], labels[idx]
    if method == "pca":
        Z = PCA(n_components=2, random_state=0).fit_transform(X)
    else:
        Z = TSNE(n_components=2, init="pca", random_state=0,
                 perplexity=min(30, max(5, X.shape[0] // 50))).fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    palette = sns.color_palette("tab10", n_colors=max(len(np.unique(labels)), 1))
    for i, c in enumerate(np.unique(labels)):
        m = labels == c
        ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.7, label=str(c), color=palette[i % len(palette)])
    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.legend(loc="best", fontsize=8, markerscale=2)
    caption = (
        f"{method.upper()}-Projektion der Window-Features, eingefaerbt nach Cluster-Label. "
        "Dient zur visuellen Plausibilisierung der gefundenen Struktur."
    )
    _save(fig, path, caption)


def plot_robustness_curves(df: pd.DataFrame, kind: str, metric: str, path: Path) -> None:
    sub = df[df["augmentation"] == kind]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo, grp in sub.groupby("algorithm"):
        grp = grp.sort_values("level")
        ax.plot(grp["level"], grp[metric], "o-", label=algo)
    ax.set_xlabel(f"{kind} level")
    ax.set_ylabel(metric)
    ax.set_title(f"Robustheit gegen '{kind}' - {metric}")
    ax.legend(fontsize=8)
    caption = (
        f"Verlauf von {metric} pro Algorithmus unter zunehmendem '{kind}'-Stoereinfluss. "
        "Flachere Kurven deuten auf hoehere Robustheit hin (bei Stabilitaets-ARI ist "
        "ein konstant hoher Wert wuenschenswert)."
    )
    _save(fig, path, caption)


def plot_active_learning(steps_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strategy, grp in steps_df.groupby("strategy"):
        ax.plot(grp["n_labeled"], grp["ari"], "o-", label=strategy)
    ax.set_xlabel("Anzahl gelabelte Windows")
    ax.set_ylabel("ARI vs. synthetische Ground Truth")
    ax.set_title("Active-Learning-Konvergenz (GMM mit Uncertainty-Sampling)")
    ax.legend()
    caption = (
        "ARI-Verlauf des semi-supervised GMM mit zunehmender Anzahl simuliert "
        "gelabelter Windows. Uncertainty-Sampling sollte schneller konvergieren "
        "als Random-Sampling."
    )
    _save(fig, path, caption)


def plot_stress_per_cluster(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="cluster", y="stress", ax=ax,
                hue="cluster", palette="rocket", legend=False)
    ax.set_title("Stresslabels pro Cluster (post-hoc Charakterisierung)")
    caption = (
        "Verteilung der vorhandenen Stresslabels je Cluster. Signifikante "
        "Unterschiede stuetzen die behavioral-semantische Interpretation der "
        "gefundenen Cluster (Feedback-konforme Argumentation)."
    )
    _save(fig, path, caption)


def plot_feature_signature(window_df: pd.DataFrame, labels: np.ndarray,
                           top_features: list[str], path: Path) -> None:
    df = window_df.copy()
    df["cluster"] = labels
    long = df.melt(id_vars="cluster", value_vars=top_features,
                   var_name="feature", value_name="value")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=long, x="feature", y="value", hue="cluster", ax=ax, palette="tab10")
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([t.get_text() for t in ax.get_xticklabels()], rotation=30, ha="right")
    ax.set_title("Feature-Signaturen pro Cluster")
    caption = (
        "Verteilung ausgewaehlter Window-Features pro Cluster. Erlaubt die "
        "Beschreibung 'Cluster X hat hoehere Gyro-Varianz / haeufigere Stopps' "
        "ohne Persona-Etiketten zu vergeben."
    )
    _save(fig, path, caption)


def plot_consensus_matrix(consensus: np.ndarray, path: Path,
                          title: str = "Konsensmatrix") -> None:
    n = min(consensus.shape[0], 400)
    idx = np.random.default_rng(0).choice(consensus.shape[0], n, replace=False)
    sub = consensus[np.ix_(idx, idx)]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(sub, cmap="rocket_r", ax=ax, cbar_kws={"label": "Co-Assignment"})
    ax.set_title(title)
    caption = (
        "Konsensmatrix aus Bootstrap-Subsamples (Zufallsstichprobe). Hohe Werte "
        "(dunkel) entlang der Diagonalbloecke entsprechen stabilen Clustern."
    )
    _save(fig, path, caption)
