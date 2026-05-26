from __future__ import annotations

import _paths 

from pathlib import Path

import pandas as pd

import config as cfg


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


SIGNATURE_COLS = [
    "cluster",
    "speed_mps_mean",
    "speed_mps_std",
    "linacc_mag_mean",
    "linacc_mag_std",
    "gyro_mag_mean",
    "gyro_mag_std",
    "linacc_mag_cadence_energy",
    "linacc_mag_vibration_energy",
    "stop_ratio",
    "stop_transitions",
]


def _compact_signatures(sig: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in SIGNATURE_COLS if c in sig.columns]
    return sig[cols]


def _real_section(prefix: str, title: str, parts: list[str]) -> None:
    parts.append(f"## {title}\n")
    km = _read(cfg.TABLES_DIR / f"{prefix}_kmeans_selection.csv")
    gmm = _read(cfg.TABLES_DIR / f"{prefix}_gmm_selection.csv")
    ks = _read(cfg.TABLES_DIR / f"{prefix}_k_selection.csv")
    if km is not None:
        parts.append(f"**K-Means Modellauswahl ({prefix})**\n\n" + _md(km) + "\n")
        parts.append(f"![{prefix} elbow](figures/{prefix}_01_kmeans_elbow.png)\n")
    if gmm is not None:
        parts.append(f"**GMM Modellauswahl ({prefix})**\n\n" + _md(gmm) + "\n")
        parts.append(f"![{prefix} bic aic](figures/{prefix}_02_gmm_bic_aic.png)\n")
    if ks is not None:
        parts.append("k-Auswahl (Silhouette-Max + BIC-Knee):\n\n" + _md(ks) + "\n")

    bench = _read(cfg.TABLES_DIR / f"{prefix}_benchmark.csv")
    if bench is not None:
        parts.append(
            f"**Benchmark {prefix}** (Spalte `degenerate=True` markiert Pathologien: "
            "DBSCAN-1-Cluster-mit-Noise und AHC-Average-Chaining (>90% in einem Cluster); "
            "diese werden bei der 'best' Auswahl ausgeschlossen, weil ihre hohen "
            "internen Metriken Artefakte einer trivialen Partition sind):\n\n"
            + _md(bench) + "\n"
        )
        for metric in ["silhouette", "davies_bouldin", "calinski_harabasz", "stability_mean_ari"]:
            parts.append(f"![{metric}](figures/{prefix}_03_{metric}.png)\n")
        best = bench[~bench["degenerate"]].sort_values(
            "stability_mean_ari", ascending=False).iloc[0]
        parts.append(
            f"Bester nicht-degenerierter Algorithmus auf {prefix} nach Bootstrap-"
            f"Stabilitaet: `{best['algorithm']}` "
            f"(Stability ARI = {best['stability_mean_ari']:.3f}).\n"
        )

    parts.append(f"![{prefix} PCA](figures/{prefix}_05_pca.png)\n")
    sig = _read(cfg.TABLES_DIR / f"{prefix}_cluster_signatures.csv")
    if sig is not None:
        compact = _compact_signatures(sig).head(10)
        parts.append(
            f"**Cluster-Signaturen ({prefix}, kompakt - Schluesselfeatures je Cluster; "
            f"vollstaendige Tabelle in `tables/{prefix}_cluster_signatures.csv`):**\n\n"
            + _md(compact) + "\n"
        )
    parts.append(f"![{prefix} feature signatures](figures/{prefix}_07_feature_signatures.png)\n")


def main() -> None:
    parts: list[str] = []
    parts.append("# BA2 Analysebericht\n")

    _real_section("real", "1. Primaer-Track: Echtdaten", parts)

    parts.append("## 2. Externer Track: Synthetische Daten (BA1-Generator)\n")
    synth = _read(cfg.TABLES_DIR / "synth_benchmark.csv")
    if synth is not None:
        parts.append(_md(synth) + "\n")
        for metric in ["ari", "nmi", "v_measure", "fmi"]:
            parts.append(f"![synth {metric}](figures/synth_04_{metric}.png)\n")
        best = synth[~synth["degenerate"]].sort_values("ari", ascending=False).iloc[0]
        parts.append(
            f"Bester nicht-degenerierter Algorithmus nach ARI: `{best['algorithm']}` "
            f"(ARI = {best['ari']:.3f}). Dieser Track validiert, welche Algorithmen "
            "die 10 BA1-Personas (Commuter, Sport, MTB, E-Bike, Urban LastMile, ...) "
            "ueberhaupt rekonstruieren koennen.\n"
        )
    parts.append("![synth PCA](figures/synth_06_pca_truth.png)\n")

    parts.append("## 3. Robustheit gegen kontrollierte Augmentationen (Echtdaten)\n")
    rob = _read(cfg.TABLES_DIR / "robustness.csv")
    if rob is not None:
        rob_valid = rob[~rob["degenerate"]] if "degenerate" in rob.columns else rob
        # Exclude the baseline level (level=0, no perturbation): every algorithm
        # trivially scores stability_ari=1.0 there, which would inflate the
        # per-augmentation means and squash the differences we actually care about.
        rob_perturbed = rob_valid[rob_valid["level"] > 0]
        agg = rob_perturbed.groupby(["augmentation", "algorithm"]).agg(
            sil_mean=("silhouette", "mean"),
            stability_mean=("stability_ari", "mean"),
        ).reset_index()
        parts.append(
            "Aggregation nur ueber **nicht-degenerierte Laeufe** und nur ueber "
            "**Stoerstufen > 0** (level=0 ist die Baseline ohne Stoerung, dort ist "
            "stability_ari trivial = 1.0 fuer jeden Algorithmus und wuerde die "
            "Mittelwerte kuenstlich aufblasen). DBSCAN und AHC-Average sind hier "
            "deshalb meist gar nicht oder nur teilweise gelistet: ihre 'perfekte "
            "Stabilitaet' unter Augmentation ist ein Artefakt davon, dass sie unter "
            "Stoerung trivialerweise weiter eine Ein-Cluster-Partition ausgeben "
            "(DBSCAN) bzw. das Chaining konservieren (AHC-avg).\n\n"
            + _md(agg) + "\n"
        )
        for kind in ["noise", "dropout", "drift"]:
            parts.append(f"![{kind} silhouette](figures/robust_{kind}_silhouette.png)\n")
            parts.append(f"![{kind} stability](figures/robust_{kind}_stability_ari.png)\n")
        winners = (rob_perturbed.groupby(["augmentation", "algorithm"])["stability_ari"].mean()
                   .reset_index()
                   .sort_values(["augmentation", "stability_ari"], ascending=[True, False])
                   .groupby("augmentation").head(1))
        parts.append("**Robustester nicht-degenerierter Algorithmus pro Stoertyp "
                     "(mittlere Stabilitaets-ARI):**\n\n" + _md(winners) + "\n")

    parts.append("## 4. Active Learning (explorativ, synthetische Daten)\n")
    al_df = _read(cfg.TABLES_DIR / "active_learning.csv")
    if al_df is not None:
        last = al_df.sort_values(["strategy", "n_labeled"]).groupby("strategy").tail(1)
        parts.append("**Endstand pro Strategie:**\n\n" + _md(last) + "\n")
        parts.append("![AL Konvergenz](figures/active_learning.png)\n")
        gain = al_df.groupby("strategy")["ari"].agg(["first", "last"]).reset_index()
        gain["delta_ari"] = gain["last"] - gain["first"]
        parts.append("**ARI-Verbesserung erstes -> letztes Round:**\n\n" + _md(gain) + "\n")

    parts.append("## 5. Zusammenfassende Empfehlung\n")
    real_bench = _read(cfg.TABLES_DIR / "real_benchmark.csv")
    if real_bench is not None and rob is not None:
        best_real = real_bench[~real_bench["degenerate"]].sort_values(
            "stability_mean_ari", ascending=False).iloc[0]
        rob_valid = rob[~rob["degenerate"]] if "degenerate" in rob.columns else rob
        rob_perturbed = rob_valid[rob_valid["level"] > 0]
        avg_rob = rob_perturbed.groupby("algorithm")["stability_ari"].mean().sort_values(ascending=False)
        parts.append(
            f"- Bester Algorithmus nach **Bootstrap-Stabilitaet (Echtdaten)**: "
            f"`{best_real['algorithm']}`.\n"
            f"- Bester nicht-degenerierter Algorithmus nach **mittlerer Robustheits-ARI** "
            f"ueber alle Stoertypen: `{avg_rob.index[0]}` (mean = {avg_rob.iloc[0]:.3f}).\n"
        )
    if synth is not None:
        best_synth = synth[~synth["degenerate"]].sort_values("ari", ascending=False).iloc[0]
        parts.append(
            f"- Bester Algorithmus nach **ARI auf synthetischer Ground Truth** (10 Personas): "
            f"`{best_synth['algorithm']}` (ARI = {best_synth['ari']:.3f}).\n"
        )

    out = cfg.RESULTS_DIR / "analysis_report.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"[report] wrote {out}")


if __name__ == "__main__":
    main()
