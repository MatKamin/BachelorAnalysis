from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
for p in (str(SCRIPTS_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd

import config as cfg

K_VALUES = [5, 6, 7, 8]
STABILITY_COL = "stability_mean_ari"


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _kdir(k: int) -> Path:
    return cfg.RESULTS_DIR / f"k{k}"


def _available_ks() -> list[int]:
    return [k for k in K_VALUES
            if (_kdir(k) / "tables" / "real_benchmark.csv").exists()]


def _best_non_degenerate(bench: pd.DataFrame, metric: str, ascending: bool = False) -> pd.Series:
    cand = bench[~bench["degenerate"]] if "degenerate" in bench.columns else bench
    if cand.empty:
        cand = bench
    return cand.sort_values(metric, ascending=ascending).iloc[0]


def _k_selection_recap(parts: list[str], ks: list[int]) -> None:
    parts.append("## 1. k-Auswahl: Evidenz aus der Modellselektion\n")
    k0 = ks[0]
    km = _read(_kdir(k0) / "tables" / "real_kmeans_selection.csv")
    gmm = _read(_kdir(k0) / "tables" / "real_gmm_selection.csv")
    if km is not None:
        k_sil = int(km.loc[km["silhouette"].idxmax(), "k"])
        parts.append(
            f"Die Silhouette (K-Means) ist maximal bei **k = {k_sil}**.\n\n" + _md(km) + "\n")
    if gmm is not None:
        k_bic = int(gmm.loc[gmm["bic"].idxmin(), "k"])
        parts.append(
            f"Der BIC (GMM) ist minimal (bestes Modell) bei **k = {k_bic}**.\n\n" + _md(gmm) + "\n")


def _sensitivity_comparison(parts: list[str], ks: list[int]) -> None:
    parts.append("## 2. Sensitivitatsanalyse: Vergleich uber k\n")

    best_rows = []
    benches: dict[int, pd.DataFrame] = {}
    for k in ks:
        b = _read(_kdir(k) / "tables" / "real_benchmark.csv")
        if b is None:
            continue
        benches[k] = b
        best = _best_non_degenerate(b, STABILITY_COL)
        best_rows.append({
            "k": k,
            "best_algorithm": best["algorithm"],
            "stability_mean_ari": best[STABILITY_COL],
            "silhouette": best["silhouette"],
            "davies_bouldin": best["davies_bouldin"],
            "calinski_harabasz": best["calinski_harabasz"],
        })
    if best_rows:
        parts.append(
            "**Bestes nicht-degeneriertes Verfahren je k** (Kriterium: Bootstrap-"
            "Stabilitats-ARI):\n\n" + _md(pd.DataFrame(best_rows)) + "\n")
        winners = {r["best_algorithm"] for r in best_rows}
        if len(winners) == 1:
            parts.append(
                f"> Ueber alle gepruften k bleibt **{winners.pop()}** das stabilste "
                "Verfahren. Die zentrale Schlussfolgerung der Arbeit ist damit "
                "robust gegenuber der Wahl von k.\n")
        else:
            parts.append(
                "> Das beste Verfahren wechselt zwischen den k-Werten "
                f"({', '.join(sorted(winners))}); die Wahl von k beeinflusst die "
                "Schlussfolgerung und wird unten diskutiert.\n")

    def _pivot(metric: str, label: str) -> None:
        frames = []
        for k, b in benches.items():
            s = b[["algorithm", metric]].copy()
            s = s.rename(columns={metric: f"k={k}"})
            frames.append(s.set_index("algorithm"))
        if not frames:
            return
        piv = pd.concat(frames, axis=1).reset_index()
        parts.append(f"**{label} je Verfahren x k:**\n\n" + _md(piv) + "\n")

    _pivot(STABILITY_COL, "Bootstrap-Stabilitat (mean ARI)")
    _pivot("silhouette", "Silhouette")


def _robustness_across_k(parts: list[str], ks: list[int]) -> None:
    parts.append("## 3. Robustheit uber k (mittlere Stabilitats-ARI unter Stoerung)\n")
    frames = []
    for k in ks:
        rob = _read(_kdir(k) / "tables" / "robustness.csv")
        if rob is None:
            continue
        valid = rob[~rob["degenerate"]] if "degenerate" in rob.columns else rob
        perturbed = valid[valid["level"] > 0]
        agg = perturbed.groupby("algorithm")["stability_ari"].mean()
        agg.name = f"k={k}"
        frames.append(agg)
    if frames:
        piv = pd.concat(frames, axis=1).reset_index()
        parts.append(
            "Aggregiert nur uber nicht-degenerierte Laeufe und Stoerstufen > 0:\n\n"
            + _md(piv) + "\n")
    else:
        parts.append("_Keine Robustheitsdaten gefunden._\n")


def _per_k_detail(parts: list[str], ks: list[int]) -> None:
    parts.append("## 4. Detailergebnisse je k\n")
    for k in ks:
        parts.append(f"### k = {k}\n")
        rel = f"k{k}/figures"
        b = _read(_kdir(k) / "tables" / "real_benchmark.csv")
        if b is not None:
            parts.append(_md(b) + "\n")
            best = _best_non_degenerate(b, STABILITY_COL)
            parts.append(
                f"Bestes nicht-degeneriertes Verfahren: `{best['algorithm']}` "
                f"(Stability ARI = {best[STABILITY_COL]:.3f}).\n")
            for metric in ["silhouette", "davies_bouldin",
                           "calinski_harabasz", "stability_mean_ari"]:
                parts.append(f"![{metric} k{k}]({rel}/real_03_{metric}.png)\n")
        parts.append(f"![PCA k{k}]({rel}/real_05_pca.png)\n")


def _k_independent_tracks(parts: list[str]) -> None:
    flat_tables = cfg.RESULTS_DIR / "tables"
    flat_figs = "figures"

    parts.append("## 5. k-unabhaengige Tracks\n")

    synth = _read(flat_tables / "synth_benchmark.csv")
    if synth is not None:
        parts.append("### 5.1 Synthetischer externer Track\n")
        parts.append(_md(synth) + "\n")
        for metric in ["ari", "nmi", "v_measure", "fmi"]:
            parts.append(f"![synth {metric}]({flat_figs}/synth_04_{metric}.png)\n")

    al = _read(flat_tables / "active_learning.csv")
    if al is not None:
        parts.append("### 5.2 Active Learning (explorativ)\n")
        last = al.sort_values(["strategy", "n_labeled"]).groupby("strategy").tail(1)
        parts.append("**Endstand pro Strategie:**\n\n" + _md(last) + "\n")
        parts.append(f"![AL Konvergenz]({flat_figs}/active_learning.png)\n")


def main() -> None:
    ks = _available_ks()
    parts: list[str] = ["# Analysebericht\n"]
    if not ks:
        parts.append("_Keine k-Ordner gefunden. Bitte zuerst 06_run_k_sweep.py ausfuehren._\n")
    else:
        parts.append(f"Gepruefte Clusteranzahlen: k in {{{', '.join(map(str, ks))}}}.\n")
        _k_selection_recap(parts, ks)
        _sensitivity_comparison(parts, ks)
        _robustness_across_k(parts, ks)
        _per_k_detail(parts, ks)
    _k_independent_tracks(parts)

    out = cfg.RESULTS_DIR / "analysis_report_ksweep.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"[report] wrote {out}  (k folders: {ks})")


if __name__ == "__main__":
    main()
