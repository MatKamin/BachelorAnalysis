from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
for p in (str(SCRIPTS_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")

import config as cfg
from _common import load_windows

K_VALUES = [5, 6, 7, 8]


def _load_script_module(name: str, filename: str):
    """Load a digit-prefixed pipeline script as an importable module.

    __name__ is set to the module name (not '__main__'), so the script's
    main() does not auto-execute on import.
    """
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    bench = _load_script_module("bench02", "02_run_benchmark.py")
    robust = _load_script_module("robust03", "03_run_robustness.py")

    real_df = load_windows(cfg.FEATURES_DIR / "real_windows.parquet")
    print(f"[ksweep] real windows = {len(real_df)}")

    original_choose_k = bench._choose_k
    orig_tables_dir = cfg.TABLES_DIR
    orig_figures_dir = cfg.FIGURES_DIR

    for k in K_VALUES:
        k_dir = cfg.RESULTS_DIR / f"k{k}"
        tables_dir = k_dir / "tables"
        figures_dir = k_dir / "figures"
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        cfg.TABLES_DIR = tables_dir
        cfg.FIGURES_DIR = figures_dir

        def _forced_choose_k(km, gmm, _k=k):
            chosen = original_choose_k(km, gmm)
            chosen["k_used"] = _k
            return chosen

        bench._choose_k = _forced_choose_k

        print(f"\n========== k = {k} ==========")
        print(f"[ksweep] real benchmark -> {tables_dir}")
        bench.run_real_track(real_df, "real")

        print(f"[ksweep] robustness (reads k from real_best_choice.csv) -> {tables_dir}")
        robust.main()

    bench._choose_k = original_choose_k
    cfg.TABLES_DIR = orig_tables_dir
    cfg.FIGURES_DIR = orig_figures_dir
    print("\n[ksweep] done.")


if __name__ == "__main__":
    main()
