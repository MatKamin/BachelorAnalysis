"""Run the full pipeline end-to-end."""
from __future__ import annotations

import _paths

import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

PIPELINE = [
    "01_extract_features.py",
    "02_run_benchmark.py",
    "03_run_robustness.py",
    "04_run_active_learning.py",
    "05_generate_report.py",
]


def main() -> None:
    for name in PIPELINE:
        py_path = SCRIPTS_DIR / name
        print(f"\n=== {name} ===")
        t0 = time.time()
        with open(py_path, "r", encoding="utf-8") as f:
            code = compile(f.read(), str(py_path), "exec")
        exec(code, {"__name__": "__main__", "__file__": str(py_path)})
        print(f"=== {name} done in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
