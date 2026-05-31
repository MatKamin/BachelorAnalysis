# BA2 Analyse

## Setup

```bash
cd ba2_analysis
python -m pip install -r requirements.txt
```

## Ausfuehrung

```bash
python scripts/run_all.py

python scripts/01_extract_features.py
python scripts/02_run_benchmark.py
python scripts/03_run_robustness.py
python scripts/04_run_active_learning.py
python scripts/05_generate_report.py
python scripts/06_run_k_sweep.py
python scripts/07_generate_report_ksweep.py
```

`config.py` steuert Subsampling (`MAX_TRIPS_PER_USER`, `MAX_USERS`),
Window-Parameter, Augmentation-Levels und AL-Budgets. Standard ist eine
schnelle Subset-Konfiguration; fuer den finalen Lauf `MAX_TRIPS_PER_USER = None`
setzen.

## K-Sweep

`scripts/06_run_k_sweep.py` wiederholt Benchmark und Robustheit fuer jede
feste Clusterzahl aus `K_VALUES = [5, 6, 7, 8]` (definiert in `06_run_k_sweep.py`
bzw. `07_generate_report_ksweep.py`). Pro Wert entsteht ein eigener
Ergebnisordner `results/k5/`, `results/k6/`, `results/k7/`, `results/k8/`.
`scripts/07_generate_report_ksweep.py` fasst die k-Werte vergleichend zusammen.

## Outputs

- `results/features/` - extrahierte Window-Features (Parquet)
- `results/tables/`   - alle Metriken (CSV)
- `results/figures/`  - alle Plots (PNG)
- `results/analysis_report.md` - synthetisierter Bericht
- `results/k5/` ... `results/k8/` - Tabellen und Plots je Clusterzahl (K-Sweep)
- `results/analysis_report_ksweep.md` - vergleichender K-Sweep-Bericht

## Datenquelle

C. Wang, "Cyclist Stress Dataset," IEEE DataPort, May 2025, doi: https://doi.org/10.21227/hsnj-3k48.