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

## Outputs

- `results/features/` - extrahierte Window-Features (Parquet)
- `results/tables/`   - alle Metriken (CSV)
- `results/figures/`  - alle Plots (PNG)
- `results/analysis_report.md` - synthetisierter Bericht

## Datenquelle

C. Wang, "Cyclist Stress Dataset," IEEE DataPort, May 2025, doi: https://doi.org/10.21227/hsnj-3k48.