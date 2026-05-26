from __future__ import annotations

import _paths

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cfg
from src import data_loading as dl
from src import features as ft
from src import preprocessing as pp
from src import synthetic as syn


def _trip_has_usable_gps(df: pd.DataFrame) -> bool:
    """Reject trips whose median speed is implausibly high (sustained GPS-jumps)."""
    if "speed_mps" not in df.columns:
        return True
    v = df["speed_mps"].to_numpy(dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return True  # no speed signal
    return float(np.median(v)) <= cfg.MAX_TRIP_MEDIAN_SPEED_MPS


def extract_source(root: Path, source: str, out_path: Path) -> pd.DataFrame:
    trips = dl.discover_trips(
        root, source,
        max_users=cfg.MAX_USERS,
        max_trips_per_user=cfg.MAX_TRIPS_PER_USER,
    )
    print(f"[{source}] {len(trips)} trips")
    spec = ft.WindowSpec(cfg.WINDOW_SECONDS, cfg.WINDOW_OVERLAP, cfg.TARGET_HZ)

    parts: list[pd.DataFrame] = []
    dropped_gps = 0
    for trip in tqdm(trips, desc=source):
        try:
            df = dl.load_trip(trip)
            df = pp.preprocess_trip(df, cfg.TARGET_HZ, cfg.LOWPASS_CUTOFF_HZ, cfg.LOWPASS_ORDER)
            if not _trip_has_usable_gps(df):
                dropped_gps += 1
                continue
            wf = ft.windowize_trip(df, spec, cfg.CADENCE_BAND, cfg.VIBRATION_BAND)
            if wf.empty:
                continue
            wf.insert(0, "trip_id", trip.trip_id)
            wf.insert(0, "user_id", trip.user_id)
            wf.insert(0, "source", trip.source)
            parts.append(wf)
        except Exception as exc:
            print(f"[warn] {trip.key}: {exc}")
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not out.empty:
        out.to_parquet(out_path, index=False)
    print(f"[{source}] wrote {len(out)} windows from {len(trips) - dropped_gps} "
          f"trips ({dropped_gps} dropped for bad GPS) -> {out_path}")
    return out


def extract_synth(out_path: Path) -> pd.DataFrame:
    spec = ft.WindowSpec(cfg.WINDOW_SECONDS, cfg.WINDOW_OVERLAP, cfg.TARGET_HZ)
    corpus = syn.generate_synthetic_corpus(
        cfg.SYNTH_N_USERS_PER_PERSONA,
        cfg.SYNTH_TRIPS_PER_USER,
        cfg.SYNTH_TRIP_MINUTES,
        cfg.TARGET_HZ,
    )
    parts: list[pd.DataFrame] = []
    for persona, user, trip, df in tqdm(corpus, desc="synth"):
        wf = ft.windowize_trip(df, spec, cfg.CADENCE_BAND, cfg.VIBRATION_BAND)
        if wf.empty:
            continue
        wf.insert(0, "persona", persona)
        wf.insert(0, "trip_id", trip)
        wf.insert(0, "user_id", user)
        wf.insert(0, "source", "synth")
        parts.append(wf)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not out.empty:
        out.to_parquet(out_path, index=False)
    print(f"[synth] wrote {len(out)} windows -> {out_path}")
    return out


def main() -> None:
    cfg.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    extract_source(cfg.REAL_ROOT, "real-world", cfg.FEATURES_DIR / "real_windows.parquet")
    extract_synth(cfg.FEATURES_DIR / "synth_windows.parquet")


if __name__ == "__main__":
    main()
