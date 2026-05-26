from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew

FEATURE_SIGNALS = ["linacc_mag", "gyro_mag", "speed_mps"]
STOP_SPEED_THRESHOLD = 0.5


@dataclass
class WindowSpec:
    seconds: float
    overlap: float
    fs: float

    @property
    def size(self) -> int:
        return max(2, int(round(self.seconds * self.fs)))

    @property
    def step(self) -> int:
        return max(1, int(round(self.size * (1.0 - self.overlap))))


def _safe_stats(x: np.ndarray) -> dict[str, float]:
    if x.size == 0 or np.all(np.isnan(x)):
        return {"mean": 0.0, "std": 0.0, "var": 0.0, "min": 0.0, "max": 0.0,
                "skew": 0.0, "kurt": 0.0, "rms": 0.0, "iqr": 0.0}
    x = x[~np.isnan(x)]
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "var": float(np.var(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "skew": float(skew(x)) if x.size > 2 else 0.0,
        "kurt": float(kurtosis(x)) if x.size > 3 else 0.0,
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
    }


def _zero_crossing_rate(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    x = x - np.nanmean(x)
    s = np.sign(x)
    return float(np.mean(np.abs(np.diff(s)) > 0))


def _signal_magnitude_area(triple: np.ndarray) -> float:
    if triple.size == 0:
        return 0.0
    return float(np.mean(np.sum(np.abs(triple), axis=1)))


def _spectral_features(x: np.ndarray, fs: float, cadence_band: tuple[float, float],
                       vibration_band: tuple[float, float]) -> dict[str, float]:
    if x.size < 8 or np.all(np.isnan(x)):
        return {"dom_freq": 0.0, "dom_power": 0.0, "cadence_energy": 0.0,
                "vibration_energy": 0.0, "spectral_entropy": 0.0,
                "total_energy": 0.0}
    x = np.nan_to_num(x - np.nanmean(x))
    nperseg = min(len(x), 64)
    freqs, psd = welch(x, fs=fs, nperseg=nperseg)
    psd = np.clip(psd, 1e-12, None)
    total = float(np.sum(psd))
    p_norm = psd / total
    dom_idx = int(np.argmax(psd))
    cb = (freqs >= cadence_band[0]) & (freqs <= cadence_band[1])
    vb = (freqs >= vibration_band[0]) & (freqs <= vibration_band[1])
    return {
        "dom_freq": float(freqs[dom_idx]),
        "dom_power": float(psd[dom_idx]),
        "cadence_energy": float(np.sum(psd[cb])),
        "vibration_energy": float(np.sum(psd[vb])),
        "spectral_entropy": float(-np.sum(p_norm * np.log(p_norm))),
        "total_energy": total,
    }


def extract_window_features(window: pd.DataFrame, spec: WindowSpec,
                            cadence_band: tuple[float, float],
                            vibration_band: tuple[float, float]) -> dict[str, float]:
    feats: dict[str, float] = {}
    for sig in FEATURE_SIGNALS:
        if sig not in window.columns:
            continue
        arr = window[sig].to_numpy(dtype=float)
        for k, v in _safe_stats(arr).items():
            feats[f"{sig}_{k}"] = v
        feats[f"{sig}_zcr"] = _zero_crossing_rate(arr)
        for k, v in _spectral_features(arr, spec.fs, cadence_band, vibration_band).items():
            feats[f"{sig}_{k}"] = v

    for prefix, cols in [("acc", ("ACC X", "ACC Y", "ACC Z")),
                         ("gyro", ("GYRO X", "GYRO Y", "GYRO Z"))]:
        if all(c in window.columns for c in cols):
            triple = window[list(cols)].to_numpy(dtype=float)
            triple = np.nan_to_num(triple)
            feats[f"{prefix}_sma"] = _signal_magnitude_area(triple)

    if "speed_mps" in window.columns:
        v = window["speed_mps"].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        if v.size > 0:
            stops = (v < STOP_SPEED_THRESHOLD).astype(int)
            transitions = np.abs(np.diff(stops)).sum() if len(stops) > 1 else 0
            feats["stop_ratio"] = float(np.mean(stops))
            feats["stop_transitions"] = float(transitions)
        else:
            feats["stop_ratio"] = 0.0
            feats["stop_transitions"] = 0.0

    return feats


def windowize_trip(df: pd.DataFrame, spec: WindowSpec,
                   cadence_band: tuple[float, float],
                   vibration_band: tuple[float, float]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    n = len(df)
    rows = []
    for start in range(0, n - spec.size + 1, spec.step):
        win = df.iloc[start:start + spec.size]
        feats = extract_window_features(win, spec, cadence_band, vibration_band)
        feats["window_start_s"] = float(win["t"].iloc[0])
        feats["window_end_s"] = float(win["t"].iloc[-1])
        rows.append(feats)
    return pd.DataFrame(rows)
