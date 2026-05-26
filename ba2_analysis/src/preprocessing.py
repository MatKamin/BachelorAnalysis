from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

IMU_TRIPLETS = {
    "acc": ("ACC X", "ACC Y", "ACC Z"),
    "gyro": ("GYRO X", "GYRO Y", "GYRO Z"),
    "linacc": ("Linear Acc. X", "Linear Acc. Y", "Linear Acc. Z"),
    "grav": ("Gravity X", "Gravity Y", "Gravity Z"),
    "mag": ("MAG X", "MAG Y", "MAG Z"),
}


def resample_uniform(df: pd.DataFrame, target_hz: float) -> pd.DataFrame:
    if df.empty:
        return df
    dt = 1.0 / target_hz
    t0, t1 = df["t"].iloc[0], df["t"].iloc[-1]
    grid = np.arange(t0, t1 + dt, dt)
    out = pd.DataFrame({"t": grid})
    src_t = df["t"].to_numpy()
    for col in df.columns:
        if col == "t":
            continue
        y = df[col].to_numpy(dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            out[col] = np.nan
            continue
        out[col] = np.interp(grid, src_t[mask], y[mask], left=np.nan, right=np.nan)
    return out


def butter_lowpass(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    if len(data) < order * 3:
        return data
    nyq = 0.5 * fs
    wn = min(0.99, cutoff / nyq)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, data, axis=0)


def apply_lowpass(df: pd.DataFrame, fs: float, cutoff: float, order: int = 4) -> pd.DataFrame:
    out = df.copy()
    for cols in IMU_TRIPLETS.values():
        for c in cols:
            if c in out.columns:
                y = out[c].to_numpy(dtype=float)
                if np.isnan(y).all():
                    continue
                y = pd.Series(y).interpolate(limit_direction="both").to_numpy()
                out[c] = butter_lowpass(y, cutoff, fs, order)
    return out


def derive_motion_acc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if all(c in out.columns for c in IMU_TRIPLETS["linacc"]):
        lx, ly, lz = IMU_TRIPLETS["linacc"]
        out["linacc_mag"] = np.sqrt(out[lx] ** 2 + out[ly] ** 2 + out[lz] ** 2)
    elif all(c in out.columns for c in IMU_TRIPLETS["acc"]):
        ax, ay, az = IMU_TRIPLETS["acc"]
        if all(c in out.columns for c in IMU_TRIPLETS["grav"]):
            gx, gy, gz = IMU_TRIPLETS["grav"]
            out["linacc_mag"] = np.sqrt(
                (out[ax] - out[gx]) ** 2
                + (out[ay] - out[gy]) ** 2
                + (out[az] - out[gz]) ** 2
            )
        else:
            mag = np.sqrt(out[ax] ** 2 + out[ay] ** 2 + out[az] ** 2)
            out["linacc_mag"] = mag - mag.rolling(50, min_periods=1).mean()
    gx, gy, gz = IMU_TRIPLETS["gyro"]
    if all(c in out.columns for c in IMU_TRIPLETS["gyro"]):
        out["gyro_mag"] = np.sqrt(out[gx] ** 2 + out[gy] ** 2 + out[gz] ** 2)
    return out


MAX_SPEED_MPS = 25.0


def compute_speed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "latitude" not in out.columns or "longitude" not in out.columns:
        out["speed_mps"] = np.nan
        return out
    lat = out["latitude"].to_numpy(dtype=float)
    lon = out["longitude"].to_numpy(dtype=float)
    t = out["t"].to_numpy(dtype=float)
    if np.isnan(lat).all() or np.isnan(lon).all():
        out["speed_mps"] = np.nan
        return out

    valid = ~np.isnan(lat) & ~np.isnan(lon)
    if valid.sum() < 2:
        out["speed_mps"] = 0.0
        return out
    lat_v, lon_v, t_v = lat[valid], lon[valid], t[valid]

    changed = np.concatenate(
        ([True], (np.diff(lat_v) != 0) | (np.diff(lon_v) != 0))
    )
    lat_u, lon_u, t_u = lat_v[changed], lon_v[changed], t_v[changed]
    if len(t_u) < 2:
        out["speed_mps"] = 0.0
        return out

    R = 6_371_000.0
    lat1, lat2 = np.radians(lat_u[:-1]), np.radians(lat_u[1:])
    dlat = lat2 - lat1
    dlon = np.radians(lon_u[1:] - lon_u[:-1])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    d = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    dt = np.diff(t_u)
    dt[dt <= 0] = np.nan
    v = d / dt
    v[v < 0] = 0.0
    v[v > MAX_SPEED_MPS] = np.nan

    seg_idx = np.searchsorted(t_u, t, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(v) - 1)
    full = v[seg_idx]
    full = pd.Series(full).rolling(50, min_periods=1).mean().to_numpy()
    out["speed_mps"] = full
    return out


def preprocess_trip(df: pd.DataFrame, target_hz: float, cutoff: float, order: int) -> pd.DataFrame:
    df = resample_uniform(df, target_hz)
    df = apply_lowpass(df, target_hz, cutoff, order)
    df = derive_motion_acc(df)
    df = compute_speed(df)
    return df
