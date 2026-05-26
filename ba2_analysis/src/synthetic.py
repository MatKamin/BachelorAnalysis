
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import preprocessing as pp


PERSONAS: dict[str, dict[str, float]] = {
    "P01_Commuter":       {"speed_avg": 19, "speed_std": 5.0, "stops_avg": 0.10, "vib_avg": 1.0, "gyro_scale": 1.0},
    "P02_Sport_Road":     {"speed_avg": 32, "speed_std": 6.0, "stops_avg": 0.02, "vib_avg": 0.9, "gyro_scale": 1.5},
    "P03_Occasional":     {"speed_avg": 15, "speed_std": 5.5, "stops_avg": 0.15, "vib_avg": 1.0, "gyro_scale": 0.8},
    "P04_MTB":            {"speed_avg": 14, "speed_std": 8.0, "stops_avg": 0.05, "vib_avg": 2.5, "gyro_scale": 2.0},
    "P05_E_Bike":         {"speed_avg": 24, "speed_std": 4.0, "stops_avg": 0.05, "vib_avg": 0.9, "gyro_scale": 0.9},
    "P06_Urban_LastMile": {"speed_avg": 13, "speed_std": 5.0, "stops_avg": 0.45, "vib_avg": 1.1, "gyro_scale": 1.2},
    "P07_Safety":         {"speed_avg": 9,  "speed_std": 4.0, "stops_avg": 0.10, "vib_avg": 0.9, "gyro_scale": 0.5},
    "P08_Fair_Weather":   {"speed_avg": 20, "speed_std": 5.0, "stops_avg": 0.05, "vib_avg": 1.0, "gyro_scale": 1.0},
    "P09_Night_Rider":    {"speed_avg": 22, "speed_std": 5.0, "stops_avg": 0.05, "vib_avg": 1.0, "gyro_scale": 1.0},
    "P10_Strava_Hunter":  {"speed_avg": 34, "speed_std": 10.0, "stops_avg": 0.02, "vib_avg": 1.0, "gyro_scale": 2.5},
}

WAYPOINT_SECONDS = 5
KMH_TO_MPS = 1.0 / 3.6


@dataclass
class TripMood:
    stop_prob: float
    vib_base: float
    gyro_scale: float
    speed_mood: float
    speed_bias: float


def _trip_mood(traits: dict[str, float], rng: np.random.Generator) -> TripMood:
    return TripMood(
        stop_prob=float(np.clip(rng.normal(traits["stops_avg"], 0.15), 0.0, 0.9)),
        vib_base=float(np.clip(rng.normal(traits["vib_avg"], 0.3), 0.1, 4.0)),
        gyro_scale=float(np.clip(rng.normal(traits["gyro_scale"], 0.4), 0.1, 4.0)),
        speed_mood=float(rng.normal(1.0, 0.15)),
        speed_bias=float(rng.normal(1.0, 0.25)),
    )


def _waypoints(persona: str, traits: dict[str, float], mood: TripMood,
               duration_s: int, rng: np.random.Generator) -> pd.DataFrame:
    n_wp = max(1, duration_s // WAYPOINT_SECONDS)
    rows = []
    for _ in range(n_wp):
        if rng.random() < mood.stop_prob:
            speed_kmh = 0.0
        else:
            base = traits["speed_avg"] * mood.speed_bias * mood.speed_mood
            speed_kmh = max(0.0, float(rng.normal(base, traits["speed_std"])))
        rows.append({
            "speed_kmh": speed_kmh,
            "vib": mood.vib_base,
            "gyro_scale": mood.gyro_scale,
        })
    return pd.DataFrame(rows)


def _render_50hz(waypoints: pd.DataFrame, fs: float,
                 rng: np.random.Generator) -> pd.DataFrame:
    samples_per_wp = int(round(WAYPOINT_SECONDS * fs))
    n = len(waypoints) * samples_per_wp
    t = np.arange(n) / fs

    speed_mps = np.repeat(waypoints["speed_kmh"].to_numpy() * KMH_TO_MPS, samples_per_wp)
    vib = np.repeat(waypoints["vib"].to_numpy(), samples_per_wp)
    gyro_scale = np.repeat(waypoints["gyro_scale"].to_numpy(), samples_per_wp)

    # Smooth speed transitions (no instantaneous jumps between waypoints).
    speed_mps = pd.Series(speed_mps).rolling(int(fs), min_periods=1).mean().to_numpy()

    # Cadence proxy: pedal cycle ~ 1 Hz at low speed, ~ 2 Hz at high speed.
    cadence_hz = 0.8 + 0.04 * speed_mps
    phase = 2 * np.pi * np.cumsum(cadence_hz) / fs

    lin_x = vib * 0.05 * np.sin(phase) + rng.normal(0, vib * 0.05, n)
    lin_y = vib * 0.05 * np.cos(phase) + rng.normal(0, vib * 0.05, n)
    lin_z = vib * 0.1 * rng.normal(0, 1, n)

    gyro_std = 0.1 * gyro_scale
    gyro_x = rng.normal(0, gyro_std, n)
    gyro_y = rng.normal(0, gyro_std, n)
    gyro_z = rng.normal(0, gyro_std * 0.5, n)

    grav = np.full(n, 9.81)
    df = pd.DataFrame({
        "t": t,
        "unix Timestamp": (t * 1000).astype(np.int64),
        "latitude": np.nan, "longitude": np.nan,
        "ACC X": lin_x, "ACC Y": lin_y, "ACC Z": lin_z + grav,
        "GYRO X": gyro_x, "GYRO Y": gyro_y, "GYRO Z": gyro_z,
        "Light": rng.normal(5000, 500, n),
        "Gravity X": np.zeros(n), "Gravity Y": np.zeros(n), "Gravity Z": grav,
        "Orientation X": np.zeros(n), "Orientation Y": np.zeros(n), "Orientation Z": np.zeros(n),
        "MAG X": rng.normal(-50, 5, n), "MAG Y": rng.normal(50, 5, n), "MAG Z": rng.normal(-100, 5, n),
        "Linear Acc. X": lin_x, "Linear Acc. Y": lin_y, "Linear Acc. Z": lin_z,
        "Pressure": rng.normal(1000, 1, n),
        "speed_mps": speed_mps,
    })
    df = pp.derive_motion_acc(df)
    return df


def generate_synthetic_corpus(n_users_per_persona: int, trips_per_user: int,
                              minutes_per_trip: float, fs: float,
                              seed: int = 42) -> list[tuple[str, str, str, pd.DataFrame]]:
    rng = np.random.default_rng(seed)
    duration_s = int(round(minutes_per_trip * 60))
    out: list[tuple[str, str, str, pd.DataFrame]] = []
    for persona, traits in PERSONAS.items():
        for u in range(n_users_per_persona):
            user_id = f"{persona}_u{u:02d}"
            for trip_i in range(trips_per_user):
                mood = _trip_mood(traits, rng)
                wp = _waypoints(persona, traits, mood, duration_s, rng)
                df = _render_50hz(wp, fs, rng)
                out.append((persona, user_id, f"trip{trip_i}", df))
    return out


def persona_names() -> list[str]:
    return list(PERSONAS.keys())
