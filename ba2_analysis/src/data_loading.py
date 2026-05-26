from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "unix Timestamp", "latitude", "longitude",
    "ACC X", "ACC Y", "ACC Z",
    "GYRO X", "GYRO Y", "GYRO Z",
    "Light",
    "Gravity X", "Gravity Y", "Gravity Z",
    "Orientation X", "Orientation Y", "Orientation Z",
    "MAG X", "MAG Y", "MAG Z",
    "Linear Acc. X", "Linear Acc. Y", "Linear Acc. Z",
    "Pressure",
]

NUMERIC_COLUMNS = [c for c in RAW_COLUMNS if c != "unix Timestamp"]


@dataclass(frozen=True)
class Trip:
    user_id: str
    trip_id: str
    source: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.source}/{self.user_id}/{self.trip_id}"


def discover_trips(
    root: Path,
    source: str,
    max_users: int | None = None,
    max_trips_per_user: int | None = None,
) -> list[Trip]:
    trips: list[Trip] = []
    users = sorted([p for p in root.iterdir() if p.is_dir()])
    if max_users is not None:
        users = users[:max_users]
    for user_dir in users:
        trip_dirs = sorted(
            [p for p in user_dir.iterdir() if p.is_dir()],
            key=lambda p: _as_int(p.name),
        )
        if max_trips_per_user is not None:
            trip_dirs = trip_dirs[:max_trips_per_user]
        for trip_dir in trip_dirs:
            csv = trip_dir / "bike_data.csv"
            if csv.exists():
                trips.append(Trip(user_dir.name, trip_dir.name, source, csv))
    return trips


def _as_int(name: str) -> int:
    try:
        return int(name)
    except ValueError:
        return 10**9


def load_trip(trip: Trip) -> pd.DataFrame:
    df = pd.read_csv(trip.path, na_values=["null", "NaN", ""], low_memory=False)
    df = df[[c for c in RAW_COLUMNS if c in df.columns]].copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["unix Timestamp"] = pd.to_numeric(df["unix Timestamp"], errors="coerce")
    df = df.dropna(subset=["unix Timestamp"]).reset_index(drop=True)
    df["t"] = (df["unix Timestamp"] - df["unix Timestamp"].iloc[0]) / 1000.0
    return df


def load_stress_labels(trip: Trip) -> pd.Series | None:
    audio = sorted(trip.path.parent.glob("bike_audio_*.csv"))
    if not audio:
        return None
    try:
        labels = pd.read_csv(audio[0])
        col = labels.columns[0]
        return pd.to_numeric(labels[col], errors="coerce").dropna().reset_index(drop=True)
    except Exception:
        return None


def iter_trips(trips: Iterable[Trip]) -> Iterator[tuple[Trip, pd.DataFrame]]:
    for trip in trips:
        try:
            yield trip, load_trip(trip)
        except Exception as exc:
            print(f"[warn] failed to load {trip.key}: {exc}")
            continue
