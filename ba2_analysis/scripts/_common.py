from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

META_COLS = {"source", "user_id", "trip_id", "persona", "stress",
             "window_start_s", "window_end_s"}


def load_windows(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in df.columns if c not in META_COLS]
    return df[feature_cols], df[[c for c in df.columns if c in META_COLS]]


def scale(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    X_filled = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler()
    return scaler.fit_transform(X_filled.to_numpy()), scaler
