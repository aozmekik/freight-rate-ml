"""Shared feature engineering for the freight rate model.

Every dataset (train, validation, December chart inputs) goes through the
same `build_features` function so train/serve skew is impossible by
construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The raw route string (~4,000 values) was tested and dropped: it slightly
# overfit on the time-based holdout vs. city + coordinate features alone.
CATEGORICAL_FEATURES = ["equipment", "pickup", "delivery", "dow"]

NUMERIC_FEATURES = [
    "distance",
    "log_distance",
    "weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "market_index",
    "quote_signal",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "posted_rate"


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame; leaves NaN in place (LightGBM handles them)."""
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["route"] = out["pickup"] + "->" + out["delivery"]
    out["log_distance"] = np.log(out["distance"].clip(lower=1.0))
    out["dow"] = out["date"].dt.dayofweek
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("category")
    return out


def clean_training(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the ~0.3% of training rows whose rate-per-mile is physically
    implausible for truckload freight (data-entry / unit errors)."""
    rpm = frame[TARGET] / frame["distance"]
    keep = rpm.between(0.5, 10.0)
    return frame[keep].reset_index(drop=True)


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """City -> (lat, lon) lookup built from every frame that has coordinates.

    Coordinates are constant per city in this data, so a first-seen lookup is
    exact for known cities.
    """
    rows = []
    for frame in frames:
        if {"pickup", "pickup_lat", "pickup_lon"} <= set(frame.columns):
            rows.append(
                frame[["pickup", "pickup_lat", "pickup_lon"]].rename(
                    columns={
                        "pickup": "city",
                        "pickup_lat": "lat",
                        "pickup_lon": "lon",
                    }
                )
            )
        if {"delivery", "delivery_lat", "delivery_lon"} <= set(frame.columns):
            rows.append(
                frame[["delivery", "delivery_lat", "delivery_lon"]].rename(
                    columns={
                        "delivery": "city",
                        "delivery_lat": "lat",
                        "delivery_lon": "lon",
                    }
                )
            )
    lookup = pd.concat(rows, ignore_index=True).dropna()
    return lookup.drop_duplicates(subset="city").set_index("city")
