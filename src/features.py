"""ML feature generation for ETA prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical feature groups (aligned with notebook 04)
GRAPH_FEATURE_NAMES = [
    "source_pagerank",
    "destination_pagerank",
    "source_degree",
    "destination_degree",
    "source_bottleneck_score",
    "destination_bottleneck_score",
    "source_community",
    "destination_community",
    "graph_pagerank_product",
    "graph_bottleneck_sum",
    "graph_pagerank_gap",
    "same_community",
]

BASE_FEATURE_NAMES = [
    "osrm_time",
    "osrm_distance",
    "segment_osrm_time",
    "segment_osrm_distance",
    "trip_hour_sin",
    "trip_hour_cos",
    "is_weekend",
    "segment_time_share",
    "segment_distance_share",
    "segment_osrm_to_trip_osrm_time",
]


def infer_route_dummy_columns(columns: pd.Index) -> list[str]:
    return sorted(c for c in columns if str(c).startswith("route_type_"))


def get_all_feature_names(columns: pd.Index) -> list[str]:
    return BASE_FEATURE_NAMES + infer_route_dummy_columns(columns) + GRAPH_FEATURE_NAMES


def add_temporal_features(df: pd.DataFrame, hour_col: str = "trip_hour") -> pd.DataFrame:
    out = df.copy()
    if hour_col not in out.columns:
        raise KeyError(f"Missing {hour_col}")
    out["trip_hour_sin"] = np.sin(2 * np.pi * out[hour_col] / 24)
    out["trip_hour_cos"] = np.cos(2 * np.pi * out[hour_col] / 24)
    if "trip_dayofweek" in out.columns:
        out["is_weekend"] = (out["trip_dayofweek"] >= 5).astype(int)
    elif "is_weekend" not in out.columns:
        out["is_weekend"] = 0
    return out


def add_route_type_dummies(df: pd.DataFrame, route_col: str = "route_type") -> pd.DataFrame:
    out = df.copy()
    dummies = pd.get_dummies(out[route_col].astype(str), prefix="route_type", dtype=float)
    return pd.concat([out, dummies], axis=1)


def add_osrm_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """OSRM-only ratios (no post-hoc actuals)."""
    out = df.copy()
    out["segment_time_share"] = out["segment_osrm_time"] / out["osrm_time"].replace(0, np.nan)
    out["segment_distance_share"] = out["segment_osrm_distance"] / out["osrm_distance"].replace(0, np.nan)
    out["segment_osrm_to_trip_osrm_time"] = out["segment_osrm_time"] / out["osrm_time"].replace(0, np.nan)
    return out


def add_graph_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["graph_pagerank_product"] = out["source_pagerank"] * out["destination_pagerank"]
    out["graph_bottleneck_sum"] = out["source_bottleneck_score"] + out["destination_bottleneck_score"]
    out["graph_pagerank_gap"] = (out["source_pagerank"] - out["destination_pagerank"]).abs()
    out["same_community"] = (out["source_community"] == out["destination_community"]).astype(float)
    return out


def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature matrix columns for training or batch inference."""
    out = add_temporal_features(df)
    out = add_route_type_dummies(out)
    out = add_osrm_ratio_features(out)
    out = add_graph_interaction_features(out)
    return out


def expand_minimal_request(
    df: pd.DataFrame,
    *,
    osrm_minutes_per_km: float,
    default_trip_dayofweek: int = 2,
) -> pd.DataFrame:
    """
    Expand minimal inference inputs (distance, hour, route) to full OSRM fields.

    When only osrm_distance is provided, estimate osrm_time and treat trip as single-segment.
    """
    out = df.copy()
    if "osrm_time" not in out.columns or out["osrm_time"].isna().any():
        out["osrm_time"] = out["osrm_distance"] * osrm_minutes_per_km
    if "segment_osrm_time" not in out.columns:
        out["segment_osrm_time"] = out["osrm_time"]
    if "segment_osrm_distance" not in out.columns:
        out["segment_osrm_distance"] = out["osrm_distance"]
    if "trip_dayofweek" not in out.columns:
        out["trip_dayofweek"] = default_trip_dayofweek
    return out
