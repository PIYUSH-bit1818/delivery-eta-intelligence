"""Derive ML features from shipments, routes, and graph metrics."""

from typing import Optional

import pandas as pd


def temporal_features(
    df: pd.DataFrame,
    timestamp_col: str,
    *,
    prefix: str = "order",
) -> pd.DataFrame:
    """Add hour-of-day, day-of-week, and weekend flags."""
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col], utc=True)
    out[f"{prefix}_hour"] = ts.dt.hour
    out[f"{prefix}_dow"] = ts.dt.dayofweek
    out[f"{prefix}_is_weekend"] = (out[f"{prefix}_dow"] >= 5).astype(int)
    return out


def merge_graph_features(
    df: pd.DataFrame,
    node_metrics: pd.DataFrame,
    *,
    node_col: str,
    metric_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Join precomputed graph centrality or load metrics onto shipment rows."""
    cols = metric_cols or [c for c in node_metrics.columns if c != "node_id"]
    return df.merge(
        node_metrics.rename(columns={"node_id": node_col}),
        on=node_col,
        how="left",
    )
