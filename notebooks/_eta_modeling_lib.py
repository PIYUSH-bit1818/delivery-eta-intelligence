"""Backward-compatible re-exports for notebooks 04+."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.graph_features import build_lane_graph_tables, merge_hub_features


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def metrics_row(name: str, y_true, y_pred) -> dict:
    return {"Model": name, **regression_metrics(y_true, y_pred)}
