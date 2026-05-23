"""Evaluate ETA model and graph analytics outputs."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics for ETA predictions."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def metrics_to_frame(metrics: dict[str, float], *, model_name: str = "model") -> pd.DataFrame:
    """Format metrics dict as a single-row DataFrame for logging."""
    row = {"model": model_name, **metrics}
    return pd.DataFrame([row])
